# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The evaluator service and its client.

Two of the three FIRM dimensions call a frozen evaluator, and neither can be read off the
trainer's own reference forward: that scores the rollout's own tokens under the rollout's own
prompt, whereas both of these ask a NEW question -- "is this supported?", "correct this
memory" -- under a different prompt. So both are forwards against a separate service.

Launch it with plain vLLM, one or more replicas::

    vllm serve $MODEL --served-model-name judge --port 8100 \\
        --allowed-local-media-path / --limit-mm-per-prompt image=8
    export MEMLIFE_JUDGE_ENDPOINTS=http://host1:8100,http://host2:8100

The service is deliberately a separate process from the trainer. Two vLLM builds' kernels can
move a probability by ~5e-3 while either reproduces *itself* to ~1e-8, so sharing a process
would couple reward values to the trainer's dependency pins for no benefit. Nothing is shared
but HTTP.

Two traps this module closes, both silent rather than loud:

**Thinking is asserted off, not merely requested.** With thinking on, token position 0 is the
opening tag and every probability read from it is garbage -- measured as a -0.466 mean shift in
faithfulness, with every row disagreeing with the offline reference. The check lives in the one
function that posts, so no body can reach the wire without it.

**Frame URLs are built with ``Path.as_uri()``.** A segment id contains ``#``, which is the URL
fragment delimiter, so ``file:///.../vid#00042/0.jpg`` parses as the directory ``/.../vid`` --
a path that EXISTS. The failure is a wrong image or a uniformly failing evaluator, never an
exception. ``as_uri()`` percent-encodes it.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from ..core import llm as mllm
from ..core import sampling as sp

#: Comma-separated base URLs, or a directory holding one ``<rank>.txt`` per replica. No
#: default: a reward silently scoring against nothing is worse than a run that refuses to start.
ENDPOINT_ENV = "MEMLIFE_JUDGE_ENDPOINTS"
#: ``--served-model-name`` on the replicas. A name, not a path, so the client does not have to
#: agree with the launcher about where the weights live.
MODEL_ENV = "MEMLIFE_JUDGE_MODEL"
SERVED_MODEL = "judge"

#: Frames per segment on this service. Must equal ``memory.frames_per_segment``: lower here and
#: every call is rejected, higher there and the evaluator silently scores a SUBSET of the
#: pixels the writer saw. :func:`memlife.config._check` pins the two together.
N_FRAMES = 8

#: Concurrent requests per :func:`post_batch`, across all replicas. Right for one-token reads;
#: see :mod:`memlife.memopt.faithfulness` for why the audit must pass a much smaller value.
MAX_INFLIGHT = 64
#: Generous on purpose: a request queued behind a step's work is normal; one that has actually
#: died is caught by the retry loop, not by this.
TIMEOUT_S = 600.0
#: Retries ROTATE across replicas, because a dead engine stays dead.
RETRIES = 6

_YESNO = sp.profile("qwen3.5", "judge_yesno")

_lock = threading.Lock()
_endpoints: list[str] | None = None
_tokenize_cache: dict[str, list[int]] = {}


def served_model() -> str:
    return os.environ.get(MODEL_ENV) or SERVED_MODEL


def endpoints() -> list[str]:
    """Base URLs, one per replica.

    Resolved ONCE per process: a replica that dies mid-run must surface as a request error,
    not as a silently shrinking pool, and one that arrives late must not change which replicas
    a step is scored against halfway through it.
    """
    global _endpoints
    with _lock:
        if _endpoints is None:
            raw = (os.environ.get(ENDPOINT_ENV) or "").strip()
            if not raw:
                raise RuntimeError(
                    f"{ENDPOINT_ENV} is unset -- the faithfulness and informativeness rewards "
                    f"both need an evaluator service. Set it to a comma-separated list of "
                    f"base URLs, or to a directory holding one <rank>.txt per replica.")
            if Path(raw).is_dir():
                eps = [p.read_text().strip() for p in sorted(Path(raw).glob("*.txt"))]
            else:
                eps = [u.strip() for u in raw.split(",") if u.strip()]
            if bad := [u for u in eps if not u.startswith("http")]:
                raise RuntimeError(f"{ENDPOINT_ENV}: {bad} are not http(s) URLs")
            if not eps:
                raise RuntimeError(f"{ENDPOINT_ENV}={raw!r} resolved to no endpoints")
            _endpoints = eps
            print(f"[judge] {len(eps)} endpoint(s): {eps}", flush=True)
    return _endpoints


# ==================================================================================== media

def frame_url(path: str | Path) -> str:
    """A ``file://`` URL for one frame. Never an f-string -- see the module docstring."""
    p = Path(path)
    if not p.is_absolute():
        raise ValueError(
            f"frame path must be absolute, got {p}. `file://relative/x.jpg` parses `relative` "
            f"as the URL AUTHORITY, so the server resolves against the wrong root -- and "
            f"because it would fail for every segment alike, the dimension would read as flat "
            f"rather than as an error.")
    return p.as_uri()


def image_part(path: str | Path) -> dict:
    return {"type": "image_url", "image_url": {"url": frame_url(path)}}


def segment_frames(segment_id: str, frames_root: str | Path) -> list[dict]:
    """The image parts for one segment, from the extracted frame tree.

    One function, so the faithfulness audit and anything else that looks at pixels cannot
    address a segment differently. The files are deliberately NOT stat'd: eight stats per
    rollout is real filesystem latency, and the server's error names the path it could not read.
    """
    d = Path(frames_root) / segment_id
    return [image_part(d / f"{i}.jpg") for i in range(N_FRAMES)]


# ==================================================================================== transport

def _session():
    """A ``requests.Session`` that ignores the environment's proxy settings.

    Clusters commonly export a proxy that cannot route to compute nodes; ``requests`` picks it
    up automatically and every call dies with a proxy error.
    """
    import requests

    s = requests.Session()
    s.trust_env = False
    return s


def check_thinking(body: dict) -> None:
    """Refuse any body that has not switched thinking off. See the module docstring."""
    if (body.get("chat_template_kwargs") or {}).get("enable_thinking") is not False:
        raise ValueError(
            "every request to the evaluator must carry "
            "chat_template_kwargs={'enable_thinking': False}. With thinking on, position 0 is "
            "the opening tag and both the Yes/No logprobs and the audit's prompt logprobs are "
            f"read at the wrong position. Got {body.get('chat_template_kwargs')!r}.")


def post_batch(bodies: Sequence[dict], *, path: str = "/v1/chat/completions",
               max_inflight: int | None = None) -> list[dict]:
    """POST each body, fanned out over the replicas.

    Retries ROTATE to the next replica rather than repeating the same URL, and the first
    attempts do not back off -- a dead engine fails instantly, so sleeping there would cost a
    second on every call unlucky enough to be routed at it.

    Raises if a body fails everywhere it was tried. Returning ``None`` instead would let a dead
    evaluator quietly rewrite the reward; raising kills the step, and the trainer resumes from
    its last checkpoint.
    """
    for b in bodies:
        check_thinking(b)
    eps = endpoints()
    out: list[Any] = [None] * len(bodies)
    local = threading.local()

    def one(i: int) -> None:
        if not hasattr(local, "s"):
            local.s = _session()
        s = local.s
        last, url, attempt = "not attempted", "", -1
        for attempt in range(RETRIES):
            url = eps[(i + attempt) % len(eps)] + path
            try:
                r = s.post(url, json=bodies[i], timeout=TIMEOUT_S)
            except Exception as e:                  # noqa: BLE001 -- transport, retry anything
                last = repr(e)
            else:
                if r.status_code < 300:
                    out[i] = r.json()
                    return
                # The BODY, not just the status line: vLLM puts the reason there -- a frame
                # outside --allowed-local-media-path, a field this build rejects -- and
                # raise_for_status throws it away.
                last = f"HTTP {r.status_code}: {r.text[:400]!r}"
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    break          # a malformed request is malformed on every replica
            if 2 <= attempt < RETRIES - 1:
                time.sleep(min(2 ** (attempt - 2), 15))
        raise RuntimeError(
            f"evaluator request {i} failed after {attempt + 1} attempt(s) over "
            f"{min(attempt + 1, len(eps))} of {len(eps)} replica(s); last was {url} -> {last}")

    if bodies:
        cap = MAX_INFLIGHT if max_inflight is None else max_inflight
        with ThreadPoolExecutor(max_workers=max(1, min(cap, len(bodies)))) as ex:
            list(ex.map(one, range(len(bodies))))
    return out


# ==================================================================================== reads

def yesno_body(messages: Sequence[dict]) -> dict:
    """The one-token Yes/No request, spelled for the OpenAI API.

    Every field is read off the role profile rather than typed, so a change to the role reaches
    the wire. ``top_k`` and ``min_p`` are omitted rather than sent as the profile's sentinel
    values: the server's defaults already mean "consider every token", and the spelling of
    those two has churned between releases -- an inert field beats a 400 on every call.
    """
    return {
        "model": served_model(), "messages": list(messages),
        "max_tokens": _YESNO.max_tokens, "temperature": _YESNO.temperature,
        "top_p": _YESNO.top_p, "presence_penalty": _YESNO.presence_penalty,
        "repetition_penalty": _YESNO.repetition_penalty,
        "logprobs": True, "top_logprobs": _YESNO.logprobs,
        "chat_template_kwargs": {"enable_thinking": _YESNO.thinking},
    }


def p_yes(resp: dict) -> float | None:
    """P(Yes) from one response, or ``None`` when the read is unusable.

    The pairs go in as a LIST, never a dict: several distinct token ids decode to "Yes", and
    keying by the decoded string keeps only the last and discards real mass.
    """
    ch = (resp.get("choices") or [{}])[0]
    content = (ch.get("logprobs") or {}).get("content") or []
    top = content[0].get("top_logprobs") if content else None
    if not top:
        # Distinct from an unreadable Yes/No, which is a legitimate None: this is a 200 with no
        # top-k in it, i.e. not the request we think we sent. Both consumers fill None with
        # 0.0, so a replica that stopped honouring `top_logprobs` would make two dimensions
        # identically zero on every rollout -- contributing exactly no advantage, with no error.
        raise RuntimeError(f"evaluator returned a completion with no top_logprobs: "
                           f"{str(ch)[:300]}")
    return mllm.p_yes_from_logprobs(
        [(t["token"], float(t["logprob"])) for t in top])


def p_yes_batch(conversations: Sequence[Sequence[dict]]) -> list[float | None]:
    return [p_yes(r) for r in post_batch([yesno_body(c) for c in conversations])]


def tokenize(text: str) -> list[int]:
    """Token ids for ``text`` under the evaluator's own tokenizer, memoised.

    Used to locate the audited span by ids rather than by decoded text. Asking the service
    rather than loading a tokenizer locally is what keeps the two from disagreeing.
    """
    if text not in _tokenize_cache:
        r = post_batch([{"model": served_model(), "prompt": text,
                         "add_special_tokens": False,
                         "chat_template_kwargs": {"enable_thinking": False}}],
                       path="/tokenize")[0]
        ids = r.get("tokens")
        if not ids:
            raise RuntimeError(f"/tokenize returned no tokens for {text!r}: {str(r)[:200]}")
        _tokenize_cache[text] = [int(t) for t in ids]
    return _tokenize_cache[text]
