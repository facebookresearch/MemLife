# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The trainer's reward hook: score ONE sampled memory on all three FIRM dimensions.

    reward.custom_reward_function.path=pkg://memlife.memopt.reward
    reward.custom_reward_function.name=compute_score

``pkg://``, not a file path: loading this file directly binds it under a synthetic module name
with no package, and its first relative import dies. Going through the package also guarantees
ONE module object, and therefore one loaded pack, rather than a second copy alongside any
normal import.

Environment, all of it set by ``scripts/train_memopt.sh``::

    MEMLIFE_PACK                   retrievability pack stem -- changes at every epoch
    MEMLIFE_KEYFACTS               the keyfacts.jsonl defining Q(c)
    MEMLIFE_FRAMES                 extracted frame tree, for the faithfulness audit
    MEMLIFE_MAX_RESPONSE_LENGTH    pad width for the token array; MUST equal
                                   data.max_response_length
    MEMLIFE_JUDGE_ENDPOINTS        the evaluator replicas (memlife.memopt.judge)

**One call per rollout at a time, per worker.** verl hands a plain ``def`` to a thread pool, so
the number of concurrent audits would be the worker count times the pool size -- and the audit
is the request shape that kills a replica when too many are in flight. A coroutine that never
yields is awaited in place instead, which makes total concurrency exactly
``reward.num_workers``. Hence ``async def`` with nothing awaited inside: that is deliberate.

What this returns is **not** the gradient. ``score`` is a scalar for logging only; the three
per-dimension keys are what :mod:`memlife.memopt.advantage` reads.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from ..core import io as mio
from . import advantage as adv
from . import faithfulness as mfaith
from . import informativeness as minf
from . import retrievability as mret

_pack: mret.Pack | None = None
_facts: dict[str, dict[str, str]] | None = None


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is unset. See memlife/memopt/reward.py for the contract.")
    return v


def max_response_length() -> int:
    return int(_env("MEMLIFE_MAX_RESPONSE_LENGTH"))


def frames_root() -> Path:
    return Path(_env("MEMLIFE_FRAMES"))


def pack() -> mret.Pack:
    """The retrievability table, loaded once per process.

    Unlocked: a reward worker is single-threaded by design (see the module docstring), so the
    worst a race could do is load it twice.
    """
    global _pack
    if _pack is None:
        _pack = mret.Pack.load(Path(_env("MEMLIFE_PACK")))
        print(f"[reward] pack {_pack.meta}", flush=True)
    return _pack


def key_facts() -> dict[str, dict[str, str]]:
    global _facts
    if _facts is None:
        out: dict[str, dict[str, str]] = {}
        for r in mio.read_jsonl(Path(_env("MEMLIFE_KEYFACTS"))):
            if not r["none"]:
                out.setdefault(r["segment_id"], {})[r["qid"]] = r["key_fact"]
        _facts = out
        print(f"[reward] {len(out)} segments in the training pool", flush=True)
    return _facts


def _faith_array(rewards, width: int) -> np.ndarray:
    """Lay the audited span's per-token rewards onto the policy's response positions.

    Exact only because the evaluator and the policy are the same model -- which
    :func:`memlife.config._check` enforces.

    **Positions past the end of the span are padded with 1.0, i.e. perfectly faithful.** That
    is right for padding outside the response (it is masked away), and it is the hazard worth
    knowing about for an in-mask tail: under the product such a token carries the full
    magnitude of whatever gates it. Only the LENGTH of a disagreement is checkable, and
    :func:`compute_score` reports it.
    """
    out = np.ones(width, dtype="float32")
    n = min(len(rewards), width)
    out[:n] = np.asarray(rewards[:n], dtype="float32")
    return out


async def compute_score(data_source=None, solution_str: str = "", ground_truth=None,
                        extra_info: dict | None = None, **kwargs) -> dict:
    """Score one sampled memory.

    ``extra_info`` carries ``segment_id`` and the segment's ``transcript``, written by
    :mod:`memlife.memopt.data`. Everything else comes from the pack and the key facts, so the
    hot path opens no store, loads no index and reads no questions.
    """
    info = extra_info or {}
    segment_id = info.get("segment_id")
    if not segment_id:
        raise KeyError("extra_info carries no segment_id; the rollout dataset is malformed")
    description = (solution_str or "").strip()
    width = max_response_length()

    facts = list(key_facts().get(segment_id, {}).values())
    inf = minf.score([(description, facts)])[0]
    ret = mret.score(segment_id, description, pack())
    f = mfaith.score([(segment_id, info.get("transcript", ""), description)],
                     frames_root=frames_root())[0]

    if f.ok:
        faith = _faith_array(f.reward, width)
    else:
        # NaN, never a constant. An unscorable row must not silently become all-1.0 -- that is
        # the best signal this dimension can emit, and a candidate could farm it. The trainer
        # fills NaN with the group's scorable mean, which is exactly neutral.
        faith = np.full(width, np.nan, dtype="float32")

    return {
        # Logging only; the gradient reads the three keys below. Kept finite, because verl
        # averages it into the step's reward statistics.
        "score": (float(np.nanmean(faith) * inf.value * ret.value)
                  if f.ok and math.isfinite(inf.value) else 0.0),
        adv.FAITH_KEY: faith.tolist(),
        adv.INF_KEY: float(inf.value),
        adv.RET_KEY: float(ret.value),
        # Diagnostics. `n_tokens` against the response mask is the only checkable form of
        # "did the audited span cover the response": a short span leaves the tail scored as
        # perfectly faithful, a long one has its real margins masked away.
        "n_tokens": float(len(f.tokens)),
        "faith_scorable": float(f.ok),
        "inf_n_q": float(inf.n_q),
        "inf_unreadable": float(inf.n_unreadable),
        "ret_from_fetch": float(ret.from_fetch),
    }
