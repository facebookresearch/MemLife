"""The vLLM engine wrapper, and the one-token Yes/No logprob read the evaluators use.

Roles are grouped into *buckets* that share an engine, so a job that both writes and judges
pays one model load rather than two. Thinking mode, temperature and the rest come from the
role's profile and **cannot be overridden per call** -- that coupling is the point of
:mod:`memlife.core.sampling`.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import sampling as sp

#: role -> engine bucket.
BUCKET = {
    "writer": "writer",
    "reader": "reader",
    "reader_fallback": "reader",
    "keyfact": "judge",
    "judge_yesno": "judge",
}


@dataclass
class ModelSpec:
    """The subset of ``configs/model/<name>.yaml`` the engine needs."""

    id: str
    local_path: str | None
    family: str
    grid_factor: int
    max_pixels: int
    tp: dict                      #: bucket -> tensor-parallel size
    context: dict                 #: bucket -> max_model_len
    limit_mm_per_prompt: dict
    gpu_memory_utilization: float
    max_num_seqs: int
    additional_config: dict = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, m) -> "ModelSpec":
        return cls(
            id=m["id"], local_path=m.get("local_path"), family=m["family"],
            grid_factor=int(m["grid_factor"]), max_pixels=int(m["max_pixels"]),
            tp=dict(m["tp"]), context=dict(m["context"]),
            limit_mm_per_prompt=dict(m["limit_mm_per_prompt"]),
            gpu_memory_utilization=float(m["gpu_memory_utilization"]),
            max_num_seqs=int(m["max_num_seqs"]),
            additional_config=dict(m.get("additional_config") or {}))

    def path(self) -> str:
        """The staged local checkpoint, or the hub id when none is configured.

        A configured-but-missing path RAISES rather than falling back. Falling back looks
        harmless and is not: with ``HF_HUB_OFFLINE=1`` and a populated cache, vLLM would load
        the cached BASE snapshot and the run would complete -- so a MemOpt checkpoint whose
        path had a typo would be silently evaluated as the untrained model.
        """
        p = os.path.expandvars(self.local_path) if self.local_path else None
        if not p:
            return self.id
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"model.local_path does not exist: {p!r}. Refusing to fall back to the hub id "
                f"{self.id!r} -- offline mode would quietly load a cached BASE snapshot "
                f"instead of this checkpoint.")
        return p


class _Unset:
    """Distinguishes 'seed not specified' from an explicit ``seed=None``."""


_UNSET = _Unset()


class LLMEngine:
    """One vLLM engine, serving every role in a bucket."""

    #: Fields a call site may NOT override: they define what the role IS. A judge call with
    #: thinking on reads logprobs at the wrong token position.
    _LOCKED = ("thinking", "logprobs")

    def __init__(self, spec: ModelSpec, bucket: str, *, seed: int = 0, verbose: bool = True):
        if bucket not in set(BUCKET.values()):
            raise KeyError(f"unknown bucket {bucket!r}; known: {sorted(set(BUCKET.values()))}")
        self.spec = spec
        self.bucket = bucket
        self.seed = seed
        self._llm = None
        self._verbose = verbose

    @property
    def llm(self):
        if self._llm is None:
            from vllm import LLM

            kw = dict(
                model=self.spec.path(),
                dtype="bfloat16",
                tensor_parallel_size=int(self.spec.tp[self.bucket]),
                max_model_len=int(self.spec.context[self.bucket]),
                limit_mm_per_prompt={"image": int(self.spec.limit_mm_per_prompt[self.bucket])},
                gpu_memory_utilization=self.spec.gpu_memory_utilization,
                # Set explicitly: vLLM's default is generous, and a hybrid model carries a
                # large per-SEQUENCE recurrent state that reserves memory before any KV cache.
                max_num_seqs=self.spec.max_num_seqs,
                # A cap, not a transform -- preparation already resized the frames.
                mm_processor_kwargs={"max_pixels": self.spec.max_pixels,
                                     "min_pixels": 4 * self.spec.grid_factor ** 2},
                seed=self.seed,
                trust_remote_code=True,
                additional_config=dict(self.spec.additional_config),
            )
            if self._verbose:
                print(f"[llm] bucket={self.bucket} model={kw['model']} "
                      f"tp={kw['tensor_parallel_size']} ctx={kw['max_model_len']} "
                      f"imgs={kw['limit_mm_per_prompt']}", flush=True)
            self._llm = LLM(**kw)
        return self._llm

    def profile(self, role: str, **over: Any) -> sp.Profile:
        self._check_role(role)
        if bad := sorted(set(over) & set(self._LOCKED)):
            raise ValueError(
                f"{bad} cannot be overridden per call -- they define what the role IS. Change "
                f"the profile in memlife/core/sampling.py, or use a different role.")
        if "max_tokens" in over and int(over["max_tokens"]) < 1:
            raise ValueError(f"max_tokens must be >= 1, got {over['max_tokens']}")
        return sp.profile(self.spec.family, role, **over)

    def _check_role(self, role: str) -> None:
        if BUCKET.get(role) != self.bucket:
            raise ValueError(
                f"role {role!r} belongs to bucket {BUCKET.get(role)!r}, not {self.bucket!r}. "
                f"Its context, tensor-parallel size and image limits would be wrong.")

    def chat(self, conversations: Sequence[Sequence[dict]], role: str, *,
             seed: int | None | _Unset = _UNSET, **profile_over: Any) -> list:
        """Run a batch of chat conversations under ``role``'s profile.

        Returns vLLM ``RequestOutput``s, so callers can read logprobs.

        ``seed`` omitted -> the engine's seed (reproducible builds). ``seed=None`` -> genuinely
        unseeded, which MemOpt rollouts need: N identical prompts under one fixed seed return N
        identical completions, giving zero reward variance and a run that logs a smooth curve
        while learning nothing.
        """
        prof = self.profile(role, **profile_over)
        eff = self.seed if isinstance(seed, _Unset) else seed
        return self.llm.chat([list(c) for c in conversations],
                             sampling_params=prof.vllm_params(seed=eff), **prof.chat_kwargs)

    def generate_text(self, conversations, role: str, *, seed=_UNSET, **over) -> list[str]:
        """:meth:`chat` reduced to strings, with the thinking trace stripped.

        For non-thinking roles that is the whole story. A THINKING role whose trace was
        truncated has no terminator, so the strip returns the reasoning itself -- fine for
        logging, wrong for parsing. The reader therefore parses
        :func:`memlife.core.sampling.action_region` directly instead of using this helper.
        """
        outs = self.chat(conversations, role, seed=seed, **over)
        return [sp.strip_thinking(o.outputs[0].text).strip() for o in outs]


# ---------------------------------------------------------------------------- yes/no scoring

#: Characters a model may wrap a one-word answer in. Stripped before an EXACT comparison.
_AFFIX = " \t\n\r*_`\"'()[]{}.,!:;-▁ "


def polarity(tok: str | None) -> str | None:
    """Classify a decoded token as yes / no / neither: **exact match after normalisation**.

    Two simpler rules each fail, in opposite directions:

    * A fixed list of exact strings misses real single tokens in the vocabulary -- ``"Yes,"``,
      ``"**Yes"``, ``"\\nYes"``, ``"▁Yes"``. When one of those is top-1 and ``"No"`` is in
      the top-20, the yes mass is floored and a true 0.97 comes back as 0.0003.
    * **Prefix matching over-corrects.** Against a real 248k-token Qwen vocabulary,
      ``startswith("no")`` claims 717 tokens that mean nothing of the kind -- ``' now'``,
      ``'None'``, ``' nor'``, ``' normal'``, ``' November'``, ``' node'``, ``' noise'`` -- and
      ``startswith("yes")`` claims ``'Yesterday'``. Each is a *confident* fabricated verdict,
      and it disables the "top-1 is neither polarity -> unreadable" guard below.
      ``' yesterday'`` is the sharp end: this judge reads memories about yesterday.
    """
    if tok is None:
        return None
    t = tok.strip(_AFFIX).lower()
    return t if t in ("yes", "no") else None


def p_yes_from_logprobs(pairs: Sequence[tuple[str, float]]) -> float | None:
    """P(Yes) = softmax over the pooled Yes and No logprobs, or ``None`` if unreadable.

    ``pairs`` must be a **list of (decoded_token, logprob)**, never a dict. Several distinct
    token ids decode to ``"Yes"``; keying by the decoded string keeps only the last and drops
    real probability mass.

    ``None`` rather than a plausible 0.5 in three unreadable cases, because a fabricated
    verdict averaged into a reward is worse than a gap the caller can count:

    * neither polarity is present at all;
    * the **top-1** token is neither -- the read is about something else, and flooring the
      missing side would manufacture a confident answer;
    * only one polarity is present AND it is the weakest token returned, so the flooring bound
      carries no information.

    A missing ``No`` is otherwise floored to the smallest returned logprob: it fell outside the
    top-k, so it cannot exceed that, and the bound errs in the right direction.
    """
    if not pairs:
        return None
    if polarity(max(pairs, key=lambda p: p[1])[0]) is None:
        return None

    ys = [lp for tok, lp in pairs if polarity(tok) == "yes"]
    ns = [lp for tok, lp in pairs if polarity(tok) == "no"]
    if not ys and not ns:
        return None

    def lse(xs: Sequence[float]) -> float:
        m = max(xs)
        return m + math.log(sum(math.exp(x - m) for x in xs))

    floor = min(lp for _, lp in pairs)
    if (not ys or not ns) and max(ys or ns) <= floor:
        return None
    y = lse(ys) if ys else floor
    n = lse(ns) if ns else floor
    return 1.0 / (1.0 + math.exp(max(-700.0, min(700.0, n - y))))

