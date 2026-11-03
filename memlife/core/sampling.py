# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sampling profiles per role, and thinking-mode control per model family.

Every generation in this repository goes through one of the named roles below, so "what
temperature did the writer use?" is a table lookup rather than a grep.

Qwen specifics that are easy to get wrong
-----------------------------------------
* Qwen3.5 **thinks by default**; the only lever is
  ``chat_template_kwargs={"enable_thinking": False}``.
* The card's ``presence_penalty`` of 1.5 is for *free generation*. Applying it to a one-token
  Yes/No logprob read distorts the very distribution being measured, so the judge roles pin
  ``presence_penalty=0.0``. A deliberate departure from the card, not an oversight.
* Judge calls **must** disable thinking. With thinking on, token position 0 is ``<think>`` and
  the Yes/No logprobs come from the wrong position -- measured as a -0.466 mean shift in
  faithfulness with every row disagreeing with the offline reference.
* Multi-turn history carries the final answer only, never the thinking trace.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

#: Every role. Anything that generates text picks exactly one.
ROLES = ("writer", "reader", "reader_fallback", "keyfact", "judge_yesno")


@dataclass(frozen=True)
class Profile:
    thinking: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_tokens: int
    logprobs: int | None = None
    note: str = ""

    def vllm_params(self, seed: int | None = None, **over: Any):
        """Build a ``vllm.SamplingParams``. Imported lazily so CPU-only tests can import us."""
        from vllm import SamplingParams

        kw: dict[str, Any] = dict(
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            min_p=self.min_p, presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens, seed=seed)
        if self.logprobs is not None:
            kw["logprobs"] = self.logprobs
        kw.update(over)
        return SamplingParams(**kw)

    @property
    def chat_kwargs(self) -> dict:
        return {"chat_template_kwargs": {"enable_thinking": self.thinking}}


# Card: thinking / general.
_Q35_THINK = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                  presence_penalty=1.5, repetition_penalty=1.0)
# Card: instruct (non-thinking) / general.
_Q35_INSTRUCT = dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0,
                     presence_penalty=1.5, repetition_penalty=1.0)
# Not from the card: greedy single-token scoring. See the module docstring on presence_penalty.
_GREEDY = dict(temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0,
               presence_penalty=0.0, repetition_penalty=1.0)

#: Decoding modes selectable from config (``writer.decoding`` / ``reader.decoding``), expressed
#: as overrides on top of whatever profile the role already has. A mode is NOT a profile:
#: ``max_tokens``, thinking and the rest still come from the role.
#:
#: ``greedy`` -- **the reported configuration** -- turns off the SAMPLING STEP and nothing else,
#: exactly as HuggingFace's ``do_sample=False`` does. ``presence_penalty`` therefore stays at
#: the card's value, keeping the repetition guard the card says greedy decoding most needs.
#:
#: ``sampled`` is the family's card profile, for the temperature-1.0 stability study. MemOpt
#: rollouts are sampled by the trainer, with the ``train`` section's settings.
DECODING: dict[str, dict] = {
    "greedy": dict(temperature=0.0, top_p=1.0, top_k=-1),
    "sampled": {},
}

_QWEN35: dict[str, Profile] = {
    "writer": Profile(
        thinking=False, **_Q35_INSTRUCT, max_tokens=512,
        note="Extraction, not reasoning -- and during MemOpt this role IS the policy, so the "
             "response must stay inside train.max_response_length, which thinking would blow "
             "past."),
    "reader": Profile(
        thinking=True, **_Q35_THINK, max_tokens=8192,
        note="Per ROUND of the agentic loop."),
    "reader_fallback": Profile(
        thinking=False, **_Q35_INSTRUCT, max_tokens=96,
        note="Recovers a single TOOL:/ANSWER: line when a round produced no parseable action."),
    "keyfact": Profile(
        thinking=False, **{**_Q35_INSTRUCT, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
        max_tokens=192,
        note="GREEDY extraction of one key fact. NOT `_GREEDY`: that zeroes presence_penalty, "
             "which is right for a one-token logprob read and wrong for a free generation."),
    "judge_yesno": Profile(
        thinking=False, **_GREEDY, max_tokens=1, logprobs=20,
        note="One token, read via logprobs. presence_penalty=0 -- the card's 1.5 is for free "
             "generation and would distort the distribution being measured."),
}

# Qwen3.6 differs in one place: its card gives presence_penalty 0.0 for thinking mode.
_QWEN36 = {k: (replace(v, presence_penalty=0.0) if v.thinking else v)
           for k, v in _QWEN35.items()}

_FAMILIES: dict[str, dict[str, Profile]] = {"qwen3.5": _QWEN35, "qwen3.6": _QWEN36}


def decoding_over(mode: str) -> dict:
    if mode not in DECODING:
        raise KeyError(f"unknown decoding {mode!r}; expected one of {sorted(DECODING)}")
    return dict(DECODING[mode])


def profile(family: str, role: str, **over: Any) -> Profile:
    if role not in ROLES:
        raise KeyError(f"unknown role {role!r}; expected one of {ROLES}")
    if family not in _FAMILIES:
        raise KeyError(
            f"no sampling profiles for model family {family!r}. Add them from that model's "
            f"card rather than reusing another family's -- known: {sorted(_FAMILIES)}")
    p = _FAMILIES[family][role]
    return replace(p, **over) if over else p


# ------------------------------------------------------------------------------ thinking

THINK_CLOSE = ("</think>",)
THINK_OPEN = ("<think>",)


def find_think_close(text: str, *, last: bool = True) -> int:
    """Index just PAST a thinking terminator, or -1 if the text holds none."""
    best = -1
    for d in THINK_CLOSE:
        i = text.rfind(d) if last else text.find(d)
        if i == -1:
            continue
        end = i + len(d)
        if best == -1 or (end > best if last else end < best):
            best = end
    return best


def strip_thinking(text: str) -> str:
    """Drop the reasoning trace from a thinking-mode response.

    The chat template seeds the assistant turn with the opener, so a thinking response contains
    a closing terminator but usually no opening tag. If none appears the trace was truncated by
    ``max_tokens`` -- return the text unchanged so the caller still has something to parse.
    """
    if not text:
        return text
    i = find_think_close(text, last=False)
    return text if i < 0 else text[i:].lstrip()


def action_region(text: str) -> str:
    """The part of a response after the LAST thinking terminator -- where an action line lives.

    Parsing the whole response would let a letter mentioned inside the reasoning ("option C
    looks wrong") be read as the answer.
    """
    if not text:
        return text
    i = find_think_close(text, last=True)
    return text if i < 0 else text[i:]
