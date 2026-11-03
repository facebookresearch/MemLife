# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Faithfulness: is every token of this memory supported by its source segment?

    R_faith[i,t] = 1 - ( max_w p_audit(w | c, y_i, y_i,<t) - p_audit(y_i,t | c, y_i, y_i,<t) )

The evaluator is asked a DIFFERENT question from the writer's -- "correct this memory" -- and
the candidate's own tokens are teacher-forced through that prompt. Where the audit's own
top-1 disagrees with the realised token, the margin is large; that disagreement is the
per-token unfaithfulness signal. The reward is in [0, 1] and equals 1 exactly where the
candidate token is the auditor's most probable continuation.

**The prompt asymmetry is the whole mechanism.** Reading the margin off the trainer's own
reference forward would be free, but that only works if the auditor's prompt IS the writer's
prompt -- and it is not.

**The evaluator sees the writer's own frames, nothing more.** Prompt privilege only, not
resolution privilege: the same eight frames at the same 704 px. An auditor with extra pixels
would penalise the writer for not stating what it structurally could not see.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import judge

# ==================================================================================== the audit

AUDIT_SYSTEM = (
    "You are correcting a written memory of one 30-second segment of a first-person video. You "
    "are given the segment's frames and transcript, and ONE memory of that segment. A claim is "
    "unsupported if it asserts an object, person, place, action, quantity, or spoken content "
    "that is not actually visible in the segment's frames or audible in its transcript. A "
    "memory that is vague, brief or incomplete is still supported.\n\n"
    "If every claim in the memory is supported, repeat the memory back EXACTLY as given, "
    "character for character. If some claim is not supported, output the memory with those "
    "claims removed, or weakened to what the segment actually shows, changing as few words as "
    "possible. Do not add detail and do not improve the writing.\n\n"
    "Output only the memory text. No preamble, no explanation, no quotation marks."
)
AUDIT_HEAD = "## The segment"
AUDIT_ASR = 'What is said in this segment:\n"""{transcript}"""'
AUDIT_CANDIDATE = (
    '## Memory of this segment\n"{description}"\n\n'
    "Output this memory with any unsupported claims corrected, or unchanged if every claim is "
    "supported.\nMemory:"
)

#: Concurrent audit requests. **Not** :data:`judge.MAX_INFLIGHT`. ``prompt_logprobs``
#: materialises logprobs for EVERY prompt position, so one multimodal audit needs gigabytes of
#: transient over the vocabulary; dozens in flight will kill the replica outright. Raise this
#: only with a measurement.
AUDIT_INFLIGHT = 4

#: Chat-template anchors delimiting the assistant turn's CONTENT. The audited span begins at
#: the first content delimiter at or after the last assistant header. Both are template
#: constants no candidate text can perturb -- the thinking tag cannot appear inside the
#: content half, and the header is a special token the trainer strips from rollouts.
#: Their token ids are resolved from the evaluator's own tokenizer at run time.
ASSISTANT_HEADER = "<|im_start|>assistant\n"
CONTENT_DELIMITER = "</think>\n\n"


class SpanError(RuntimeError):
    """The audited span could not be located, or a position inside it is unreadable.

    Raised per row and handled per row: killing a whole optimiser step over one candidate
    would be far worse than scoring that candidate as unscorable.
    """


def build_messages(segment_id: str, *, transcript: str, description: str,
                   frames_root: str | Path) -> list[dict]:
    """head -> frames -> transcript -> the memory under audit, then the memory as the
    ASSISTANT turn.

    The candidate appears twice on purpose: once as the thing being audited, once as the
    tokens whose probabilities are read. That compresses the margins -- the audit distribution
    hugs the realised token -- which is why the signal must be used as a group-relative RANK
    and never thresholded against an absolute value.
    """
    content: list[dict] = [{"type": "text", "text": AUDIT_HEAD}]
    content += judge.segment_frames(segment_id, frames_root)
    content.append({"type": "text", "text": AUDIT_ASR.format(transcript=transcript)})
    content.append({"type": "text", "text": AUDIT_CANDIDATE.format(description=description)})
    return [{"role": "system", "content": AUDIT_SYSTEM},
            {"role": "user", "content": content},
            {"role": "assistant", "content": description}]


def audit_body(messages: Sequence[dict]) -> dict:
    """The teacher-forcing request: score the prompt, do not continue it.

    ``add_generation_prompt=False`` with ``continue_final_message=True`` is what puts the
    candidate at the very END of the prompt; the one sampled token is discarded.
    """
    return {
        "model": judge.served_model(), "messages": list(messages),
        "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1,
        "add_generation_prompt": False, "continue_final_message": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


@dataclass(frozen=True)
class _Entry:
    tid: int
    logprob: float
    rank: int
    text: str


def _positions(resp: dict) -> list[dict | None]:
    """The per-position logprob dicts out of one response.

    Read from the envelope and, failing that, from the choice: vLLM has put
    ``prompt_logprobs`` in both places, and pinning one spelling would turn a version
    difference into a run that scores nothing.
    """
    try:
        lp = resp.get("prompt_logprobs")
        if lp is None:
            lp = (resp.get("choices") or [{}])[0].get("prompt_logprobs")
        if lp is not None:
            return list(lp)
    except (AttributeError, TypeError, KeyError, IndexError) as e:
        raise SpanError(f"the evaluator's response is not shaped like a chat completion "
                        f"({type(e).__name__}: {e}): {str(resp)[:300]}") from None
    raise SpanError(
        "the evaluator returned no prompt_logprobs. Token-level faithfulness cannot be scored "
        "without it -- check that the replica accepts `prompt_logprobs`.")


def _top1_and_realized(pos: dict) -> tuple[_Entry, _Entry]:
    """Split one position into (top-1, realised token).

    At ``prompt_logprobs=1`` a position carries one entry when the realised token IS the
    top-1, and two when it is not. ``rank`` is the only thing that distinguishes them -- dict
    order is not, because vLLM has emitted the sampled entry first in some releases and folded
    into the top-k in others. Everything unexpected raises rather than guesses: swapping the
    two silently negates the margin, which is a per-token advantage with the wrong sign.
    """
    if not isinstance(pos, dict):
        raise SpanError(f"prompt_logprobs position is {type(pos).__name__}, not a mapping")
    es = []
    for tid, v in pos.items():
        if (not isinstance(v, dict)
                or any(v.get(k) is None for k in ("rank", "decoded_token", "logprob"))):
            raise SpanError(f"incomplete prompt_logprobs entry for token {tid}: {v!r}")
        try:
            e = _Entry(int(tid), float(v["logprob"]), int(v["rank"]), str(v["decoded_token"]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise SpanError(f"prompt_logprobs entry {tid!r} -> {v!r} has the wrong types "
                            f"({type(exc).__name__}: {exc})") from None
        # The margin exponentiates this one frame later. Above ~709 that raises; a NaN is
        # worse, because it raises nothing and the row scores as perfectly faithful. A
        # log-probability is <= 0 by definition; the tolerance is for rounding at p = 1.
        if not math.isfinite(e.logprob) or e.logprob > 1e-6:
            raise SpanError(f"prompt_logprobs entry {tid!r} has logprob {v['logprob']!r}, "
                            f"which is not a log-probability")
        es.append(e)
    if not es:
        raise SpanError("empty prompt_logprobs position: no entry for the realised token")
    if len(es) == 1:
        if es[0].rank != 1:
            raise SpanError(f"a lone entry must be the top-1 one, got rank {es[0].rank}")
        return es[0], es[0]
    if len(es) > 2:
        raise SpanError(f"{len(es)} entries at one position; prompt_logprobs=1 returns <= 2")
    top, real = sorted(es, key=lambda e: e.rank)
    if top.rank != 1:
        raise SpanError(f"no rank-1 entry at this position: {es!r}")
    if real.rank == 1:
        # Two entries tied at rank 1. The margin would be 0 either way, but which text is the
        # realised token becomes unrecoverable, and span location needs it.
        raise SpanError(f"two rank-1 entries; realised token ambiguous: {es!r}")
    if real.logprob > top.logprob:
        # rank and logprob disagreeing means one of this function's two assumptions is wrong,
        # and the consequence is a NEGATIVE margin -- a reward above 1 and an inverted
        # per-token gradient.
        raise SpanError(f"the rank-1 entry is not the most likely one: {es!r}")
    return top, real


def _find_ids(tids: list[int], needle: list[int], *, start: int = 0, last: bool = False) -> int:
    rng = range(len(tids) - len(needle), start - 1, -1) if last \
        else range(start, len(tids) - len(needle) + 1)
    for i in rng:
        if tids[i:i + len(needle)] == needle:
            return i
    return -1


def span_start(positions: Sequence[dict | None]) -> int:
    """Index of the first audited token: just past the assistant turn's content delimiter.

    Located **by token id**, never by decoded text. Decoded text does not round-trip through
    per-position detokenization of a byte-level BPE, and a text-matching walk dies on any
    candidate the writer drifted into another script -- which is survivable for one row and
    fatal when it kills the step.
    """
    tids = []
    for i, p in enumerate(positions):
        if p is None:                       # position 0 has no distribution
            tids.append(-1)
            continue
        tids.append(_top1_and_realized(p)[1].tid)
    header = judge.tokenize(ASSISTANT_HEADER)
    delim = judge.tokenize(CONTENT_DELIMITER)
    h = _find_ids(tids, header, last=True)
    if h < 0:
        raise SpanError(
            f"no assistant header {ASSISTANT_HEADER!r} in the audited prompt. The chat "
            f"template does not match what memlife.memopt.faithfulness expects; set "
            f"ASSISTANT_HEADER and CONTENT_DELIMITER for this model family.")
    d = _find_ids(tids, delim, start=h)
    if d < 0:
        raise SpanError(
            f"no content delimiter {CONTENT_DELIMITER!r} after the assistant header. Set "
            f"CONTENT_DELIMITER for this model family.")
    return d + len(delim)


def margins(positions: Sequence[dict | None]) -> tuple[tuple[str, float], ...]:
    """``(realised token text, margin)`` for every audited token, in order.

    The margin is in PROBABILITY space, not log space: ``exp(lp_top1) - exp(lp_realised)``.
    Bounded in [0, 1] and exactly 0 where the auditor's own top-1 is the realised token.

    The TEXT comes back with it because these positions are the EVALUATOR's tokenization,
    while the reward array has to lie on the POLICY's response positions. Detokenize-then-
    retokenize is not the identity, so without the text there is nothing to align by -- and a
    shifted array would be silently zero-padded and used.
    """
    lo = span_start(positions)
    out = []
    for pos in positions[lo:]:
        top, real = _top1_and_realized(pos)
        out.append((real.text, math.exp(top.logprob) - math.exp(real.logprob)))
    return tuple(out)


@dataclass(frozen=True)
class Faith:
    """One candidate's token-level faithfulness.

    ``tokens`` is the alignment key for ``margin``; ``reward`` is ``1 - margin`` per token.

    ``ok`` is False exactly when the span could not be located. The caller must branch on it
    **before building the token array**, not only before using a scalar: an unscorable row has
    no margins at all, which pads to all-zeros -- i.e. "no unfaithful token anywhere", the best
    signal this dimension can emit, which a candidate could then farm.
    """

    tokens: tuple[str, ...]
    margin: tuple[float, ...]
    ok: bool

    @property
    def reward(self) -> tuple[float, ...]:
        return tuple(1.0 - m for m in self.margin)


def score(items: Sequence[tuple[str, str, str]], *, frames_root: str | Path,
          max_inflight: int = AUDIT_INFLIGHT) -> list[Faith]:
    """``(segment_id, transcript, description)`` per candidate -> one :class:`Faith` each.

    A span that cannot be located fails SOFT, per row. Raising would kill every rollout of the
    step over one candidate. A blank candidate asserts nothing, so it is vacuously faithful
    without an audit (the chat template drops an empty assistant turn's delimiter anyway); its
    other two dimensions are what score it.
    """
    todo = [i for i, (_s, _a, d) in enumerate(items) if (d or "").strip()]
    bodies = [audit_body(build_messages(items[i][0], transcript=items[i][1],
                                        description=items[i][2], frames_root=frames_root))
              for i in todo]
    out = [Faith((), (), True)] * len(items)
    for i, resp in zip(todo, judge.post_batch(bodies, max_inflight=max_inflight)):
        try:
            pairs = margins(_positions(resp))
        except SpanError as e:
            print(f"[faith] unscorable row: {e}", flush=True)
            out[i] = Faith((), (), False)
        else:
            out[i] = Faith(tuple(t for t, _ in pairs), tuple(m for _, m in pairs), True)
    return out
