"""The MemLife reader: a time-indexed agentic loop over the memory store.

Ten rounds, thinking on, one action per round. Four properties are the reader, and each was
learned by getting it wrong:

**No thinking content in history.** Assistant turns carry the ACTION LINE alone, never the
trace that produced it -- which is what the model card prescribes and what keeps a ten-round
context affordable.

**Only the newest ``fetch_video``'s pixels stay attached.** Older ones collapse to
"[N frames omitted]" with their text kept. At ~484 tokens per frame, keeping three 30-frame
fetches costs ~43k tokens to re-show frames the model has already reasoned over.

**Two-stage recovery, not one.** A thinking model regularly spends its whole budget reasoning
and never emits the action line. Stage 1 re-parses the region after the thinking terminator;
stage 2 makes one short NON-THINKING call that shows the model its own reasoning and asks only
for the line.

**A tool error is a turn, not a crash.** The model gets the error text back and can correct.
It is counted as a model error; it never silently becomes "no memory".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..core import frames as mframes
from ..core import mcq
from ..core import sampling as sp
from . import prompts, tools as T

#: Always available. ``fetch_video`` is added by MemLife-V.
BASE_TOOLS = ("search_memory", "fetch_memory")


@dataclass
class ReaderConfig:
    rounds: int = 10
    #: MemLife-V: adds ``fetch_video`` to the prompt and to the allowed set, together.
    source_video: bool = False
    #: Per ROUND; the reader gets ``rounds`` of them.
    max_tokens: int = 8192
    fallback_max_tokens: int = 96
    #: Sampling overrides from ``reader.decoding``, applied to BOTH the reader and the
    #: fallback -- leaving one sampled would make a greedy arm a mixture of two decoders.
    over: dict = field(default_factory=dict)
    #: Stop attaching new frames, and start compacting, once the estimate passes this.
    context_guard_tokens: int = 240_000
    #: Characters per token for the estimate. **2.0, not 4.0.** Prose is ~4.2, but what fills
    #: this context is structured: a hit header is ~1.9 and a timestamp row ~1.6. At 4.0 the
    #: estimate ran nearly half low, so compaction would first fire well past the real context
    #: limit -- i.e. never, because the request had already been rejected.
    chars_per_token: float = 2.0
    tokens_per_image: int = 484
    #: Length of a day on this dataset's clock, or None when the clock is not day-based.
    day_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f"reader.rounds must be >= 1, got {self.rounds}")

    @property
    def tools(self) -> tuple[str, ...]:
        return BASE_TOOLS + (("fetch_video",) if self.source_video else ())


@dataclass
class Trace:
    """Everything needed to score, debug or replay one question."""

    question_id: str
    letter: str | None = None
    rounds_used: int = 0
    forced_action: bool = False
    no_action: bool = False
    parse_failed: bool = False
    #: The loop ended with the round budget spent and no answer. Distinct from ``no_action``
    #: (never emitted a parseable action) and ``parse_failed`` (answered with a bad letter);
    #: without it such a row has ``letter=None`` and no flag naming the cause.
    rounds_exhausted: bool = False
    frames_dropped: int = 0
    calls: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)


def _estimate(msgs: list[dict], cfg: ReaderConfig) -> int:
    """Conservative token count. It must OVER-estimate: an estimate that runs low does not
    trigger compaction, and the round dies on a hard context-length error instead."""
    n = 0
    for m in msgs:
        c = m["content"]
        if isinstance(c, str):
            n += int(len(c) / cfg.chars_per_token)
        else:
            for part in c:
                n += (cfg.tokens_per_image if part["type"] == "image_pil"
                      else int(len(part["text"]) / cfg.chars_per_token))
    return n


def _compact(msgs: list[dict], cfg: ReaderConfig, frames_turn: int | None) -> None:
    """Shrink the oldest tool results to their head line until the estimate fits.

    Oldest first: the model has already reasoned over them, and the newest result is the one
    it is acting on. The turn holding live frames is skipped -- collapsing its text while its
    pixels are still attached would describe frames the model can see as omitted.
    """
    if _estimate(msgs, cfg) <= cfg.context_guard_tokens:
        return
    # Start at 2. msgs[0] is the system prompt and msgs[1] is the QUESTION, whose head line is
    # `Question: ...` -- head-only truncation would throw away the OPTIONS and the clock, then
    # ask for a letter for choices the model can no longer see.
    for i in range(2, len(msgs)):
        if i == frames_turn or msgs[i]["role"] != "user":
            continue
        c = msgs[i]["content"]
        if not (isinstance(c, list) and len(c) == 1 and c[0]["type"] == "text"):
            continue
        head = c[0]["text"].split("\n", 1)[0]
        if head == c[0]["text"]:
            continue                                   # already one line
        msgs[i] = {"role": "user",
                   "content": [{"type": "text", "text": head + "\n  [compacted]"}]}
        if _estimate(msgs, cfg) <= cfg.context_guard_tokens:
            return


def answer(engine, tl: T.Tools, record, *, cfg: ReaderConfig) -> Trace:
    """Run one question to an answer."""
    tr = Trace(record.question_id)
    system = prompts.reader_system(top_k=tl.cfg.top_k, source_video=cfg.source_video,
                                   fetch_video_frames=tl.cfg.fetch_video_frames)
    allowed = set(cfg.tools)

    msgs: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": prompts.question_block(
            record.question, mcq.format_options(record.choices),
            prompts.time_block(tl.now, day_seconds=cfg.day_seconds))}]},
    ]
    #: Index of the user turn holding live frames. Exactly one at a time.
    frames_turn: int | None = None
    frames_text = ""

    for rnd in range(cfg.rounds):
        tr.rounds_used = rnd + 1
        final = rnd == cfg.rounds - 1
        if final:
            msgs.append({"role": "user",
                         "content": [{"type": "text", "text": prompts.FINAL_ROUND}]})

        # Immediately before the call, not only after a tool result: the question turn plus a
        # single large result can already exceed the budget on round 0.
        _compact(msgs, cfg, frames_turn)
        raw = engine.chat([msgs], "reader", max_tokens=cfg.max_tokens,
                          **cfg.over)[0].outputs[0].text
        tr.raw.append(raw)
        act = T.parse_action(sp.action_region(raw), final_round=final)

        if act is None:                                   # stage 2: forced action
            tr.forced_action = True
            # Append to the LAST USER TURN rather than inventing an assistant turn the model
            # never produced, and pass the WHOLE cut-off trace: a short recovery call deciding
            # from the last few hundred characters loses the evidence recap earlier in it.
            reasoning = sp.strip_thinking(raw).strip() or raw
            instr = {"type": "text", "text": prompts.FORCE_ACTION.format(reasoning=reasoning)}
            tail = msgs[-1]
            tail_parts = (list(tail["content"]) if isinstance(tail["content"], list)
                          else [{"type": "text", "text": tail["content"]}])
            probe = msgs[:-1] + [{"role": tail["role"], "content": tail_parts + [instr]}]
            forced = engine.generate_text([probe], "reader_fallback",
                                          max_tokens=cfg.fallback_max_tokens, **cfg.over)[0]
            tr.raw.append(f"[forced-action probe] {forced}")
            act = T.parse_action(forced, final_round=final)
            if act is None:
                tr.no_action = True
                break

        kind, payload = act
        if kind == "answer":
            tr.letter = mcq.parse_answer(f"ANSWER: {payload}", record.n_choices)
            tr.parse_failed = tr.letter is None
            break

        # -- a tool call ------------------------------------------------------------
        # json.dumps, NOT an f-string: a Python dict repr replays as `"args": {'query': 'x'}`
        # -- single quotes, True, None. The model imitates its own history from round 2 on,
        # the JSON parse then rejects it, and the run degrades into fallbacks it caused itself.
        msgs.append({"role": "assistant",
                     "content": "TOOL: " + json.dumps({"name": kind, "args": payload})})
        if kind not in allowed:
            err = (f"[round {rnd + 1}] {kind!r} is not an available tool. Available: "
                   f"{', '.join(sorted(allowed))}.")
            tr.errors.append(err)
            msgs.append({"role": "user", "content": [{"type": "text", "text": err}]})
            continue
        try:
            result, imgs = tl.call(kind, payload)
        except (T.ToolError, ValueError, TypeError, KeyError, mframes.DecodeError) as e:
            # Not just ToolError: `{"num_frames": "max"}` raises ValueError from int(), an
            # undecodable window raises DecodeError. Letting those escape would drop the
            # QUESTION from the denominator entirely -- and the questions that provoke the
            # most tool use are the ones that would drop, a silent and biased sample. They are
            # model errors: hand them back and let it correct.
            err = f"[round {rnd + 1}] {kind} failed: {type(e).__name__}: {e}"
            tr.errors.append(err)
            msgs.append({"role": "user", "content": [{"type": "text", "text": err}]})
            continue

        tr.calls.append({"round": rnd, "name": kind, "args": payload,
                         "n_hits": len(result.get("hits", [])),
                         "hit_ids": [h["id"] for h in result.get("hits", [])],
                         # The RESOLVED window and the EFFECTIVE budget, not the model's raw
                         # arguments -- MemOpt replays these actions against new candidates.
                         "time_anchor": result.get("time_anchor"),
                         "top_k": result.get("top_k"),
                         "n_frames": len(imgs)})

        # Retire the live-frame turn as soon as ANY further round happens, not only when new
        # pixels arrive: keeping 30 frames alive across six later search rounds carries ~14.5k
        # vision tokens of context the model has finished with.
        if frames_turn is not None:
            msgs[frames_turn] = {"role": "user",
                                 "content": [{"type": "text", "text": frames_text}]}
            frames_turn = None
        _compact(msgs, cfg, frames_turn)

        # `[round N]` on every result: with all results kept in full, two 32-hit blocks are
        # otherwise unattributable to the actions that produced them.
        tag = f"[round {rnd + 1}] "
        text = tag + T.render(result, frames_attached=bool(imgs),
                              asr_chars=tl.cfg.asr_render_chars_video)
        if not imgs:
            msgs.append({"role": "user", "content": [{"type": "text", "text": text}]})
            continue
        dropped = tag + T.render(result, frames_attached=False, n_frames_dropped=len(imgs),
                                 asr_chars=tl.cfg.asr_render_chars_video)
        if _estimate(msgs, cfg) + len(imgs) * cfg.tokens_per_image > cfg.context_guard_tokens:
            # NOT tr.errors: that feeds the tool-error rate, and a long-context run would
            # report an inflated one from events that are not tool errors.
            tr.frames_dropped += len(imgs)
            msgs.append({"role": "user", "content": [{"type": "text", "text": dropped}]})
        else:
            frames_turn = len(msgs)
            frames_text = dropped
            msgs.append({"role": "user",
                         "content": [{"type": "text", "text": text}]
                         + [{"type": "image_pil", "image_pil": im} for im in imgs]})
    else:
        # The loop ran to completion, i.e. every round was spent on tool calls.
        tr.rounds_exhausted = tr.letter is None

    return tr
