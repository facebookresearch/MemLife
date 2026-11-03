# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Every prompt MemLife uses. One file, so a prompt change is one reviewable diff.

**The wording is the method.** These are the strings the reported numbers were produced with;
paraphrasing one silently changes what is being measured. They are reproduced verbatim in the
paper's appendix, including its use of "clip" where the rest of this codebase says "segment"
or "entry".

Only the reader's prompt is assembled rather than written out, and only along one axis: with
source-video access the tool list gains a third entry, which has to change the count, the
numbering and the examples together. Advertising a tool the model cannot call would make that
comparison measure prompt confusion rather than the value of re-watching video.
"""
from __future__ import annotations

# ======================================================================= the episodic writer

#: The writer sees ONE segment: eight frames and that segment's speech, no predecessor and no
#: question. So nothing here may name another segment -- an earlier draft said "the CURRENT
#: segment" three times over, naming a contrast the prompt never draws.
WRITER_SYSTEM = """\
You are writing a memory entry for one short segment of an egocentric video. This entry is the \
only record of this moment that a later question-answering system will see, so capture the \
concrete, factual details it might be asked about.

Write in the FIRST PERSON as the person wearing the camera (the video is shot from their point \
of view): refer to yourself as 'I'/'my'/'me', not as 'the camera wearer', 'the camera \
operator', or 'the person'.

The evidence lives in TWO equally important channels, and you must use both:
1. What is SEEN — the people, their actions, the objects they handle (brand names, text on \
labels, colors), and the place. Refer to any OTHER person by name when they are a named \
speaker in the transcript (tags like [B] or [Shure]); add stable visual attributes (clothing, \
hair, glasses) as a complement to the name, and use visual attributes alone only when no name \
is available.
2. What is SAID or HEARD — who speaks and the substance of what they say; paraphrase salient \
statements, questions, decisions, plans, names, numbers, times, and places mentioned in the \
transcript.

Treat the spoken transcript as a primary source of fact, not background noise: if something is \
only stated aloud, it still belongs in the memory. Write ONE paragraph of plain prose, up to \
~450 tokens; finish every sentence and do not truncate. No JSON, no bullets, no headers. \
Describe only this segment; do not speculate beyond the evidence.

Write a COMPLETE, SELF-CONTAINED first-person paragraph describing what happens in this \
segment (actions, objects, places, people, state changes)."""

#: The user turn: the frames come FIRST, so the model reads the segment before it is told what
#: to produce.
WRITER_USER = """\
What is said in this segment:
\"\"\"{transcript}\"\"\"

Write the memory entry, grounding it in the frames and transcript above."""


# ========================================================================== the agentic reader

_TOOL_SEARCH = """\
1. search_memory(query: str, time_anchor: [t_start, t_end] = [0, <now>], top_k: int = {k}): \
retrieve the memory entries most relevant to the query, restricted to clips within \
[t_start, t_end]. Returns the top_k most similar clips in time order — each with its \
description and time range. Omit time_anchor to search all memory up to the current time; pass \
a narrower [t_start, t_end] to focus on a period (e.g. a specific day)."""

_TOOL_FETCH_MEMORY = """\
2. fetch_memory(time_anchor: [t_start, t_end]): return every memory clip within \
[t_start, t_end] in time order, each with its description."""

_TOOL_FETCH_VIDEO = """\
3. fetch_video(time_anchor: [t_start, t_end], num_frames: int = {frames}): sample num_frames \
frames and the transcript from the video within [t_start, t_end]."""

_EXAMPLES = (
    'Q: "How often do I cook dinner?" -> TOOL: {"name": "search_memory", '
    '"args": {"query": "cooking dinner"}}',
    'Q: "What did I buy yesterday?" -> TOOL: {"name": "search_memory", "args": '
    '{"query": "buying purchase shopping", "time_anchor": [yesterday_start, yesterday_end]}}',
    'Q: "List everything I bought today." -> TOOL: {"name": "fetch_memory", '
    '"args": {"time_anchor": [t_start, t_end]}}',
)
_EXAMPLE_FETCH_VIDEO = ('Q: "What was on my desk when I left?" -> TOOL: {"name": '
                        '"fetch_video", "args": {"time_anchor": [t_start, t_end], '
                        '"num_frames": 30}}')


def reader_system(*, top_k: int, source_video: bool, fetch_video_frames: int) -> str:
    """The reader's system prompt. ``source_video`` is the only thing that varies.

    The tool count, the numbering and the example list all move with it, so the MemLife and
    MemLife-V prompts differ by exactly the capability they differ by.
    """
    blocks = [_TOOL_SEARCH.format(k=top_k), _TOOL_FETCH_MEMORY]
    examples = list(_EXAMPLES)
    if source_video:
        blocks.append(_TOOL_FETCH_VIDEO.format(frames=fetch_video_frames))
        examples.append(_EXAMPLE_FETCH_VIDEO)
    return (
        "You are a memory reader. You answer multiple-choice questions about an egocentric "
        "video by calling tools to inspect a pre-built memory.\n\n"
        f"You have {'three' if source_video else 'two'} tools:\n\n"
        + "\n\n".join(blocks)
        + "\n\nEach turn, respond with EXACTLY ONE of these two lines:\n"
          '  TOOL: {"name": "<tool>", "args": {...}}\n'
          "  ANSWER: <letter>\n\n"
          "Emit no other text on the response line. End your final round with "
          "`ANSWER: <letter>` once you have enough evidence.\n\nExamples:\n"
        + "\n".join(examples))


def time_block(now: float, *, day_seconds: float | None) -> str:
    """What time it is, and -- where the clock is day-based -- where the day boundaries are.

    Not decoration. The reader's own example instructs
    ``"time_anchor": [yesterday_start, yesterday_end]``; without the day table those bounds are
    guesses, and a guessed window returns confidently-scoped entries from the wrong day rather
    than an error.
    """
    if not day_seconds:
        return f"Current time: {now:.1f} s.\nMemory time window: [0.0, {now:.1f}] seconds."
    d = int(now // day_seconds) + 1
    r = now % day_seconds
    h, m, sec = int(r // 3600), int((r % 3600) // 60), int(r % 60)
    days = "\n".join(
        f"  DAY{k}: [{(k - 1) * day_seconds:.0f}, {k * day_seconds:.0f}) s"
        + (" <- current day" if k == d else "")
        for k in range(1, d + 1))
    return (f"Current time: {now:.1f} s (DAY{d}, {h:02d}:{m:02d}:{sec:02d}).\n"
            f"Day time ranges (relative seconds):\n{days}\n"
            f"Memory time window: [0.0, {now:.1f}] seconds.")


def question_block(question: str, options: str, time_text: str) -> str:
    """The reader's first user turn. **No answer directive.**

    The system prompt already owns the protocol ("respond with EXACTLY ONE of these two
    lines"). Adding "end your reply with ANSWER: <letter>" on top tells the model to answer
    immediately, before it has looked at anything -- and premature answering with zero tool
    calls is exactly the behaviour the agentic reader exists to avoid.
    """
    return f"Question: {question}\n\nOptions:\n{options}\n\n{time_text}"


FINAL_ROUND = ("This is your final round. Based on what you've gathered, output a single line "
               "`ANSWER: <letter>`.")

FORCE_ACTION = """\
Your analysis so far (it was cut off before you emitted an action):

{reasoning}

Based on the analysis above, output ONLY the next line: `TOOL: {{...}}` or `ANSWER: <letter>`."""
