"""SuperMemory-LVQA: the same 623 test questions against a ten-subject, ~119-day memory.

Every other benchmark here caps at seven days of history (EgoLifeQA) or a fraction of a day (a
single SuperMemory test subject). Nothing measures what happens when the memory itself is an
order of magnitude longer. This loader builds that condition **without writing a single new
question**: concatenate all ten SuperMemory subjects onto one monotonic clock and ask the
existing test questions of it::

    abs clock ->  S1 S2 S3 S4 S5 S6 S7 S8 | S9 | S10
                  <---- 8 filler subjects ---->  ^     S9  sees ~5.7k prior entries
                                                        S10 sees ~6.0k prior entries

S9 and S10 go **last**, in that order, so all 623 questions get the full filler history and
"S9 earlier than S10" still holds among the test subjects. Putting S9 first would leave half
the benchmark with its original short memory and measure nothing new.

The question ids are unchanged, so every LVQA cell is **paired** with its SuperMemory-VQA cell
and the delta is the result rather than the absolute number.

The recordings, transcripts and written memory entries are SuperMemory's (``STORE_DATASET``):
a segment's description does not depend on the clock it is filed under, so an LVQA build
writes only the subjects not yet written and re-times the rest when it indexes.
"""
from __future__ import annotations

import functools
import dataclasses

from . import prepared, splits, supermemory as sm
from .schema import QARecord, Stream

DATASET = "supermemory_lvqa"
STORE_DATASET = "supermemory"

#: Concatenation order. Changing it changes every question's causal cutoff and every stored
#: timestamp, so an existing store must be rebuilt.
ORDER: tuple[str, ...] = tuple(f"supermemory/S{i}" for i in range(1, 11))

#: One day between subjects. Not cosmetic: :class:`~memlife.data.schema.Stream` raises on any
#: overlap, and a day-sized gap also stops the reader's day table folding two subjects into one
#: "day".
GAP_S = 86400.0

STREAM_ID = "supermemory_lvqa/ALL"


@functools.lru_cache(maxsize=1)
def _offsets() -> dict[str, float]:
    """stream_id -> the shift added to all of its absolute timestamps.

    Derived from each subject's measured span, so adding a subject to :data:`ORDER` needs no
    other change.
    """
    out, cursor = {}, 0.0
    for sid in ORDER:
        st = prepared.get_stream("supermemory", sid)
        if not st.videos:
            continue
        out[sid] = cursor - st.videos[0].abs_start
        cursor += st.duration_s + GAP_S
    return out


@functools.lru_cache(maxsize=1)
def _stream() -> Stream:
    off = _offsets()
    vids = []
    for sid, shift in off.items():
        for v in prepared.get_stream("supermemory", sid).videos:
            vids.append(dataclasses.replace(v, abs_start=v.abs_start + shift))
    return Stream(STREAM_ID, DATASET, tuple(vids))


def get_stream(stream_id: str) -> Stream:
    if stream_id != STREAM_ID:
        raise KeyError(f"{DATASET} has one stream, {STREAM_ID!r}; got {stream_id!r}")
    return _stream()


def load(split: str = "test", *, limit: int | None = None) -> list[QARecord]:
    if split != "test":
        raise KeyError(f"{DATASET} has no split {split!r}; it reuses supermemory/test only")
    off = _offsets()
    out = []
    for r in sm.load("test"):
        shift = off[r.stream_id]
        out.append(dataclasses.replace(
            r, dataset=DATASET, stream_id=STREAM_ID, now_abs=r.now_abs + shift,
            evidence=tuple(dataclasses.replace(e, t_start_abs=e.t_start_abs + shift,
                                               t_end_abs=e.t_end_abs + shift)
                           for e in r.evidence),
            meta={**r.meta, "source_stream_id": r.stream_id, "lvqa_offset_s": shift}))
    if limit is None:
        splits.check_count(DATASET, "test", len(out))
    return out[:limit] if limit else out
