"""EgoLifeQA: 500 questions over one subject's continuous seven-day recording. Test only.

A day is one prepared file, built by concatenating the release's ~30 s fragments. Those
fragments' *starts* are 30 s apart while their *durations* are 17.6--30.0 s, so the day file's
clock runs slow against wall-clock time -- which is why ``videos.json`` carries a ``members``
map and the loader produces a :class:`~memlife.data.schema.PiecewiseVideoRef`. See
:mod:`memlife.data.prepared`.

The absolute clock is ``(day - 1) * 86400 + seconds_of_day``.
"""
from __future__ import annotations

from . import prepared, splits
from .schema import QARecord, Stream

DATASET = "egolifeqa"

#: Length of a day on this clock. Drives the reader's day-boundary table, which is the only way
#: it can turn "yesterday" into a time-scoped search.
SECONDS_PER_DAY = 86400.0


def get_stream(stream_id: str) -> Stream:
    return prepared.get_stream(DATASET, stream_id)


def load(split: str = "test", *, limit: int | None = None) -> list[QARecord]:
    want = set(splits.stream_ids(DATASET, split))
    out = [r for r in prepared.questions(DATASET) if r.stream_id in want]
    if limit is None:
        splits.check_count(DATASET, split, len(out))
    return out[:limit] if limit else out
