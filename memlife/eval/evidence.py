# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Gold retrieval targets: which memory entries a question's answer actually lives in.

Retrieval recall needs a set of segment ids per question, but the benchmarks annotate time, so
the mapping depends on the segment grid the store was built with. It is therefore computed
from the same plan the writer used, never cached alongside the questions.

Two forms of annotation, handled by one rule:

* an **interval** ``[t_start_abs, t_end_abs)`` -- every segment OVERLAPPING it;
* an **instant** (``t_start_abs == t_end_abs``, as EgoLife-EQA publishes) -- the single segment
  CONTAINING it.

Gold is never shown to the model. It exists only to score what the reader retrieved.
"""
from __future__ import annotations

import functools

from ..data.schema import QARecord, Stream
from ..memory import segments as mseg


@functools.lru_cache(maxsize=64)
def _grid(stream: Stream, segment_s: float) -> tuple:
    return tuple(mseg.plan(stream, segment_s=segment_s))


def gold(rec: QARecord, stream: Stream, *, segment_s: float = mseg.SEGMENT_S) -> set[str]:
    """Segment ids holding this question's answer, on the grid the store was built with."""
    grid = _grid(stream, segment_s)
    out: set[str] = set()
    for ev in rec.evidence:
        if ev.t_end_abs > ev.t_start_abs:
            out.update(s.segment_id for s in grid
                       if s.abs_start < ev.t_end_abs and s.abs_end > ev.t_start_abs)
        else:
            t = ev.t_start_abs
            out.update(s.segment_id for s in grid if s.abs_start <= t < s.abs_end)
    return out


def visible_gold(rec: QARecord, stream: Stream,
                 *, segment_s: float = mseg.SEGMENT_S) -> set[str]:
    """Gold a causal reader could actually reach: segments that ENDED by the cutoff.

    Recall must be reported against this, not against all gold. Some questions' annotated
    evidence straddles their own cutoff, and scoring those against unreachable segments
    measures the benchmark's labelling rather than the method. A question with no visible gold
    is excluded from the recall denominator and counted separately.
    """
    by_id = {s.segment_id: s for s in _grid(stream, segment_s)}
    return {sid for sid in gold(rec, stream, segment_s=segment_s)
            if by_id[sid].abs_end <= rec.now_abs}
