# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Plan the segment grid over a stream. One code path for every dataset.

Two rules that look like details and are not:

**A segment never crosses a video boundary.** Windows are planned per video file, so frame
sampling always has one file to read and the file-relative clock is always well defined.

**A trailing remainder shorter than ``min_tail_s`` is folded into its predecessor** rather than
kept as its own entry: a 0.3 s segment yields eight near-identical frames and a description
about nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.schema import Stream, VideoRef

#: Segment length. The write unit of the whole method.
SEGMENT_S = 30.0
MIN_TAIL_S = 1.0


@dataclass(frozen=True)
class Segment:
    """One write unit, carrying BOTH clocks so no caller has to convert."""

    segment_id: str
    video_id: str
    path: Path
    t_start: float          #: file-local, for frame sampling and transcript slicing
    t_end: float
    abs_start: float        #: stream clock, for the index and the causal filter
    abs_end: float
    order: int              #: position in the stream, 0-based and globally monotonic

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def plan_video(video: VideoRef, *, segment_s: float = SEGMENT_S,
               min_tail_s: float = MIN_TAIL_S, first_order: int = 0) -> list[Segment]:
    if segment_s <= 0:
        raise ValueError(f"segment_s must be > 0, got {segment_s}")
    dur = float(video.duration_s)
    if dur <= 0:
        raise ValueError(f"{video.video_id}: duration {dur} <= 0")

    edges: list[tuple[float, float]] = []
    t = 0.0
    while t < dur - 1e-9:
        end = min(t + segment_s, dur)
        edges.append((t, end))
        t = end
    if len(edges) > 1 and edges[-1][1] - edges[-1][0] < min_tail_s:
        last = edges.pop()
        edges[-1] = (edges[-1][0], last[1])

    return [Segment(segment_id=f"{video.video_id}#{i:05d}", video_id=video.video_id,
                    path=video.path, t_start=a, t_end=b,
                    abs_start=video.to_abs(a), abs_end=video.to_abs(b),
                    order=first_order + i)
            for i, (a, b) in enumerate(edges)]


def plan(stream: Stream, *, segment_s: float = SEGMENT_S,
         min_tail_s: float = MIN_TAIL_S) -> list[Segment]:
    """The segment grid over a whole stream, in order, with a global ``order``."""
    out: list[Segment] = []
    for v in stream.videos:
        out.extend(plan_video(v, segment_s=segment_s, min_tail_s=min_tail_s,
                              first_order=len(out)))
    return out


def by_video(segments: list[Segment]) -> dict[str, list[Segment]]:
    out: dict[str, list[Segment]] = {}
    for s in segments:
        out.setdefault(s.video_id, []).append(s)
    return out
