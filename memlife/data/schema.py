"""The record types every loader produces and every method consumes.

Three clocks, and mixing them is the single easiest way to get a silently wrong result:

``t_file``   seconds inside one video file. What :mod:`memlife.core.frames` and
             :mod:`memlife.core.asr` take.
``t_abs``    seconds on the dataset's own monotonic stream clock. What the index, the causal
             filter and every tool argument use. **Not unix:** unix times are ~1.77e9, whose
             float32 ulp is 128 s -- four segment lengths -- so one float32 cast anywhere would
             merge four segments. Every loader therefore rebases to a per-stream epoch.
``unix``     raw release metadata only. Never crosses into this module.

:class:`VideoRef` owns the ``t_file`` <-> ``t_abs`` conversion, so no call site has to.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path

from ..core.mcq import LETTERS, canon


@dataclass(frozen=True)
class VideoRef:
    """One prepared video file, placed on the stream clock."""

    video_id: str
    path: Path
    abs_start: float          #: stream-clock time of this file's frame 0
    duration_s: float
    width: int
    height: int
    fps: float

    @property
    def abs_end(self) -> float:
        return self.to_abs(self.duration_s)

    def to_abs(self, t_file: float) -> float:
        return self.abs_start + t_file

    def to_file(self, t_abs: float) -> float:
        return t_abs - self.abs_start


@dataclass(frozen=True)
class PiecewiseVideoRef(VideoRef):
    """A video whose file clock and wall clock advance at different rates.

    EgoLife publishes ~30 s fragments whose *starts* are 30 s apart but whose *durations* are
    17.6--30.0 s, so a concatenated day runs slow against wall-clock time -- by seconds, by the
    end of a long run. Under the base linear map the store's time axis and the question's time
    axis then agree only at the first frame, which misassigns boundary segments and lets
    post-cutoff frames into a causal window.

    Overriding the conversion here rather than at each call site keeps segment planning, window
    resolution and evidence matching correct without any of them knowing this exists.
    """

    offsets: tuple[float, ...] = ()        #: each member's start, file clock
    abs_starts: tuple[float, ...] = ()     #: the same member's start, wall clock

    @staticmethod
    def _seg(xs: tuple[float, ...], v: float) -> int:
        return max(0, bisect.bisect_right(xs, v) - 1)

    def to_abs(self, t_file: float) -> float:
        i = self._seg(self.offsets, t_file)
        return self.abs_starts[i] + (t_file - self.offsets[i])

    def to_file(self, t_abs: float) -> float:
        i = self._seg(self.abs_starts, t_abs)
        return self.offsets[i] + (t_abs - self.abs_starts[i])


@dataclass(frozen=True)
class EvidenceSpan:
    """An annotated interval that supports a question's answer, on the stream clock."""

    video_id: str
    t_start_abs: float
    t_end_abs: float
    modalities: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stream:
    """One subject's whole recording history, on one monotonic clock."""

    stream_id: str
    dataset: str
    videos: tuple[VideoRef, ...]

    def __post_init__(self) -> None:
        vs = sorted(self.videos, key=lambda v: v.abs_start)
        for a, b in zip(vs, vs[1:]):
            if b.abs_start < a.abs_end - 1e-6:
                raise ValueError(
                    f"{self.stream_id}: {b.video_id} starts at {b.abs_start} before "
                    f"{a.video_id} ends at {a.abs_end}. Overlapping videos make chronological "
                    f"ordering ill-defined and let one window resolve to two files.")
        object.__setattr__(self, "videos", tuple(vs))

    @property
    def duration_s(self) -> float:
        return self.videos[-1].abs_end - self.videos[0].abs_start if self.videos else 0.0

@dataclass(frozen=True)
class QARecord:
    """One benchmark question.

    ``correct_letter`` is the **canonical set string** from :func:`memlife.core.mcq.canon`:
    ``"C"`` for single-select, ``"E,F"`` for multi-select. Scoring is then exact set match by
    plain string comparison, and single-select is unchanged.
    """

    question_id: str
    question: str
    choices: tuple[str, ...]
    correct_letter: str
    dataset: str
    now_abs: float            #: the causal cutoff -- memory after this is invisible
    stream_id: str
    evidence: tuple[EvidenceSpan, ...] = ()
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 2 <= len(self.choices) <= len(LETTERS):
            raise ValueError(
                f"{self.question_id}: {len(self.choices)} choices, expected 2..{len(LETTERS)}")
        got = [x for x in str(self.correct_letter).split(",") if x]
        ok = set(LETTERS[:len(self.choices)])
        if not got or any(g not in ok for g in got):
            raise ValueError(
                f"{self.question_id}: correct_letter {self.correct_letter!r} is not a subset of "
                f"{LETTERS[:len(self.choices)]!r} ({len(self.choices)} choices)")
        if canon(got) != self.correct_letter:
            # An uncanonical gold ("F,E") never equals a canonicalised prediction, so the row
            # would score 0 for every method, silently.
            raise ValueError(
                f"{self.question_id}: correct_letter {self.correct_letter!r} is not canonical; "
                f"expected {canon(got)!r}")

    @property
    def n_choices(self) -> int:
        return len(self.choices)
