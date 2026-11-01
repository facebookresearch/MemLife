"""The prepared-data contract, and the generic loader that reads it.

Every benchmark is reduced to the same three things under ``data/<dataset>/``::

    videos.json          the recording timeline
    qa.json              the questions
    transcripts/<video_id>.json     speech, if the dataset has any

so there is one loader rather than one per benchmark, and adding a dataset is a conversion
script plus a split table -- never new reader or writer code.

``videos.json`` -- a list of::

    {
      "video_id":   "A1_JAKE_DAY1",      unique across the dataset
      "stream_id":  "egolifeqa/A1_JAKE", which recording history this file belongs to
      "path":       "videos/A1_JAKE_DAY1.mp4",   relative to data/<dataset>/, or absolute
      "abs_start":  0.0,                 stream-clock seconds of this file's frame 0
      "duration_s": 83312.4,
      "width": 704, "height": 704, "fps": 12.0,

      # OPTIONAL, for a file concatenated from fragments whose starts are further apart than
      # their durations (EgoLife). Without it the file clock and the wall clock are assumed to
      # advance together; with it the conversion is piecewise. See PiecewiseVideoRef.
      "members": [{"offset_s": 0.0, "abs_start_s": 0.0}, ...]
    }

``qa.json`` -- a list of::

    {
      "question_id":    "A1_JAKE_0001",
      "question":       "...",
      "choices":        ["...", "..."],     UNPREFIXED: no leading "A. "
      "correct_letter": "C"  or  "E,F",     canonical: sorted, comma-joined, no spaces
      "stream_id":      "egolifeqa/A1_JAKE",
      "query_abs_s":    61861.0,            THE CAUSAL CUTOFF -- memory after it is invisible
      "evidence": [{"video_id": "...", "t_start_abs": 1.0, "t_end_abs": 31.0}],
      "meta":     {...}                     free-form
    }

``query_abs_s`` is the moment the question is asked, and getting it wrong flatters the method:
pin it to the end of the *answer* evidence instead and the gold segment sits immediately behind
the causal frontier, so a reader that ignores the question and fetches the most recent entries
scores near-perfect retrieval recall.

An evidence entry with ``t_start_abs == t_end_abs`` is a grounding **instant**; the segment
containing it counts as gold (see :mod:`memlife.eval.evidence`).
"""
from __future__ import annotations

import functools
from pathlib import Path

from .. import paths
from ..core import io as mio
from ..core.mcq import canon, strip_letter_prefix
from .schema import EvidenceSpan, PiecewiseVideoRef, QARecord, Stream, VideoRef

VIDEOS_FILE = "videos.json"
QA_FILE = "qa.json"
TRANSCRIPTS_DIR = "transcripts"


def _resolve(dataset: str, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else paths.dataset_dir(dataset) / path


def _video_ref(dataset: str, row: dict) -> VideoRef:
    common = dict(
        video_id=row["video_id"], path=_resolve(dataset, row["path"]),
        abs_start=float(row["abs_start"]), duration_s=float(row["duration_s"]),
        width=int(row["width"]), height=int(row["height"]), fps=float(row["fps"]))
    members = row.get("members")
    if not members:
        return VideoRef(**common)
    ms = sorted(members, key=lambda m: float(m["offset_s"]))
    return PiecewiseVideoRef(
        **common,
        offsets=tuple(float(m["offset_s"]) for m in ms),
        abs_starts=tuple(float(m["abs_start_s"]) for m in ms))


@functools.lru_cache(maxsize=None)
def streams(dataset: str) -> dict[str, Stream]:
    """Every stream in a prepared dataset, keyed by ``stream_id``."""
    rows = mio.read_json(paths.dataset_dir(dataset) / VIDEOS_FILE)
    by_stream: dict[str, list[VideoRef]] = {}
    for r in rows:
        by_stream.setdefault(r["stream_id"], []).append(_video_ref(dataset, r))
    return {sid: Stream(sid, dataset, tuple(vs)) for sid, vs in by_stream.items()}


def get_stream(dataset: str, stream_id: str) -> Stream:
    try:
        return streams(dataset)[stream_id]
    except KeyError:
        raise KeyError(
            f"{dataset}: no stream {stream_id!r} in {VIDEOS_FILE}; known: "
            f"{sorted(streams(dataset))[:8]}") from None


@functools.lru_cache(maxsize=None)
def questions(dataset: str) -> tuple[QARecord, ...]:
    """Every question in a prepared dataset, in ``question_id`` order."""
    out = [to_record(dataset, q) for q in mio.read_json(paths.dataset_dir(dataset) / QA_FILE)]
    out.sort(key=lambda r: r.question_id)
    return tuple(out)


def to_record(dataset: str, q: dict) -> QARecord:
    return QARecord(
        question_id=str(q["question_id"]),
        question=q["question"],
        # Stripped here rather than trusted: an already-prefixed option would be
        # double-prefixed by `mcq.format_options` and the model would see "A. A. red".
        choices=tuple(strip_letter_prefix(c) for c in q["choices"]),
        correct_letter=canon(str(q["correct_letter"]).split(",")),
        dataset=dataset,
        now_abs=float(q["query_abs_s"]),
        stream_id=q["stream_id"],
        evidence=tuple(EvidenceSpan(
            video_id=e.get("video_id") or "",
            t_start_abs=float(e["t_start_abs"]), t_end_abs=float(e["t_end_abs"]),
            modalities=tuple(e.get("modalities", []))) for e in q.get("evidence", ())),
        meta=dict(q.get("meta") or {}))


def transcripts_for(dataset: str, stream: Stream, *, expected: bool) -> dict[str, list]:
    """``video_id`` -> turns, empty where the dataset has no audio."""
    from ..core import asr as masr

    d = paths.dataset_dir(dataset) / TRANSCRIPTS_DIR
    return {v.video_id: masr.load(d / f"{v.video_id}.json", expected=expected)
            for v in stream.videos}
