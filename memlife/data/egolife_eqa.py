# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EgoLife-EQA: 100 test questions about recurring events across days of the EgoLife recordings.

Three question types, 45 / 31 / 24 of them:

``frequency_counting``  "On which days did I go grocery shopping? Choose all that apply."
                        8 options (DAY 1..DAY 7, NONE), **multi-select**, 1--5 correct.
``routine``             "What do I usually check on my phone in the morning?" 2--4 options.
``comparison``          "Which did I do on more days -- X or Y?" 3 options.

A multi-select prediction is correct only if it matches the complete annotated set, which is
exactly what comparing two canonical set strings does (:mod:`memlife.core.mcq`).

**It reuses EgoLifeQA's recordings.** The questions are new; the video is not. So the timeline,
the transcripts and the built memory store all come from ``egolifeqa``. The questions also
cover A3_TASHA, which the EgoLifeQA test build does not write; building this dataset under the
same tag writes only that subject.

The questions ship with the repository (``datasets/egolife_eqa/test.jsonl``) and are converted
to ``data/egolife_eqa/qa.json`` on first use.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from ..core import io as mio
from ..core.mcq import canon
from . import prepared, splits
from .schema import QARecord, Stream

DATASET = "egolife_eqa"

#: Which prepared dataset owns the recordings, the transcripts and the memory store.
STORE_DATASET = "egolifeqa"

SECONDS_PER_DAY = 86400.0

#: The published question file, shipped with the code rather than under ``$MEMLIFE_ROOT``.
SOURCE = Path(__file__).resolve().parents[2] / "datasets" / "egolife_eqa" / "test.jsonl"


def to_contract(rows: list[dict]) -> list[dict]:
    """Published EgoLife-EQA records -> :mod:`memlife.data.prepared` question records.

    Evidence is published as grounding **instants** (``abs_s``), not intervals, because what an
    annotator verified is a moment in the recording. A zero-width span is the faithful
    representation: :func:`memlife.eval.evidence.visible_gold` counts the segment containing an
    instant as gold, so no interval is invented here.

    One published evidence entry carries ``abs_s: null`` -- the annotator confirmed the day but
    it could not be resolved to a timestamp. It is skipped: an entry with no place on the clock
    cannot name a gold segment, and inventing one would put recall against the wrong entry.
    """
    return [{
        "question_id": r["question_id"],
        "question": r["question"],
        "choices": list(r["choices"]),
        "correct_letter": canon(str(r["correct_letter"]).split(",")),
        "stream_id": f"{STORE_DATASET}/{r['subject']}",
        "query_abs_s": float(r["query_abs_s"]),
        "evidence": [{"video_id": e.get("video_id") or "",
                      "t_start_abs": float(e["abs_s"]),
                      "t_end_abs": float(e["abs_s"])}
                     for e in r.get("evidence", ()) if e.get("abs_s") is not None],
        "meta": {k: r[k] for k in
                 ("type", "subject", "activity", "source", "answer_value",
                  "multi_select", "n_gold") if k in r},
    } for r in rows]


def ensure() -> Path:
    """Make sure ``data/egolife_eqa/qa.json`` exists. Returns its path."""
    qa = paths.dataset_dir(DATASET) / "qa.json"
    if not qa.exists():
        rows = [json.loads(ln) for ln in SOURCE.read_text().splitlines() if ln.strip()]
        paths.ensure(qa.parent)
        mio.write_json(qa, to_contract(rows), indent=1)
        print(f"[data] wrote {len(rows)} questions to {qa}", flush=True)
    return qa


def get_stream(stream_id: str) -> Stream:
    return prepared.get_stream(STORE_DATASET, stream_id)


def load(split: str = "test", *, limit: int | None = None) -> list[QARecord]:
    if split not in ("test", "all"):
        raise KeyError(f"egolife_eqa has no split {split!r}; it is test-only (100 questions)")
    ensure()
    want = set(splits.stream_ids(DATASET, "test", prefix=STORE_DATASET))
    out = [r for r in prepared.questions(DATASET) if r.stream_id in want]
    if limit is None:
        splits.check_count(DATASET, "test", len(out))
    return out[:limit] if limit else out
