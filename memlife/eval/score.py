# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Score a finished prediction directory. **Always recomputed from the files, never cached.**

A run commits rows continuously, so any number remembered from a previous look is stale. Every
figure a report prints is produced by re-reading ``results/*.json`` at report time.

Three things are reported beside accuracy, because leaving them out is how an accuracy becomes
misleading:

``n`` and the DENOMINATOR -- a resumed sweep with 611 of 623 rows has an accuracy over 611, and
the gap must be visible rather than inferred.
``no_letter`` -- questions where the strict parser found no answer. These count as WRONG, which
is the honest reading, but a method that fails to parse on 20 % of questions is a different
object from one that answers them wrong.
``recall`` -- did retrieval ever surface a gold segment. An accuracy that moves without recall
moving is the reader changing, not the memory.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import json
from dataclasses import dataclass
from pathlib import Path

from ..data.schema import QARecord, Stream
from . import evidence as ev

#: Concurrent opens in :func:`load_rows`. These are hundreds of small files, and on a network
#: filesystem the cost is latency, not bandwidth -- so the fix is concurrency, not a cache.
#: Deliberately NOT a cache: the contract is that every number is recomputed at report time.
_READ_THREADS = 32


@dataclass
class Scores:
    n: int
    expected: int
    accuracy: float
    correct: int
    no_letter: int
    recall: float | None = None
    recall_n: int = 0                  #: questions with at least one REACHABLE gold segment
    no_visible_gold: int = 0
    #: What a model scores by ALWAYS abstaining, where that option exists. ``None`` otherwise.
    #: On SuperMemory test this is 32.4 %, so an accuracy quoted without it makes abstention
    #: bias look like memory.
    abstain_floor: float | None = None

    def line(self, label: str) -> str:
        miss = "" if self.n == self.expected else f" (MISSING {self.expected - self.n})"
        rec = "" if self.recall is None else f"  recall={self.recall:6.2%} (n={self.recall_n})"
        return (f"{label:<34s} acc={self.accuracy:6.2%}  {self.correct}/{self.n}{miss}"
                f"  no_letter={self.no_letter}{rec}")


def load_rows(pred_dir: Path) -> list[dict]:
    """Every committed prediction. ``pred_dir`` is the directory holding ``results/``."""
    d = Path(pred_dir)
    d = d if d.name == "results" else d / "results"
    if not d.exists():
        raise FileNotFoundError(f"no predictions at {d}")
    files = sorted(d.glob("*.json"))
    if not files:
        return []
    with cf.ThreadPoolExecutor(min(_READ_THREADS, len(files))) as pool:
        return [json.loads(t) for t in pool.map(Path.read_text, files)]


def score(rows: list[dict], records: list[QARecord], streams: dict[str, Stream],
          *, segment_s: float = 30.0) -> Scores:
    """Accuracy, parse rate and retrieval recall over ``records``.

    ``rows`` may be a subset (a resumed sweep); ``records`` is the full split, so the
    denominator is the split's size and a partial run is visibly partial.
    """
    by_qid = {r["question_id"]: r for r in rows}
    scored: list[QARecord] = []
    correct = no_letter = 0
    rec_hit = rec_n = no_gold = 0

    for q in records:
        row = by_qid.get(q.question_id)
        if row is None:
            continue
        scored.append(q)
        letter = row.get("letter")
        # Exact set match: both sides are canonical comma-joined letters, so this is
        # single-select equality and multi-select set equality in one comparison.
        correct += int(letter == q.correct_letter)
        # A missing letter is a WRONG answer, and separately a reported parse failure.
        # Dropping these rows would let a method raise its accuracy by not answering.
        if letter is None:
            no_letter += 1

        st = streams.get(q.stream_id)
        if st is not None:
            g = ev.visible_gold(q, st, segment_s=segment_s)
            if not g:
                no_gold += 1
            else:
                rec_n += 1
                got = {sid for c in row.get("calls", []) for sid in c.get("hit_ids", ())}
                rec_hit += int(bool(g & got))

    ab = [q for q in scored if (q.meta or {}).get("abstain_letter")]
    return Scores(n=len(scored), expected=len(records),
                  accuracy=correct / len(scored) if scored else 0.0,
                  correct=correct, no_letter=no_letter,
                  recall=(rec_hit / rec_n) if rec_n else None,
                  recall_n=rec_n, no_visible_gold=no_gold,
                  abstain_floor=(sum(q.correct_letter == q.meta["abstain_letter"] for q in ab)
                                 / len(scored) if ab and scored else None))


def trace_stats(rows: list[dict], *, round_cap: int | None = None) -> dict:
    """How the reader actually behaved. Read these beside any accuracy."""
    n = len(rows)
    if not n:
        return {}
    calls = collections.Counter(c["name"] for r in rows for c in r.get("calls", ()))
    return {
        "n": n,
        "forced_action_rate": round(sum(bool(r.get("forced_action")) for r in rows) / n, 4),
        "no_action_rate": round(sum(bool(r.get("no_action")) for r in rows) / n, 4),
        "parse_failed_rate": round(sum(bool(r.get("parse_failed")) for r in rows) / n, 4),
        # Against the CONFIGURED cap, not the largest observed one: with max(observed), a run
        # where every question finished in three rounds reports a large cap-hit rate when the
        # true rate is zero.
        "hit_round_cap_rate": (round(sum(r.get("rounds_used", 0) >= round_cap
                                         for r in rows) / n, 4) if round_cap else None),
        "rounds_exhausted_rate": round(sum(bool(r.get("rounds_exhausted"))
                                           for r in rows) / n, 4),
        "mean_rounds": round(sum(r.get("rounds_used", 0) for r in rows) / n, 2),
        "tool_calls": dict(calls),
        "tool_error_rate": round(sum(bool(r.get("n_errors")) for r in rows) / n, 4),
    }

