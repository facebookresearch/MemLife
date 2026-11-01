"""SuperMemory-VQA: ten subjects, split S1--S6 / S7--S8 / S9--S10 (:mod:`memlife.data.splits`).

One subject is one memory stream -- all of that person's sessions on a single monotonic clock.

Every question carries a fixed "This question can not be answered." option. 202 of the 623 test
questions have it as gold, so a model that always abstains scores 32.4 % -- which is why
:mod:`memlife.eval.score` prints that floor beside any SuperMemory accuracy.
"""
from __future__ import annotations

from ..core.mcq import LETTERS
from . import prepared, splits
from .schema import QARecord, Stream

DATASET = "supermemory"

ABSTAIN_TEXT = "This question can not be answered."


def get_stream(stream_id: str) -> Stream:
    return prepared.get_stream(DATASET, stream_id)


def _abstain_letter(choices) -> str | None:
    want = ABSTAIN_TEXT.rstrip(".").lower()
    return next((LETTERS[i] for i, c in enumerate(choices)
                 if c.strip().rstrip(".").lower() == want), None)


def load(split: str = "test", *, limit: int | None = None) -> list[QARecord]:
    want = set(splits.stream_ids(DATASET, split))
    out = [r for r in prepared.questions(DATASET) if r.stream_id in want]
    for r in out:
        r.meta.setdefault("abstain_letter", _abstain_letter(r.choices))
    if limit is None:
        splits.check_count(DATASET, split, len(out))
    return out[:limit] if limit else out
