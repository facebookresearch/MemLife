# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark splits. **The single source of truth** -- no loader may hardcode a subject list.

SuperMemory-VQA is split **by subject**, not by question, because the unit of memory is a
person's whole recording history: two questions about the same subject share one store, so
splitting by question would leak training memory into test.

``S7--S8 is validation, not a second test set.`` Anything selected on it -- a checkpoint, a
threshold, a prompt variant -- is fitted, and reporting it as a result overstates the method.
S9--S10 exists to be looked at once, at the end.

:data:`EXPECTED` pins every split's question count. A loader whose count drifts raises rather
than quietly reporting an accuracy over a different denominator.
"""
from __future__ import annotations

#: dataset -> split -> the stream_id suffixes that belong to it.
SUBJECTS: dict[str, dict[str, tuple]] = {
    "supermemory": {
        "train": ("S1", "S2", "S3", "S4", "S5", "S6"),
        "val": ("S7", "S8"),
        "test": ("S9", "S10"),
    },
    # Only A1_JAKE has public QA, so EgoLifeQA is test-only.
    "egolifeqa": {"test": ("A1_JAKE",)},
    # EgoLife-EQA asks about two subjects; the whole benchmark is evaluation.
    "egolife_eqa": {"test": ("A3_TASHA", "A1_JAKE")},
    # One concatenated stream holding every SuperMemory subject.
    "supermemory_lvqa": {"test": ("ALL",)},
}

#: (dataset, split) -> expected question count.
EXPECTED: dict[tuple[str, str], int] = {
    ("supermemory", "train"): 3425,
    ("supermemory", "val"): 723,
    ("supermemory", "test"): 623,
    ("egolifeqa", "test"): 500,
    ("supermemory_lvqa", "test"): 623,
    ("egolife_eqa", "test"): 100,
}


def stream_ids(dataset: str, split: str, *, prefix: str | None = None) -> tuple[str, ...]:
    try:
        keys = SUBJECTS[dataset][split]
    except KeyError:
        raise KeyError(
            f"{dataset} has no split {split!r}; known: {sorted(SUBJECTS.get(dataset, ()))}"
        ) from None
    p = dataset if prefix is None else prefix
    return tuple(f"{p}/{k}" for k in keys)


def check_count(dataset: str, split: str, n: int) -> None:
    want = EXPECTED.get((dataset, split))
    if want is not None and n != want:
        raise ValueError(
            f"{dataset}/{split}: loaded {n} questions, expected {want}. A silently different "
            f"denominator makes every accuracy incomparable with the reported one. Re-run the "
            f"data preparation, or update memlife/data/splits.py if the change is intended.")
