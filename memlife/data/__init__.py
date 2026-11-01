"""Benchmark loaders. Every one exposes ``load(split, *, limit)`` and ``get_stream(id)``.

``STORE_DATASET`` names the prepared dataset whose recordings, transcripts and memory store a
benchmark uses. It differs from the benchmark name for EgoLife-EQA, which asks new questions
about EgoLifeQA's recordings, and for SuperMemory-LVQA, which re-times SuperMemory's.
"""
from __future__ import annotations

from . import egolife_eqa, egolifeqa, supermemory, supermemory_lvqa

LOADERS = {
    "supermemory": supermemory,
    "egolifeqa": egolifeqa,
    "supermemory_lvqa": supermemory_lvqa,
    "egolife_eqa": egolife_eqa,
}


def store_dataset(dataset: str) -> str:
    """Which prepared dataset holds the videos, transcripts and store for ``dataset``."""
    return getattr(LOADERS[dataset], "STORE_DATASET", dataset)


__all__ = ["LOADERS", "store_dataset"]
