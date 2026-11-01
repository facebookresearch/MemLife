"""The memory store: one entry per segment, plus the index over their descriptions.

An :class:`Entry` is what the writer produced for one 30 s segment and everything the reader
can be shown about it. The store owns the index and the two retrieval operations the tools
expose -- similarity search and interval enumeration -- so the tools stay thin and the
causality rule lives in exactly one place (:mod:`memlife.core.index`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..core import embed
from ..core import io as mio
from ..core.index import NO_CUTOFF, Index


@dataclass
class Entry:
    """One segment's memory."""

    segment_id: str
    video_id: str
    order: int
    t_start: float          #: file-local
    t_end: float
    abs_start: float        #: stream clock
    abs_end: float
    description: str = ""
    transcript: str = ""

    def to_json(self) -> dict:
        return asdict(self)


class MemoryStore:
    """Entries plus the index over their descriptions, for one stream."""

    def __init__(self, stream_id: str, entries: list[Entry] | None = None):
        self.stream_id = stream_id
        self.entries: list[Entry] = list(entries or [])
        self.by_id: dict[str, Entry] = {e.segment_id: e for e in self.entries}
        self.index: Index | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, e: Entry) -> None:
        if e.segment_id in self.by_id:
            raise KeyError(f"duplicate segment_id {e.segment_id!r} in {self.stream_id}")
        self.entries.append(e)
        self.by_id[e.segment_id] = e

    # ------------------------------------------------------------------ vectors

    def build_index(self, *, device: str = "cpu", vecs: np.ndarray | None = None) -> None:
        """Embed the descriptions and build the index.

        CPU by default and **not a free choice**: MemOpt's retrievability reward compares a
        candidate's cosine against cached competitor scores at margins measured as tight as
        7e-06, and a CPU/GPU split alone can flip a hit. ``vecs`` lets a caller hand over
        vectors it has already computed rather than re-encoding every description.
        """
        self.entries.sort(key=lambda e: (e.order, e.abs_start))
        if vecs is None:
            vecs = embed.encode([e.description for e in self.entries], device=device)
        self.index = Index.build(
            vecs, [e.segment_id for e in self.entries],
            [e.abs_start for e in self.entries], [e.abs_end for e in self.entries],
            [e.order for e in self.entries])

    def _need_index(self) -> Index:
        if self.index is None:
            raise RuntimeError(
                f"{self.stream_id}: index not built. Call build_index() or load(). Retrieval "
                f"against an unbuilt store returns nothing, which reads as 'empty memory'.")
        return self.index

    # ------------------------------------------------------------------ retrieval

    def search(self, query: str, k: int, *, now, anchor=None,
               device: str = "cpu") -> list[tuple[Entry, float]]:
        ix = self._need_index()
        q = embed.encode_one(query, device=device)
        return [(self.by_id[h.entry_id], float(h.score))
                for h in ix.search(q, k, now=now, anchor=anchor)]

    def in_window(self, anchor, *, now, limit: int | None = None,
                  sample: str = "even") -> tuple[list[Entry], int]:
        """Backs ``fetch_memory``. Returns ``(entries, n_dropped)`` -- the caller MUST render
        the drop, or a truncated enumeration reads as "nothing happened then"."""
        w = self._need_index().in_window(anchor, now=now, limit=limit, sample=sample)
        return [self.by_id[h.entry_id] for h in w], w.n_dropped

    # ------------------------------------------------------------------ persistence

    def save(self, d: Path, *, build_id: str = "") -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        mio.write_jsonl(d / "entries.jsonl", [e.to_json() for e in self.entries])
        if self.index is not None:
            self.index.save(d / "index", build_id=build_id)
        mio.write_json(d / "meta.json", {"stream_id": self.stream_id, "n": len(self.entries),
                                         "build_id": build_id})

    @classmethod
    def load(cls, d: Path, *, expect_build_id: str | None = None) -> "MemoryStore":
        d = Path(d)
        meta = mio.read_json(d / "meta.json")
        s = cls(meta["stream_id"], [Entry(**r) for r in mio.read_jsonl(d / "entries.jsonl")])
        if len(s) != meta["n"]:
            raise ValueError(f"{d}: meta says {meta['n']} entries, entries.jsonl has {len(s)}")
        s.index = Index.load(d / "index", expect_build_id=expect_build_id)
        if len(s.index) != len(s):
            raise ValueError(f"{d}: index has {len(s.index)} vectors for {len(s)} entries")
        return s


__all__ = ["MemoryStore", "Entry", "NO_CUTOFF"]
