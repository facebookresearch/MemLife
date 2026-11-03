# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Exact cosine retrieval over memory entries, with a causal time filter.

A dense matmul over L2-normalised vectors: exact, no training, no recall cliff. These corpora
run to tens of thousands of entries, not millions, so an approximate index would trade
correctness for speed we do not need -- and a recall cliff is indistinguishable from bad memory.

Two invariants this module exists to hold:

**A filtered search must never under-return.** Scoring everything and masking afterwards is
the only way to guarantee that. Over-fetching a multiple of ``k`` from a global ranking and
then filtering looks fine on a small corpus and fails completely on a large one: with 6k
entries, a mid-stream window of 1.2k eligible entries can return *zero* hits at k=32 because
none of them are in the global top-2048. The reader then concludes "no memory about this" and
answers from priors -- plausible accuracy, wrong reason.

**A causal cutoff means fully in the past.** ``t_end <= now``, not overlap. With 30 s segments
and overlap semantics the segment straddling the question is eligible, and its description was
written from frames spanning up to 30 s *after* the question was asked. On these benchmarks the
answer event usually sits just before the query time, so that straddling segment is exactly the
one most likely to be retrieved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import embed as _embed
from . import io as mio
from .embed import DIM


class _NoCutoff:
    def __repr__(self) -> str:
        return "NO_CUTOFF"


#: Pass as ``now=`` to disable the causal filter ON PURPOSE.
#:
#: ``now`` is required precisely so that forgetting it is a TypeError rather than a silent
#: revert to non-causal retrieval -- and a causality leak RAISES accuracy, so it does not look
#: like a bug.
NO_CUTOFF = _NoCutoff()


@dataclass(frozen=True)
class Hit:
    idx: int
    #: Cosine similarity, or ``None`` for an enumeration (:meth:`Index.in_window`) where there
    #: is no query. Synthesising 1.0 there would make every rank-based reward constant.
    score: float | None
    entry_id: str


@dataclass(frozen=True)
class Window:
    """An enumeration result, carrying what was DROPPED as well as what was kept.

    A bare list lets a renderer silently show the last 10 of 500 matching entries, so the model
    concludes "nothing happened then" from a truncation it was never told about. Callers must
    render :attr:`n_dropped`.
    """

    hits: tuple[Hit, ...]
    n_total: int

    @property
    def n_dropped(self) -> int:
        return self.n_total - len(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)


def _cutoff(now) -> float | None:
    if isinstance(now, _NoCutoff):
        return None
    if now is None:
        raise TypeError(
            "`now` is required. Pass the question's causal cutoff, or index.NO_CUTOFF to "
            "disable the filter deliberately. Omitting it silently reverts to non-causal "
            "retrieval, which RAISES accuracy and so does not look like a bug.")
    v = float(now)
    if math.isnan(v):
        raise ValueError("causal cutoff `now` is NaN")
    return v


def _valid_window(w: tuple[float, float] | None) -> tuple[float, float] | None:
    """Reject inverted, empty or NaN windows loudly.

    All three otherwise return ``[]``, which is the same signature as "nothing in memory yet".
    """
    if w is None:
        return None
    lo, hi = float(w[0]), float(w[1])
    if math.isnan(lo) or math.isnan(hi):
        raise ValueError(f"time window contains NaN: {w!r}")
    if hi <= lo:
        raise ValueError(f"empty or inverted time window {w!r} (hi must exceed lo)")
    return lo, hi


def is_eligible(t_start: float, t_end: float,
                anchor: tuple[float, float] | None, now: float | None) -> bool:
    """THE eligibility rule, in one place; ``anchor``/``now`` already normalised."""
    if now is not None and t_end > now:
        return False                                   # causal: fully in the past only
    if anchor is not None and not (t_start < anchor[1] and t_end > anchor[0]):
        return False                                   # anchor: overlap is what a user means
    return True


class Index:
    """Vectors plus the metadata needed to filter them and join back to entries.

    Times are on the dataset's **absolute** clock. File-relative seconds belong to
    :mod:`memlife.core.frames` and :mod:`memlife.core.asr`, never here.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.vecs = np.zeros((0, dim), dtype="float32")
        self.ids: list[str] = []
        self.t_start: np.ndarray = np.zeros(0, dtype="float64")
        self.t_end: np.ndarray = np.zeros(0, dtype="float64")
        self.order: np.ndarray = np.zeros(0, dtype="int64")

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, vecs: np.ndarray, ids: Sequence[str], t_start: Sequence[float],
              t_end: Sequence[float], order: Sequence[int]) -> "Index":
        n = len(ids)
        if not (len(t_start) == len(t_end) == len(order) == n == vecs.shape[0]):
            raise ValueError("Index.build(): ragged inputs")
        self = cls(dim=int(vecs.shape[1]) if n else DIM)
        self.vecs = np.ascontiguousarray(vecs, dtype="float32")
        self.ids = list(ids)
        self.t_start = np.asarray(t_start, dtype="float64")
        self.t_end = np.asarray(t_end, dtype="float64")
        self.order = np.asarray(order, dtype="int64")
        self.check_order()
        return self

    # ------------------------------------------------------------------ eligibility

    def eligible_mask(self, anchor, now) -> np.ndarray:
        anchor, now = _valid_window(anchor), _cutoff(now)
        m = np.ones(len(self), dtype=bool)
        if now is not None:
            m &= self.t_end <= now
        if anchor is not None:
            m &= (self.t_start < anchor[1]) & (self.t_end > anchor[0])
        return m

    # ------------------------------------------------------------------ queries

    def search(self, query: np.ndarray, k: int, *, now, anchor=None) -> list[Hit]:
        """Top-``k`` eligible entries by cosine, returned in CHRONOLOGICAL order.

        Those are two separate things and the distinction is load-bearing: membership is
        decided purely by relevance, and the time ordering is applied afterwards to the chosen
        set. Returning them ranked instead changes what the reader can reason about -- a
        timeline it can read as a sequence of events, rather than a list it has to re-sort.

        ``anchor`` is intersected with the causal cutoff, never substituted for it.
        """
        q = np.asarray(query, dtype="float32").reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != index dim {self.dim}")
        a, n = _valid_window(anchor), _cutoff(now)
        if a is not None and n is not None and a[0] >= n:
            raise ValueError(
                f"anchor {a} starts at or after the causal cutoff {n}: the query asks only "
                f"about the future and can never return anything. Treat it as a model error "
                f"and count it, rather than reading [] as 'no memory'.")
        if not len(self) or k <= 0:
            return []

        mask = self.eligible_mask(anchor, now)
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            return []
        scores = self.vecs[cand] @ q
        take = min(k, cand.size)
        # argpartition then sort the survivors: O(n) instead of O(n log n) on the full corpus.
        top = cand[np.argpartition(-scores, take - 1)[:take]]
        top = top[np.argsort(-(self.vecs[top] @ q), kind="stable")]
        hits = [Hit(int(i), float(self.vecs[i] @ q), self.ids[int(i)]) for i in top]
        hits.sort(key=lambda h: (self.order[h.idx], self.t_start[h.idx]))
        return hits

    def in_window(self, anchor, *, now, limit: int | None = None,
                  sample: str = "even") -> Window:
        """Every eligible entry overlapping ``anchor``, chronologically. Backs ``fetch_memory``.

        ``Hit.score`` is ``None``: there is no query, so there is no similarity.

        ``sample`` decides which survive the cap. ``"even"`` spreads them across the whole
        window -- right for "list everything I bought today", which ``"recent"`` answers with
        the tail of the day while looking complete. ``"recent"`` keeps the last ``limit``.
        """
        a, n = _valid_window(anchor), _cutoff(now)
        if a is None:
            raise ValueError("in_window needs a time window")
        if n is not None and a[0] >= n:
            raise ValueError(
                f"anchor {a} starts at or after the causal cutoff {n}: the query asks only "
                f"about the future. Count it as a model error; do not read [] as empty.")
        idx = np.flatnonzero(self.eligible_mask(anchor, now))
        idx = idx[np.lexsort((self.t_start[idx], self.order[idx]))]
        out = [Hit(int(i), None, self.ids[int(i)]) for i in idx]
        n_total = len(out)
        if limit is not None:
            if limit < 0:
                raise ValueError(f"limit must be >= 0, got {limit}")
            if sample not in ("recent", "even"):
                raise ValueError(f"sample must be 'recent' or 'even', got {sample!r}")
            if not limit:
                out = []
            elif limit < n_total:
                if sample == "recent":
                    out = out[n_total - limit:]        # NOT out[-limit:]; limit>0 here, but
                else:                                  # the explicit form documents the intent
                    keep = (sorted({round(i * (n_total - 1) / (limit - 1)) for i in range(limit)})
                            if limit > 1 else [n_total - 1])
                    out = [out[i] for i in keep]
        return Window(tuple(out), n_total)

    # ------------------------------------------------------------------ persistence

    def save(self, stem: Path, *, build_id: str = "") -> None:
        """Vectors then metadata, metadata LAST so a torn write is visible on load."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        vec_p = Path(str(stem) + ".npy")
        self.check_order()
        tmp = Path(str(vec_p) + ".partial")
        np.save(tmp, self.vecs)
        # np.save appends .npy when the name lacks it; ours ends in .partial, so it does.
        Path(str(tmp) + ".npy").replace(vec_p)
        mio.write_json(Path(str(stem) + ".meta.json"), {
            "dim": self.dim, "ids": self.ids,
            "t_start": self.t_start.tolist(), "t_end": self.t_end.tolist(),
            "order": self.order.tolist(),
            # Provenance. Without it a killed rebuild leaves the previous run's vectors beside
            # this run's metadata: identical ids, identical count, every vector wrong.
            "build_id": build_id,
            "embed_model": _embed.MODEL_ID,
            "embed_device": _embed.device_in_use(),
        })

    @classmethod
    def load(cls, stem: Path, *, expect_build_id: str | None = None) -> "Index":
        stem = Path(stem)
        meta = mio.read_json(Path(str(stem) + ".meta.json"))
        self = cls(dim=int(meta["dim"]))
        self.vecs = np.ascontiguousarray(np.load(Path(str(stem) + ".npy")), dtype="float32")
        self.ids = list(meta["ids"])
        self.t_start = np.asarray(meta["t_start"], dtype="float64")
        self.t_end = np.asarray(meta["t_end"], dtype="float64")
        self.order = np.asarray(meta["order"], dtype="int64")
        if self.vecs.shape[0] != len(self.ids):
            raise ValueError(
                f"{stem}: {self.vecs.shape[0]} vectors for {len(self.ids)} ids -- the two files "
                f"are from different builds")
        if self.vecs.shape[1] != self.dim:
            raise ValueError(f"{stem}: vector dim {self.vecs.shape[1]} != metadata {self.dim}")
        if (got := meta.get("embed_model")) and got != _embed.MODEL_ID:
            raise ValueError(
                f"{stem}: built with embedder {got!r}, this process uses {_embed.MODEL_ID!r}. "
                f"Any other 1024-d encoder passes the dim and count checks while silently "
                f"changing every retrieval.")
        if (got := meta.get("embed_device")) and (cur := _embed.device_in_use()) and got != cur:
            raise ValueError(
                f"{stem}: built with the embedder on {got!r}, this process has it on {cur!r}. "
                f"CPU/GPU divergence flips retrieval hits at ~1e-6 margins.")
        self.check_order()
        if expect_build_id is not None and meta.get("build_id") != expect_build_id:
            raise ValueError(
                f"{stem}: build_id {meta.get('build_id')!r} != expected {expect_build_id!r} -- "
                f"a rebuild was interrupted, or this store belongs to a different config.")
        return self

    def check_order(self) -> None:
        """``order`` is the primary sort key for a render the prompt calls chronological, so it
        must agree with time. A builder that restarted ``order`` per session would otherwise put
        session 5's first segment ahead of session 1's fourth -- well-formed on disk, scrambled
        in the prompt."""
        if len(self.order) != len(set(self.order.tolist())):
            raise ValueError("index `order` values are not unique")
        by_order = np.argsort(self.order, kind="stable")
        ts, te = self.t_start[by_order], self.t_end[by_order]
        # Non-decreasing in time. Equal starts across two videos are legal; an inversion is not.
        bad = np.flatnonzero((ts[:-1] > ts[1:]) | ((ts[:-1] == ts[1:]) & (te[:-1] > te[1:])))
        if bad.size:
            i, j = int(by_order[bad[0]]), int(by_order[bad[0] + 1])
            raise ValueError(
                f"index `order` disagrees with time: {self.ids[i]!r} (order {self.order[i]}, "
                f"t={self.t_start[i]}) precedes {self.ids[j]!r} (order {self.order[j]}, "
                f"t={self.t_start[j]}). Chronological rendering would show a scrambled timeline.")

