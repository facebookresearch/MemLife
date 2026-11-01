"""Retrievability: would this memory actually be returned at question time?

    R_ret(c, y) = mean over q in Q(c) of  1[ y in C_q ]

Running the whole agentic reader for every candidate of every segment is not affordable, and
it would also make the reward depend on the reader's own sampling. Instead the reader's
actions are **cached once per epoch** (:mod:`memlife.memopt.trace`) and replayed:

* a **search** action returns the candidate when its similarity to that action's query ranks
  inside that action's retrieval budget, among the entries eligible under that action's causal
  and temporal constraints;
* a **fetch** action returns the candidate when the action's interval contained its segment --
  which does not depend on what the candidate says, so it is read straight off the trace.

**The whole replay is one dot product and one comparison per action.** Ranking inside a budget
``k`` is equivalent to beating the ``k``-th best competitor, so an action needs exactly two
numbers on disk: its query vector and that threshold. Nothing here opens a store, loads an
index or reads the questions -- which is what makes it affordable inside the trainer loop.

Ties are resolved the way the index resolves them: rank counts *strictly greater* scores, so
``delivered`` is ``cos >= threshold``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import embed


@dataclass(frozen=True)
class Pack:
    """The replay table. Built by :mod:`memlife.memopt.trace`, read in the reward loop.

    ``query_vecs``  ``(n_events, dim)`` L2-normalised search queries.
    ``thresholds``  ``(n_events,)`` the similarity a candidate must reach to be returned by
                    that action. ``-inf`` when the action's window held fewer entries than its
                    budget, i.e. everything eligible was returned.
    ``pairs``       one per ``(segment, question)``: whether a fetch already delivered it, and
                    which search events belong to it.
    """

    query_vecs: np.ndarray
    thresholds: np.ndarray
    pairs: list[dict]
    by_segment: dict[str, list[int]]
    meta: dict

    def segments(self) -> list[str]:
        return sorted(self.by_segment)

    # ------------------------------------------------------------------ persistence

    def save(self, stem: Path) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(stem) + ".npz", query_vecs=self.query_vecs, thresholds=self.thresholds)
        Path(str(stem) + ".json").write_text(json.dumps(
            {"pairs": self.pairs, "by_segment": self.by_segment, "meta": self.meta}))

    @classmethod
    def load(cls, stem: Path) -> "Pack":
        stem = Path(stem)
        z = np.load(str(stem) + ".npz")
        j = json.loads(Path(str(stem) + ".json").read_text())
        return cls(query_vecs=np.ascontiguousarray(z["query_vecs"], dtype="float32"),
                   thresholds=np.asarray(z["thresholds"], dtype="float64"),
                   pairs=j["pairs"], by_segment=j["by_segment"], meta=j["meta"])


@dataclass(frozen=True)
class Ret:
    """One candidate's retrievability.

    ``from_fetch`` is the share of :attr:`value` that no candidate can move -- questions whose
    memory a fetch action already delivered. A dimension diluted by those produces small
    within-group differences while still looking healthy on its mean, so it belongs in the
    per-step log.
    """

    value: float
    n_q: int
    from_fetch: float


def score(segment_id: str, description: str, pack: Pack, *, device: str = "cpu") -> Ret:
    """Replay every cached action for one candidate description of one segment.

    ``Q(c)`` is non-empty by construction, so this is defined for every segment in the pack --
    unlike a rank-based proxy, it never has to be imputed.
    """
    idx = pack.by_segment.get(segment_id)
    if not idx:
        raise KeyError(
            f"{segment_id!r} is not in the retrievability pack. Its rollouts cannot be scored "
            f"on this dimension; the rollout dataset and the pack disagree about the pool.")
    v = embed.encode_one(description or "", device=device).astype("float64")
    hit = fetch_hit = 0
    for i in idx:
        p = pack.pairs[i]
        if p["fetch_delivered"]:
            # Caption-independent: the enumeration returned this segment whatever it says.
            hit += 1
            fetch_hit += 1
            continue
        ev = p["events"]
        if ev and np.any(pack.query_vecs[ev].astype("float64") @ v >= pack.thresholds[ev]):
            hit += 1
    n = len(idx)
    return Ret(value=hit / n, n_q=n, from_fetch=fetch_hit / n)

