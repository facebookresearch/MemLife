# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build the retrievability replay table from the reader's own cached actions.

    # once per epoch, after rebuilding the memory bank with the current writer
    python -m memlife.eval --dataset supermemory --split train --tag e1 --store-tag e1
    python -m memlife.memopt.trace --dataset supermemory --split train \\
        --pred-tag e1 --store-tag e1 --out runs/memopt/supermemory_train/pack_e1

The reader has already run over the training questions -- that is what produced the prediction
directory -- and every one of its memory accesses is in the saved trace. This stage turns
those into a table that can decide, for a *new* candidate description, whether that action
would have returned it.

**An action needs exactly two numbers.** Ranking inside a budget ``k`` is the same as beating
the ``k``-th best competitor, so a search action reduces to its query vector and that
threshold. The competitor set EXCLUDES the segment being scored, because replay replaces only
that segment's entry and leaves the rest of the bank alone.

A fetch action needs no numbers at all: whether it returned a segment does not depend on what
that segment's description says, so it is read straight off the trace.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import config as mcfg
from .. import paths
from ..core import embed
from ..core.index import is_eligible
from ..data import LOADERS
from ..eval import score as msc
from ..memory.build import build_id, stream_dir
from ..memory.store import MemoryStore
from . import keyfacts
from .retrievability import Pack


def build(cfg, dataset: str, split: str, *, pred_tag: str, store_tag: str) -> Pack:
    mod = LOADERS[dataset]
    pool = keyfacts.load_pool(dataset, split)
    records = {r.question_id: r for r in mod.load(split)}
    rows = {r["question_id"]: r for r in msc.load_rows(paths.pred_dir(dataset, split, pred_tag))}
    print(f"[trace] {len(pool)} pooled segments, {len(rows)} reader traces", flush=True)

    stores: dict[str, MemoryStore] = {}
    row_indices: dict[int, dict[str, int]] = {}

    def _row_index(ix) -> dict[str, int]:
        """``segment_id -> index`` for one store, built once. A linear scan per pair would be
        quadratic over thousands of pairs and thousands of entries."""
        return row_indices.setdefault(id(ix), {s: i for i, s in enumerate(ix.ids)})

    def store_for(stream_id: str) -> MemoryStore:
        if stream_id not in stores:
            stores[stream_id] = MemoryStore.load(
                stream_dir(dataset, stream_id, store_tag),
                expect_build_id=build_id(cfg, store_tag))
        return stores[stream_id]

    query_vecs: list[np.ndarray] = []
    thresholds: list[float] = []
    pairs: list[dict] = []
    by_segment: dict[str, list[int]] = {}
    n_missing_trace = 0

    for segment_id in sorted(pool):
        for qid in sorted(pool[segment_id]):
            row = rows.get(qid)
            rec = records.get(qid)
            if row is None or rec is None:
                n_missing_trace += 1
                continue
            store = store_for(rec.stream_id)
            ix = store.index
            rows_by_id = _row_index(ix)
            entry = store.by_id.get(segment_id)
            if entry is None:
                # The segment is in the pool but not in this store: the bank was built with a
                # different segment grid. Scoring it would compare against the wrong entries.
                raise KeyError(
                    f"{segment_id} is in the key-fact pool but not in the store at "
                    f"{stream_dir(dataset, rec.stream_id, store_tag)}. The pool and the memory "
                    f"bank disagree about the segment grid.")
            row_idx = rows_by_id[segment_id]

            # The trace records the MODEL-FACING tool names, which are what the reader parsed
            # out of its own action line -- not the short keys `reader.tools` uses.
            fetch_delivered = any(
                segment_id in (c.get("hit_ids") or ())
                for c in row.get("calls", ()) if c["name"] == "fetch_memory")

            events: list[int] = []
            if not fetch_delivered:
                for c in row.get("calls", ()):
                    if c["name"] != "search_memory":
                        continue
                    anchor = c.get("time_anchor")
                    anchor = tuple(anchor) if anchor else None
                    # Eligibility depends only on time, so it is the same for every candidate
                    # -- an action that could not have reached this segment contributes
                    # nothing rather than a zero.
                    if not is_eligible(entry.abs_start, entry.abs_end, anchor, rec.now_abs):
                        continue
                    query = (c.get("args") or {}).get("query") or ""
                    if not query.strip():
                        continue
                    qv = embed.encode_one(query)
                    mask = ix.eligible_mask(anchor, rec.now_abs)
                    mask[row_idx] = False               # replay replaces THIS entry
                    comp = ix.vecs[mask] @ qv
                    k = int(c.get("top_k") or 0)
                    if k <= 0:
                        continue
                    # Fewer competitors than the budget: everything eligible comes back, so
                    # the candidate is returned whatever it says.
                    thr = (-np.inf if comp.size < k
                           else float(np.partition(comp, comp.size - k)[comp.size - k]))
                    query_vecs.append(qv)
                    thresholds.append(thr)
                    events.append(len(thresholds) - 1)

            by_segment.setdefault(segment_id, []).append(len(pairs))
            pairs.append({"segment_id": segment_id, "qid": qid,
                          "fetch_delivered": bool(fetch_delivered), "events": events})

    if n_missing_trace:
        print(f"[trace] {n_missing_trace} pooled pair(s) have no reader trace and were "
              f"dropped -- re-run the evaluation over {dataset}/{split} if that is unexpected",
              flush=True)
    vecs = (np.stack(query_vecs).astype("float32") if query_vecs
            else np.zeros((0, embed.DIM), dtype="float32"))
    pack = Pack(query_vecs=vecs, thresholds=np.asarray(thresholds, dtype="float64"),
                pairs=pairs, by_segment=by_segment,
                meta={"dataset": dataset, "split": split, "pred_tag": pred_tag,
                      "store_tag": store_tag, "n_pairs": len(pairs),
                      "n_events": len(thresholds), "n_segments": len(by_segment)})
    n_fetch = sum(p["fetch_delivered"] for p in pairs)
    print(f"[trace] {len(by_segment)} segments, {len(pairs)} pairs, {len(thresholds)} search "
          f"events, {n_fetch} pair(s) already delivered by a fetch "
          f"({n_fetch / max(1, len(pairs)):.1%} of the dimension is caption-independent)",
          flush=True)
    return pack


def main() -> int:
    ap = mcfg.add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--split", default="train")
    ap.add_argument("--pred-tag", required=True, help="the reader run to read actions from")
    ap.add_argument("--store-tag", required=True, help="the memory bank that run read")
    ap.add_argument("--out", required=True, type=Path, help="pack stem, no suffix")
    a = ap.parse_args()
    pack = build(mcfg.from_args(a), a.dataset, a.split,
                 pred_tag=a.pred_tag, store_tag=a.store_tag)
    pack.save(a.out)
    print(f"[trace] wrote {a.out}.npz / {a.out}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
