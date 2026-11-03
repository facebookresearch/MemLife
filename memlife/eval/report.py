# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Print the numbers for one or more finished runs.

    python -m memlife.eval.report --dataset egolife_eqa --tag memlife --tag memlife_memopt

Every figure is recomputed from ``results/*.json`` here and now -- see :mod:`memlife.eval.score`
for why there is no cache.
"""
from __future__ import annotations

import argparse

from .. import paths
from ..data import LOADERS
from . import score as msc


def report(dataset: str, split: str, tags: list[str], *, segment_s: float) -> int:
    mod = LOADERS[dataset]
    records = mod.load(split)
    streams = {sid: mod.get_stream(sid) for sid in {r.stream_id for r in records}}

    print(f"\n{dataset}/{split}  ({len(records)} questions)")
    print("-" * 78)
    for tag in tags:
        d = paths.pred_dir(dataset, split, tag)
        try:
            rows = msc.load_rows(d)
        except FileNotFoundError:
            print(f"{tag:<34s} no predictions at {d}")
            continue
        s = msc.score(rows, records, streams, segment_s=segment_s)
        print(s.line(tag))
        if s.abstain_floor is not None:
            print(f"{'  always-abstain (free floor)':<34s} acc={s.abstain_floor:6.2%}")
        if s.no_visible_gold:
            print(f"{'  no reachable gold evidence':<34s} {s.no_visible_gold} question(s), "
                  f"excluded from the recall denominator")
        ts = msc.trace_stats(rows)
        if ts:
            print(f"{'  reader':<34s} mean_rounds={ts['mean_rounds']} "
                  f"forced={ts['forced_action_rate']:.2%} no_action={ts['no_action_rate']:.2%} "
                  f"tool_errors={ts['tool_error_rate']:.2%} calls={ts['tool_calls']}")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="supermemory", choices=sorted(paths.DATASETS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", action="append", required=True, dest="tags")
    ap.add_argument("--segment-seconds", type=float, default=30.0)
    a = ap.parse_args()
    return report(a.dataset, a.split, a.tags, segment_s=a.segment_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
