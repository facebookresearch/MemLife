"""Build memory stores.

    python -m memlife.memory.build --dataset egolifeqa --split test --tag base

Every segment is written independently, so the build shards two ways and a killed worker costs
one video, not the dataset:

* ``--num-shards``/``--shard`` over STREAMS when a dataset has many of them, otherwise over the
  VIDEOS inside each stream. Sharding two streams across eight workers would idle six;
* within a shard, each video's descriptions land under ``by_video/<video_id>/`` behind a
  ``.done`` marker carrying :func:`build_id`, so a re-run skips what is finished.

``--stage index`` is the CPU-only closer: it assembles the finished parts, embeds them and
writes the store. Keeping it separate means eight workers can write in parallel without racing
on one index.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .. import config as mcfg
from .. import paths
from ..core import io as mio
from ..core import sampling as sp
from ..core.llm import LLMEngine, ModelSpec
from ..data import LOADERS, prepared, store_dataset
from . import segments as mseg
from .store import Entry, MemoryStore
from .writer import WriterConfig, write


def build_id(cfg, tag: str) -> str:
    """A digest of the config a stored DESCRIPTION depends on.

    This is the guard against store collision: reuse a tag under a different config and the old
    descriptions sit behind a ``.done`` marker, the build skips every video, and the evaluation
    reads one config's memory while the run is labelled with another's. ``model.id`` is in here
    because it is NOT part of the path. ``model.local_path`` is not: a MemOpt store is written
    by a trained checkpoint and read by the base reader, so each checkpoint gets its own tag.
    """
    return mio.fingerprint({
        "model": cfg["model"]["id"], "max_pixels": cfg["model"]["max_pixels"],
        "memory": mcfg.to_plain(cfg["memory"]), "writer": mcfg.to_plain(cfg["writer"]),
        "seed": cfg["seed"], "tag": tag,
    })


def writer_config(cfg) -> WriterConfig:
    return WriterConfig(
        n_frames=int(cfg["memory"]["frames_per_segment"]),
        max_tokens=int(cfg["writer"]["max_tokens"]),
        over=sp.decoding_over(str(cfg["writer"]["decoding"])))


def streams_for(dataset: str, split: str) -> list:
    """The streams a split needs, derived from the QUESTIONS rather than a subject table.

    A stream nobody asks about does not need a store, and one that is asked about must have
    one. Taking the set of ``stream_id``s the split's questions carry gets both right.
    """
    mod = LOADERS[dataset]
    seen: list[str] = []
    for r in mod.load(split):
        if r.stream_id not in seen:
            seen.append(r.stream_id)
    return [mod.get_stream(sid) for sid in seen]


def stream_dir(dataset: str, stream_id: str, tag: str) -> Path:
    return paths.store_dir(store_dataset(dataset), tag) / stream_id.replace("/", "_")


def parts_dir(dataset: str, tag: str) -> Path:
    """Written descriptions, one directory per video, shared by every stream of the store.

    A description depends on the video and the writer, never on the stream clock, so a stream
    that re-times the same recordings (SuperMemory-LVQA) reuses them instead of rewriting.
    """
    return paths.store_dir(store_dataset(dataset), tag) / "by_video"


def _shard(items: list, shard: int, num_shards: int, key) -> list:
    """Stable hash partition, so a worker's slice does not move when the list does."""
    import hashlib

    return [x for x in items
            if int(hashlib.sha256(str(key(x)).encode()).hexdigest(), 16) % num_shards == shard]


def run_write(cfg, engine, dataset: str, stream, turns, *, tag: str,
              shard: int, num_shards: int) -> dict:
    wcfg = writer_config(cfg)
    plan = mseg.plan(stream, segment_s=float(cfg["memory"]["segment_seconds"]),
                     min_tail_s=float(cfg["memory"]["min_tail_seconds"]))
    groups = mseg.by_video(plan)
    out_root = parts_dir(dataset, tag)
    fp = build_id(cfg, tag)

    todo = sorted(groups)
    if num_shards > 1:
        # Longest video first, so the partition does not hand one worker every long recording.
        todo = [v for _, v in sorted((-len(groups[v]), v) for v in todo)]
        todo = _shard(todo, shard, num_shards, key=lambda v: v)

    t0, n_written = time.time(), 0
    for i, vid in enumerate(todo):
        d = out_root / vid
        marker = mio.DoneMarker(d)
        if marker.done(fp):
            continue
        segs = groups[vid]
        descriptions = write(engine, segs, turns.get(vid, []), cfg=wcfg,
                             batch_size=int(cfg["writer"]["batch_size"]))
        mio.write_jsonl(d / "entries.jsonl", [
            {"segment_id": s.segment_id, "video_id": s.video_id, "order": s.order,
             "t_start": s.t_start, "t_end": s.t_end,
             "abs_start": s.abs_start, "abs_end": s.abs_end,
             "description": desc,
             # Stored but never shown on a retrieved row -- it is what MemOpt's faithfulness
             # audit scores the memory against. See memlife/memory/tools.py.
             "transcript": _transcript(turns.get(vid, []), s)}
            for s, desc in zip(segs, descriptions)])
        marker.mark(fp, {"segments": len(segs)})
        n_written += len(segs)
        el = time.time() - t0
        print(f"[write] {stream.stream_id} {vid}: {len(segs)} segments "
              f"({i + 1}/{len(todo)} videos, {el / max(1, n_written):.2f} s/segment)", flush=True)
    return {"videos": len(todo), "segments_written": n_written,
            "seconds": round(time.time() - t0, 1)}


def _transcript(turns, seg) -> str:
    from ..core import asr as masr

    return masr.render(masr.slice_window(turns, seg.t_start, seg.t_end))


def collect(cfg, dataset: str, stream, tag: str) -> MemoryStore:
    """Assemble the finished per-video parts. Fails on a missing or STALE part.

    Building a store from a partial or mixed set would evaluate against a memory with holes in
    it, and nothing downstream would notice. The clock fields are re-derived from THIS stream,
    since a part may have been written for another stream over the same video.
    """
    root = parts_dir(dataset, tag)
    fp = build_id(cfg, tag)
    grid = {g.segment_id: g for g in mseg.plan(
        stream, segment_s=float(cfg["memory"]["segment_seconds"]),
        min_tail_s=float(cfg["memory"]["min_tail_seconds"]))}
    s = MemoryStore(stream.stream_id)
    missing = []
    for v in stream.videos:
        d = root / v.video_id
        if not mio.DoneMarker(d).done(fp):
            missing.append(v.video_id)
            continue
        for r in mio.read_jsonl(d / "entries.jsonl"):
            g = grid[r["segment_id"]]
            s.add(Entry(**{**r, "order": g.order, "abs_start": g.abs_start,
                           "abs_end": g.abs_end}))
    if missing:
        raise RuntimeError(
            f"{stream.stream_id}: {len(missing)} video(s) have no part matching this config "
            f"(e.g. {missing[:3]}). Re-run --stage write with this config.")
    return s


def run_index(cfg, dataset: str, stream, *, tag: str) -> dict:
    s = collect(cfg, dataset, stream, tag)
    s.build_index()
    s.save(stream_dir(dataset, stream.stream_id, tag), build_id=build_id(cfg, tag))
    return {"entries": len(s)}


def main() -> int:
    ap = mcfg.add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--split", default="test")
    ap.add_argument("--stage", required=True, choices=["write", "index"])
    ap.add_argument("--stream", default=None, help="one stream_id; default: all in the split")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--tag", default="default", help="names this store")
    a = ap.parse_args()

    cfg = mcfg.from_args(a)
    ds = a.dataset
    src = store_dataset(ds)
    streams = streams_for(ds, a.split)
    if a.stream:
        streams = [s for s in streams if s.stream_id == a.stream]
        if not streams:
            raise SystemExit(f"no stream {a.stream!r} in {ds}/{a.split}")

    shard, num_shards = a.shard, a.num_shards
    if num_shards > 1 and len(streams) >= num_shards:
        streams = _shard(streams, shard, num_shards, key=lambda s: s.stream_id)
        print(f"[build] sharding by stream: {len(streams)} on this worker", flush=True)
        shard, num_shards = 0, 1
    if a.stage == "index" and num_shards > 1:
        raise SystemExit(
            "--stage index cannot be sharded: it reads every video's part for a stream and "
            "writes one index, so workers would race on the store directory. Shard across "
            "streams with --stream instead.")

    print(f"[build] {mcfg.describe(cfg)}  stage={a.stage} tag={a.tag}", flush=True)
    engine = (LLMEngine(ModelSpec.from_cfg(cfg["model"]), "writer", seed=int(cfg["seed"]))
              if a.stage == "write" else None)
    has_asr = bool(cfg["dataset"].get("has_asr", True))
    for s in streams:
        if a.stage == "write":
            turns = prepared.transcripts_for(src, s, expected=has_asr)
            print(f"[write] {s.stream_id}: {len(s.videos)} videos", flush=True)
            st = run_write(cfg, engine, ds, s, turns, tag=a.tag, shard=shard,
                           num_shards=num_shards)
            print(f"[write] {st}", flush=True)
        else:
            print(f"[index] {s.stream_id}: {run_index(cfg, ds, s, tag=a.tag)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
