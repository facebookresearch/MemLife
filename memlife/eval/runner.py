"""**The** evaluation loop. Every dataset, every split, both reader variants.

    python -m memlife.eval --dataset egolife_eqa --tag memlife --store-tag base

One loop rather than one per configuration is the point: it owns the things that must not vary
between runs -- the causal cutoff, the per-question output contract, resume, and what counts as
a completed run -- so a run can only differ in how it answers.

Resume is per question, not per shard, so workers can be added or rebalanced mid-sweep: a
question that failed is simply absent and gets retried.
"""
from __future__ import annotations

import argparse
import time
import traceback

from .. import config as mcfg
from .. import paths
from ..core import io as mio
from ..core import sampling as sp
from ..core.llm import LLMEngine, ModelSpec
from ..data import LOADERS, prepared, store_dataset
from ..data.schema import QARecord
from ..memory import reader as mreader
from ..memory import tools as T
from ..memory.build import build_id, stream_dir
from ..memory.store import MemoryStore

#: Consecutive per-question failures that mean the environment is broken, not the questions.
_FAIL_STREAK = 3


class MemLife:
    """A prebuilt store plus the agentic reader."""

    def __init__(self, cfg, store_tag: str, store_cfg=None):
        self.cfg = cfg
        self.store_tag = store_tag
        # The config the STORE was written under. Normally this one; it differs only for a
        # cross-backbone cell, where one model's reader is pointed at another's memory.
        self.store_cfg = cfg if store_cfg is None else store_cfg
        self.engine = LLMEngine(ModelSpec.from_cfg(cfg["model"]), "reader",
                                seed=int(cfg["seed"]))
        r = cfg["reader"]
        self.tool_cfg = T.ToolConfig(
            top_k=int(r["top_k"]),
            fetch_memory_limit=int(r["fetch_memory_limit"]),
            source_video=bool(r["source_video"]),
            fetch_video_frames=int(r["fetch_video_frames"]),
            max_fetch_video_frames=int(r["max_fetch_video_frames"]),
            asr_render_chars_video=int(r["asr_render_chars_video"]),
        )
        self.reader_cfg = mreader.ReaderConfig(
            over=sp.decoding_over(str(r["decoding"])),
            rounds=int(r["rounds"]),
            source_video=bool(r["source_video"]),
            max_tokens=int(r["round_max_tokens"]),
            context_guard_tokens=int(r["context_guard_tokens"]),
            day_seconds=cfg["dataset"].get("seconds_per_day"),
            # The UPPER BOUND for any frame under the pixel cap. A guard that under-estimates
            # never fires and the run dies on a context overflow; over-estimating costs only a
            # little headroom.
            tokens_per_image=(int(cfg["model"]["max_pixels"])
                              // int(cfg["model"]["grid_factor"]) ** 2),
        )
        self._stores: dict[str, MemoryStore] = {}

    def store(self, dataset: str, stream_id: str) -> MemoryStore:
        """The store for one stream. ``dataset`` is the BENCHMARK name; ``stream_dir``
        resolves which prepared dataset actually owns the store, so EgoLife-EQA reuses
        EgoLifeQA's rather than demanding an identical rebuild."""
        if stream_id not in self._stores:
            d = stream_dir(dataset, stream_id, self.store_tag)
            if not (d / "meta.json").exists():
                raise FileNotFoundError(
                    f"no store at {d}. Build it first:\n"
                    f"  python -m memlife.memory.build --dataset {dataset} "
                    f"--tag {self.store_tag} --stage write\n"
                    f"  python -m memlife.memory.build --dataset {dataset} "
                    f"--tag {self.store_tag} --stage index")
            # The build id must match, or the run reads a store written under a different
            # writer, model or seed while being reported under this config. The path and tag
            # alone do not encode that.
            self._stores[stream_id] = MemoryStore.load(
                d, expect_build_id=build_id(self.store_cfg, self.store_tag))
        return self._stores[stream_id]

    def warm(self) -> None:
        """Force the engine up now, so an init failure is an init failure."""
        _ = self.engine.llm

    def predict(self, rec: QARecord, stream, turns) -> dict:
        tl = T.Tools(self.store(rec.dataset, rec.stream_id), stream, turns,
                     now=rec.now_abs, cfg=self.tool_cfg)
        tr = mreader.answer(self.engine, tl, rec, cfg=self.reader_cfg)
        return {"question_id": tr.question_id, "letter": tr.letter,
                "rounds_used": tr.rounds_used, "forced_action": tr.forced_action,
                "no_action": tr.no_action, "parse_failed": tr.parse_failed,
                "rounds_exhausted": tr.rounds_exhausted, "frames_dropped": tr.frames_dropped,
                # Always saved: retrieval recall and MemOpt's retrievability replay are both
                # computed from it, so a run without it cannot be scored or trained against.
                "calls": tr.calls,
                "n_errors": len(tr.errors), "errors": tr.errors[:5], "raw": tr.raw}


def run_id(cfg, *, tag: str, store_tag: str, store_cfg=None) -> str:
    """A digest of everything the ANSWER can depend on.

    A config change therefore cannot be resumed into an existing directory -- half the rows
    would come from one setting and half from another, and the mixed accuracy would look like
    a real number.
    """
    return mio.fingerprint({
        "model": cfg["model"]["id"], "memory": mcfg.to_plain(cfg["memory"]),
        "reader": mcfg.to_plain(cfg["reader"]), "writer": mcfg.to_plain(cfg["writer"]),
        "seed": cfg["seed"], "tag": tag, "store_tag": store_tag,
        "overrides": cfg["_resolved"]["overrides"],
        **({"store_model": store_cfg["model"]["id"]} if store_cfg is not None else {}),
    })


def _shard(items: list, shard: int, num_shards: int, key) -> list:
    import hashlib

    return [x for x in items
            if int(hashlib.sha256(str(key(x)).encode()).hexdigest(), 16) % num_shards == shard]


def run(cfg, *, dataset: str, split: str, tag: str, store_tag: str,
        shard: int = 0, num_shards: int = 1, limit: int | None = None,
        store_cfg=None) -> dict:
    mod = LOADERS[dataset]
    src = store_dataset(dataset)
    records = mod.load(split, limit=limit)

    out_dir = paths.pred_dir(dataset, split, tag)
    fp = run_id(cfg, tag=tag, store_tag=store_tag, store_cfg=store_cfg)
    writer = mio.PerQidWriter(out_dir, fp)
    save_raw = bool(cfg["reader"]["save_raw"])

    mine = records if num_shards == 1 else _shard(records, shard, num_shards,
                                                  key=lambda r: r.question_id)
    todo = [r for r in mine if not writer.already_done(r.question_id)]
    print(f"[eval] {dataset}/{split} shard {shard}/{num_shards}: {len(todo)} of {len(mine)} "
          f"questions to do ({len(records)} in split)", flush=True)

    method = MemLife(cfg, store_tag, store_cfg)
    stream_ids = {r.stream_id for r in todo}
    streams = {sid: mod.get_stream(sid) for sid in stream_ids}
    has_asr = bool(cfg["dataset"].get("has_asr", True))
    turns = {sid: prepared.transcripts_for(src, streams[sid], expected=has_asr)
             for sid in stream_ids}

    # Load every store and the engine BEFORE the loop. Both are lazy, so without this the
    # first question is where an out-of-memory error, a wedged engine or a build-id mismatch
    # surfaces -- and the per-question handler below would treat that as "this question
    # failed" and retry the same slow failure for every remaining record.
    if todo:
        for sid in stream_ids:
            method.store(dataset, sid)
        method.warm()

    t0, n_ok, n_fail, streak = time.time(), 0, 0, 0
    for i, rec in enumerate(todo):
        try:
            row = method.predict(rec, streams[rec.stream_id], turns[rec.stream_id])
            row.update(correct_letter=rec.correct_letter,
                       correct=(row.get("letter") == rec.correct_letter),
                       now_abs=rec.now_abs, stream_id=rec.stream_id, meta=rec.meta)
            # The generation goes to a SIDE directory, never into results/: a ten-round trace
            # is large, and folding it in would multiply the cost of every report, which is
            # dominated by reading those files. Scoring never needs it; debugging always does.
            raw = row.pop("raw", None)
            if raw is not None and save_raw:
                mio.write_json(out_dir / "raw" / f"{rec.question_id}.json",
                               {"question_id": rec.question_id, "letter": row.get("letter"),
                                "raw": raw})
            writer.write(rec.question_id, row)
            n_ok, streak = n_ok + 1, 0
        except Exception as e:                           # noqa: BLE001 -- one row must not end it
            # The row is NOT written, so a resume retries it and the completeness check still
            # sees the gap.
            n_fail, streak = n_fail + 1, streak + 1
            print(f"[eval] FAIL {rec.question_id}: {type(e).__name__}: {e}", flush=True)
            if n_fail <= 3:
                traceback.print_exc()
            if streak >= _FAIL_STREAK:
                print(f"[eval] ABORT: {streak} consecutive failures -- this is the "
                      f"environment, not the questions. {n_ok} rows committed; the rest stay "
                      f"absent so a resume retries them.", flush=True)
                break
        if (i + 1) % 20 == 0:
            print(f"[eval] {i + 1}/{len(todo)}  {(time.time() - t0) / (i + 1):.1f} s/q  "
                  f"ok={n_ok} fail={n_fail}", flush=True)

    done = writer.count()
    # `--limit` truncates `records`, so `done == len(records)` is true for a 20-question smoke
    # run. Marking that complete would write a stamp claiming the sweep finished.
    if num_shards == 1 and limit is None and done == len(records):
        writer.mark_complete(len(records))
    return {"todo": len(todo), "ok": n_ok, "failed": n_fail, "on_disk": done,
            "expected": len(mine), "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = mcfg.add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="default", help="names this RUN's prediction directory")
    ap.add_argument("--store-tag", default=None,
                    help="which memory store to READ; defaults to --tag. Separate them when "
                         "only the reader changes -- MemLife and MemLife-V read ONE store -- so "
                         "that an identical store is not rebuilt and the two runs' predictions "
                         "do not land in the same directory.")
    ap.add_argument("--store-model", default=None, metavar="NAME",
                    help="the configs/model/<NAME>.yaml the STORE was built with; defaults to "
                         "--model. Set it only to read one model's memory with another's "
                         "reader, which is how the writer's contribution is separated from "
                         "the reader's.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()

    cfg = mcfg.from_args(a)
    # Resolve the store's config through the SAME loader with the SAME overlays, so memory,
    # writer and seed all still have to agree -- only the model differs.
    store_cfg = None
    if a.store_model and a.store_model != a.model:
        store_cfg = mcfg.load(dataset=a.dataset, model=a.store_model,
                              overrides=a.overrides)
        print(f"[eval] cross-backbone: reading a store built by {store_cfg['model']['id']} "
              f"with the {cfg['model']['id']} reader", flush=True)

    print(f"[eval] {mcfg.describe(cfg)}  tag={a.tag}", flush=True)
    st = run(cfg, dataset=a.dataset, split=a.split, tag=a.tag,
             store_tag=a.store_tag or a.tag, shard=a.shard, num_shards=a.num_shards,
             limit=a.limit, store_cfg=store_cfg)
    print(f"[eval] {st}", flush=True)
    return 0 if st["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
