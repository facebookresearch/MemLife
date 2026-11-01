"""Build the MemOpt rollout dataset: one row per pooled segment.

    python -m memlife.memopt.data --dataset supermemory --split train

Two artifacts, and the point of building them together is that there is exactly ONE copy of
each segment's pixels:

``frames/<segment_id>/{0..7}.jpg``   extracted once, sampled exactly as the writer samples
                                    them. The trainer hands these to the policy and the
                                    faithfulness audit reads the same files, so the evaluator
                                    scores **the pixels the writer actually saw**. A second,
                                    independently resized copy would not be a faithfulness
                                    score.
``rollouts.parquet``                the prompts, built through
                                    :func:`memlife.memory.writer.build_messages` -- the same
                                    function the deployed build calls. That seam is what makes
                                    "train == deploy" true by construction rather than by
                                    review.

Only segments that survived key-fact extraction are included: a segment with no question left
has no informativeness and no retrievability signal, so training on it would be optimising
faithfulness alone under the full objective's name.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .. import config as mcfg
from .. import paths
from ..core import asr as masr
from ..core import frames as mframes
from ..data import LOADERS, prepared, store_dataset
from ..memory import segments as mseg
from ..memory.writer import WriterConfig, build_messages
from . import keyfacts


def frames_root(dataset: str, split: str) -> Path:
    return keyfacts.out_dir(dataset, split) / "frames"


def extract_frames(cfg, dataset: str, split: str, segment_ids: set[str], *,
                   shard: int = 0, num_shards: int = 1) -> int:
    """Write each pooled segment's frames to disk, skipping those already there.

    ``num_shards > 1`` takes only this worker's segments (a stable hash of the segment id).
    """
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} out of range for --num-shards {num_shards}")
    segment_ids = {s for s in segment_ids
                   if int(hashlib.sha256(s.encode()).hexdigest(), 16) % num_shards == shard}
    mod = LOADERS[dataset]
    n_frames = int(cfg["memory"]["frames_per_segment"])
    root = paths.ensure(frames_root(dataset, split))
    want_by_video: dict[str, set[str]] = {}
    for sid in segment_ids:
        # A segment id is "<video_id>#<index>", and a video belongs to exactly one stream.
        want_by_video.setdefault(sid.split("#")[0], set()).add(sid)

    n = 0
    for stream_id in {r.stream_id for r in mod.load(split)}:
        stream = mod.get_stream(stream_id)
        vids = {v.video_id for v in stream.videos}
        grid = {s.segment_id: s for s in mseg.plan(
            stream, segment_s=float(cfg["memory"]["segment_seconds"]),
            min_tail_s=float(cfg["memory"]["min_tail_seconds"]))}
        for vid in vids & set(want_by_video):
            for sid in sorted(want_by_video[vid]):
                d = root / sid
                if (d / f"{n_frames - 1}.jpg").exists():
                    continue
                seg = grid.get(sid)
                if seg is None:
                    raise KeyError(
                        f"{sid} is in the pool but not on {stream_id}'s segment grid -- the "
                        f"key facts were extracted at a different memory.segment_seconds.")
                imgs = mframes.sample_frames(seg.path, seg.t_start, seg.t_end, n_frames,
                                             allow_pad=False)
                d.mkdir(parents=True, exist_ok=True)
                for i, im in enumerate(imgs):
                    im.save(d / f"{i}.jpg", quality=95)
                n += 1
                if n % 200 == 0:
                    print(f"[data] extracted {n} segments", flush=True)
    print(f"[data] frames for {len(segment_ids)} segments under {root} ({n} new)", flush=True)
    return n


def build_rows(cfg, dataset: str, split: str, segment_ids: set[str]) -> list[dict]:
    """One parquet row per pooled segment."""
    mod = LOADERS[dataset]
    src = store_dataset(dataset)
    wcfg = WriterConfig(n_frames=int(cfg["memory"]["frames_per_segment"]),
                        max_tokens=int(cfg["writer"]["max_tokens"]))
    # Resolved once: per-frame resolve() walks the whole path on a network filesystem.
    root = frames_root(dataset, split).resolve()
    has_asr = bool(cfg["dataset"].get("has_asr", True))
    rows: list[dict] = []

    for stream_id in sorted({r.stream_id for r in mod.load(split)}):
        stream = mod.get_stream(stream_id)
        turns = prepared.transcripts_for(src, stream, expected=has_asr)
        for seg in mseg.plan(stream, segment_s=float(cfg["memory"]["segment_seconds"]),
                             min_tail_s=float(cfg["memory"]["min_tail_seconds"])):
            if seg.segment_id not in segment_ids:
                continue
            paths_for_seg = [str(root / seg.segment_id / f"{i}.jpg")
                             for i in range(wcfg.n_frames)]
            # The SAME assembly the deployed writer uses; only the image parts differ,
            # carrying paths instead of pixels.
            msgs = build_messages(seg, turns.get(seg.video_id, []), cfg=wcfg,
                                  sample=lambda *_a, **_k: paths_for_seg)
            prompt = [{"role": m["role"], "content": _flatten(m["content"])} for m in msgs]
            transcript = masr.render(masr.slice_window(
                turns.get(seg.video_id, []), seg.t_start, seg.t_end))
            rows.append({
                "data_source": f"memlife/{dataset}",
                "prompt": prompt,
                "images": paths_for_seg,
                "ability": "memory_writing",
                # The trainer requires the field; MemOpt has no reference answer, and its
                # absence is the point -- the writer never sees a question.
                "reward_model": {"style": "custom", "ground_truth": ""},
                "extra_info": {"segment_id": seg.segment_id, "transcript": transcript,
                               "stream_id": stream_id},
            })
    return rows


IMAGE_TOKEN = "<image>"


def _flatten(content) -> str:
    """Chat content for the trainer: ONE string, each image part a literal ``<image>``.

    verl splices the row's images in only at ``<image>`` markers inside STRING content; list
    content is passed through untouched, so the policy would see no frames at all. The marker's
    POSITION is where the pixels go, so it comes from the same assembly the deployed writer uses.
    Text parts are joined with nothing between them, which is what the chat template does with
    separate content items. A literal ``<image>`` in the text (a transcript can carry one) would
    consume an image and shift every later frame, so it is refused.
    """
    parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
    out = []
    for p in parts:
        if p["type"] in ("image_pil", "image"):
            out.append(IMAGE_TOKEN)
        elif IMAGE_TOKEN in p["text"]:
            raise ValueError(f"text part contains the literal {IMAGE_TOKEN!r}, which the trainer "
                             f"reads as an image slot: {p['text'][:120]!r}")
        else:
            out.append(p["text"])
    return "".join(out)


def main() -> int:
    ap = mcfg.add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=None, help="default: <memopt dir>/rollouts.parquet")
    ap.add_argument("--shard", type=int, default=0,
                    help="with --num-shards > 1: extract this worker's frames only, no parquet")
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()

    cfg = mcfg.from_args(a)
    pool = keyfacts.load_pool(a.dataset, a.split)
    ids = set(pool)
    print(f"[data] {len(ids)} segments in the pool, "
          f"{sum(len(v) for v in pool.values())} (segment, question) pairs", flush=True)
    extract_frames(cfg, a.dataset, a.split, ids, shard=a.shard, num_shards=a.num_shards)
    if a.num_shards > 1:
        return 0

    rows = build_rows(cfg, a.dataset, a.split, ids)
    if len(rows) != len(ids):
        raise RuntimeError(
            f"built {len(rows)} rows for {len(ids)} pooled segments. A missing row would "
            f"silently shrink the training set; an extra one is a duplicate segment id.")

    import pandas as pd

    out = a.out or (keyfacts.out_dir(a.dataset, a.split) / "rollouts.parquet")
    paths.ensure(out.parent)
    pd.DataFrame(rows).to_parquet(out)
    print(f"[data] wrote {len(rows)} rows to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
