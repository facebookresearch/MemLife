"""Turn your own recordings into memory, then ask it questions.

    python -m memlife.demo build --videos my_videos/ --out runs/demo
    python -m memlife.demo ask --out runs/demo "Where did I leave my keys?" \\
        --choices "On the desk" "In the car" "In the kitchen" "In my bag"

``build`` re-encodes every ``*.mp4`` under ``--videos`` (needs ffmpeg), places them back-to-back
in file-name order on one clock, and writes one memory entry per 30 s segment. Speech is
optional: pass ``--transcripts`` with one ``<video name>.json`` per video, a list of
``{"start", "end", "text"}`` in seconds from the start of that file.

``ask`` answers one multiple-choice question from the memory, as of the end of the recording
or as of ``--at`` seconds. ``--source-video`` lets the reader re-watch footage (MemLife-V).
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import config as mcfg
from .core import asr as masr
from .core import io as mio
from .core.llm import LLMEngine, ModelSpec
from .core.mcq import LETTERS
from .data.prepare.transcode import transcode_one
from .data.schema import QARecord, Stream, VideoRef
from .eval.runner import MemLife
from .memory import reader as mreader
from .memory import segments as mseg
from .memory import tools as T
from .memory.build import writer_config
from .memory.store import Entry, MemoryStore
from .memory.writer import write

STREAM_ID = "custom/me"


def _stream(out: Path) -> Stream:
    return Stream(STREAM_ID, "custom", tuple(
        VideoRef(**{**v, "path": out / v["path"]}) for v in mio.read_json(out / "videos.json")))


def _transcripts(out: Path, stream: Stream) -> dict[str, list]:
    return {v.video_id: masr.load(out / "transcripts" / f"{v.video_id}.json", expected=False)
            for v in stream.videos}


def build(cfg, videos: Path, out: Path, transcripts: Path | None) -> None:
    srcs = sorted(videos.rglob("*.mp4"))
    if not srcs:
        raise SystemExit(f"no .mp4 under {videos}")
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        prepared = list(ex.map(transcode_one, srcs,
                               [out / "videos" / f"{s.stem}.mp4" for s in srcs]))
    rows, t = [], 0.0
    for src, r in zip(srcs, prepared):
        rows.append({"video_id": src.stem, "path": f"videos/{src.stem}.mp4", "abs_start": t,
                     "duration_s": r["out_duration_s"], "width": r["width"],
                     "height": r["height"], "fps": r["fps"]})
        t += r["out_duration_s"]
        print(f"[demo] prepared {src.name}: {r['out_duration_s']:.0f} s", flush=True)
    mio.write_json(out / "videos.json", rows, indent=1)
    if transcripts:
        for r in rows:
            p = transcripts / f"{r['video_id']}.json"
            if p.exists():
                masr.save(out / "transcripts" / p.name, masr.load(p), allow_empty=True)

    stream = _stream(out)
    turns = _transcripts(out, stream)
    plan = mseg.plan(stream, segment_s=float(cfg["memory"]["segment_seconds"]),
                     min_tail_s=float(cfg["memory"]["min_tail_seconds"]))
    engine = LLMEngine(ModelSpec.from_cfg(cfg["model"]), "writer", seed=int(cfg["seed"]))
    store = MemoryStore(STREAM_ID)
    for vid, segs in mseg.by_video(plan).items():
        vturns = turns[vid]
        descs = write(engine, segs, vturns, cfg=writer_config(cfg),
                      batch_size=int(cfg["writer"]["batch_size"]))
        for s, d in zip(segs, descs):
            store.add(Entry(s.segment_id, s.video_id, s.order, s.t_start, s.t_end,
                            s.abs_start, s.abs_end, d,
                            masr.render(masr.slice_window(vturns, s.t_start, s.t_end))))
        print(f"[demo] wrote {len(segs)} memory entries for {vid}", flush=True)
    store.build_index()
    store.save(out / "store")
    print(f"[demo] memory: {len(store)} entries over {stream.duration_s:.0f} s -> {out}",
          flush=True)


def ask(cfg, out: Path, question: str, choices: list[str], at: float | None) -> None:
    stream = _stream(out)
    now = stream.videos[-1].abs_end if at is None else at
    # The reader never sees correct_letter; QARecord just requires a valid one.
    rec = QARecord("q", question, tuple(choices), "A", "custom", now, STREAM_ID)
    m = MemLife(cfg, store_tag="demo")
    tools = T.Tools(MemoryStore.load(out / "store"), stream, _transcripts(out, stream),
                    now=now, cfg=m.tool_cfg)
    tr = mreader.answer(m.engine, tools, rec, cfg=m.reader_cfg)

    for c in tr.calls:
        args = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
        print(f"  round {c['round'] + 1}: {c['name']}({args}) -> {c['n_hits']} entries")
    if tr.letter is None:
        print("No answer within the round budget.")
        return
    picks = [choices[LETTERS.index(x)] for x in tr.letter.split(",")]
    print(f"ANSWER: {tr.letter} ({'; '.join(picks)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="qwen3.5-9b")
    common.add_argument("--set", action="append", default=[], dest="overrides", metavar="K=V")
    common.add_argument("--out", type=Path, default=Path("runs/demo"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", parents=[common], help="write memory for a folder of videos")
    b.add_argument("--videos", type=Path, required=True)
    b.add_argument("--transcripts", type=Path, default=None)
    q = sub.add_parser("ask", parents=[common],
                       help="answer one multiple-choice question from the memory")
    q.add_argument("question")
    q.add_argument("--choices", nargs="+", required=True)
    q.add_argument("--at", type=float, default=None,
                   help="ask as of this many seconds into the recording; default: the end")
    q.add_argument("--source-video", action="store_true", help="MemLife-V")
    a = ap.parse_args()

    overrides = list(a.overrides)
    if a.cmd == "ask" and a.source_video:
        overrides.append("reader.source_video=true")
    cfg = mcfg.load(dataset="custom", model=a.model, overrides=overrides)
    out = a.out.resolve()
    if a.cmd == "build":
        build(cfg, a.videos, out, a.transcripts)
    else:
        ask(cfg, out, a.question, a.choices, a.at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
