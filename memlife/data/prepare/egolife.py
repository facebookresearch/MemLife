"""Prepare the EgoLife recordings: merge the clips into runs, and put the speech on their clock.

    python -m memlife.data.prepare egolife --src raw/EgoLife

``--src`` is the EgoLife release: one folder per subject, plus ``EgoLifeCap/Transcript/`` and
``EgoLifeQA/``. It writes ``videos.json``, ``transcripts/`` and the EgoLifeQA ``qa.json``;
EgoLife-EQA's questions ship in ``datasets/egolife_eqa/`` (:mod:`memlife.data.egolife_eqa`).

**Why merge at all.** EgoLife ships one mp4 per ~30 s -- tens of thousands of them. Merging
into contiguous runs at preparation time means every consumer afterwards sees one long
recording, so segment planning, window resolution and frame sampling have a single code path
rather than per-clip stitching and day-boundary special cases.

A run is cut at any gap **> 31 s between consecutive clip STARTS**: clips are ~30 s and start
~30 s apart, so a larger gap means the recording actually stopped. In practice that is a few
runs per day, separated by hours.

The clock
---------
Absolute seconds are ``(day - 1) * 86400 + HH*3600 + MM*60 + SS + FF/100``, read from the
filename. Inside a run, clip *k* sits at the cumulative sum of its predecessors' real
durations -- which is **not** its wall-clock offset, because clips are 17.6-30.0 s long while
their starts are ~30 s apart. A merged run therefore runs slow against the day, by seconds by
the end of a long one.

So ``videos.json`` records every member's ``offset_s`` (position in the merged file) beside
its ``abs_start_s`` (wall clock), and :class:`~memlife.data.schema.PiecewiseVideoRef` converts
between the two piecewise. Using a single scalar offset instead puts the store's time axis and
the question's time axis on two clocks that agree only at the first frame -- which misassigns
boundary segments and lets post-cutoff frames into a causal window.
"""
from __future__ import annotations

import argparse
import bisect
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ... import paths
from ...core import asr as masr
from ...core import io as mio
from ...core.frames import MAX_PIXELS, fit_dims
from .transcode import encode_args, ffmpeg_bin, probe

#: ``DAY3_A1_JAKE_15553000.mp4`` -> day 3, subject A1_JAKE, 15:55:30.00
_CLIP = re.compile(r"^DAY(\d+)_(.+)_(\d{8})$")
#: ``A1_JAKE_DAY1_11000000.srt`` -- named for a wall-clock HOUR, not for a clip.
_SRT = re.compile(r"^(.+)_DAY(\d+)_(\d{8})$")

SECONDS_PER_DAY = 86400
#: Consecutive clip STARTS further apart than this end the run.
GAP_S = 31.0
#: The release's frame rate. Forced on the merged file, because concat-demuxing members with
#: differing timebases is exactly how variable frame rate gets in.
FPS = 20.0


def _hhmmssff(t: str) -> float:
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6]) + int(t[6:8]) / 100.0


def clip_abs_start(stem: str) -> tuple[str, int, float]:
    """``DAY3_A1_JAKE_15553000`` -> ``(subject, day, absolute seconds)``."""
    m = _CLIP.match(stem)
    if not m:
        raise ValueError(f"unparseable EgoLife clip name: {stem!r}")
    day, subject, t = int(m.group(1)), m.group(2), m.group(3)
    return subject, day, (day - 1) * SECONDS_PER_DAY + _hhmmssff(t)


def srt_abs_start(stem: str) -> float:
    m = _SRT.match(stem)
    if not m:
        raise ValueError(f"unparseable EgoLife SRT name: {stem!r}")
    return (int(m.group(2)) - 1) * SECONDS_PER_DAY + _hhmmssff(m.group(3))


def plan_runs(src_root: Path, subject: str) -> list[list[Path]]:
    """Group one subject's clips into contiguous runs, in absolute-time order."""
    clips = sorted((clip_abs_start(p.stem)[2], p)
                   for p in (src_root / subject).rglob("*.mp4"))
    if not clips:
        return []
    runs: list[list[Path]] = [[clips[0][1]]]
    for (prev_t, _), (t, p) in zip(clips, clips[1:]):
        if t - prev_t > GAP_S:
            runs.append([])
        runs[-1].append(p)
    return runs


def merge_one(members: list[Path], dst: Path, *, max_pixels: int = MAX_PIXELS,
              factor: int = 32) -> dict:
    """Concat, scale and encode in ONE ffmpeg pass. Returns the ``videos.json`` row."""
    probes = [probe(p) for p in members]
    geos = {(q.width, q.height) for q in probes}
    if len(geos) != 1:
        raise RuntimeError(f"{dst.name}: members disagree on geometry: {sorted(geos)}")
    sw, sh = geos.pop()
    w, h = fit_dims(sw, sh, max_pixels, factor)

    subject, day, run_abs0 = clip_abs_start(members[0].stem)
    offset, rows = 0.0, []
    for p, q in zip(members, probes):
        rows.append({"src": p.name, "offset_s": round(offset, 3),
                     "abs_start_s": round(clip_abs_start(p.stem)[2], 3)})
        offset += q.duration_s
    drift = max(abs((r["abs_start_s"] - run_abs0) - r["offset_s"]) for r in rows)

    row = {"video_id": dst.stem, "stream_id": f"egolifeqa/{subject}", "path": str(dst),
           "abs_start": round(run_abs0, 3), "duration_s": round(offset, 3),
           "width": w, "height": h, "fps": FPS,
           "day": day, "n_members": len(members),
           "src_width": sw, "src_height": sh,
           # How far the merged clock drifts from the wall clock by the end of this run. It is
           # a diagnostic only: `members` below is what the loader actually uses, and it is
           # exact whatever this says.
           "max_drift_s": round(drift, 3),
           "members": rows}

    if dst.exists():
        got = probe(dst)
        if (got.width, got.height) == (w, h) and abs(got.duration_s - offset) < 2.0:
            row["skipped"] = True
            return row
        dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     dir=str(dst.parent)) as lf:
        for p in members:
            lf.write(f"file '{p.resolve()}'\n")
        listfile = lf.name
    tmp = dst.with_name(f".{dst.stem}.partial")
    tmp.unlink(missing_ok=True)
    try:
        cmd = [ffmpeg_bin(), "-y", "-nostdin", "-loglevel", "error", "-threads", "1",
               "-f", "concat", "-safe", "0", "-i", listfile,
               *encode_args(w, h, FPS), "-f", "mp4", str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            r = subprocess.run([c for c in cmd if c not in ("-c:a", "copy")],
                               capture_output=True, text=True)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg concat failed for {dst.name}:\n{r.stderr[-2000:]}")
        got = probe(tmp)
        if (got.width, got.height) != (w, h):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{dst.name}: asked {w}x{h}, got {got.width}x{got.height}")
        # Losing a member silently would shorten the timeline and misplace every later clip.
        if abs(got.duration_s - offset) > max(2.0, 0.01 * offset):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{dst.name}: merged {got.duration_s:.1f}s but members sum to "
                               f"{offset:.1f}s -- a member was dropped or truncated")
        tmp.replace(dst)
    finally:
        Path(listfile).unlink(missing_ok=True)
    row["skipped"] = False
    return row


def _worker(args) -> dict:
    members, dst = [Path(m) for m in args[0]], Path(args[1])
    try:
        return merge_one(members, dst)
    except Exception as e:                              # noqa: BLE001 -- report, don't abort
        return {"video_id": dst.stem, "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------------------ transcripts

def convert_transcripts(src_root: Path, out_dir: Path, rows: list[dict]) -> int:
    """Per-hour SRTs -> one prepared transcript per merged run, on that run's own clock.

    This is the subtle half, and getting it wrong is silent and total: a mis-based transcript
    makes every window slice return nothing, every segment renders "(no speech)", and the
    writer produces fluent vision-only descriptions for a whole subject with nothing logged.

    Three facts force the shape:

    1. an SRT is named for a wall-clock HOUR, not for a clip, so ``cue_abs = hour_abs +
       cue_local``;
    2. a merged run may start mid-hour, and a run boundary may fall INSIDE an hour file;
    3. therefore cues are assigned to runs by **absolute time**, never by file, and only then
       converted to run-local. A single scalar offset per file cannot express this.

    Run-local is itself piecewise -- the run clock drifts against the wall clock -- so a cue is
    placed through the same member map the loader reads.
    """
    n = 0
    for subject in sorted({r["stream_id"].split("/")[1] for r in rows if "error" not in r}):
        n += _subject_transcripts(src_root, out_dir, subject, rows)
    return n


def _subject_transcripts(src_root: Path, out_dir: Path, subject: str, rows: list[dict]) -> int:
    # Only this subject's speech: the subjects record the same hours, and the release's
    # DenseCaption folder holds SRTs with the same names that are captions, not speech.
    runs = sorted((r for r in rows if "error" not in r and r["stream_id"].endswith(f"/{subject}")),
                  key=lambda r: r["abs_start"])
    starts = [r["abs_start"] for r in runs]
    ends = [_wall_end(r) for r in runs]
    buckets: dict[str, list[masr.Turn]] = {r["video_id"]: [] for r in runs}

    srts = sorted((src_root / "EgoLifeCap" / "Transcript" / subject).rglob("*.srt"))
    if not srts:
        raise SystemExit(f"no transcripts under {src_root}/EgoLifeCap/Transcript/{subject}")
    for srt in srts:
        base = srt_abs_start(srt.stem)
        for turn in masr.parse_srt(srt.read_text(errors="replace"), offset_s=base):
            i = bisect.bisect_right(starts, turn.start) - 1
            if i < 0 or turn.start > ends[i]:
                continue                     # falls in a gap between runs
            r = runs[i]
            # The end is clamped: a cue may run past the last member, and a turn stamped
            # beyond the file would never overlap any segment window.
            buckets[r["video_id"]].append(masr.Turn(
                _to_file(r, turn.start), min(_to_file(r, turn.end), r["duration_s"]),
                turn.text, turn.speaker))

    n = 0
    for r in runs:
        turns = sorted(buckets[r["video_id"]], key=lambda t: t.start)
        n += masr.save(out_dir / f"{r['video_id']}.json", turns, allow_empty=True)
    return n


def _wall_end(row: dict) -> float:
    """The run's last wall-clock instant.

    NOT ``abs_start + duration_s``: ``duration_s`` is the sum of member durations, and the run
    clock runs slow against the day, so that would end the run early and drop the tail of its
    speech. The last member starts at its own ``abs_start_s`` and runs for whatever is left of
    the file after its ``offset_s``.
    """
    last = row["members"][-1]
    return last["abs_start_s"] + (row["duration_s"] - last["offset_s"])


def _to_file(row: dict, t_abs: float) -> float:
    """Wall clock -> position in the merged file, through this run's member map."""
    abs_starts = [m["abs_start_s"] for m in row["members"]]
    i = max(0, bisect.bisect_right(abs_starts, t_abs) - 1)
    return row["members"][i]["offset_s"] + (t_abs - abs_starts[i])


# ------------------------------------------------------------------------------- questions

def _day(d: str) -> int:
    return int(str(d).upper().replace("DAY", ""))      # one entry is "Day1"


def _target_abs(tt: dict) -> list[float]:
    """Every annotated target instant. 36 questions carry ``time_list`` instead of ``time``,
    some unsorted, and one value has a second timestamp glued on (the first 8 digits win)."""
    if not tt:
        return []
    base = (_day(tt["date"]) - 1) * SECONDS_PER_DAY
    raw = tt.get("time_list") or ([tt["time"]] if tt.get("time") else [])
    return sorted(base + _hhmmssff(str(v)[:8]) for v in raw)


def convert_questions(src: Path, dst: Path) -> int:
    """EgoLifeQA's 500 questions, for ``A1_JAKE``. Its target times are grounding instants."""
    out = []
    for r in mio.read_json(src):
        qt = r["query_time"]
        targets = _target_abs(r.get("target_time") or {})
        out.append({
            "question_id": str(r["ID"]),
            "question": r["question"],
            "choices": [r[f"choice_{c}"] for c in "abcd"],
            "correct_letter": str(r["answer"]).strip().upper(),
            "stream_id": "egolifeqa/A1_JAKE",
            "query_abs_s": (_day(qt["date"]) - 1) * SECONDS_PER_DAY + _hhmmssff(qt["time"]),
            "evidence": [{"video_id": "", "t_start_abs": t, "t_end_abs": t} for t in targets],
            "meta": {"type": r.get("type")}})
    mio.write_json(dst, out, indent=1)
    return len(out)


# ------------------------------------------------------------------------------------ driver

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True, help="the EgoLife release root")
    # The two subjects EgoLifeQA and EgoLife-EQA ask about.
    ap.add_argument("--subjects", nargs="+", default=["A1_JAKE", "A3_TASHA"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    out = paths.ensure(paths.dataset_dir("egolifeqa"))
    videos = paths.ensure(out / "videos")

    jobs = []
    for s in a.subjects:
        for run in plan_runs(a.src, s):
            first = run[0].stem
            jobs.append(([str(p) for p in run], str(videos / f"{first}.mp4")))
    if not jobs:
        raise SystemExit(f"no EgoLife clips under {a.src}")
    print(f"[egolife] {len(a.subjects)} subject(s), {len(jobs)} runs "
          f"({sum(len(j[0]) for j in jobs)} clips)", flush=True)

    rows, errs = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            row = f.result()
            rows.append(row)
            errs += "error" in row
            if i % 10 == 0 or i == len(jobs):
                print(f"[egolife] merged {i}/{len(jobs)}  errors={errs}", flush=True)

    rows.sort(key=lambda r: (r.get("stream_id", ""), r.get("abs_start", 0.0)))
    # videos.json carries only the contract's fields plus the member map; the merge
    # diagnostics stay beside it so a later question about a run can be answered.
    mio.write_json(out / "videos.json",
                   [{k: v for k, v in r.items() if k not in ("skipped", "error")}
                    for r in rows if "error" not in r], indent=1)
    mio.write_jsonl(out / "merge_manifest.jsonl", rows)

    n_turns = convert_transcripts(a.src, paths.ensure(out / "transcripts"), rows)
    n_q = convert_questions(a.src / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json", out / "qa.json")
    print(f"[egolife] wrote {out}/videos.json ({len(rows) - errs} runs), {n_turns} transcript "
          f"turns and {n_q} questions; {errs} error(s)", flush=True)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
