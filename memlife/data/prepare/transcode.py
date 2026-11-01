"""Re-encode every source video to ``fit_dims(w, h, MAX_PIXELS)`` -- once, at preparation time.

    python -m memlife.data.prepare transcode --src raw/SuperMemory-VQA/data/video \
        --dst data/supermemory/videos

This is the resize the whole system assumes. Doing it here rather than at run time is what
makes the processor's own resize a no-op, gives every method identical pixels for a given
video, and -- since the released footage is low-resolution -- is also what keeps the stored
video small.

**Geometry is per video, not per dataset.** A release can carry a dozen distinct source
geometries; each file is probed and sized on its own, and the manifest records what actually
happened. Comparing two config values cannot catch a config/data disagreement; a manifest can.

**Constant frame rate is mandatory.** Frame sampling maps seconds to indices through the
average frame rate, which assumes CFR. A 0.1 % error over a 4500 s recording is 4.5 s of drift
by the end -- a sixth of a segment, enough to describe the wrong half of it. ``-r`` is
therefore always passed, at the source's own average rate unless ``--fps`` overrides it.

Four flags that look incidental and are not:

``flags=lanczos``  the resampler. Anything else changes the pixels the model sees.
``setsar=1``       stops ffmpeg writing a compensating sample aspect ratio, which would make
                   the container report a different size than the decoder returns.
``-threads 1``     **after** ``-i``. Option position is semantic: before the input it
                   configures only the decoder, and the encoder then auto-selects a thread per
                   core -- so N worker processes oversubscribe the machine by N times and
                   everything thrashes. Parallelism here is across FILES, not within one.
``-c:a copy``      audio is kept. Transcripts are derived separately, but a prepared file with
                   no audio cannot be re-derived from.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ...core import io as mio
from ...core.frames import MAX_PIXELS, fit_dims, n_vision_tokens

CRF = 20
PRESET = "medium"
#: Keyframe every ~2-3 s, so seeking stays cheap on multi-hour recordings.
GOP = 60


@dataclass
class Probe:
    width: int
    height: int
    fps: float
    duration_s: float
    codec: str


def probe(path: Path) -> Probe:
    """Geometry, average frame rate and duration, via ffprobe.

    The AVERAGE rate, deliberately: that is what the decoder reports and what frame sampling
    divides by, so reading the nominal rate instead would silently disagree with it.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,codec_name",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    num, _, den = st["avg_frame_rate"].partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return Probe(int(st["width"]), int(st["height"]), fps,
                 float(j["format"]["duration"]), st.get("codec_name", ""))


def ffmpeg_bin() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("no ffmpeg on PATH")
    return ff


def _tmp(dst: Path) -> Path:
    """In-progress output: hidden, and matching no video glob.

    Writing straight to the final name means a killed job leaves a truncated mp4 that is
    indistinguishable from a short video. The temp name carries no extension, so ffmpeg is
    told the container explicitly with ``-f mp4``.
    """
    return dst.with_name(f".{dst.stem}.partial")


def _encode(cmd: list[str], tmp: Path, src) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        # `-c:a copy` fails on a file with no audio stream. Retry without it rather than
        # losing the video.
        r = subprocess.run([c for c in cmd if c not in ("-c:a", "copy")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed on {src}:\n{r.stderr[-2000:]}")


def video_filter(w: int, h: int) -> str:
    return f"scale={w}:{h}:flags=lanczos,setsar=1"


def encode_args(w: int, h: int, fps: float) -> list[str]:
    """The encoder settings shared by this step and the EgoLife merge."""
    return [
        "-vf", video_filter(w, h),
        "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
        # Output-side; see the module docstring on option position.
        "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1:sliced-threads=0",
        "-pix_fmt", "yuv420p", "-g", str(GOP), "-r", f"{fps:.6f}",
        "-c:a", "copy", "-movflags", "+faststart",
    ]


def transcode_one(src: Path, dst: Path, *, max_pixels: int = MAX_PIXELS,
                  factor: int = 32, fps: float | None = None) -> dict:
    """Re-encode one file. Idempotent: an existing output at the right geometry is left alone."""
    p = probe(src)
    w, h = fit_dims(p.width, p.height, max_pixels, factor)
    out_fps = p.fps if fps is None else fps

    row = {"video_id": src.stem, "src": str(src), "path": str(dst),
           "src_width": p.width, "src_height": p.height, "src_codec": p.codec,
           "width": w, "height": h, "fps": round(out_fps, 6),
           "duration_s": round(p.duration_s, 3),
           "tokens_per_frame": n_vision_tokens(w, h, factor)}

    if dst.exists():
        got = probe(dst)
        if (got.width, got.height) == (w, h):
            row["skipped"] = True
            row["out_duration_s"] = round(got.duration_s, 3)
            return row
        dst.unlink()                      # wrong geometry: a stale build, redo it

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(dst)
    tmp.unlink(missing_ok=True)
    _encode([ffmpeg_bin(), "-y", "-nostdin", "-loglevel", "error",
             "-threads", "1", "-i", str(src), *encode_args(w, h, out_fps),
             "-f", "mp4", str(tmp)], tmp, src)

    got = probe(tmp)
    if (got.width, got.height) != (w, h):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{src}: asked {w}x{h}, ffmpeg produced {got.width}x{got.height}")
    tmp.replace(dst)
    row["skipped"] = False
    row["out_duration_s"] = round(got.duration_s, 3)
    return row


def _worker(args) -> dict:
    src, dst = Path(args[0]), Path(args[1])
    try:
        return transcode_one(src, dst)
    except Exception as e:                              # noqa: BLE001 -- report, don't abort
        return {"video_id": src.stem, "src": str(src), "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: <dst>/manifest.jsonl -- what was actually produced")
    ap.add_argument("--fps", type=float, default=None,
                    help="force an output rate; default is each source's own average")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    # One ffmpeg per file, one thread each. More workers than cores oversubscribes badly; see
    # the module docstring.
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    jobs = [(str(p), str(a.dst / f"{p.stem}.mp4")) for p in sorted(a.src.rglob("*.mp4"))]
    if not jobs:
        raise SystemExit(f"no .mp4 under {a.src}")
    print(f"[transcode] {len(jobs)} videos -> {a.dst} ({a.workers} workers)", flush=True)

    rows, errs = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            row = f.result()
            rows.append(row)
            errs += "error" in row
            if i % 25 == 0 or i == len(jobs):
                print(f"[transcode] {i}/{len(jobs)}  errors={errs}", flush=True)

    rows.sort(key=lambda r: r["video_id"])
    out = a.manifest or (a.dst / "manifest.jsonl")
    mio.write_jsonl(out, rows)
    geos = sorted({(r["width"], r["height"]) for r in rows if "width" in r})
    print(f"[transcode] wrote {out}; prepared geometries {geos}; {errs} error(s)", flush=True)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
