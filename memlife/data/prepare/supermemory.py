"""Prepare SuperMemory-VQA: questions, speech and the recording timeline, from the release.

    python -m memlife.data.prepare transcode \
        --src raw/SuperMemory-VQA/data/video --dst data/supermemory/videos
    python -m memlife.data.prepare supermemory --src raw/SuperMemory-VQA

Run ``transcode`` first: this step reads its manifest for each video's duration and geometry.

The clock is per subject, in seconds since that subject's first session (unix time in float32
has a 128 s step). A question's cutoff is the moment the person asks (``question_evidence``);
the two questions without one fall back to the end of their primary session. Questions whose
videos are not in the release are dropped (82 of 4,853).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ... import paths
from ...core import asr as masr
from ...core import io as mio
from ...core.mcq import LETTERS, strip_letter_prefix

DATASET = "supermemory"


def _subject(video_id: str) -> int:
    return int(video_id.split("_")[1])


def _clock_to_s(v) -> float | None:
    """``"16:32"`` / ``"01:29:28"`` -> seconds. Numbers pass through."""
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).split(":")
    if not 1 <= len(parts) <= 3 or not all(p.strip().isdigit() for p in parts):
        return None
    s = 0.0
    for p in parts:
        s = s * 60 + int(p)
    return s


def _query_unix(r: dict) -> float | None:
    """When the person starts asking. Six early questions use an older schema, with clock
    strings against the block's own start; two of those carry a ``00:00`` placeholder."""
    qe = r.get("question_evidence") or {}
    spans = qe.get("time_spans") or []
    if spans:
        return min(float(s["video_start_time_unix"]) + float(s["start_time"]) for s in spans)
    span = qe.get("time_span") or {}
    off, end = _clock_to_s(span.get("start_time")), _clock_to_s(span.get("end_time"))
    if qe.get("start_time") is None or off is None or (off == 0.0 and end == 0.0):
        return None
    return float(qe["start_time"]) + off


def _evidence(r: dict) -> list[dict]:
    return (r.get("answer_evidence") or {}).get("evidence_list", [])


def convert(src: Path, out: Path) -> dict:
    manifest = {r["video_id"]: r for r in mio.read_jsonl(out / "videos" / "manifest.jsonl")
                if "error" not in r}
    raw = mio.read_json(src / "data" / "json" / "all_qa.json")
    kept = [r for r in raw if r["metadata"]["primary_video_id"] in manifest
            and all(e["video_id"] in manifest for e in _evidence(r))]

    # Session starts: from evidence rows first, then each question's own start, which is its
    # primary session's start.
    start: dict[str, float] = {}
    for r in kept:
        for e in _evidence(r):
            start.setdefault(e["video_id"], float(e["start_time"]))
    for r in kept:
        start.setdefault(r["metadata"]["primary_video_id"], float(r["start_time"]))
    epoch: dict[int, float] = {}
    for vid, t in start.items():
        epoch[_subject(vid)] = min(epoch.get(_subject(vid), t), t)

    videos = []
    for vid in sorted(start, key=lambda v: (_subject(v), start[v])):
        m = manifest[vid]
        videos.append({
            "video_id": vid, "stream_id": f"{DATASET}/S{_subject(vid)}",
            "path": f"videos/{Path(m['path']).name}",
            "abs_start": start[vid] - epoch[_subject(vid)],
            "duration_s": float(m.get("out_duration_s", m["duration_s"])),
            "width": m["width"], "height": m["height"], "fps": m["fps"]})
    end = {v["video_id"]: v["abs_start"] + v["duration_s"] for v in videos}

    questions = []
    for r in kept:
        s, e0 = int(r["subject"]), epoch[int(r["subject"])]
        q = _query_unix(r)
        questions.append({
            "question_id": str(r["question_id"]),
            "question": r["question"],
            "choices": [strip_letter_prefix(c) for c in r["choices"]],
            "correct_letter": LETTERS[int(r["correct_option_index"])],
            "stream_id": f"{DATASET}/S{s}",
            "query_abs_s": q - e0 if q is not None else end[r["metadata"]["primary_video_id"]],
            "evidence": [{"video_id": e["video_id"],
                          "t_start_abs": float(e["start_time"]) - e0
                          + float(e["time_span"]["start_time"]),
                          "t_end_abs": float(e["start_time"]) - e0
                          + float(e["time_span"]["end_time"]),
                          "modalities": e.get("modalities", [])} for e in _evidence(r)],
            "meta": {"subject": s, "skill": r["metadata"].get("skill"),
                     "is_answerable": bool(r.get("is_answerable", True)),
                     "has_query_time": q is not None}})

    mio.write_json(out / "videos.json", videos, indent=1)
    mio.write_json(out / "qa.json", questions, indent=1)
    return {"questions": len(questions), "dropped": len(raw) - len(kept), "videos": len(videos),
            "transcript_turns": convert_transcripts(src, paths.ensure(out / "transcripts"))}


def convert_transcripts(src: Path, out_dir: Path) -> int:
    """The release's Gemini-aligned transcripts, speech only: the other entries are sound
    events such as ``[Sound of poker chips clicking]``, which the writer would read as words."""
    n = 0
    for f in sorted((src / "data" / "transcripts").rglob("*_gemini_aligned_transcript.json")):
        # The release lowercases the person directory; video ids are capitalised.
        vid = re.sub(r"^person_", "Person_", f.name.replace("_gemini_aligned_transcript.json", ""))
        doc = mio.read_json(f)
        rows = doc["transcript"] if isinstance(doc, dict) else doc
        turns = sorted((masr.Turn(float(r["start"]), float(r["end"]), r["text"].strip(),
                                  str(r.get("person") or "").strip())
                        for r in rows
                        if r.get("kind") == "speech" and (r.get("text") or "").strip()),
                       key=lambda t: (t.start, t.end))
        n += masr.save(out_dir / f"{vid}.json", turns, allow_empty=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True,
                    help="the SuperMemory-VQA release root (holds data/json and data/transcripts)")
    a = ap.parse_args()
    st = convert(a.src, paths.ensure(paths.dataset_dir(DATASET)))
    print(f"[supermemory] {st}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
