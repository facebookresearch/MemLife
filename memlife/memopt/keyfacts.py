# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Key facts: what does ONE evidence segment actually contribute to ONE question?

    python -m memlife.memopt.keyfacts --dataset supermemory --split train

This defines the MemOpt training pool. For every ``(segment, question)`` pair where the
segment is **reachable** gold evidence, a frozen evaluator is asked what fact that segment
really shows. Pairs it answers NONE for drop out of ``Q(c)``; a segment left with no surviving
question drops out of the pool entirely. The ``none`` rate printed at the end IS the pool size.

Three decisions that look like details and are not:

**Reachable gold, not all gold.** Evidence that had not happened yet at the question's own
cutoff is dropped. Rewarding a memory for a segment the reader could never retrieve would be
rewarding retrieval of the future.

**The segment's own eight frames, nothing else.** The writer sees no neighbouring context
either, so an extractor given some would credit the segment with facts the writer had no way
to state -- and the reward would then ask the policy for something it structurally cannot
produce.

**Frames come straight from the video**, sampled exactly as the writer samples them. This
stage therefore depends on no memory store and no extracted frame tree, and can run before
either exists.

Output, one row per pair, in ``runs/memopt/<dataset>_<split>/keyfacts.jsonl``::

    {"segment_id", "qid", "supports": bool, "key_fact": str|None, "none": bool}

with the invariant ``key_fact is None`` exactly when ``none``, so the two spellings of "this
pair is out of the pool" cannot diverge in a consumer.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

from .. import config as mcfg
from .. import paths
from ..core import asr as masr
from ..core import frames as mframes
from ..core import io as mio
from ..core import mcq
from ..core.llm import LLMEngine, ModelSpec
from ..data import LOADERS, prepared, store_dataset
from ..data.schema import QARecord
from ..eval.evidence import visible_gold
from ..memory import segments as mseg

# ==================================================================================== prompt

SYSTEM = (
    "You are auditing ONE 30-second segment of a first-person video against a question about "
    "the wearer's day and that question's known-correct answer. The segment has been labelled "
    "as evidence for the question. State, in one sentence, the KEY FACT that is actually "
    "visible or audible in THIS segment and that helps establish the correct answer.\n\n"
    "Rules:\n"
    "1. The key fact must be something you can SEE in this segment's frames or HEAR in its "
    "transcript. Never infer it from the question or from the answer.\n"
    "2. Write it as a standalone declarative sentence about the segment, naming the concrete "
    "objects, people, places, actions or spoken words involved. Do NOT mention the question, "
    "the options, or the answer, and do not reuse an option's wording.\n"
    "3. If this segment does NOT actually contain anything supporting the correct answer - the "
    "evidence label is wrong, or the segment is unrelated background - say so instead of "
    "inventing a link.\n"
    "4. A segment can be genuine evidence while showing only PART of what the answer needs; in "
    "that case state the part it does show.\n\n"
    "Output EXACTLY two lines and nothing else:\n"
    "SUPPORTS: YES\n"
    "KEY FACT: <one sentence>\n"
    "or\n"
    "SUPPORTS: NO\n"
    "KEY FACT: NONE"
)
USER_HEAD = "## This segment"
USER_ASR = 'What is said in this segment:\n"""{transcript}"""'
USER_TAIL = (
    '## Question a later reader will ask about this day\n"{question}"\n'
    "Options:\n{options}\n"
    "Correct answer: {letter}. {answer}\n\n"
    "What key fact does THIS segment contribute toward that correct answer? Output the two "
    "lines."
)


def build_messages(frames: list, *, transcript: str, rec: QARecord) -> list[dict]:
    """head -> frames -> transcript -> question.

    The order is load-bearing in the same way the writer's is: the question arrives AFTER the
    pixels, so the model reads the segment before it learns what it is being audited against.
    Putting the question first invites it to answer from the option text.
    """
    # A multi-select question has several gold letters; name them all.
    letters = rec.correct_letter.split(",")
    answer = "; ".join(rec.choices[mcq.letter_to_index(x)] for x in letters)
    content: list[dict] = [{"type": "text", "text": USER_HEAD}]
    content += [{"type": "image_pil", "image_pil": im} for im in frames]
    content.append({"type": "text", "text": USER_ASR.format(transcript=transcript)})
    content.append({"type": "text", "text": USER_TAIL.format(
        question=rec.question, options=mcq.format_options(rec.choices),
        letter=rec.correct_letter, answer=answer)})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


# ==================================================================================== parsing

# Both labels are LINE-ANCHORED, tolerating the bullet / number / emphasis junk a chat model
# wraps a two-line answer in: a model that bolds the two lines it was told to emit HAS
# complied, and every spelling we fail to match is a pair silently dropped from the pool for a
# formatting habit. Anchoring also stops a mid-sentence "the sign supports: yes" overwriting
# the verdict.
_PREFIX = r"^[ \t]*(?:[-*+>#]+\s*|\d+[.)]\s*)?[*_`]*\s*"
_COLON = r"[*_`]*\s*:\s*"
_SUPPORTS_RE = re.compile(_PREFIX + r"SUPPORTS" + _COLON + r"[*_`\"']*[ \t]*(YES|NO)\b",
                          re.IGNORECASE | re.MULTILINE)
_KEYFACT_RE = re.compile(_PREFIX + r"KEY[ _]?FACT" + _COLON + r"(.*)$",
                         re.IGNORECASE | re.MULTILINE)

#: Shorter than this and it is a label fragment, not a fact.
MIN_FACT_CHARS = 15


def parse(text: str) -> tuple[bool, str | None, bool]:
    """``(supports, key_fact, parse_failed)``.

    The LAST verdict wins, because a model that restates ("...on reflection: SUPPORTS: NO")
    means the restatement. A missing verdict line is ``parse_failed`` -- which is NOT the same
    claim as "this segment does not support the answer", and is counted separately so an
    unreadable answer stays visible instead of being filed as a substantive NO.
    """
    sm = None
    for sm in _SUPPORTS_RE.finditer(text):
        pass
    fm = None
    for fm in _KEYFACT_RE.finditer(text):
        pass
    if sm is None:
        return (False, None, True)
    supports = sm.group(1).upper() == "YES"
    fact = (fm.group(1).strip().strip("*_`\"'") if fm else "")
    if not supports or fact.upper().startswith("NONE") or len(fact) < MIN_FACT_CHARS:
        return (supports, None, False)
    return (True, fact, False)


# ==================================================================================== driver

def out_dir(dataset: str, split: str) -> Path:
    return paths.memopt_dir(f"{dataset}_{split}")


def _mine(records: list, shard: int, num_shards: int) -> list:
    """This worker's questions: a stable hash of the question id, as the evaluation shards."""
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} out of range for --num-shards {num_shards}")
    return [r for r in records
            if int(hashlib.sha256(r.question_id.encode()).hexdigest(), 16) % num_shards == shard]


def run(cfg, dataset: str, split: str, *, limit: int | None, batch_size: int,
        shard: int = 0, num_shards: int = 1) -> dict:
    """Extract key facts, resuming per pair.

    With ``num_shards > 1`` this worker writes its own ``keyfacts.shard<i>.jsonl``. The unsharded
    run that follows folds every shard file into ``keyfacts.jsonl`` and does whatever is left, so
    the pool is one file however the work was split. Pairs already written are read from ALL of
    those files, so re-running with a different shard count never writes a pair twice.
    """
    mod = LOADERS[dataset]
    src = store_dataset(dataset)
    records = _mine(mod.load(split, limit=limit), shard, num_shards)
    n_frames = int(cfg["memory"]["frames_per_segment"])
    segment_s = float(cfg["memory"]["segment_seconds"])
    min_tail = float(cfg["memory"]["min_tail_seconds"])
    has_asr = bool(cfg["dataset"].get("has_asr", True))

    root = paths.ensure(out_dir(dataset, split))
    path = root / "keyfacts.jsonl"
    shard_files = sorted(root.glob("keyfacts.shard*.jsonl"))
    if num_shards == 1 and shard_files:
        have = ({(r["segment_id"], r["qid"]) for r in mio.read_jsonl(path)}
                if path.exists() else set())
        for f in shard_files:
            fresh = [r for r in mio.read_jsonl(f) if (r["segment_id"], r["qid"]) not in have]
            have |= {(r["segment_id"], r["qid"]) for r in fresh}
            mio.append_jsonl(path, fresh)
            f.unlink()
        shard_files = []
    out = path if num_shards == 1 else root / f"keyfacts.shard{shard}.jsonl"
    seen = {(r["segment_id"], r["qid"])
            for f in [path, *shard_files] if f.exists() for r in mio.read_jsonl(f)}
    print(f"[keyfacts] shard {shard}/{num_shards}: {len(records)} questions, "
          f"{len(seen)} pair(s) already done", flush=True)

    engine = None
    rows: list[dict] = []
    pending: list[tuple[str, str, list, str, QARecord]] = []
    t0 = time.time()

    def flush() -> None:
        nonlocal engine
        if not pending:
            return
        if engine is None:
            engine = LLMEngine(ModelSpec.from_cfg(cfg["model"]), "judge", seed=int(cfg["seed"]))
        texts = engine.generate_text(
            [build_messages(f, transcript=a, rec=r) for _s, _q, f, a, r in pending], "keyfact")
        for (sid, qid, _f, _a, _r), text in zip(pending, texts):
            supports, fact, failed = parse(text)
            rows.append({"segment_id": sid, "qid": qid, "supports": supports,
                         "key_fact": fact, "none": fact is None, "parse_failed": failed})
        pending.clear()

    for n, rec in enumerate(records):
        # A question whose gold answer is "can not be answered" has no supporting fact by
        # construction, so any YES the extractor gives is spurious.
        if rec.correct_letter == rec.meta.get("abstain_letter"):
            continue
        stream = mod.get_stream(rec.stream_id)
        gold = visible_gold(rec, stream, segment_s=segment_s)
        if not gold:
            continue
        todo = [sid for sid in sorted(gold) if (sid, rec.question_id) not in seen]
        if not todo:
            continue
        turns = prepared.transcripts_for(src, stream, expected=has_asr)
        grid = {s.segment_id: s for s in mseg.plan(stream, segment_s=segment_s,
                                                   min_tail_s=min_tail)}
        for sid in todo:
            seg = grid[sid]
            try:
                frames = mframes.sample_frames(seg.path, seg.t_start, seg.t_end, n_frames,
                                               allow_pad=False)
            except mframes.DecodeError as e:
                print(f"[keyfacts] skip {sid}: {e}", flush=True)
                continue
            transcript = masr.render(masr.slice_window(
                turns.get(seg.video_id, []), seg.t_start, seg.t_end))
            pending.append((sid, rec.question_id, frames, transcript, rec))
            if len(pending) >= batch_size:
                flush()
        # Appended continuously, so a killed job costs the current batch and a re-run picks up
        # where it stopped.
        mio.append_jsonl(out, rows)
        rows.clear()
        if (n + 1) % 50 == 0:
            print(f"[keyfacts] {n + 1}/{len(records)} questions  "
                  f"{time.time() - t0:.0f}s", flush=True)
    flush()
    mio.append_jsonl(out, rows)
    if num_shards > 1:
        return {"shard": shard}

    all_rows = list(mio.read_jsonl(path))
    none = sum(bool(r["none"]) for r in all_rows)
    pool = {r["segment_id"] for r in all_rows if not r["none"]}
    print(f"[keyfacts] {len(all_rows)} pairs, {none} NONE ({none / max(1, len(all_rows)):.1%}), "
          f"{len(pool)} segments in the training pool -> {path}", flush=True)
    return {"pairs": len(all_rows), "none": none, "pool": len(pool)}


def load_pool(dataset: str, split: str) -> dict[str, dict[str, str]]:
    """``segment_id -> {qid: key_fact}`` over the surviving pairs -- i.e. ``Q(c)``."""
    path = out_dir(dataset, split) / "keyfacts.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no key facts at {path}. Run `python -m memlife.memopt.keyfacts --dataset "
            f"{dataset} --split {split}` first: it defines the MemOpt training pool.")
    out: dict[str, dict[str, str]] = {}
    for r in mio.read_jsonl(path):
        if not r["none"]:
            out.setdefault(r["segment_id"], {})[r["qid"]] = r["key_fact"]
    return out


def main() -> int:
    ap = mcfg.add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    run(mcfg.from_args(a), a.dataset, a.split, limit=a.limit, batch_size=a.batch_size,
        shard=a.shard, num_shards=a.num_shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
