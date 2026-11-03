# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""JSON/JSONL I/O, content fingerprints, and the per-question output contract.

Writes go through a temp file and a rename, so a job killed mid-write leaves either the old
file or the new one -- never a truncated file that a later job reads as valid.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, obj: Any, *, indent: int | None = None) -> None:
    write_atomic(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    write_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Append rows to a JSONL file, creating it if needed.

    Not atomic, deliberately: this is the resume log for a long stage, so the point is that
    completed work is on disk the moment it finishes. A torn final line is one row to redo.
    """
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def fingerprint(obj: Any) -> str:
    """A short, stable digest of a JSON-able object. Key order never matters."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


class DoneMarker:
    """``<dir>/.done``, carrying the fingerprint of the config that produced the directory.

    A marker without the fingerprint would let a rebuild under new settings skip every item
    that an older build had already written, producing a store that is half one configuration
    and half another with nothing on disk recording it.
    """

    def __init__(self, d: Path):
        self.path = Path(d) / ".done"

    def done(self, fp: str | None = None) -> bool:
        if not self.path.exists():
            return False
        if fp is None:
            return True
        try:
            return json.loads(self.path.read_text()).get("fingerprint") == fp
        except (ValueError, OSError):
            return False

    def mark(self, fp: str, extra: dict | None = None) -> None:
        write_json(self.path, {"fingerprint": fp, **(extra or {})})


class PerQidWriter:
    """One ``results/<question_id>.json`` per question.

    Per-question files rather than per-shard JSONLs so workers can be added or rebalanced
    mid-sweep without shard-partition surgery: a question that failed is simply absent, and a
    resume retries exactly those.
    """

    def __init__(self, out_dir: Path, run_fp: str):
        self.dir = Path(out_dir) / "results"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_fp = run_fp
        stamp = Path(out_dir) / "run.json"
        if stamp.exists():
            got = read_json(stamp).get("run_fp")
            if got != run_fp:
                raise RuntimeError(
                    f"{out_dir} holds rows from a run fingerprinted {got!r}, this run is "
                    f"{run_fp!r}. Resuming would average two configurations into one accuracy. "
                    f"Use a different --tag.")
        else:
            write_json(stamp, {"run_fp": run_fp})

    def already_done(self, qid: str) -> bool:
        return (self.dir / f"{qid}.json").exists()

    def write(self, qid: str, row: dict) -> None:
        write_json(self.dir / f"{qid}.json", row)

    def count(self) -> int:
        return sum(1 for _ in self.dir.glob("*.json"))

    def mark_complete(self, expected: int) -> None:
        write_json(self.dir.parent / ".complete", {"n": expected, "run_fp": self.run_fp})
