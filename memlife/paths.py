"""Every filesystem path resolves from here. Nothing else may hardcode one.

One environment variable, ``MEMLIFE_ROOT``, anchors the tree. It defaults to the repository
root, so a fresh clone works with no configuration::

    <root>/data/<dataset>/      prepared benchmark data   (see docs in README.md)
    <root>/models/              staged model checkpoints
    <root>/runs/stores/         built memory stores
    <root>/runs/predictions/    evaluation output
    <root>/runs/memopt/         MemOpt training artifacts

Import must never touch the filesystem -- only :func:`ensure` creates anything.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "MEMLIFE_ROOT"

#: The benchmarks the loaders in ``memlife.data`` know about.
DATASETS = ("supermemory", "egolifeqa", "supermemory_lvqa", "egolife_eqa")


def root() -> Path:
    v = os.environ.get(ROOT_ENV)
    return Path(v).resolve() if v else Path(__file__).resolve().parent.parent


def data_root() -> Path:
    return root() / "data"


def models_root() -> Path:
    return root() / "models"


def runs_root() -> Path:
    return root() / "runs"


def configs_root() -> Path:
    return root() / "configs"


def dataset_dir(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}")
    return data_root() / dataset


def store_dir(dataset: str, tag: str = "default") -> Path:
    """Where a built memory store lives.

    ``tag`` separates stores that differ in anything the reader can see -- writer model, writer
    ablation, trained checkpoint. Reusing a tag across two different builds is the easiest way
    to silently evaluate against stale memory, which is why :func:`memlife.memory.build.build_id`
    also fingerprints the writer config into the store and the reader checks it on load.
    """
    return runs_root() / "stores" / dataset / tag


def pred_dir(dataset: str, split: str, tag: str = "default") -> Path:
    return runs_root() / "predictions" / dataset / split / tag


def memopt_dir(name: str) -> Path:
    return runs_root() / "memopt" / name


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
