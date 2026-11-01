"""Configuration: one base file, plus overlays for the model and the dataset.

    cfg = config.load(dataset="supermemory", model="qwen3.5-9b")

Resolution order, each layer deep-merged over the last::

    configs/default.yaml
    configs/model/<model>.yaml
    configs/dataset/<dataset>.yaml
    --set key.path=value              (in the order given)

Two rules are enforced rather than documented, and both exist so that a run always matches the
settings it is labelled with:

**An overlay or --set may only change a key that already exists.** A typo (``reader.topk`` for
``reader.top_k``) would otherwise add a dead key, run the default, and be reported under the
new name.

**Neither may change a key's type.** ``source_video: "false"`` is a non-empty string and
therefore truthy; without this check that run would quietly enable the tool it meant to leave
off.

The resolved config is frozen. Mutating it after load would mean the fingerprint stamped into a
store or a prediction directory no longer describes the run that produced it.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

from . import paths

# --------------------------------------------------------------------------------- container


class Cfg(dict):
    """A dict with attribute access that refuses writes once frozen."""

    _frozen = False

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setitem__(self, k, v):
        if self._frozen:
            raise TypeError(
                f"config is frozen (tried to set {k!r}). The resolved config is fingerprinted "
                f"into store and prediction directories; mutating it would make that stamp "
                f"describe a different run. Use --set, or a config overlay.")
        super().__setitem__(k, v)

    def freeze(self) -> "Cfg":
        for v in self.values():
            if isinstance(v, Cfg):
                v.freeze()
        object.__setattr__(self, "_frozen", True)
        return self


def _wrap(obj: Any) -> Any:
    """Recursively turn dicts into ``Cfg`` and lists into tuples.

    Lists become tuples so a frozen config cannot be mutated through one.
    """
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_wrap(v) for v in obj)
    return obj


def to_plain(obj: Any) -> Any:
    """Inverse of :func:`_wrap`, for YAML/JSON dumps."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


# ----------------------------------------------------------------------------------- merging


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    return yaml.safe_load(path.read_text()) or {}


def _deep_merge(base: dict, over: dict) -> dict:
    """``over`` wins. Dicts merge recursively; **lists REPLACE, never concatenate.**

    Replacement is the only rule that lets an overlay shorten a list; concatenation would make
    "use fewer of these" unexpressible.
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _same_type(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return True                      # a nullable key may be filled in or cleared
    if isinstance(a, bool) != isinstance(b, bool):
        return False                     # bool is an int; do not let True replace 1
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return True
    return type(a) is type(b) or (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)))


#: Sections whose contents are passed through to another library verbatim, so this file
#: cannot know their keys. The section itself must still exist; only its interior is opaque.
_OPAQUE = frozenset({"model.additional_config"})


def _check_overlay(base: dict, over: dict, source: str, prefix: str = "") -> None:
    for k, v in over.items():
        path = f"{prefix}{k}"
        if path in _OPAQUE:
            continue
        if k not in base:
            raise KeyError(
                f"{source} sets {path!r}, which does not exist in the base config. An overlay "
                f"may only CHANGE existing keys -- otherwise a typo adds a dead key, the run "
                f"uses the default, and it is reported under this overlay's name.")
        if isinstance(v, dict) and isinstance(base[k], dict):
            _check_overlay(base[k], v, source, f"{path}.")
        elif not _same_type(base[k], v):
            raise TypeError(
                f"{source} sets {path}={v!r} ({type(v).__name__}) over a "
                f"{type(base[k]).__name__}. A string 'false' is truthy, so this run would use "
                f"the default under the overlay's name.")


def _coerce(old: Any, text: str) -> Any:
    """Parse a ``--set`` value in the type the existing key already has."""
    if isinstance(old, bool):
        low = text.strip().lower()
        if low not in ("true", "false"):
            raise ValueError(f"expected true/false, got {text!r}")
        return low == "true"
    if isinstance(old, int):
        return int(text)
    if isinstance(old, float):
        return float(text)
    if isinstance(old, (list, tuple)):
        inner = text.strip().strip("[]")
        return tuple(x.strip() for x in inner.split(",") if x.strip())
    if old is None:
        return None if text.strip().lower() in ("null", "none", "") else text
    return text


def _apply_set(cfg: dict, spec: str) -> None:
    if "=" not in spec:
        raise ValueError(f"--set expects key.path=value, got {spec!r}")
    key, _, text = spec.partition("=")
    node, parts = cfg, key.strip().split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"--set {key}: no section {p!r}")
        node = node[p]
    leaf = parts[-1]
    if leaf not in node:
        raise KeyError(
            f"--set {key}: no such key. --set may only CHANGE existing keys, for the same "
            f"reason an overlay may: a typo would otherwise run the default under a new name.")
    node[leaf] = _coerce(node[leaf], text)


# ------------------------------------------------------------------------------- invariants


def _check(cfg: dict) -> None:
    """The cross-key conditions no single overlay can check for itself."""
    w, r, rw = cfg["writer"], cfg["reader"], cfg["rewards"]

    if w["max_tokens"] > cfg["train"]["max_response_length"]:
        raise ValueError(
            f"writer.max_tokens={w['max_tokens']} exceeds "
            f"train.max_response_length={cfg['train']['max_response_length']}. During MemOpt the "
            f"writer IS the policy, so a memory longer than the response budget is truncated "
            f"mid-sentence and then scored as if the writer had stopped there.")

    if r["source_video"] and not cfg["dataset"].get("has_video", True):
        raise ValueError(
            f"reader.source_video is set but dataset {cfg['dataset']['name']!r} ships no source "
            f"video. The prompt would advertise a tool every call fails on.")
    if r["fetch_video_frames"] > r["max_fetch_video_frames"]:
        raise ValueError(
            f"reader.fetch_video_frames={r['fetch_video_frames']} exceeds "
            f"max_fetch_video_frames={r['max_fetch_video_frames']}, so the value the prompt "
            f"advertises is one the tool immediately clamps.")
    if r["rounds"] < 1:
        raise ValueError(f"reader.rounds must be >= 1, got {r['rounds']}")

    # The evaluator tokenizes the candidate it audits and the result is laid down positionally
    # on the policy's own response tokens, so the two must be the same tokenizer.
    if rw["judge_model"] not in (None, cfg["model"]["id"]):
        raise ValueError(
            f"rewards.judge_model={rw['judge_model']!r} differs from model.id="
            f"{cfg['model']['id']!r}. Faithfulness margins are indexed by the EVALUATOR's "
            f"tokenization and applied to the POLICY's response positions; two tokenizers "
            f"silently misalign them.")

    if cfg["memory"]["segment_seconds"] <= cfg["memory"]["min_tail_seconds"]:
        raise ValueError("memory.min_tail_seconds must be shorter than memory.segment_seconds")

    # The faithfulness evaluator serves a fixed number of images per segment, so a writer that
    # saw more would be audited against a SUBSET of the pixels it used.
    from .memopt.judge import N_FRAMES

    if int(cfg["memory"]["frames_per_segment"]) != N_FRAMES:
        raise ValueError(
            f"memory.frames_per_segment={cfg['memory']['frames_per_segment']} but the "
            f"evaluator serves {N_FRAMES} images per segment (memlife/memopt/judge.py). Lower "
            f"there and every audit is rejected; higher there and the audit silently scores a "
            f"subset of what the writer saw. Change both together.")


def _derive(cfg: dict) -> None:
    """Fill the keys whose value is a function of other keys."""
    if cfg["reader"]["context_guard_tokens"] is None:
        # Leave the reader a working margin below the model's real context so compaction fires
        # before vLLM rejects the prompt outright.
        cfg["reader"]["context_guard_tokens"] = max(
            8192, int(cfg["model"]["context"]["reader"])
            - int(cfg["reader"]["context_headroom_tokens"]))
    if cfg["rewards"]["judge_model"] is None:
        cfg["rewards"]["judge_model"] = cfg["model"]["id"]


# ------------------------------------------------------------------------------------ load


def load(*, dataset: str = "supermemory", model: str = "qwen3.5-9b",
         overrides: Iterable[str] = ()) -> Cfg:
    root = paths.configs_root()
    cfg = _read(root / "default.yaml")

    for kind, name in (("model", model), ("dataset", dataset)):
        over = _read(root / kind / f"{name}.yaml")
        _check_overlay(cfg[kind], over, f"configs/{kind}/{name}.yaml", f"{kind}.")
        cfg[kind] = _deep_merge(cfg[kind], over)

    overrides = tuple(overrides)
    for spec in overrides:
        _apply_set(cfg, spec)

    _derive(cfg)
    _check(cfg)
    # Provenance, so a store or prediction directory records how it was configured.
    cfg["_resolved"] = {"dataset": dataset, "model": model,
                        "overrides": list(overrides)}
    return _wrap(cfg).freeze()


# -------------------------------------------------------------------------------------- cli


def add_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--dataset", default="supermemory", choices=sorted(paths.DATASETS))
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--set", action="append", default=[], dest="overrides", metavar="K=V",
                    help="override one existing key, e.g. --set reader.top_k=64")
    return ap


def from_args(a: argparse.Namespace) -> Cfg:
    return load(dataset=a.dataset, model=a.model, overrides=a.overrides)


def describe(cfg: Cfg) -> str:
    r = cfg["_resolved"]
    bits = [f"dataset={r['dataset']}", f"model={r['model']}"]
    if r["overrides"]:
        bits.append("set=" + " ".join(r["overrides"]))
    return "  ".join(bits)
