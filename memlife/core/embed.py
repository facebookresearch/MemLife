"""Text embeddings. One model, one normalisation, one device policy.

``BAAI/bge-large-en-v1.5``, 1024-d, L2-normalised -- so an inner product *is* cosine and the
index can be an exact flat IP index with no recall loss.

**CPU by default, and CPU is mandatory for MemOpt reward scoring.** Retrievability compares a
candidate's cosine against cached competitor scores at margins measured as tight as 7e-06, well
inside CPU-vs-GPU float divergence. Build the table on one device and query it on the other and
hits flip. Pinning CPU everywhere costs little and removes the whole class of problem.
"""
from __future__ import annotations

import functools
import threading
from typing import Sequence

import numpy as np

MODEL_ID = "BAAI/bge-large-en-v1.5"
#: Directory under ``$MEMLIFE_ROOT/models/`` holding a self-contained copy, preferred over the
#: hub id so an offline compute node does not have to resolve the hub cache layout.
LOCAL_DIRNAME = "bge-large-en-v1.5"
DIM = 1024

_BATCH = 32
_CACHE = 10_000

_lock = threading.Lock()
_model = None
_model_device: str | None = None


def device_in_use() -> str | None:
    """The device the embedder was loaded on, or None if it has not been used yet."""
    return _model_device


def _norm_device(device: str) -> str:
    """Canonicalise so ``cuda`` and ``cuda:0`` are one device."""
    try:
        import torch

        d = torch.device(device)
    except Exception:
        return device
    if d.type in ("cuda", "xpu", "hpu", "mps"):
        return f"{d.type}:{0 if d.index is None else d.index}"
    return d.type


def _resolve() -> str:
    from .. import paths

    d = paths.models_root() / LOCAL_DIRNAME
    return str(d) if (d / "config.json").exists() else MODEL_ID


def _get(device: str):
    global _model, _model_device
    with _lock:
        if _model is not None and _model_device != device:
            raise RuntimeError(
                f"embedder already loaded on {_model_device!r}, refusing to also serve "
                f"{device!r}. Mixing devices flips retrieval hits at ~1e-6 margins.")
        if _model is None:
            from sentence_transformers import SentenceTransformer

            src = _resolve()
            print(f"[embed] loading {src} on {device}", flush=True)
            _model = SentenceTransformer(src, device=device)
            _model_device = device
        return _model


def encode(texts: Sequence[str], *, device: str = "cpu") -> np.ndarray:
    """``(n, 1024)`` float32, L2-normalised. No instruction prefix -- BGE symmetric usage."""
    if not texts:
        return np.zeros((0, DIM), dtype="float32")
    v = _get(_norm_device(device)).encode(
        list(texts), batch_size=_BATCH, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False)
    return np.ascontiguousarray(v, dtype="float32")


@functools.lru_cache(maxsize=_CACHE)
def _encode_one_cached(text: str, device: str) -> np.ndarray:
    v = encode([text], device=device)[0]
    v.flags.writeable = False           # a shared cache entry must not be mutable by a caller
    return v


def encode_one(text: str, *, device: str = "cpu") -> np.ndarray:
    """Single string, memoised -- queries repeat constantly across questions and rounds."""
    return _encode_one_cached(text, _norm_device(device))
