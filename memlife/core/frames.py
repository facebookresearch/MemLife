# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Frame sampling. ``n`` frames spread evenly over a file-relative time window.

**Times here are FILE-RELATIVE seconds, never the dataset's absolute clock.** Passing the
absolute clock used to succeed silently: every index clamped to the last frame, the tail-repeat
padded the result back up to ``n``, and the arity check passed -- producing a whole store of
"eight copies of the final frame" captions that looks completely normal on disk. The window
checks below exist to make that loud.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np

#: Qwen vision geometry: patch 16 x spatial merge 2 -> one token per 32x32 px.
GRID_FACTOR = 32
#: 512 tokens. The prepared 704x704 frames are 22x22 = 484 tokens, so this is a cap, not a
#: transform -- preparation already resized.
MAX_PIXELS = 512 * GRID_FACTOR * GRID_FACTOR

#: A tail window may legitimately be short, and repeating the last frame beats failing. A
#: window mostly outside the file is a clock mixup, so cap how much padding is tolerated.
_MAX_PAD_FRACTION = 0.5
#: ...and require most of the request to exist at all: [99, 129) on a 100 s file otherwise
#: clamps to one second of video and returns eight well-formed, wholly mislabelled frames.
_MIN_INSIDE_FRACTION = 0.5

_READER_CACHE = 2


class DecodeError(RuntimeError):
    """A window that cannot be sampled. Reported to the reader as a tool error, never as
    an empty frame list -- "no frames" reads as "nothing happened here"."""


@functools.lru_cache(maxsize=_READER_CACHE)
def _reader(path: str):
    import decord

    return decord.VideoReader(path, num_threads=1)


def fit_dims(w: int, h: int, max_pixels: int = MAX_PIXELS,
             factor: int = GRID_FACTOR) -> tuple[int, int]:
    """Largest ``factor``-aligned size within ``max_pixels``. The preparation geometry.

    This mirrors the SHRINK half of Qwen's own ``smart_resize``, and mirroring it exactly is
    the point: preparation resizes once to the size the processor would have chosen, so at run
    time the processor's resize is a no-op and every method sees identical pixels.

    Round-to-nearest per axis first, so an already-conforming size is a fixed point; then, if
    that is over budget, a uniform scale followed by **independent floor-to-grid per axis**.
    Independent flooring is what Qwen does, so the slight aspect squeeze it can cause is the
    reference behaviour rather than a deviation. The upscale half is deliberately not
    implemented -- nothing here benefits from inventing pixels.

    >>> fit_dims(1408, 1408)
    (704, 704)
    >>> fit_dims(720, 958)
    (608, 832)
    >>> fit_dims(*fit_dims(1408, 1408)) == fit_dims(1408, 1408)   # idempotent
    True
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"bad source geometry {w}x{h}")
    wb = max(factor, round(w / factor) * factor)
    hb = max(factor, round(h / factor) * factor)
    if wb * hb > max_pixels:
        beta = math.sqrt(w * h / max_pixels)
        wb = max(factor, math.floor(w / beta / factor) * factor)
        hb = max(factor, math.floor(h / beta / factor) * factor)
    return wb, hb


def n_vision_tokens(w: int, h: int, factor: int = GRID_FACTOR) -> int:
    return (w // factor) * (h // factor)


def frame_indices(total: int, fps: float, t_start: float, t_end: float, n: int,
                  *, allow_pad: bool = True, tag: str = "") -> np.ndarray:
    """The exact indices :func:`sample_frames` decodes. Pure; opens nothing.

    Factored out so an offline replay (MemOpt's retrievability trace, say) can reproduce what
    the agent really saw without a video reader. A hand-rolled ``linspace`` at each call site
    is the drift this function exists to prevent.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (math.isfinite(t_start) and math.isfinite(t_end)):
        raise DecodeError(f"{tag}: non-finite window [{t_start}, {t_end}]")
    if t_end <= t_start:
        raise DecodeError(f"{tag}: empty or inverted window [{t_start:.3f}, {t_end:.3f}]")
    if total == 0:
        raise DecodeError(f"{tag}: zero frames")
    if not (fps > 0 and math.isfinite(fps)):
        raise DecodeError(f"{tag}: bad frame rate {fps!r}; every segment would decode frame 0")

    duration = total / fps
    if t_start >= duration or t_end <= 0:
        raise DecodeError(
            f"{tag} [{t_start:.2f}, {t_end:.2f}]: window lies outside the file "
            f"(duration {duration:.2f}s). Are these file-relative seconds?")
    inside = min(t_end, duration) - max(t_start, 0.0)
    if inside < _MIN_INSIDE_FRACTION * (t_end - t_start):
        raise DecodeError(
            f"{tag} [{t_start:.2f}, {t_end:.2f}]: only {inside:.2f}s of a "
            f"{t_end - t_start:.2f}s window is inside the file (duration {duration:.2f}s).")

    lo = max(0, min(total - 1, int(round(t_start * fps))))
    hi = max(lo, min(total - 1, int(round(t_end * fps)) - 1))
    idx = np.unique(np.linspace(lo, hi, n).round().astype(int))
    if len(idx) < n:
        if not allow_pad:
            raise DecodeError(
                f"{tag} [{t_start:.2f}, {t_end:.2f}]: only {len(idx)} distinct frames for n={n} "
                f"and allow_pad=False. The writer must never caption duplicated frames.")
        if len(idx) < n * (1.0 - _MAX_PAD_FRACTION):
            raise DecodeError(
                f"{tag} [{t_start:.2f}, {t_end:.2f}]: only {len(idx)} distinct frames for n={n} "
                f"(fps {fps:.2f}, duration {duration:.2f}s). Too degenerate to pad.")
        idx = np.concatenate([idx, np.full(n - len(idx), idx[-1])])
    return idx


def sample_frames(path: str | Path, t_start: float, t_end: float, n: int,
                  *, allow_pad: bool = True) -> list:
    """``n`` PIL frames over ``[t_start, t_end)``, at the file's native geometry.

    Native == prepared == what the model sees: preparation already applied :func:`fit_dims`.

    ``allow_pad=False`` refuses to repeat a frame, and **the writer always passes it**: the
    shortest legal segment still yields far more than ``frames_per_segment`` distinct frames,
    so padding there can only mean a bug. Only ``fetch_video``, whose window the model chooses,
    may legitimately pad.
    """
    from PIL import Image

    vr = _reader(str(path))
    idx = frame_indices(len(vr), float(vr.get_avg_fps()), t_start, t_end, n,
                        allow_pad=allow_pad, tag=str(path))
    batch = vr.get_batch(idx.tolist()).asnumpy()
    if batch.shape[0] != n:
        raise DecodeError(f"{path} [{t_start:.2f},{t_end:.2f}]: asked {n} frames, "
                          f"got {batch.shape[0]}")
    return [Image.fromarray(f) for f in batch]

