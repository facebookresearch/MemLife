# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Turn the upstream releases into the contract :mod:`memlife.data.prepared` describes.

    python -m memlife.data.prepare transcode   --src raw/SuperMemory-VQA/data/video \
                                               --dst data/supermemory/videos
    python -m memlife.data.prepare supermemory --src raw/SuperMemory-VQA
    python -m memlife.data.prepare egolife     --src raw/EgoLife

``transcode`` is the generic step and the one that matters for reproducibility: **every video
is re-encoded once, to the geometry the model's own processor would have chosen.** After that
the processor's resize is a no-op, every method sees identical pixels for a given video, and
nothing is resampled at run time.

``supermemory`` converts the release's questions and transcripts onto per-subject clocks.
``egolife`` merges EgoLife's ~30 s clips into contiguous runs with the same encoder, writes
``videos.json`` with the member map that makes the run clock exact, and converts the per-hour
SRTs and the EgoLifeQA questions onto it.
"""
