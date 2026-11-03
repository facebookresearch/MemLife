# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MemLife — episodic memory for long-horizon egocentric video QA, and MemOpt, its writer
reinforcement-learning framework.

Two entry points matter:

* ``memlife.memory.build``  — turn a video stream into a memory store (the MemLife writer)
* ``memlife.eval``          — answer a benchmark's questions from that store (the MemLife reader)

and one more if you are training the writer:

* ``memlife.memopt.train``  — GRPO over the FIRM reward (MemOpt)
"""

__version__ = "1.0.0"
