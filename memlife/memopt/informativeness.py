# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Informativeness: does this memory still carry the answer-relevant evidence its segment
supplied?

    R_inf(c, y) = mean over q in Q(c) of  P_ent( y => k_{c,q} )

One text-only entailment call per (candidate, question) pair, against the source-grounded key
fact :mod:`memlife.memopt.keyfacts` extracted for that pair. The candidate is the only thing
that varies inside a rollout group, so this is the dimension that asks "did the writer keep
what a later reader will need?" **without ever showing the writer a question.**

Text-only by construction: the two texts are the key fact and the candidate. The evaluator
sees no pixels at all here -- grounding is faithfulness's job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import judge

ENTAIL_SYSTEM = (
    "You are a strict entailment judge. You are given a TARGET FACT about one 30-second "
    "segment of a first-person video, and a MEMORY that was written for that same segment. "
    "Decide whether the memory states the target fact, or states something from which the "
    "target fact follows directly. A paraphrase counts. A vaguer statement that omits the "
    "specific detail in the target fact does not. Use only these two texts - no outside "
    "knowledge and no guessing. Answer with exactly one word: Yes or No."
)
ENTAIL_USER = (
    '## TARGET FACT\n"{key_fact}"\n'
    '\n## MEMORY\n"{description}"\n\n'
    "Does the memory state or directly entail the target fact? Answer Yes or No.\nAnswer:"
)


def build_messages(key_fact: str, description: str) -> list[dict]:
    return [{"role": "system", "content": ENTAIL_SYSTEM},
            {"role": "user", "content": ENTAIL_USER.format(key_fact=key_fact,
                                                           description=description)}]


@dataclass(frozen=True)
class Inf:
    """One candidate's informativeness, already averaged over ``Q(c)``."""

    value: float
    n_q: int
    n_unreadable: int


def score(items: Sequence[tuple[str, Sequence[str]]]) -> list[Inf]:
    """``(description, key_facts)`` per candidate -> one :class:`Inf` each.

    Every (candidate, key fact) pair goes out in ONE fan-out, so a step's entailment calls are
    a single batch rather than one round trip per candidate.

    An unreadable probability counts as 0.0 **and is counted**. Filling it with the mean would
    invent a verdict; dropping the question from the denominator would make the denominator
    itself depend on the candidate. A row with NOTHING readable is an evaluator failure rather
    than a verdict, so it is NaN, which the trainer fills with the group's mean.
    """
    # A bare str IS a Sequence[str], so ("memory", "milk") would fan out four calls over
    # 'm','i','l','k' and report a perfectly plausible score over letters.
    if bad := [i for i, (_d, facts) in enumerate(items) if isinstance(facts, str)]:
        raise TypeError(f"item(s) {bad[:5]}: key facts must be a list, not a bare string")
    jobs = [(i, kf) for i, (_d, facts) in enumerate(items) for kf in facts]
    # Q(c) is DEFINED as the pairs that have a key fact, so an empty one here means a caller
    # read the key-fact rows without filtering -- and the evaluator would be asked, for a whole
    # run, whether the memory entails the empty string, which returns an ordinary probability.
    # Fatal, where an empty LIST below is not: a missing question is one row scoring 0.0,
    # whereas this is systematic and silent.
    if bad := [i for i, kf in jobs if not (kf or "").strip()]:
        raise ValueError(
            f"empty key fact for {len(bad)} pair(s), e.g. item {bad[0]}: Q(c) must exclude "
            f"pairs the extractor returned NONE for (see memlife.memopt.keyfacts).")

    ps = judge.p_yes_batch([build_messages(kf, items[i][0] or "") for i, kf in jobs])
    total = [0.0] * len(items)
    unread = [0] * len(items)
    for (i, _kf), p in zip(jobs, ps):
        if p is None:
            unread[i] += 1
        else:
            total[i] += p
    # max(n, 1): the pool guarantees |Q(c)| >= 1, so a zero here means the key-fact rows and
    # the rollout dataset disagree about the pool. Scoring it 0.0 keeps the row constant
    # within its group -- contributing exactly zero advantage -- rather than raising inside the
    # trainer loop.
    return [Inf(value=(math.nan if facts and unread[i] == len(facts)
                       else total[i] / max(len(facts), 1)),
                n_q=len(facts), n_unreadable=unread[i])
            for i, (_d, facts) in enumerate(items)]
