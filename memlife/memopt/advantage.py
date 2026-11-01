"""The MemOpt advantage: token-level, group-relative, length-fair.

For each training segment the writer samples a group of ``G`` candidates. The three FIRM
dimensions combine per token::

    x[i,t] = R_faith[i,t] * R_inf[i] * R_ret[i]

``R_inf`` and ``R_ret`` are per-candidate scalars broadcast across that candidate's tokens;
only faithfulness varies along the token axis. The product means a token earns credit only
when it is faithful **and** its memory is informative **and** retrievable -- any one of the
three at zero zeroes the token.

The group statistics are then **length-fair**::

    x̄_i      = (1/L_i) * sum_t x[i,t]                  each candidate's own mean
    mu_g     = (1/G)   * sum_i x̄_i                     MACRO over candidates
    sigma_g² = (1/G)   * sum_i (1/L_i) * sum_t (x[i,t] - mu_g)²
    Ã[i,t]   = clip( (x[i,t] - mu_g) / (sigma_g + eps), -kappa, +kappa )

Three details in those four lines each separate this from something that trains differently:

**Averaging within a candidate before the group statistics.** Pooling every token of every
candidate into one mean weights a 400-token candidate four times as heavily as a 100-token
one -- and candidate length *grows under training*, so the lengthening candidates would
increasingly define the baseline they are judged against.

**The squared deviation is from ``mu_g``, not from ``x̄_i``.** The mean *within*-candidate
variance drops the between-candidate variance entirely -- which, under a group-relative
advantage, is where almost all of the signal lives: informativeness and retrievability are
sequence constants, so every bit of their signal is between-candidate. Deviating from
``mu_g`` gives the pooled variance, macro-weighted: it keeps that term and still removes the
length bias.

**Re-centring after the clip.** The clip is one-sided in practice -- deviations run far in one
direction only -- so it perturbs the group mean, and the batch-wide whitening that follows
then reads that perturbation as a *between-group* signal with the wrong sign. An unclipped
z-score sums to zero over its group; subtracting the post-clip mask-weighted mean restores
that. Pooled rather than macro, because the whitening subtracts a mask-weighted batch mean and
only a mask-weighted correction neutralises it.

The arithmetic is plain numpy and importable without a trainer, so it can be tested directly.
:func:`register` installs it into verl when verl is present.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

#: The registry name, and the value ``algorithm.adv_estimator`` must carry.
NAME = "memopt"

#: Advantage clip, kappa. Not a safety rail: for a sparse non-negative deviation field the
#: standard deviation shrinks as violations get rarer, so each survivor is punished harder --
#: unbounded in the limit. The clip turns a gradient spike into a bounded one. The cost is
#: that in the sparse regime the dimension degenerates toward a binary flag, which
#: :func:`clip_fraction` is the direct read on.
CLIP = 3.0
EPS = 1e-6

#: ``non_tensor_batch`` keys the reward function writes. ``faith`` is the ``[B, T]`` token
#: array; the other two are ``[B]`` scalars.
FAITH_KEY = "faith"
INF_KEY = "inf"
RET_KEY = "ret"


# ============================================================================== the reward field

def combine(faith: np.ndarray, inf: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """The ``[B, T]`` reward field: the product of the three dimensions.

    No weights. Under a pure product a weight collapses to one global scalar on the array, and
    the z-score below divides its magnitude straight back out -- so it would be recorded in the
    run's configuration while changing nothing about the gradient (or, if negative, inverting
    all of it).
    """
    faith = np.asarray(faith, dtype="float64")
    inf = np.asarray(inf, dtype="float64").reshape(-1, 1)
    ret = np.asarray(ret, dtype="float64").reshape(-1, 1)
    if inf.shape[0] != faith.shape[0] or ret.shape[0] != faith.shape[0]:
        raise ValueError(f"inf {inf.shape[0]} / ret {ret.shape[0]} rows against "
                         f"{faith.shape[0]} rows of token-level faithfulness")
    return faith * inf * ret


# ============================================================================== the normaliser

def _groups(index: Sequence) -> dict:
    """Row indices per group id, in first-appearance order. ``index`` is one id per prompt,
    repeated once per candidate."""
    out: dict = {}
    for i, g in enumerate(index):
        out.setdefault(g, []).append(i)
    return out


def _moments(vals: np.ndarray, sel: np.ndarray) -> tuple[float, float] | None:
    """``(mu_g, sigma_g)`` for one group, or ``None`` when every candidate is empty.

    The one place the recipe's arithmetic lives, so a diagnostic reports the numbers the
    gradient used rather than a second implementation of them.

    A candidate with no in-mask token is excluded from ``G`` outright: it has no mean to
    contribute, and counting it as a zero would drag ``mu_g`` toward 0 for a row that receives
    no advantage anyway.
    """
    counts = sel.sum(axis=1)
    live = counts > 0
    if not live.any():
        return None
    counts = counts[live]
    # np.where, not vals * sel: out-of-mask positions hold whatever the padding held.
    means = np.where(sel, vals, 0.0).sum(axis=1)[live] / counts
    mu = float(means.mean())
    # THE DEVIATION IS FROM mu, NOT FROM means[i] -- see the module docstring.
    dev2 = np.where(sel, (vals - mu) ** 2, 0.0).sum(axis=1)[live] / counts
    return mu, float(np.sqrt(dev2.mean()))


def group_advantage(x: np.ndarray, mask: np.ndarray, index: Sequence, *,
                    clip: float = CLIP, eps: float = EPS) -> np.ndarray:
    """``Ã`` for one batch: ``[B, T]`` float64, already zeroed outside ``mask``."""
    if x.shape != mask.shape:
        raise ValueError(f"values {x.shape} against response mask {mask.shape}")
    if len(index) != x.shape[0]:
        raise ValueError(f"{len(index)} group ids for {x.shape[0]} rows")
    if not np.isfinite(x).all():
        # One NaN takes out the WHOLE BATCH, not one row: it poisons its group's statistics,
        # and the whitening cannot exclude it because `nan * 0 = nan`. Worse, clip_fraction
        # reports a healthy 0.0 for an all-NaN array, so the step log would say it was fine.
        raise ValueError(
            f"{int((~np.isfinite(x)).sum())} non-finite value(s) in the reward field. `inf` "
            f"and `ret` reach here non-finite only if imputation did not run; faithfulness is "
            f"finite whenever the audit margin is.")

    out = np.zeros(x.shape, dtype="float64")
    for rows in _groups(index).values():
        sel = np.asarray(mask[rows], dtype=bool)
        vals = np.asarray(x[rows], dtype="float64")
        moments = _moments(vals, sel)
        if moments is None:
            continue                 # every candidate empty: nothing to normalise or reward
        mu, sigma = moments
        # A flat group gives (x - mu) = 0 exactly, hence 0/(0 + eps) = 0 for every token:
        # zero advantage for the whole group, with no guard needed.
        z = (vals - mu) / (sigma + eps)
        # np.where, not * sel: a multiply leaves a non-finite z on the padding too, and the
        # whitening would carry it into every row of the batch.
        out[rows] = np.where(sel, np.clip(z, -clip, clip), 0.0)
        # RE-CENTRE AFTER CLIPPING, pooled. See the module docstring.
        n_sel = int(sel.sum())
        out[rows] -= (out[rows] * sel).sum() / n_sel * sel
    return out


def clip_fraction(a: np.ndarray, mask: np.ndarray, *, clip: float = CLIP) -> float:
    """Share of in-mask positions whose advantage saturated. Read it beside any reward curve:
    a high value means the signal has degenerated toward a sign function."""
    sel = np.asarray(mask, dtype=bool)
    n = int(sel.sum())
    if n == 0:
        return 0.0
    return float((np.abs(np.asarray(a)[sel]) >= clip - 1e-9).sum()) / n


def group_sigmas(x: np.ndarray, mask: np.ndarray, index: Sequence) -> list[float]:
    """``sigma_g`` per group, in first-appearance order. ``0.0`` means a flat group, which
    contributes exactly no gradient."""
    return [0.0 if m is None else m[1]
            for m in (_moments(np.asarray(x[rows], dtype="float64"),
                               np.asarray(mask[rows], dtype=bool))
                      for rows in _groups(index).values())]


# =============================================================================== the estimator

def compute_memopt_advantage(token_level_rewards, response_mask, index,
                             epsilon: float = EPS, config=None,
                             non_tensor_batch: dict | None = None,
                             batch: dict | None = None, **kwargs):
    """The estimator the trainer calls once per optimiser step."""
    if non_tensor_batch is None:
        # NOT a fallback to the scalar reward. Falling back would run this arm as plain GRPO
        # on a summed scalar and report it under the MemOpt name -- the token axis that the
        # whole objective exists to preserve would simply be absent.
        raise ValueError(
            f"{NAME} was called without non_tensor_batch, so the per-dimension rewards are "
            f"unreachable. memlife.memopt.train widens the trainer's forwarding to pass them; "
            f"reaching here means the run did not go through it. Refusing to fall back to the "
            f"scalar reward.")
    for key in (FAITH_KEY, INF_KEY, RET_KEY):
        if key not in non_tensor_batch:
            raise ValueError(
                f"{NAME}: reward key {key!r} is not in non_tensor_batch. Available: "
                f"{sorted(non_tensor_batch)}. The reward function must return all three.")

    import torch
    import verl.utils.torch_functional as verl_F

    faith = np.asarray(non_tensor_batch[FAITH_KEY], dtype="float64")
    if faith.ndim != 2 or faith.shape != tuple(response_mask.shape):
        # A ragged list-of-lists arrives as an object array of shape (B,), and a pad width
        # that disagrees with max_response_length arrives as (B, wrong). Both would otherwise
        # broadcast or truncate into a silently misaligned per-token signal.
        raise ValueError(
            f"{FAITH_KEY!r} has shape {faith.shape}, expected {tuple(response_mask.shape)}. "
            f"The reward function pads to data.max_response_length; the two must agree.")

    mask = response_mask.detach().to("cpu").numpy() > 0
    x = combine(faith, non_tensor_batch[INF_KEY], non_tensor_batch[RET_KEY])
    a = group_advantage(x, mask, list(index), clip=CLIP, eps=epsilon)
    adv = torch.tensor(a, dtype=token_level_rewards.dtype,
                       device=token_level_rewards.device)
    adv = verl_F.masked_whiten(adv, response_mask) * response_mask
    return adv, adv


def register() -> bool:
    """Put :data:`NAME` in verl's estimator registry. Returns False where verl is absent.

    Called at import, so the module stays importable (and testable) without a trainer. In the
    trainer the miss is not silent -- the registry raises before the first optimiser step.

    ``find_spec``, not ``except ImportError``: a broken verl install raises ImportError too,
    and swallowing that would turn a fixable dependency problem into "unknown advantage
    estimator" much later. The membership check is not an optimisation either -- re-registering
    a different function object raises, which happens on a module reload.
    """
    import importlib.util

    if importlib.util.find_spec("verl") is None:
        return False
    from verl.trainer.ppo import core_algos

    if NAME not in core_algos.ADV_ESTIMATOR_REGISTRY:
        core_algos.register_adv_est(NAME)(compute_memopt_advantage)
    return True


REGISTERED = register()
