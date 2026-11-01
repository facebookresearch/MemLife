"""The MemOpt trainer entry point.

    python -m memlife.memopt.train <the same hydra overrides verl.trainer.main_ppo takes>
    python -m memlife.memopt.train --preflight          # check the wiring in seconds

MemOpt runs on `verl <https://github.com/volcengine/verl>`_'s GRPO trainer. Three things have
to happen in the process that computes advantages, and stock verl does none of them:

**1. Register the estimator where it is used.** The registry is per-process, and advantages are
computed inside the driver actor -- not the shell that ran ``python -m``, and not the reward
workers. Importing :mod:`memlife.memopt.advantage` from the reward function therefore does not
help. :class:`TaskRunner` below is an actor defined in this module whose ``run`` installs
everything in its own process and then hands off to verl's own runner verbatim.

**2. Forward the per-dimension rewards.** verl widens an estimator's keyword arguments for one
built-in name only; a custom estimator receives the summed scalar and nothing else -- i.e.
exactly the three arrays MemOpt needs are unreachable. :func:`install` binds them onto the
registry entry for the duration of one call, rather than editing verl's source.

**3. Impute, and log, per step.** The reward function is per-rollout, so a candidate whose
faithfulness audit failed can only mark it NaN; the fill needs the whole group and therefore
has to happen here. Zero-filling is NOT the neutral choice a reader might assume -- in a group
scoring ~0.8 a 0.0 is one of the strongest negative signals in the batch, handed out for an
evaluator hiccup. The scorable members' mean gives exactly zero advantage, which is what
"no information" should be worth.

If verl changes, the failure is loud: ``--preflight`` fails in seconds rather than forty
minutes into an allocation.
"""
from __future__ import annotations

import functools
import os
import sys

import numpy as np

from . import advantage as madv


# ------------------------------------------------------------------------------ imputation

def _groups(uids) -> dict:
    out: dict = {}
    for i, g in enumerate(uids):
        out.setdefault(g, []).append(i)
    return out


def impute(non_tensor_batch: dict, uids, response_mask: np.ndarray) -> dict:
    """Fill unscorable rewards with their group's scorable mean. Returns per-step counts.

    Order matters: the token array's own NaN mask is read before it is filled, and the scalar
    dimensions are filled independently -- a row can be scorable on one and not another.
    """
    stats = {}
    n = len(uids)
    for key in (madv.INF_KEY, madv.RET_KEY):
        col = np.asarray(non_tensor_batch[key], dtype="float64").reshape(-1)
        bad = ~np.isfinite(col)
        for rows in _groups(uids).values():
            rows = np.asarray(rows)
            good = rows[~bad[rows]]
            # A group with no scorable member gets 0.0 -- constant within the group, hence
            # exactly zero advantage, rather than a fabricated ranking.
            col[rows[bad[rows]]] = float(col[good].mean()) if good.size else 0.0
        non_tensor_batch[key] = col
        stats[f"{key}_imputed"] = int(bad.sum())

    faith = np.asarray(non_tensor_batch[madv.FAITH_KEY], dtype="float64")
    if faith.shape != response_mask.shape:
        raise ValueError(
            f"'{madv.FAITH_KEY}' has shape {faith.shape}, expected {response_mask.shape}. "
            f"MEMLIFE_MAX_RESPONSE_LENGTH must equal data.max_response_length.")
    bad_rows = ~np.isfinite(faith).all(axis=1)
    for rows in _groups(uids).values():
        rows = np.asarray(rows)
        good = rows[~bad_rows[rows]]
        if good.size:
            sel = response_mask[good] > 0
            fill = float(np.where(sel, faith[good], 0.0).sum() / max(1, int(sel.sum())))
        else:
            fill = 0.0
        faith[rows[bad_rows[rows]]] = fill
    non_tensor_batch[madv.FAITH_KEY] = faith
    stats["faith_imputed"] = int(bad_rows.sum())
    stats["n"] = n
    return stats


def log_step(non_tensor_batch: dict, uids, response_mask: np.ndarray) -> dict:
    """Per-step dimension statistics. Read these beside any reward curve.

    ``flat_group_frac`` is the number that matters most: a dimension that is constant within a
    group contributes exactly nothing to that group's gradient, however healthy its mean
    looks.
    """
    out: dict = {}
    for key in (madv.INF_KEY, madv.RET_KEY):
        col = np.asarray(non_tensor_batch[key], dtype="float64").reshape(-1)
        flat = sum(float(np.ptp(col[rows])) < 1e-12 for rows in _groups(uids).values())
        out[f"memopt/{key}_mean"] = float(col.mean())
        out[f"memopt/{key}_flat_group_frac"] = flat / max(1, len(_groups(uids)))
    faith = np.asarray(non_tensor_batch[madv.FAITH_KEY], dtype="float64")
    mask = response_mask > 0
    out["memopt/faith_mean"] = float(faith[mask].mean()) if mask.any() else 0.0

    x = madv.combine(faith, non_tensor_batch[madv.INF_KEY], non_tensor_batch[madv.RET_KEY])
    a = madv.group_advantage(x, mask, list(uids))
    sig = madv.group_sigmas(x, mask, list(uids))
    out["memopt/sigma_g_mean"] = float(np.mean(sig)) if sig else 0.0
    out["memopt/sigma_g_zero_frac"] = (sum(s == 0.0 for s in sig) / len(sig)) if sig else 0.0
    out["memopt/clip_frac"] = madv.clip_fraction(a, mask)

    # The only checkable form of "did the audited span cover the response".
    nt = non_tensor_batch.get("n_tokens")
    if nt is not None:
        nt = np.asarray(nt, dtype="float64").reshape(-1)
        lens = mask.sum(axis=1)
        # The response includes an end-of-turn token the audited span does not carry, so
        # n_tokens == len - 1 is the aligned case.
        out["memopt/span_short_frac"] = float((nt < lens - 1).mean())
        out["memopt/span_long_frac"] = float((nt > lens).mean())
    return out


# ------------------------------------------------------------------------------ installation

def _numpy(t):
    return t.detach().to("cpu").numpy() if hasattr(t, "detach") else np.asarray(t)


def _wrap_compute_advantage(original):
    @functools.wraps(original)
    def compute_advantage(data, adv_estimator, *args, **kwargs):
        if str(adv_estimator) not in (madv.NAME, f"AdvantageEstimator.{madv.NAME}"):
            return original(data, adv_estimator, *args, **kwargs)
        from verl.trainer.ppo import core_algos

        ntb = data.non_tensor_batch
        uids = list(ntb["uid"])
        mask = _numpy(data.batch["response_mask"])
        stats = impute(ntb, uids, mask)
        metrics = log_step(ntb, uids, mask)
        print(f"[memopt] {stats} {({k: round(v, 4) for k, v in metrics.items()})}", flush=True)
        if hasattr(data, "meta_info"):
            data.meta_info.setdefault("metrics", {}).update(metrics)

        # Bind the two arguments verl does not forward to a custom estimator, for the
        # duration of this one call. Binding rather than re-assembling the whole kwarg set
        # keeps index, config and everything else exactly as verl built them.
        reg = core_algos.ADV_ESTIMATOR_REGISTRY
        base = reg[madv.NAME]
        reg[madv.NAME] = functools.partial(base, non_tensor_batch=ntb, batch=data.batch)
        try:
            return original(data, adv_estimator, *args, **kwargs)
        finally:
            reg[madv.NAME] = base

    return compute_advantage


_INSTALLED = False


def install() -> None:
    """Register the estimator and widen advantage computation, in THIS process. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    if not madv.register():
        raise ImportError("verl is not installed; MemOpt training needs it. "
                          "Install it with: pip install -e '.[train]'")
    from verl.trainer.ppo import ray_trainer

    if not hasattr(ray_trainer, "compute_advantage"):
        raise RuntimeError(
            "verl.trainer.ppo.ray_trainer has no `compute_advantage`. This verl version does "
            "not match what memlife.memopt.train patches; see the module docstring for the "
            "three things that have to happen and where.")
    ray_trainer.compute_advantage = _wrap_compute_advantage(ray_trainer.compute_advantage)
    _INSTALLED = True
    print(f"[memopt] installed: estimator={madv.NAME}, advantage forwarding widened",
          flush=True)


def preflight() -> int:
    """Check the wiring without allocating anything. Fails in seconds, not minutes."""
    ok = True
    try:
        install()
    except Exception as e:                                # noqa: BLE001 -- report every fault
        print(f"[preflight] FAIL install: {type(e).__name__}: {e}")
        return 1
    from verl.trainer.ppo import core_algos

    if madv.NAME not in core_algos.ADV_ESTIMATOR_REGISTRY:
        print(f"[preflight] FAIL {madv.NAME} is not registered")
        ok = False
    try:
        _verl_task_runner()
    except Exception as e:                                # noqa: BLE001 -- report every fault
        print(f"[preflight] FAIL TaskRunner hook: {e}")
        ok = False
    for var in ("MEMLIFE_PACK", "MEMLIFE_KEYFACTS", "MEMLIFE_FRAMES",
                "MEMLIFE_MAX_RESPONSE_LENGTH", "MEMLIFE_JUDGE_ENDPOINTS"):
        if not os.environ.get(var):
            print(f"[preflight] FAIL {var} is unset")
            ok = False

    # The arithmetic, on a synthetic group: two candidates, the second strictly more faithful.
    faith = np.array([[0.2, 0.9], [1.0, 1.0]])
    a = madv.group_advantage(
        madv.combine(faith, np.array([1.0, 1.0]), np.array([1.0, 1.0])),
        np.ones_like(faith, dtype=bool), ["g", "g"])
    if not a[1].mean() > a[0].mean():
        print(f"[preflight] FAIL the more faithful candidate does not score higher: {a}")
        ok = False
    print(f"[preflight] {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------ entry point

def _verl_task_runner():
    """verl's v0 ``TaskRunner`` class, unwrapped from its ``@ray.remote`` decorator.

    A Ray actor class cannot be instantiated locally; Ray keeps the real class at
    ``__ray_metadata__.modified_class``.
    """
    from verl.trainer import main_ppo_v0

    impl = getattr(getattr(main_ppo_v0.TaskRunner, "__ray_metadata__", None), "modified_class",
                   None)
    if impl is None or not callable(getattr(impl, "run", None)):
        raise RuntimeError(
            "cannot reach the class behind verl's @ray.remote TaskRunner "
            "(verl.trainer.main_ppo_v0.TaskRunner.__ray_metadata__.modified_class). This verl or "
            "Ray version does not match what memlife.memopt.train wraps.")
    if getattr(impl, "__memlife__", False):
        raise RuntimeError("main_ppo_v0.TaskRunner is already MemOpt's wrapper: this process is "
                           "both driver and actor (ray local_mode?)")
    return impl


class TaskRunner:
    """The actor verl creates instead of its own: install, then verl's ``run`` verbatim.

    ``run`` imports this module BY NAME: under ``python -m`` this file is ``__main__``, so Ray
    ships the class by value and nothing else would import the module in the actor.
    """

    __memlife__ = True

    def run(self, config):
        from memlife.memopt.train import _verl_task_runner, install

        install()
        return _verl_task_runner()().run(config)


def main() -> None:
    if "--preflight" in sys.argv:
        raise SystemExit(preflight())

    import ray
    from verl.trainer import main_ppo as verl_main
    from verl.trainer import main_ppo_v0

    # The v0 trainer (trainer.use_v1=False): verl's main imports main_ppo_v0.TaskRunner inside
    # the function, so rebinding the module attribute is enough. The v1 trainer never calls
    # install(), and the run would die on "unknown advantage estimator".
    _verl_task_runner()
    main_ppo_v0.TaskRunner = ray.remote(num_cpus=1)(TaskRunner)
    verl_main.main()


if __name__ == "__main__":
    main()
