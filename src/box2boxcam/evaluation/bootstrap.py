"""Paired bootstrap over frames for the difference between two detectors."""

import numpy as np

from box2boxcam.evaluation.matching import (
    count_small_tp,
    count_tp_fp,
    max_tp_under_fp_budget,
)


def stat_functions(thresholds, budgets, small, small_threshold):
    """Named statistics ``f(matches, frames) -> int``; ``frames=None`` means all frames."""
    stats = {}
    for t in thresholds:
        stats[f"TP@{t}"] = lambda m, fr, t=t: count_tp_fp(m, t, fr)[0]
        stats[f"FP@{t}"] = lambda m, fr, t=t: count_tp_fp(m, t, fr)[1]
    stats[f"smallTP@{small_threshold}"] = lambda m, fr: count_small_tp(
        m, small, small_threshold, fr
    )
    for budget in budgets:
        stats[f"maxTP@FP<={budget}"] = lambda m, fr, b=budget: max_tp_under_fp_budget(
            m, b, fr
        )
    return stats


def paired_bootstrap(matches_a, matches_b, stats, resamples=2000, seed=0):
    """Frames are resampled with replacement, the same draw for both models.

    Returns ``{name: {"observed", "low", "high", "p_le_0"}}`` for ``A - B``;
    ``low``/``high`` is the 95 % percentile interval, ``p_le_0`` the share of
    resamples with a difference <= 0.
    """
    if len(matches_a) != len(matches_b):
        raise ValueError("both models need the same frames")
    n = len(matches_a)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, n, n) for _ in range(resamples)]
    out = {}
    for name, fn in stats.items():
        observed = fn(matches_a, None) - fn(matches_b, None)
        values = np.array([fn(matches_a, d) - fn(matches_b, d) for d in draws])
        out[name] = {
            "observed": int(observed),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
            "p_le_0": float(np.mean(values <= 0)),
        }
    return out
