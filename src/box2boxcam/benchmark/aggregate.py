"""Segment loading and per-variant aggregation with bootstrap over segments."""

import json
from pathlib import Path

import numpy as np

KEYS = [
    "n_in",
    "frac_in",
    "lt3",
    "ac_dx",
    "ac_in",
    "ev_in",
    "ev_dx",
    "ev_m",
    "ev_ac_in",
    "lag",
    "speed",
    "reversals_pm",
    "ball_used",
]
PAIRED_KEYS = (
    "lt3",
    "ac_dx",
    "lag",
    "reversals_pm",
    "ev_ac_in",
    "ev_in",
    "ev_dx",
    "ev_m",
)
RESAMPLES = 2000


def segments(segments_file, which):
    """Segments of one split (TUNE/TEST/FINAL), all of them (ALL) or the reference (REF)."""
    data = json.loads(Path(segments_file).read_text())
    if which == "REF":
        return data["reference"]
    return [s for s in data["segments"] if s["set"] == which or which == "ALL"]


def bootstrap_mean(values, rng, resamples=RESAMPLES):
    """Mean of ``values`` and its 95 % percentile bootstrap interval (resampling items)."""
    values = np.asarray(values, float)
    boots = [
        values[rng.integers(0, len(values), len(values))].mean()
        for _ in range(resamples)
    ]
    return [
        float(values.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    ]


def aggregate(per_seg, names, rng=None):
    """Mean [95 % CI] per metric over segments, pooled lag, paired differences vs ``names[0]``."""
    rng = np.random.default_rng(0) if rng is None else rng
    agg = {}
    for v in names:
        agg[v] = {}
        for k in KEYS:
            vals = np.array([s[v][k] for s in per_seg if s[v][k] is not None], float)
            if not len(vals):
                agg[v][k] = None
                continue
            agg[v][k] = bootstrap_mean(vals, rng)
        # pooled lag over all transitions, and paired differences vs the first variant
        lags = [lag for s in per_seg for lag in s[v]["lags"]]
        agg[v]["lag_pooled"] = float(np.mean(lags)) if lags else None
        agg[v]["n_trans"] = len(lags)
    base = names[0]
    for v in names[1:]:
        agg[v]["paired_vs_" + base] = {}
        for k in PAIRED_KEYS:
            d = np.array(
                [
                    s[v][k] - s[base][k]
                    for s in per_seg
                    if s[v][k] is not None and s[base][k] is not None
                ]
            )
            if not len(d):
                continue
            agg[v]["paired_vs_" + base][k] = bootstrap_mean(d, rng)
    return agg
