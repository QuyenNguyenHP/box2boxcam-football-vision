"""Evaluation table and bootstrap report for one or two detectors."""

from box2boxcam.evaluation.bootstrap import paired_bootstrap, stat_functions
from box2boxcam.evaluation.matching import (
    all_balls,
    count_tp_fp,
    max_tp_under_fp_budget,
    recovery,
    small_balls,
)

DEFAULT_THRESHOLDS = (0.25, 0.35, 0.45, 0.55)
DEFAULT_BUDGETS = (25, 60, 100, 200)


def summarise(matches, gts, thresholds, budgets, small):
    """Metrics of one model: TP/FP/small-TP per threshold, max TP per FP budget."""
    rows = []
    for t in thresholds:
        tp, fp, found = count_tp_fp(matches, t)
        rows.append(
            {"threshold": t, "tp": tp, "fp": fp, "small_tp": len(found & small)}
        )
    return {
        "balls": len(all_balls(gts)),
        "small_balls": len(small),
        "frames": len(gts),
        "thresholds": rows,
        "max_tp_under_fp_budget": {
            str(b): max_tp_under_fp_budget(matches, b) for b in budgets
        },
    }


def markdown_table(results, thresholds, budgets):
    """One row per model: ``TP/FP (small TP)`` per threshold and the budget columns."""
    head = (
        "| model | "
        + " | ".join(f"{t:.2f} TP/FP (small)" for t in thresholds)
        + " | max TP at FP<="
        + "/".join(str(b) for b in budgets)
        + " |"
    )
    lines = [head, "|---|" + "---|" * (len(thresholds) + 1)]
    for name, r in results.items():
        cells = [f"{x['tp']}/{x['fp']} ({x['small_tp']})" for x in r["thresholds"]]
        budget_cell = "/".join(
            str(r["max_tp_under_fp_budget"][str(b)]) for b in budgets
        )
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {budget_cell} |")
    return "\n".join(lines)


def bootstrap_table(boot):
    lines = ["| statistic | A - B | 95 % CI | P(diff <= 0) |", "|---|---|---|---|"]
    for name, v in boot.items():
        lines.append(
            f"| {name} | {v['observed']:+d} | [{v['low']:+.0f}, {v['high']:+.0f}] "
            f"| {v['p_le_0']:.3f} |"
        )
    return "\n".join(lines)


def evaluate(
    gts,
    matches_by_model,
    thresholds=DEFAULT_THRESHOLDS,
    budgets=DEFAULT_BUDGETS,
    small_max_side=16.0,
    resamples=2000,
    seed=0,
):
    """Full report. The first model is A; if a second one exists it is B and the
    bootstrap gives ``A - B`` over frames."""
    small = small_balls(gts, small_max_side)
    names = list(matches_by_model)
    results = {
        n: summarise(matches_by_model[n], gts, thresholds, budgets, small)
        for n in names
    }
    report = {"models": results, "table": markdown_table(results, thresholds, budgets)}
    if len(names) == 2:
        a, b = (matches_by_model[n] for n in names)
        stats = stat_functions(thresholds, budgets, small, thresholds[0])
        report["bootstrap"] = {
            "A": names[0],
            "B": names[1],
            "resamples": resamples,
            "seed": seed,
            "stats": paired_bootstrap(a, b, stats, resamples, seed),
        }
        report["recovery_at_first_threshold"] = recovery(
            a, b, gts, thresholds[0], small
        )
    return report
