"""Figures for the README and model card: precision-recall curves and training curves.

``pr`` reads cached detections written by ``scripts/evaluate_detector.py --cache-dir`` and
plots, for each model, precision against recall over all confidence thresholds (IoU >= 0.5).
``curves`` plots the per-epoch ``results.csv`` written by Ultralytics training.

Examples::

    python scripts/make_report_figures.py pr --frames-dir heldout/images --labels heldout/labels \\
        --model "box2boxcam v1.0=cache/box2boxcam-v1.0_960_0.05.json" \\
        --model "stock YOLO26s=cache/stock-yolo26s_960_0.05_c32.json" --out pr.png
    python scripts/make_report_figures.py curves --results-csv train/results.csv --out curves.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from box2boxcam.evaluation.detect import load_detections
from box2boxcam.evaluation.ground_truth import load_ground_truth
from box2boxcam.evaluation.matching import all_balls, match_all

COLORS = ["#1b6ca8", "#d9822b", "#7a7a7a"]


def pr_points(matches, n_balls):
    """Precision and recall after each detection, by decreasing confidence."""
    events = sorted((c, j >= 0) for frame in matches for c, j in frame)[::-1]
    flags = np.array([e[1] for e in events], dtype=float)
    tp = np.cumsum(flags)
    fp = np.cumsum(1 - flags)
    return tp / max(n_balls, 1), tp / np.maximum(tp + fp, 1)


def plot_pr(args):
    frames = load_ground_truth(args.frames_dir, args.labels)
    gts = [f.boxes for f in frames]
    ids = [f.id for f in frames]
    n_balls = len(all_balls(gts))
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    for k, spec in enumerate(args.model):
        name, _, path = spec.partition("=")
        matches = match_all(gts, load_detections(path, ids))
        recall, precision = pr_points(matches, n_balls)
        ax.plot(recall, precision, color=COLORS[k % 3], lw=2, label=name)
        for t, marker in ((0.25, "o"), (0.55, "s")):
            tp = sum(1 for fr in matches for c, j in fr if c >= t and j >= 0)
            n = sum(1 for fr in matches for c, _ in fr if c >= t)
            if n:
                ax.plot(tp / n_balls, tp / n, marker, color=COLORS[k % 3], ms=7)
    ax.plot([], [], "o", color="k", label="confidence 0.25")
    ax.plot([], [], "s", color="k", label="confidence 0.55")
    ax.set(
        xlabel=f"Recall (share of the {n_balls} balls found)",
        ylabel="Precision",
        xlim=(0, 0.95),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out)


def plot_curves(args):
    with open(args.results_csv) as f:
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    epoch = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=130)
    panels = [
        ("Box loss", [("train/box_loss", "train"), ("val/box_loss", "validation")]),
        ("Class loss", [("train/cls_loss", "train"), ("val/cls_loss", "validation")]),
        (
            "Validation mAP",
            [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")],
        ),
    ]
    for ax, (ylabel, series) in zip(axes, panels):
        for k, (key, label) in enumerate(series):
            ax.plot(epoch, [r[key] for r in rows], color=COLORS[k], lw=2, label=label)
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pr")
    a.add_argument("--frames-dir", type=Path, required=True)
    a.add_argument("--labels", type=Path, required=True)
    a.add_argument("--model", action="append", required=True, metavar="NAME=CACHE.json")
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(fn=plot_pr)
    c = sub.add_parser("curves")
    c.add_argument("--results-csv", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(fn=plot_curves)
    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
