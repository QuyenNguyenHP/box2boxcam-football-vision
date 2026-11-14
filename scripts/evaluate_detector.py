"""Evaluate a ball detector on a held-out set of frames, optionally against a second model.

Reports, at IoU >= 0.5: true/false positives at several confidence thresholds, true
positives on small balls (longest side <= 16 px), the maximum number of true positives
reachable under fixed false-positive budgets, and, when --compare is given, a paired
bootstrap over frames of the difference model A (--weights) minus model B (--compare).

Ground truth (pixel boxes) is read from --labels, either a directory of YOLO .txt files or a
JSON file; see box2boxcam.evaluation.ground_truth for the exact layouts.

Example::

    python scripts/evaluate_detector.py --weights new.pt --compare reference.pt \\
        --frames-dir heldout/images --labels heldout/labels --out-json report.json

Frames of a held-out match must never be used for training or model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from box2boxcam.evaluation.detect import load_detections, run_detector, save_detections
from box2boxcam.evaluation.ground_truth import load_ground_truth
from box2boxcam.evaluation.matching import IOU_THRESHOLD, SMALL_MAX_SIDE, match_all
from box2boxcam.evaluation.report import (
    DEFAULT_BUDGETS,
    DEFAULT_THRESHOLDS,
    bootstrap_table,
    evaluate,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", type=Path, required=True, help="Model A (.pt).")
    p.add_argument(
        "--compare", type=Path, help="Model B (.pt): adds the A - B bootstrap."
    )
    p.add_argument(
        "--frames-dir", type=Path, required=True, help="Directory of evaluation images."
    )
    p.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="YOLO label directory or JSON ground-truth file.",
    )
    p.add_argument("--imgsz", type=int, default=960, help="Inference size.")
    p.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Lowest confidence kept at inference; thresholds are applied afterwards.",
    )
    p.add_argument(
        "--class-filter",
        action="append",
        default=[],
        metavar="STEM=ID[,ID]",
        help="Keep only these class ids for the model whose file stem is STEM, e.g. "
        "yolo26s=32 for the COCO sports ball of a stock model. Repeatable.",
    )
    p.add_argument("--device", help="Torch device (default: chosen by Ultralytics).")
    p.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS)
    )
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="Matching IoU.")
    p.add_argument(
        "--small-px",
        type=float,
        default=SMALL_MAX_SIDE,
        help="A ball is small when its longest side is at most this many pixels.",
    )
    p.add_argument("--resamples", type=int, default=2000, help="Bootstrap resamples.")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache raw detections here (one JSON per weights file) to skip inference on re-runs.",
    )
    p.add_argument("--out-json", type=Path, help="Write the full report as JSON.")
    return p


def parse_class_filters(specs):
    """``["stem=32,33"]`` -> ``{"stem": [32, 33]}``."""
    out = {}
    for spec in specs:
        stem, _, ids = spec.partition("=")
        if not stem or not ids:
            raise SystemExit(f"--class-filter expects STEM=ID[,ID], got {spec!r}")
        out[stem] = [int(i) for i in ids.split(",")]
    return out


def detections_for(weights, frames, args):
    classes = parse_class_filters(args.class_filter).get(weights.stem)
    ids = [f.id for f in frames]
    cache = None
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        tag = "" if classes is None else "_c" + "-".join(map(str, classes))
        key = f"{weights.stem}_{args.imgsz}_{args.conf}{tag}.json"
        cache = args.cache_dir / key
        if cache.exists():
            return load_detections(cache, ids)
    dets = run_detector(
        weights,
        [f.path for f in frames],
        args.imgsz,
        args.conf,
        args.device,
        classes,
    )
    if cache:
        save_detections(cache, ids, dets)
    return dets


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare and args.weights.stem == args.compare.stem:
        raise SystemExit("--weights and --compare must have different file names")
    frames = load_ground_truth(args.frames_dir, args.labels)
    gts = [f.boxes for f in frames]
    print(
        f"{len(frames)} frames, {sum(len(g) for g in gts)} balls",
        file=sys.stderr,
    )
    matches = {}
    for weights in [args.weights] + ([args.compare] if args.compare else []):
        matches[weights.stem] = match_all(
            gts, detections_for(weights, frames, args), args.iou
        )
    report = evaluate(
        gts,
        matches,
        thresholds=tuple(args.thresholds),
        budgets=tuple(args.budgets),
        small_max_side=args.small_px,
        resamples=args.resamples,
        seed=args.seed,
    )
    print(report["table"])
    if "bootstrap" in report:
        b = report["bootstrap"]
        print(f"\nBootstrap over frames, A = {b['A']}, B = {b['B']}:")
        print(bootstrap_table(b["stats"]))
        print(
            "\nRecovery at the first threshold:", report["recovery_at_first_threshold"]
        )
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
