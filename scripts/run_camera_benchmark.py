"""Closed-loop camera-guidance benchmark on SoccerTrack v2 (see docs/camera_guidance_benchmark.md).

Each variant drives its own virtual 16:9 camera over the panorama and only sees what the
detectors find inside its current crop. Metrics are computed against the official
ground-truth boxes of the 22 players (the ball is only used at annotated event times).

Example (one segment, quick check)::

    python scripts/run_camera_benchmark.py --soccertrack-root /data/soccertrack-v2 \\
        --set REF --variants current guard --out runs/ref.json \\
        --people-weights yolo26s.pt --pose-weights yolo26s-pose.pt --ball-weights ball.pt

Splits are fixed in configs/camera_benchmark_segments.json. TUNE is for development, TEST may
be looked at a few times, and FINAL has already been consumed by the published numbers:
re-running it is a reproduction, not a held-out test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTS = REPO / "configs" / "camera_benchmark_segments.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--soccertrack-root",
        type=Path,
        help="Local snapshot of the SoccerTrack v2 dataset (release v1.2): needs "
        "mot/<match>.txt, bas/<match>/<match>_12_class_events.json, "
        "ball/<match>_1st_ball.npz, raw/<match>/<match>_keypoints.json and the video.",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=DEFAULT_SEGMENTS,
        help="Segment split file (default: %(default)s).",
    )
    parser.add_argument(
        "--set",
        dest="which",
        default="TUNE",
        choices=["TUNE", "TEST", "FINAL", "REF", "ALL"],
        help="Split to run. REF is a single 30 s sanity segment.",
    )
    parser.add_argument(
        "--segment-id",
        nargs="+",
        help="Only run these segment ids of the chosen split (e.g. 117092_a).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["current", "guard"],
        help="Variant names (see --list-variants). The first one is the baseline "
        "for paired differences. Default: %(default)s.",
    )
    parser.add_argument(
        "--list-variants", action="store_true", help="Print variant names and exit."
    )
    parser.add_argument("--out", type=Path, help="Result JSON path.")
    parser.add_argument(
        "--video-template",
        default="{root}/mot/clips/{match}.mp4",
        help="Panorama video of a match; {root} and {match} are substituted. "
        "Default: %(default)s (the official 240 s MOT clips).",
    )
    parser.add_argument("--people-weights", type=Path, help="COCO person detector.")
    parser.add_argument("--pose-weights", type=Path, help="Pose model (17 keypoints).")
    parser.add_argument(
        "--ball-weights", type=Path, help="Ball detector (class 0 = ball)."
    )
    parser.add_argument(
        "--device", help="Torch device (default: mps when available, else cpu)."
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Instead of scoring, run the body-orientation audit on the first "
        "variant against GT player motion and print a summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from box2boxcam.benchmark.variants import VARIANTS

    if args.list_variants:
        print("\n".join(VARIANTS))
        return 0
    missing = [
        flag
        for flag in (
            "soccertrack_root",
            "out",
            "people_weights",
            "pose_weights",
            "ball_weights",
        )
        if getattr(args, flag) is None and not (flag == "out" and args.audit)
    ]
    if missing:
        build_parser().error(
            "missing required arguments: "
            + ", ".join("--" + m.replace("_", "-") for m in missing)
        )
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        build_parser().error(f"unknown variants {unknown}; see --list-variants")

    import cv2

    from box2boxcam.benchmark import audit, closed_loop
    from box2boxcam.benchmark.aggregate import segments
    from box2boxcam.benchmark.gt import GT

    if args.which == "FINAL":
        print(
            "Note: the FINAL split was consumed by the published results; "
            "this is a reproduction, not a held-out test.",
            file=sys.stderr,
        )
    import torch

    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    device = args.device or closed_loop.default_device()
    models = closed_loop.load_models(
        args.people_weights, args.pose_weights, args.ball_weights
    )
    if args.audit:
        bodies, heads = [], []
        chosen = segments(args.segments, args.which)
        if args.segment_id:
            chosen = [s for s in chosen if s["id"] in args.segment_id]
        for seg in chosen:
            _, rows = closed_loop.simulate_segment(
                args.soccertrack_root,
                args.video_template,
                seg,
                args.variants[:1],
                models,
                device,
            )
            gt = GT(seg["match"], args.soccertrack_root)
            audit.audit_segment(gt, seg, rows[args.variants[0]], bodies, heads)
            print(seg["id"], len(bodies), len(heads), flush=True)
        audit.summarize(bodies, heads)
        if args.out:
            args.out.write_text(json.dumps({"bodies": bodies, "heads": heads}))
        return 0
    closed_loop.run_benchmark(
        args.soccertrack_root,
        args.segments,
        args.which,
        args.variants,
        args.out,
        models,
        device,
        video_template=args.video_template,
        only_ids=args.segment_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
