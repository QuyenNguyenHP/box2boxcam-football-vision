"""Annotate the extracted six-video set, resumably, without copying images.

Each video has its own batch/state; temporal neighbours never cross videos.
The merged output retains extracted image IDs, original timestamps and URLs.
No failures are automatically resubmitted and no annotations are human-approved.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from box2boxcam.data.luna_batch import (
    TERMINAL_BATCH_STATUSES,
    LunaBatchConfig,
    _atomic_json,
    build_plan,
    collect_batch,
    refresh_status,
    submit_batch,
)


def prepare(config_path: Path, root: Path):
    raw = json.loads(config_path.read_text())
    downloaded_path = root / raw["download_manifest"]
    downloaded = json.loads(downloaded_path.read_text())
    with (root / raw["frames_manifest"]).open() as stream:
        rows = list(csv.DictReader(stream))
    output_root = root / raw["output_root"]
    configs = []
    for video in downloaded["videos"]:
        video_path = downloaded_path.parent / video["file"]
        video_id = video_path.stem
        if hashlib.sha256(video_path.read_bytes()).hexdigest() != video["sha256"]:
            raise ValueError(f"Source changed: {video_path}")
        selected = [row for row in rows if row["video_id"] == video_id]
        if not selected or any(not (root / row["image_path"]).is_file() for row in selected):
            raise ValueError(f"Missing extracted images for {video_id}")
        config = LunaBatchConfig(
            project_root=root, config_path=config_path.resolve(),
            video_path=video_path, output_root=output_root / video_id,
            model=raw["model"], reasoning_effort=raw["reasoning_effort"],
            image_detail=raw["image_detail"], jpeg_quality=raw["jpeg_quality"],
            temporal_offset=raw["temporal_offset"],
            max_output_tokens=raw["max_output_tokens"], frame_stride=4,
            max_samples=None,
            center_frames=tuple(int(row["source_frame"]) for row in selected),
            retry_failures_from=None, reuse_uploads_from=None,
        )
        plan = build_plan(config)
        if any(int(row["width"]) != plan["video"]["width"] or
               int(row["height"]) != plan["video"]["height"] for row in selected):
            raise ValueError(f"Image dimensions differ: {video_id}")
        configs.append(config)
    if sum(len(config.center_frames) for config in configs) != len(rows):
        raise ValueError("Extracted images are not uniquely covered by the videos")
    _atomic_json(output_root / "run_config.json", raw)
    print(f"{len(rows)} cases, {len(configs)} videos, {raw['model']}, "
          f"reasoning={raw['reasoning_effort']}, detail={raw['image_detail']}, "
          f"max_output_tokens={raw['max_output_tokens']}", flush=True)
    return raw, rows, configs


def merge_results(rows, configs, output_root):
    predictions = {}
    for config in configs:
        path = config.output_root / "predictions.json"
        if path.exists():
            for prediction in json.loads(path.read_text())["predictions"]:
                predictions[(config.video_path.stem, prediction["source_frame"])] = prediction
    merged, manifest = [], []
    for source in rows:
        row = dict(source)
        prediction = predictions.get((row["video_id"], int(row["source_frame"])))
        if prediction:
            merged.append({**prediction, **source, "status": "luna_proposed",
                           "review_required": True})
            row["status"] = "luna_positive" if prediction["ball_in_play_visible"] else "luna_negative"
            row["source_label"] = str(
                (output_root / row["video_id"] / "labels" /
                 f"frame_{int(row['source_frame']):08d}.txt").relative_to(configs[0].project_root)
            )
        else:
            row["status"], row["source_label"] = "api_error", ""
        manifest.append(row)
    _atomic_json(output_root / "predictions.json", {"predictions": merged})
    with (output_root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "planned_cases": len(rows), "valid_predictions": len(merged),
        "positive_predictions": sum(p["ball_in_play_visible"] for p in merged),
        "negative_predictions": sum(not p["ball_in_play_visible"] for p in merged),
        "missing_or_failed_predictions": len(rows) - len(merged),
        "review_required": True,
        "images_copied": False, "splits": "unassigned",
    }
    _atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status", "collect"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw, rows, configs = prepare(args.config, root)
    output_root = root / raw["output_root"]
    if args.action == "plan":
        return
    if args.action == "run":
        run_lock = (output_root / "run.lock").open("a")
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This run is already active; use status instead") from error
        # Bound both memory and network concurrency on the 8 GB Mac.
        with ThreadPoolExecutor(max_workers=int(raw["upload_workers"])) as pool:
            list(pool.map(submit_batch, configs))
    pending = list(configs)
    errors = []
    while pending:
        for config in list(pending):
            state = refresh_status(config)
            if args.action == "status":
                pending.remove(config)
                continue
            if state["batch_status"] == "completed":
                if not (config.output_root / "summary.json").exists():
                    collect_batch(config)
                pending.remove(config)
            elif state["batch_status"] in TERMINAL_BATCH_STATUSES:
                errors.append({"video_id": config.video_path.stem,
                               "status": state["batch_status"]})
                pending.remove(config)
        if args.action != "run":
            break
        if pending:
            time.sleep(int(raw["poll_seconds"]))
    if args.action != "status":
        merge_results(rows, configs, output_root)
        _atomic_json(output_root / "batch_failures.json", {"failures": errors})
        if errors:
            raise RuntimeError(f"Failed batches (not resubmitted): {errors}")


if __name__ == "__main__":
    main()
