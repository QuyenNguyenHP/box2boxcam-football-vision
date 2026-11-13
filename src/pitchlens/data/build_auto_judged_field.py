"""Freeze automatically judged ball candidates as a train-only source.

A vision model judged zoomed crops of detector/Luna candidates as "ball",
"not_ball" or "unsure". NO human reviewed these labels. "unsure" objects are
painted neutral grey so they are neither positives nor background; frames with
objects too large for the crop judge are dropped. Inputs are only read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from pitchlens.data.audit_yolo import audit
from pitchlens.data.build_mobile_dataset import sha256
from pitchlens.data.label_studio_review import _yolo_line
from pitchlens.data.reserved_sessions import assert_no_reserved_training

LABEL_SOURCE = "auto_judged_claude_sonnet"
MAX_SIDE = 64
MASK_SCALE = 1.5
MASK_MIN_SIDE = 24
MASK_COLOR = (114, 114, 114)
VERDICTS = {"ball", "not_ball", "unsure"}


def clamp_box(box, width, height):
    x0, y0, x1, y1 = box
    return [min(max(x0, 0), width), min(max(y0, 0), height),
            min(max(x1, 0), width), min(max(y1, 0), height)]


def mask_rectangle(box, width, height):
    """Integer rectangle enlarged by 50%, at least 24 px per side, inside the image."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = max((x1 - x0) * MASK_SCALE, MASK_MIN_SIDE) / 2
    half_h = max((y1 - y0) * MASK_SCALE, MASK_MIN_SIDE) / 2
    return clamp_box([math.floor(cx - half_w), math.floor(cy - half_h),
                      math.ceil(cx + half_w), math.ceil(cy + half_h)], width, height)


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def judge_frame(frame, candidates, verdicts):
    """Apply the labelling policy to one frame; never reads pixels."""
    width, height = frame["image_size"]
    judged = []
    for cell in frame["cells"]:
        candidate = candidates[str(cell)]
        if candidate["frame_path"] != frame["path"] or candidate["frame_sha256"] != frame["sha256"]:
            raise ValueError(f"Candidate {cell} does not belong to {frame['item_id']}")
        if verdicts.get(cell) not in VERDICTS:
            raise ValueError(f"Missing or invalid verdict for candidate {cell}")
        judged.append({"cell": cell, "verdict": verdicts[cell], "box_xyxy_pixels": candidate["box_xyxy_pixels"],
                       "sources": candidate["sources"], "detector_confidence": candidate["detector_confidence"],
                       "luna_confidence": candidate["luna_confidence"]})
    boxes = [clamp_box(c["box_xyxy_pixels"], width, height) for c in judged if c["verdict"] == "ball"]
    masks = [mask_rectangle(c["box_xyxy_pixels"], width, height) for c in judged if c["verdict"] == "unsure"]
    if any(b[2] <= b[0] or b[3] <= b[1] for b in boxes):
        raise ValueError(f"Empty ball box after clamping: {frame['item_id']}")
    reason = None
    if any(max(x1 - x0, y1 - y0) > MAX_SIDE for x0, y0, x1, y1 in (c["box_xyxy_pixels"] for c in judged)):
        reason = "candidate_side_over_64px"
    elif any(overlaps(box, mask) for box in boxes for mask in masks):
        reason = "ball_overlaps_unsure_mask"
    return {"status": "excluded" if reason else ("positive" if boxes else "negative"),
            "split": "excluded" if reason else "unassigned", "drop_reason": reason,
            "boxes_pixels": boxes, "masks_pixels": masks, "masked": bool(masks) and not reason,
            "candidates": judged}


def load_inputs(inputs: Path):
    frames = json.loads((inputs / "manifest/frames.json").read_text())
    candidates = json.loads((inputs / "manifest/candidates.json").read_text())
    verdicts = {}
    for path in sorted(inputs.glob("verdicts_part*.json")):
        for entry in json.loads(path.read_text()):
            if entry["cell"] in verdicts:
                raise ValueError(f"Duplicate verdict for candidate {entry['cell']}")
            verdicts[entry["cell"]] = entry["verdict"]
    cells = [cell for frame in frames for cell in frame["cells"]]
    if len(cells) != len(set(cells)) or set(cells) != {int(c) for c in candidates} or set(cells) != set(verdicts):
        raise ValueError("Frames, candidates and verdicts do not cover the same candidates")
    if len({frame["item_id"] for frame in frames}) != len(frames):
        raise ValueError("Duplicate frame")
    return frames, candidates, verdicts


def video_sources(root: Path) -> dict:
    """clip_id -> source video path and SHA-256, from the frozen extraction manifests."""
    field = json.loads((root / "data/interim/gpt6_luna_field_all_balls_20261014/source_manifest.json").read_text())
    videos = {Path(v["file"]).stem: {"source_video": v["source_video"], "source_video_sha256": v["sha256"]}
              for v in field["videos"]}
    for row in json.loads((root / "data/processed/lou_mistrau_train_v001/review_manifest.json").read_text()):
        videos[row["video_id"]] = {"source_video": row["source_video"],
                                   "source_video_sha256": row["source_video_sha256"]}
    return videos


def build(root: Path, inputs: Path, output: Path, videos: dict | None = None) -> dict:
    root = root.resolve()
    if output.exists():
        raise FileExistsError("Frozen export exists; use a new version")
    frames, candidates, verdicts = load_inputs(inputs)
    videos = video_sources(root) if videos is None else videos
    rows = []
    for frame in sorted(frames, key=lambda f: (f["clip_id"], f["frame_index"])):
        image = Path(frame["path"]).resolve()
        source_image = str(image.relative_to(root))
        width, height = frame["image_size"]
        row = {"item_id": frame["item_id"], "source_frame": frame["frame_index"],
               "timestamp_seconds": frame["timestamp_seconds"], "video_id": frame["clip_id"],
               "match_id": frame["match_id"], **videos[frame["clip_id"]], "width": width, "height": height,
               "source_image": source_image, "source_image_sha256": frame["sha256"],
               "label_source": LABEL_SOURCE, "human_reviewed": False,
               "annotation_scope": "all_visible", "intended_split": "train",
               "priority_class": frame["priority_class"], "luna_status": frame["luna_status"],
               "luna_frame_confidence": frame["luna_frame_confidence"]}
        rows.append(row | judge_frame(frame, candidates, verdicts))
    # Check every input frame, dropped or not, against reserved sessions as training.
    assert_no_reserved_training(root, [dict(row, split=row["intended_split"]) for row in rows])
    selected = [row for row in rows if row["split"] == "unassigned"]
    if not selected:
        raise ValueError("No frame left to train on")
    for row in selected:
        image = root / row["source_image"]
        if sha256(image) != row["source_image_sha256"]:
            raise ValueError(f"Source JPEG changed since judging: {image}")
        with Image.open(image) as pixels:
            if pixels.size != (row["width"], row["height"]):
                raise ValueError(f"Image size differs from manifest: {image}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    (staging / "images").mkdir()
    (staging / "labels").mkdir()
    for row in selected:
        target = staging / "images" / f"{row['item_id']}.jpg"
        if row["masked"]:
            with Image.open(root / row["source_image"]) as pixels:
                painted = pixels.convert("RGB")
            draw = ImageDraw.Draw(painted)
            for x0, y0, x1, y1 in row["masks_pixels"]:
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=MASK_COLOR)
            painted.save(target, format="JPEG", quality=95)
        else:
            os.link(root / row["source_image"], target)
        label = staging / "labels" / f"{row['item_id']}.txt"
        label.write_text("".join(_yolo_line(box, row["width"], row["height"]) for box in row["boxes_pixels"]))
        row["image_sha256"] = sha256(target)
        row["label_sha256"] = sha256(label)
    validation = audit(staging, target_size=960)
    if validation["issues_count"]:
        raise ValueError(f"Invalid export: {validation['issues_count']} issues; staging preserved")
    status = Counter(row["status"] for row in selected)
    kept_candidates = [c for row in selected for c in row["candidates"]]
    report = {
        "created_at": datetime.now(UTC).isoformat(), "dataset": str(output), "label_source": LABEL_SOURCE,
        "human_reviewed": False, "input_frames": len(rows), "retained_images": len(selected),
        "dropped_frames": dict(Counter(row["drop_reason"] for row in rows if row["drop_reason"])),
        "positive_images": status["positive"], "negative_images": status["negative"],
        "boxes": sum(len(row["boxes_pixels"]) for row in selected),
        "multi_box_images": sum(len(row["boxes_pixels"]) > 1 for row in selected),
        "masked_images": sum(row["masked"] for row in selected),
        "masks": sum(len(row["masks_pixels"]) for row in selected),
        "kept_candidate_verdicts": dict(Counter(c["verdict"] for c in kept_candidates)),
        "kept_not_ball_detector_conf_ge_0_25": sum(
            c["verdict"] == "not_ball" and (c["detector_confidence"] or 0) >= 0.25 for c in kept_candidates),
        "retained_by_match": dict(Counter(row["match_id"] for row in selected)),
        "policy": {"max_candidate_side_px": MAX_SIDE, "ball": "candidate box clamped to the image, class 0",
                   "not_ball": "no label (background)",
                   "unsure": f"grey {MASK_COLOR} rectangle, box enlarged x{MASK_SCALE}, min {MASK_MIN_SIDE} px side; JPEG quality 95 copy",
                   "ball_overlapping_mask": "frame dropped", "unmasked_images": "hardlinks to original JPEGs"},
        "judge_calibration": "Blind check vs human labels on detector candidates: 98.7% correct when confident; 1 wrong not_ball in 272, 3 wrong ball in 45",
        "known_limitation": "A ball proposed neither by the detector nor by Luna stays unlabelled (false negative label)",
        "split_policy": "Train only; every source session already trains elsewhere or is train-only",
        "input_files_sha256": {path.relative_to(inputs).as_posix(): sha256(path) for path in
                               [inputs / "manifest/frames.json", inputs / "manifest/candidates.json",
                                *sorted(inputs.glob("verdicts_part*.json"))]},
        "existing_datasets_modified": False,
    }
    validation["root"] = str(output)
    judge_inputs = staging / "judge_inputs"
    judge_inputs.mkdir()
    for name in report["input_files_sha256"]:
        shutil.copyfile(inputs / name, judge_inputs / Path(name).name)
    for name, data in (("review_manifest.json", rows), ("build_report.json", report), ("audit.json", validation)):
        (staging / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (staging / "README.md").write_text(
        f"# Auto-judged balls ({output.name}) — NO human review\n\n"
        f"**No human verified these labels** (no human reviewed these labels). "
        f"{len(selected)} train images, {status['positive']} positives, {status['negative']} negatives, "
        f"{report['boxes']} boxes, {report['masked_images']} masked images.\n\n"
        "Candidates from the detector and from Luna were judged on zoomed crops by a vision model "
        "(`ball` / `not_ball` / `unsure`). `ball` yields a YOLO box; `not_ball` stays background; "
        "`unsure` is covered by a grey rectangle (114, 114, 114), enlarged by 50% (at least 24 px), "
        "in a JPEG copy at quality 95. Images with a candidate larger than 64 px on a side, or in which a "
        "ball overlaps a mask, are dropped.\n\n"
        "Blind calibration against human labels, on detector candidates: 98.7% of "
        "answers correct when the judge is confident (1 wrong `not_ball` out of 272; 3 wrong `ball` out of 45). "
        "Known limitation: a ball proposed by neither the detector nor Luna stays unannotated.\n\n"
        "Train only. `review_manifest.json` keeps the full provenance (source path and SHA-256, "
        "match, clip, verdict, sources and confidences of each candidate, masking). The judge files are "
        "copied into `judge_inputs/`.\n")
    if output.exists():
        raise FileExistsError("Output appeared during export; staging preserved")
    staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="Directory with manifest/frames.json, manifest/candidates.json, verdicts_part*.json")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "data/processed/auto_field_judged_v001").resolve()
    print(json.dumps(build(root, args.inputs.resolve(), output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
