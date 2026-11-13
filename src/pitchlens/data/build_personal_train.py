"""Freeze user-reviewed personal train clips without changing review submissions.

September 30 long clip: confidence >= 0.86. October 7: >= 0.88.
Human submissions override proposals and are retained below the threshold too.
Untouched negatives above the confirmed threshold are retained as real negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from pitchlens.data.audit_yolo import audit
from pitchlens.data.build_personal_test import BASE_URL, _save_json, select_review
from pitchlens.data.label_studio_review import (
    _label_studio_request,
    _yolo_line,
    download_local_export,
)

THRESHOLDS = {
    "2026-09-30_lou_mistrau_19h38m04": 0.86,
    "2026-10-07_lou_mistrau_20h40m43": 0.88,
    "2026-10-07_lou_mistrau_20h47m20": 0.88,
}


def select_train(task: dict, proposal: dict, threshold: float) -> dict:
    """Keep reviewed high-confidence proposals plus all submitted human decisions."""
    data = task["data"]
    submitted = any(not a.get("was_cancelled") for a in task.get("annotations", []))
    if not submitted and float(data["luna_confidence"]) < threshold:
        return {
            "item_id": data["item_id"], "video_id": data["video_id"],
            "source_frame": data["source_frame"],
            "timestamp_seconds": data["timestamp_seconds"],
            "source_video": data["source_video"], "source_image": data["image_path"],
            "match_id": data["match_id"], "width": data["width"], "height": data["height"],
            "luna_confidence": data["luna_confidence"],
            "confidence_threshold": threshold,
            "status": "not_selected_below_threshold", "split": "not_selected",
            "boxes_pixels": [], "rotations_canonicalized": [],
            "label_source": "not_accepted", "annotation_id": None,
        }
    row = select_review(task, proposal, confirm_unchanged=True)
    row["split"] = "excluded" if row["status"] == "excluded" else "train"
    row["confidence_threshold"] = threshold
    return row


def build(root: Path, output: Path, *, confirm_reviewed: bool) -> dict:
    if not confirm_reviewed:
        raise ValueError("Explicit user confirmation of reviewed thresholds is required")
    if output.exists():
        raise FileExistsError(f"Frozen dataset already exists: {output}")
    shared = root / "data/interim/gpt6_luna_label_studio_review"
    proposal_paths = [
        shared / "lou_mistrau_20261008/tasks_train.json",
        shared / "gpt6_luna_lou_mistrau_long_train_20261009/tasks.json",
    ]
    proposals = [p for path in proposal_paths for p in json.loads(path.read_text())]
    expected = {p["data"]["item_id"]: p for p in proposals}
    if len(expected) != 1877 or len(expected) != len(proposals):
        raise ValueError("Expected 1,877 unique train proposals")
    token = next(s.partition(": ")[2] for s in (shared / "local_login.txt").read_text().splitlines()
                 if s.startswith("API token:"))
    tasks = download_local_export(token=token, project_id=3)
    listed = []
    page = 1
    while True:
        live = _label_studio_request(
            f"{BASE_URL}/api/tasks/?project=3&page_size=1000&page={page}", token,
        )
        listed.extend(live["tasks"])
        if len(listed) >= live["total"]:
            break
        if not live["tasks"]:
            raise ValueError("Incomplete live task inventory")
        page += 1
    if any(t.get("drafts") for t in listed):
        raise ValueError("New drafts detected: resolve their contents before freezing")
    ids = [t["data"]["item_id"] for t in tasks]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Train review inventory differs from original proposals")
    live_by_id = {t["data"]["item_id"]: t for t in listed}
    if len(live_by_id) != len(listed) or set(live_by_id) != set(expected):
        raise ValueError("Live/export task inventories differ")
    test_root = root / "data/processed/lou_mistrau_test_v001"
    test_rows = json.loads((test_root / "review_manifest.json").read_text())
    if set(ids) & {row["item_id"] for row in test_rows}:
        raise ValueError("Test frames found in train")
    predictions = {}
    source_paths = [
        root / "data/interim/gpt6_luna_lou_mistrau_20261008/predictions.json",
        root / "data/interim/gpt6_luna_lou_mistrau_long_train_20261009/predictions.json",
    ]
    for path in source_paths:
        for p in json.loads(path.read_text())["predictions"]:
            if p["item_id"] in predictions:
                raise ValueError("Duplicate source prediction")
            predictions[p["item_id"]] = p
    rows = []
    for task in sorted(tasks, key=lambda t: (t["data"]["video_id"], t["data"]["source_frame"])):
        data = task["data"]
        original = expected[data["item_id"]]
        if data["intended_split"] != "train" or data["video_id"] not in THRESHOLDS:
            raise ValueError("Wrong clip or split in train")
        if data != original["data"] or data != live_by_id[data["item_id"]]["data"]:
            raise ValueError(f"Task metadata changed: {data['item_id']}")
        row = select_train(task, original, THRESHOLDS[data["video_id"]])
        row["source_video_sha256"] = predictions[row["item_id"]]["source_video_sha256"]
        if row["split"] == "train":
            image = (root / row["source_image"]).resolve()
            image.relative_to(shared.resolve())
            with Image.open(image) as pixels:
                if pixels.size != (row["width"], row["height"]):
                    raise ValueError(f"Image/annotation dimensions differ: {image}")
        rows.append(row)
    counts = Counter(r["status"] for r in rows)
    selected = [r for r in rows if r["split"] == "train"]
    by_video = {}
    for video_id, threshold in THRESHOLDS.items():
        items = [r for r in rows if r["video_id"] == video_id]
        kept = [r for r in items if r["split"] == "train"]
        by_video[video_id] = {
            "available_proposals": len(items), "confidence_threshold": threshold,
            "retained_images": len(kept),
            "positive_images": sum(r["status"] == "positive" for r in kept),
            "negative_images": sum(r["status"] == "negative" for r in kept),
            "excluded_images": sum(r["status"] == "excluded" for r in items),
            "not_selected_below_threshold": sum(r["split"] == "not_selected" for r in items),
            "submitted_below_threshold_retained": sum(
                r["label_source"] == "submitted_human"
                and r["luna_confidence"] < r["confidence_threshold"] for r in kept
            ),
        }
    failure_path = root / "data/interim/gpt6_luna_lou_mistrau_long_train_20261009/2026-09-30_lou_mistrau_19h38m04/failures.json"
    failures = json.loads(failure_path.read_text())["failures"]
    report = {
        "created_at": datetime.now(UTC).isoformat(), "dataset": str(output),
        "label_studio_project_id": 3,
        "available_proposals": len(rows), "retained_images": len(selected),
        "positive_images": counts["positive"], "negative_images": counts["negative"],
        "excluded_images": counts["excluded"],
        "not_selected_below_threshold": counts["not_selected_below_threshold"],
        "missing_api_predictions": len(failures),
        "boxes": sum(len(r["boxes_pixels"]) for r in selected),
        "multi_box_images": sum(len(r["boxes_pixels"]) > 1 for r in selected),
        "label_sources_retained": dict(Counter(r["label_source"] for r in selected)),
        "by_video": by_video,
        "selection_policy": "Submitted human decisions override proposals at any confidence; otherwise keep all reviewed proposals at or above the confirmed per-video threshold, including negatives",
        "label_studio_modified": False,
        "image_storage": "hardlinks; no duplicated JPEG payload",
        "source_proposals_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in proposal_paths},
        "benchmark_policy": {
            "test_dataset": str(test_root), "exact_item_id_overlap_with_test": 0,
            "same_match_in_train_and_test": True,
            "independent_match_generalization_test": False,
            "validation_split_not_created": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    images = staging / "train/images"
    labels = staging / "train/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for row in selected:
        os.link(root / row["source_image"], images / f"{row['item_id']}.jpg")
        (labels / f"{row['item_id']}.txt").write_text("".join(
            _yolo_line(box, row["width"], row["height"]) for box in row["boxes_pixels"]
        ))
    validation = audit(staging, target_size=640)
    if validation["issues"]:
        raise ValueError(f"Invalid dataset; staging preserved at {staging}: {validation['issues']}")
    validation["root"] = str(output)
    (staging / "dataset.yaml").write_text(
        f"path: {json.dumps(str(output))}\ntrain: train/images\nval: null\ntest: null\nnames:\n  0: ball\n"
    )
    _save_json(staging / "review_manifest.json", rows)
    _save_json(staging / "review_snapshot.json", tasks)
    _save_json(staging / "luna_proposals.json", proposals)
    _save_json(staging / "missing_predictions.json", failures)
    _save_json(staging / "build_report.json", report)
    _save_json(staging / "audit.json", validation)
    (staging / "README.md").write_text(
        "# lou_mistrau train v001\n\n"
        f"{len(selected)} reviewed images: {counts['positive']} positives and {counts['negative']} negatives.\n\n"
        "Inclusive thresholds confirmed by the user: 0.86 for the long video of "
        "30 September; 0.88 for the two videos of 7 October. Submitted corrections "
        "take priority over Luna, even below the threshold. Exclusions are removed. "
        "Unchanged proposals above the threshold are accepted after confirmation "
        "of their review; they are not artificially marked Submit.\n\n"
        "The 10 expired Luna responses remain in missing_predictions.json: "
        "neither negative images nor human exclusions. Unsubmitted cases below the threshold "
        "remain in the review, not in this train set. No paid request was re-run.\n\n"
        "This folder only defines the train set. A distinct validation set must be chosen "
        "before configuring a full training run; do not use the test set as val. "
        "The short video of 30 September remains in lou_mistrau_test_v001. "
        "The same match is therefore present in train and test, but in distinct clips: "
        "this is not an evaluation on an independent match.\n\n"
        "## Files\n\n"
        "- train/images and train/labels: YOLO export, images linked without duplicating the JPEGs.\n"
        "- dataset.yaml: train only; val and test intentionally null.\n"
        "- review_manifest.json: selection and provenance for the 1,877 proposals.\n"
        "- review_snapshot.json and luna_proposals.json: frozen decisions and proposals.\n"
        "- missing_predictions.json: the 10 expired requests.\n"
        "- build_report.json and audit.json: counters and format checks.\n\n"
        "The files are frozen; rebuild into a new version to integrate "
        "later corrections. Do not modify the shared JPEGs in place.\n"
    )
    if output.exists():
        raise FileExistsError(f"Output appeared during build; staging preserved: {staging}")
    staging.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-reviewed-thresholds", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "data/processed/lou_mistrau_train_v001").resolve()
    print(json.dumps(build(root, output, confirm_reviewed=args.confirm_reviewed_thresholds),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
