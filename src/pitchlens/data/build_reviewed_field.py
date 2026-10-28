"""Freeze submitted all-ball field annotations as train-only candidates.

Unsubmitted model proposals and pending drafts are never accepted as labels.
Earlier frozen datasets and live review tasks remain untouched.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from pitchlens.data.audit_yolo import audit
from pitchlens.data.build_mobile_dataset import sha256
from pitchlens.data.build_personal_test import select_review
from pitchlens.data.label_studio_review import _label_studio_request, _yolo_line, download_local_export

BASE = "http://127.0.0.1:8080"
SESSION = "personal_field_session_20261012"


def select_submitted(task: dict, proposal: dict) -> dict | None:
    if task.get("drafts") or not any(not a.get("was_cancelled") for a in task.get("annotations", [])):
        return None
    identity = ("item_id", "source_frame", "video_id", "source_video", "match_id", "width", "height", "image_path", "annotation_scope")
    if any(task["data"].get(key) != proposal["data"].get(key) for key in identity):
        raise ValueError("Source identity/dimensions changed during review")
    if task["data"]["annotation_scope"] != "all_visible" or task["data"]["match_id"] != SESSION:
        raise ValueError("Expected the private all-ball field session")
    if task["data"].get("intended_split") != "train":
        raise ValueError("Never use reserved evaluation images in this training source")
    row = select_review(task, proposal, confirm_unchanged=False)
    row.update(split="excluded" if row["status"] == "excluded" else "unassigned",
               annotation_scope="all_visible", intended_split="train", confidence_threshold=None)
    return row


def build(root: Path, source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Frozen export exists; use a new version")
    token = next(line.partition(": ")[2] for line in
                 (root / "data/interim/gpt6_luna_label_studio_review/local_login.txt").read_text().splitlines()
                 if line.startswith("API token:"))
    project_id = json.loads((source / "review/label_studio_project.json").read_text())["project_id"]
    project = _label_studio_request(f"{BASE}/api/projects/{project_id}/", token)
    if project["title"] != "PitchLens — Match iPhone — Tous les ballons":
        raise ValueError("Unexpected review project")
    proposals = json.loads((source / "review/tasks.json").read_text())
    expected = {task["data"]["item_id"]: task for task in proposals}
    if len(expected) != 1243 or len(expected) != len(proposals):
        raise ValueError("Unexpected planned frame inventory")
    snapshot = download_local_export(token=token, project_id=project_id)
    live, page = [], 1
    while True:
        response = _label_studio_request(f"{BASE}/api/tasks?project={project_id}&page_size=200&page={page}&fields=all", token)
        live.extend(response["tasks"])
        if len(live) >= response["total"]:
            break
        if not response["tasks"]:
            raise ValueError("Incomplete live review inventory")
        page += 1
    live_by_id = {task["data"]["item_id"]: task for task in live}
    if len(live_by_id) != len(live) or set(live_by_id) != set(expected):
        raise ValueError("Live inventory differs from planned frames")
    if len(snapshot) != len(expected) or {t["data"]["item_id"] for t in snapshot} != set(expected):
        raise ValueError("Export inventory differs from planned frames")
    sources = json.loads((source / "source_manifest.json").read_text())["videos"]
    video_hashes = {Path(v["file"]).stem: v["sha256"] for v in sources}
    frame_hashes = {video: json.loads((source / video / "frames_manifest.json").read_text())["frames"] for video in video_hashes}
    rows, omitted = [], []
    for task in sorted(snapshot, key=lambda t: (t["data"]["video_id"], t["data"]["source_frame"])):
        item = task["data"]["item_id"]
        pending_drafts = live_by_id[item].get("drafts", [])
        row = select_submitted({**task, "drafts": pending_drafts}, expected[item])
        if row is None:
            omitted.append({"item_id": item, "reason": "pending_draft" if pending_drafts else "not_submitted"})
            continue
        row["source_video_sha256"] = video_hashes[row["video_id"]]
        if row["status"] != "excluded":
            image = (root / row["source_image"]).resolve()
            image.relative_to(source.resolve())
            with Image.open(image) as pixels:
                if pixels.size != (row["width"], row["height"]):
                    raise ValueError("Image/label orientation differs")
            if sha256(image) != frame_hashes[row["video_id"]][str(row["source_frame"])]["sha256"]:
                raise ValueError("Reviewed JPEG changed since inference")
            row["image_sha256"] = sha256(image)
        rows.append(row)
    selected = [row for row in rows if row["split"] == "unassigned"]
    if not selected:
        raise ValueError("No submitted field annotations to train on")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    (staging / "images").mkdir()
    (staging / "labels").mkdir()
    for row in selected:
        os.link(root / row["source_image"], staging / "images" / f"{row['item_id']}.jpg")
        label = staging / "labels" / f"{row['item_id']}.txt"
        label.write_text("".join(_yolo_line(box, row["width"], row["height"]) for box in row["boxes_pixels"]))
        row["label_sha256"] = sha256(label)
    validation = audit(staging, target_size=960)
    if validation["issues_count"]:
        raise ValueError(f"Invalid export: {validation['issues_count']} issues; staging preserved")
    counts = Counter(row["status"] for row in rows)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset": str(output),
              "project_id": project_id, "retained_images": len(selected),
              "positive_images": counts["positive"], "negative_images": counts["negative"], "excluded_images": counts["excluded"],
              "boxes": sum(len(row["boxes_pixels"]) for row in selected),
              "multi_box_images": sum(len(row["boxes_pixels"]) > 1 for row in selected),
              "pending_or_unsubmitted": len(omitted), "annotation_scope": "all_visible",
              "selection_policy": "Submitted human annotations only; pending drafts omitted; no confidence threshold or model fallback",
              "match_group": SESSION, "split_policy": "Entire capture session is train-only, never random adjacent-frame validation",
              "existing_datasets_modified": False, "label_studio_modified": False}
    validation["root"] = str(output)
    for name, data in (("review_manifest.json", rows), ("build_report.json", report), ("audit.json", validation),
                       ("review_snapshot.json", snapshot), ("pending_review.json", omitted)):
        (staging / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (staging / "README.md").write_text(
        f"# Match iPhone — annotations humaines v001\n\n{len(selected)} images validées, {report['boxes']} boîtes. "
        "Tous les ballons visibles, y compris les ballons de réserve et immobiles. "
        "Les propositions non soumises et les brouillons restent hors entraînement. "
        "Une seule session, réservée au train ; pas de validation indépendante sur ce match.\n")
    if output.exists():
        raise FileExistsError("Output appeared during export; staging preserved")
    staging.rename(output)
    return report
