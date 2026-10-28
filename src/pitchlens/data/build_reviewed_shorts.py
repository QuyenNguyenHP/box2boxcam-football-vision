"""Freeze reviewed Luna shorts without changing Label Studio or existing splits.

Untouched proposals require explicit confirmation of review above the threshold.
Submitted human decisions take precedence at every score. Drafts must be resolved
before freezing: they are never silently replaced with a model prediction.
This is an unsplit candidate export, not an automatically merged training set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from pitchlens.data.audit_yolo import audit
from pitchlens.data.build_personal_test import BASE_URL, _save_json, select_review
from pitchlens.data.label_studio_review import (
    _label_studio_request,
    _yolo_line,
    download_local_export,
)


def select_short(task: dict, proposal: dict, *, threshold: float) -> dict:
    """Preserve human corrections/exclusions; accept confirmed proposals strictly >."""
    if task.get("drafts"):
        raise ValueError("Resolve saved drafts before freezing their final contents")
    submitted = any(not a.get("was_cancelled") for a in task.get("annotations", []))
    data = task["data"]
    if not submitted and float(data["luna_confidence"]) <= threshold:
        return {
            **data,
            "status": "not_selected_at_or_below_threshold",
            "split": "not_selected",
            "boxes_pixels": [],
            "label_source": "not_accepted",
            "annotation_id": None,
            "confidence_threshold": threshold,
            "threshold_comparison": ">",
        }
    # Shorts have no verified match ID. Do not invent one or split their frames.
    selected_task = copy.deepcopy(task)
    selected_task["data"].setdefault("match_id", "")
    row = select_review(selected_task, proposal, confirm_unchanged=True)
    row.update({
        "split": "excluded" if row["status"] == "excluded" else "unassigned",
        "confidence_threshold": threshold,
        "threshold_comparison": ">",
        "source_url": data["source_url"],
        "uploader": data["uploader"],
        "scene_id": "",
    })
    return row


def build(root: Path, output: Path, *, threshold: float, confirm_reviewed: bool) -> dict:
    if not confirm_reviewed:
        raise ValueError("Explicit confirmation of review above the threshold is required")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if output.exists():
        raise FileExistsError(f"Frozen output already exists: {output}")
    shared = root / "data/interim/gpt6_luna_label_studio_review"
    proposal_path = shared / "new_videos_20260928/tasks.json"
    prediction_path = root / "data/interim/gpt6_luna_new_videos_20260928/predictions.json"
    proposals = json.loads(proposal_path.read_text())
    expected = {p["data"]["item_id"]: p for p in proposals}
    if len(expected) != 782 or len(proposals) != len(expected):
        raise ValueError("Expected all 782 unique original proposals")
    token = next(
        line.partition(": ")[2]
        for line in (shared / "local_login.txt").read_text().splitlines()
        if line.startswith("API token:")
    )
    project = _label_studio_request(BASE_URL + "/api/projects/2/", token)
    if project["title"] != "PitchLens — Luna max — 782 nouvelles images":
        raise ValueError("Unexpected Label Studio project")
    tasks = download_local_export(token=token, project_id=2)
    listed = []
    page = 1
    while True:
        response = _label_studio_request(
            f"{BASE_URL}/api/tasks/?project=2&page_size=200&page={page}", token,
        )
        listed.extend(response["tasks"])
        if len(listed) >= response["total"]:
            break
        if not response["tasks"]:
            raise ValueError("Incomplete live review inventory")
        page += 1
    if any(task.get("drafts") for task in listed):
        raise ValueError("Saved drafts detected: preserve/resolve their contents before export")
    live = {t["data"]["item_id"]: t for t in listed}
    ids = [t["data"]["item_id"] for t in tasks]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Review export differs from original proposals")
    if len(live) != len(listed) or set(live) != set(expected):
        raise ValueError("Live/export task inventories differ")
    predictions = json.loads(prediction_path.read_text())["predictions"]
    prediction_index = {p["item_id"]: p for p in predictions}
    if len(prediction_index) != len(predictions) or set(prediction_index) != set(expected):
        raise ValueError("Source predictions differ from review inventory")
    rows = []
    for task in sorted(tasks, key=lambda t: (t["data"]["video_id"], t["data"]["source_frame"])):
        data = task["data"]
        proposal = expected[data["item_id"]]
        if data != proposal["data"] or data != live[data["item_id"]]["data"]:
            raise ValueError(f"Review metadata changed: {data['item_id']}")
        row = select_short(task, proposal, threshold=threshold)
        row["source_video_sha256"] = prediction_index[data["item_id"]]["source_video_sha256"]
        if row["split"] == "unassigned":
            image_path = (root / row["source_image"]).resolve()
            image_path.relative_to(root / "data/interim")
            with Image.open(image_path) as image:
                if image.size != (row["width"], row["height"]):
                    raise ValueError(f"Image/annotation dimensions differ: {image_path}")
        rows.append(row)
    selected = [row for row in rows if row["split"] == "unassigned"]
    if not selected:
        raise ValueError("No reviewed images retained")
    counts = Counter(row["status"] for row in rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(output),
        "label_studio_project_id": 2,
        "available_proposals": len(rows),
        "confidence_threshold": threshold,
        "threshold_comparison": ">",
        "user_confirmed_review_above_threshold": True,
        "retained_images": len(selected),
        "positive_images": counts["positive"],
        "negative_images": counts["negative"],
        "excluded_images": counts["excluded"],
        "not_selected_at_or_below_threshold": counts["not_selected_at_or_below_threshold"],
        "boxes": sum(len(row["boxes_pixels"]) for row in selected),
        "multi_box_images": sum(len(row["boxes_pixels"]) > 1 for row in selected),
        "submitted_at_or_below_threshold_retained": sum(
            row["label_source"] == "submitted_human" and row["luna_confidence"] <= threshold
            for row in selected
        ),
        "label_sources_retained": dict(Counter(row["label_source"] for row in selected)),
        "by_video": {
            video: dict(Counter(row["status"] for row in selected if row["video_id"] == video))
            for video in sorted({row["video_id"] for row in rows})
        },
        "selection_policy": "Submitted human decisions override proposals at any confidence; otherwise keep user-confirmed proposals strictly above threshold, including negatives",
        "label_studio_modified": False,
        "existing_datasets_modified": False,
        "training_started": False,
        "paid_api_calls": False,
        "image_storage": "hardlinks; no duplicated JPEG payload",
        "split_policy": "Unassigned candidates for future training; verify match/scene groups and reserved evaluation overlap before merging",
        "redistribution_rights": "not cleared; private experiment only",
        "source_proposals_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        "source_predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    (staging / "images").mkdir()
    (staging / "labels").mkdir()
    hashes = Counter()
    for row in selected:
        image_path = root / row["source_image"]
        os.link(image_path, staging / "images" / f"{row['item_id']}.jpg")
        hashes[hashlib.sha256(image_path.read_bytes()).hexdigest()] += 1
        (staging / "labels" / f"{row['item_id']}.txt").write_text("".join(
            _yolo_line(box, row["width"], row["height"]) for box in row["boxes_pixels"]
        ))
    report["exact_duplicate_images"] = sum(count - 1 for count in hashes.values())
    validation = audit(staging, target_size=960)
    if validation["issues"]:
        raise ValueError(f"Invalid labels; staging preserved at {staging}: {validation['issues']}")
    validation["root"] = str(output)
    _save_json(staging / "review_manifest.json", rows)
    _save_json(staging / "review_snapshot.json", tasks)
    _save_json(staging / "live_task_snapshot.json", listed)
    _save_json(staging / "luna_proposals.json", proposals)
    _save_json(staging / "build_report.json", report)
    _save_json(staging / "audit.json", validation)
    (staging / "README.md").write_text(
        "# Shorts Luna — revue v001\n\n"
        f"{len(selected)} images retenues, dont {counts['positive']} positives et "
        f"{counts['negative']} négatives. {counts['excluded']} exclusions.\n\n"
        f"L'utilisateur confirme avoir revu toutes les propositions de confiance > {threshold}. "
        "Les propositions inchangées sont acceptées sans falsifier le statut Submit. "
        "Les corrections et exclusions soumises ont priorité, quel que soit leur score. "
        "Aucun brouillon présent au moment de cet export. Les annotations manquantes "
        "ne deviennent jamais des négatifs par défaut.\n\n"
        "images/ partage les JPEG existants par liens physiques. labels/ contient les "
        "annotations YOLO retenues (classe 0 = ballon, fichier vide = négatif revu). "
        "review_manifest.json garde aussi les exclusions et images non retenues. "
        "Les snapshots et build_report.json documentent décisions et provenance.\n\n"
        "Les splits restent non assignés : contrôler les matchs/scènes et les recouvrements "
        "avec validation/test avant de fusionner ces candidates au prochain train. "
        "Aucun dataset existant modifié, entraînement lancé ou appel Luna payant effectué. "
        "Les droits de redistribution ne sont pas clarifiés.\n\n"
        "Export figé : créer une nouvelle version pour les corrections ultérieures. "
        "Ne pas modifier les JPEG partagés en place.\n"
    )
    if output.exists():
        raise FileExistsError(f"Output appeared during build; staging preserved at {staging}")
    staging.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--confirm-reviewed-above-threshold", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "data/processed/luna_shorts_reviewed_v001").resolve()
    print(json.dumps(build(
        root, output, threshold=args.threshold,
        confirm_reviewed=args.confirm_reviewed_above_threshold,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
