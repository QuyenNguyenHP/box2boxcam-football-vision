"""Freeze all user-reviewed close-ups except exclusions, without altering Label Studio."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from pitchlens.data.audit_yolo import audit
from pitchlens.data.build_mobile_dataset import sha256
from pitchlens.data.build_personal_test import BASE_URL, _save_json, select_review
from pitchlens.data.label_studio_review import (
    _label_studio_request,
    _yolo_line,
    download_local_export,
)


def select_closeup(task: dict, proposal: dict, *, confirm_reviewed: bool) -> dict:
    if not confirm_reviewed:
        raise ValueError("Explicit confirmation that ALL images were reviewed is required")
    if task.get("drafts"):
        raise ValueError("Saved drafts must be resolved; never discard unsent corrections")
    row = select_review(task, proposal, confirm_unchanged=True)
    row.update(
        split="excluded" if row["status"] == "excluded" else "unassigned",
        source_video_sha256=task["data"]["source_video_sha256"],
        confidence_threshold=None,
        intended_split="train",
    )
    return row


def build(root: Path, output: Path, *, confirm_reviewed: bool) -> dict:
    if not confirm_reviewed:
        raise ValueError("Confirm review of ALL images, not just submitted annotations")
    if output.exists():
        raise FileExistsError(f"Frozen export already exists: {output}")
    shared = root / "data/interim/gpt6_luna_label_studio_review"
    review = shared / "personal_closeups_20261011"
    proposals = json.loads((review / "tasks.json").read_text())
    expected = {p["data"]["item_id"]: p for p in proposals}
    if len(expected) != 456 or len(proposals) != len(expected):
        raise ValueError("Expected exactly 456 unique close-up proposals")
    project_id = json.loads((review / "review_project.json").read_text())["project_id"]
    token = next(s.partition(": ")[2] for s in (shared / "local_login.txt").read_text().splitlines()
                 if s.startswith("API token:"))
    project = _label_studio_request(f"{BASE_URL}/api/projects/{project_id}/", token)
    if project["title"] != "PitchLens — Gros plans — Luna max (11 octobre)":
        raise ValueError("Unexpected close-up review project")
    tasks = download_local_export(token=token, project_id=project_id)
    listed, page = [], 1
    while True:
        response = _label_studio_request(
            f"{BASE_URL}/api/tasks/?project={project_id}&page_size=200&page={page}", token)
        listed.extend(response["tasks"])
        if len(listed) >= response["total"]:
            break
        if not response["tasks"]:
            raise ValueError("Incomplete live inventory")
        page += 1
    live = {t["data"]["item_id"]: t for t in listed}
    exported = {t["data"]["item_id"]: t for t in tasks}
    if (len(live) != len(listed) or len(exported) != len(tasks)
            or set(live) != set(expected) or set(exported) != set(expected)):
        raise ValueError("Original/live/export inventories differ")
    rows = []
    for task in sorted(tasks, key=lambda t: (t["data"]["video_id"], t["data"]["source_frame"])):
        item_id = task["data"]["item_id"]
        if task["data"] != expected[item_id]["data"] or task["data"] != live[item_id]["data"]:
            raise ValueError(f"Task metadata changed: {item_id}")
        row = select_closeup({**task, "drafts": live[item_id].get("drafts", [])},
                             expected[item_id], confirm_reviewed=confirm_reviewed)
        if row["split"] == "unassigned":
            image = (root / row["source_image"]).resolve()
            image.relative_to(root / "data/interim")
            with Image.open(image) as pixels:
                if pixels.size != (row["width"], row["height"]):
                    raise ValueError(f"Image/annotation dimensions differ: {image}")
        rows.append(row)
    selected = [r for r in rows if r["split"] == "unassigned"]
    if not selected:
        raise ValueError("No retained close-ups")
    counts = Counter(r["status"] for r in rows)
    report = {
        "created_at": datetime.now(UTC).isoformat(), "dataset": str(output),
        "label_studio_project_id": project_id, "available_proposals": len(rows),
        "retained_images": len(selected), "positive_images": counts["positive"],
        "negative_images": counts["negative"], "excluded_images": counts["excluded"],
        "boxes": sum(len(r["boxes_pixels"]) for r in selected),
        "label_sources_retained": dict(Counter(r["label_source"] for r in selected)),
        "selection_policy": "All user-confirmed reviewed images except explicit exclusions; human corrections take priority",
        "user_confirmed_review_all": True, "confidence_threshold": None,
        "drafts_at_export": 0, "label_studio_modified": False,
        "existing_datasets_modified": False, "image_storage": "hardlinks",
        "split_policy": "All six clips belong to one capture session; training candidates only",
        "source_proposals_sha256": sha256(review / "tasks.json"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    (staging / "images").mkdir()
    (staging / "labels").mkdir()
    for row in selected:
        os.link(root / row["source_image"], staging / "images" / f"{row['item_id']}.jpg")
        (staging / "labels" / f"{row['item_id']}.txt").write_text("".join(
            _yolo_line(box, row["width"], row["height"]) for box in row["boxes_pixels"]))
    validation = audit(staging, target_size=960)
    if validation["issues"]:
        raise ValueError(f"Invalid labels; staging preserved at {staging}: {validation['issues']}")
    validation["root"] = str(output)
    for name, value in (("review_manifest.json", rows), ("review_snapshot.json", tasks),
                        ("live_task_snapshot.json", listed), ("luna_proposals.json", proposals),
                        ("build_report.json", report), ("audit.json", validation)):
        _save_json(staging / name, value)
    (staging / "README.md").write_text(
        "# Gros plans personnels — revue v001\n\n"
        f"{len(selected)} images retenues : {counts['positive']} positives, "
        f"{counts['negative']} négatives. {counts['excluded']} exclusions.\n\n"
        "Toutes les images ont été revues selon confirmation de l'utilisateur, sans seuil. "
        "Les annotations humaines soumises remplacent Luna ; les propositions inchangées "
        "restent acceptées sans modifier leur statut Submit. Aucun brouillon au moment "
        "de l'export. Les images exclues ne deviennent pas des négatifs.\n\n"
        "images/ utilise des liens physiques ; labels/ contient les boîtes YOLO (classe 0). "
        "Les snapshots et le manifeste préservent la provenance et les exclusions. "
        "Les six vidéos forment une seule session, destinée au train ; aucun découpage "
        "aléatoire de frames en validation. Export figé, sources conservées.\n")
    if output.exists():
        raise FileExistsError(f"Output appeared during build; staging preserved: {staging}")
    staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-all-reviewed", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "data/processed/luna_closeups_reviewed_v001").resolve()
    print(json.dumps(build(root, output, confirm_reviewed=args.confirm_all_reviewed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
