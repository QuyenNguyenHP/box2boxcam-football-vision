"""Freeze the reviewed short Lou Mistrau clip as a test-only YOLO dataset.

Submitted corrections win. Untouched Luna proposals require explicit confirmation
that the user reviewed them unchanged. No confidence filter applies to test.
The Label Studio project itself is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from pitchlens.data.audit_yolo import audit
from pitchlens.data.label_studio_review import (
    _box_from_result,
    _label_studio_request,
    _yolo_line,
    download_local_export,
)

VIDEO_ID = "2026-09-30_lou_mistrau_19h41m35"
BASE_URL = "http://127.0.0.1:8080"


def _axis_box(result: dict, width: int, height: int) -> list[int]:
    """Canonicalize 180° rectangles without changing their physical footprint.

    Label Studio stores the rotated anchor, not the axis-aligned top-left corner.
    Its installed SDK's convert_annotation_to_yolo_obb defines these coordinates.
    Other angles are rejected instead of silently enlarging the user's rectangle.
    """
    value = dict(result["value"])
    angle = float(value.get("rotation", 0)) % 360
    if math.isclose(angle, 180, abs_tol=1e-9):
        value["x"] = float(value["x"]) - float(value["width"])
        value["y"] = float(value["y"]) - float(value["height"])
    elif not math.isclose(angle, 0, abs_tol=1e-9):
        raise ValueError(f"Non-axis-aligned rotation requires review: {angle}")
    value["rotation"] = 0
    return _box_from_result({**result, "value": value}, width, height)


def select_review(task: dict, proposal: dict, *, confirm_unchanged: bool) -> dict:
    """Select human results, including empty negatives, before any model fallback."""
    data = task["data"]
    if data["item_id"] != proposal["data"]["item_id"]:
        raise ValueError("Task/proposal identity mismatch")
    annotations = [
        row for row in task.get("annotations", []) if not row.get("was_cancelled")
    ]
    annotation = None
    if annotations:
        annotation = max(
            annotations,
            key=lambda row: (str(row.get("updated_at") or ""), int(row.get("id") or 0)),
        )
        results = annotation.get("result") or []
        label_source = "submitted_human"
    else:
        if not confirm_unchanged:
            raise ValueError("Untouched proposals require explicit user confirmation")
        predictions = proposal.get("predictions", [])
        if len(predictions) != 1 or "result" not in predictions[0]:
            raise ValueError("Expected one complete original Luna proposal")
        results = predictions[0]["result"]
        label_source = "user_confirmed_unchanged_luna"
    excluded = any(
        result.get("from_name") == "decision"
        and "Exclure image" in result.get("value", {}).get("choices", [])
        for result in results
    )
    # Excluded frames never need usable boxes; preserve raw results in the snapshot.
    rectangles = [] if excluded else [
        result for result in results
        if result.get("from_name") == "bbox" and result.get("type") == "rectangle"
    ]
    boxes = [_axis_box(result, int(data["width"]), int(data["height"]))
             for result in rectangles]
    return {
        "item_id": data["item_id"],
        "source_frame": data["source_frame"],
        "timestamp_seconds": data["timestamp_seconds"],
        "video_id": data["video_id"],
        "source_video": data["source_video"],
        "match_id": data["match_id"],
        "width": data["width"],
        "height": data["height"],
        "status": "excluded" if excluded else "positive" if boxes else "negative",
        "split": "excluded" if excluded else "test",
        "boxes_pixels": boxes,
        "rotations_canonicalized": [
            r["value"]["rotation"] for r in rectangles
            if float(r["value"].get("rotation", 0)) != 0
        ],
        "label_source": label_source,
        "annotation_id": annotation.get("id") if annotation else None,
        "luna_confidence": data.get("luna_confidence"),
        "source_image": data["image_path"],
    }


def _save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build(root: Path, output: Path, *, confirm_unchanged: bool) -> dict:
    if not confirm_unchanged:
        raise ValueError("Confirm that all untouched proposals were reviewed unchanged")
    if output.exists():
        raise FileExistsError(f"Frozen dataset already exists: {output}")
    shared = root / "data/interim/gpt6_luna_label_studio_review"
    proposal_path = shared / "lou_mistrau_20261008/tasks_test.json"
    proposals = json.loads(proposal_path.read_text())
    expected = {p["data"]["item_id"]: p for p in proposals}
    if len(expected) != 380 or len(proposals) != 380:
        raise ValueError("Expected exactly 380 unique original test proposals")
    token = next(
        s.partition(": ")[2]
        for s in (shared / "local_login.txt").read_text().splitlines()
        if s.startswith("API token:")
    )
    tasks = download_local_export(token=token, project_id=4)
    live = _label_studio_request(
        BASE_URL + "/api/tasks/?project=4&page_size=1000&page=1", token,
    )
    listed = live["tasks"]
    if live["total"] != 380 or len(listed) != 380:
        raise ValueError("Incomplete or changed live test inventory")
    if any(task.get("drafts") for task in listed):
        raise ValueError("New drafts detected: resolve their contents before freezing")
    live_by_id = {t["data"]["item_id"]: t for t in listed}
    task_ids = [t["data"]["item_id"] for t in tasks]
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(expected):
        raise ValueError("Export does not contain exactly the expected 380 frames")
    if set(live_by_id) != set(expected):
        raise ValueError("Live tasks differ from the original test inventory")
    rows = []
    for task in sorted(tasks, key=lambda t: int(t["data"]["source_frame"])):
        data = task["data"]
        original = expected[data["item_id"]]
        if data["video_id"] != VIDEO_ID or data["intended_split"] != "test":
            raise ValueError("Wrong source clip or split in test")
        # Metadata and original proposals must not drift between review and export.
        if data != original["data"] or data != live_by_id[data["item_id"]]["data"]:
            raise ValueError(f"Task metadata changed: {data['item_id']}")
        row = select_review(task, original, confirm_unchanged=confirm_unchanged)
        if row["status"] != "excluded":
            image = (root / row["source_image"]).resolve()
            image.relative_to(shared.resolve())
            with Image.open(image) as pixels:
                if pixels.size != (row["width"], row["height"]):
                    raise ValueError(f"Image/annotation dimensions differ: {image}")
        rows.append(row)
    source_predictions = root / "data/interim/gpt6_luna_lou_mistrau_20261008/predictions.json"
    records = json.loads(source_predictions.read_text())["predictions"]
    video_hashes = {p["source_video_sha256"] for p in records if p["video_id"] == VIDEO_ID}
    if len(video_hashes) != 1:
        raise ValueError("Ambiguous source-video provenance")
    counts = Counter(row["status"] for row in rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(output),
        "label_studio_project_id": 4,
        "source_video": rows[0]["source_video"],
        "source_video_sha256": next(iter(video_hashes)),
        "original_proposals_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        "sampled_frames": len(rows),
        "frame_stride": 4,
        "retained_images": len(rows) - counts["excluded"],
        "positive_images": counts["positive"],
        "negative_images": counts["negative"],
        "excluded_images": counts["excluded"],
        "boxes": sum(len(row["boxes_pixels"]) for row in rows),
        "multi_box_images": sum(len(row["boxes_pixels"]) > 1 for row in rows),
        "axis_rotations_canonicalized": sum(len(r["rotations_canonicalized"]) for r in rows),
        "label_sources_all_frames": dict(Counter(r["label_source"] for r in rows)),
        "label_sources_retained": dict(Counter(
            r["label_source"] for r in rows if r["status"] != "excluded"
        )),
        "selection_policy": "all user-reviewed frames except explicit exclusions",
        "user_confirmed_unchanged_proposals": True,
        "confidence_threshold": None,
        "image_storage": "hardlinks; no duplicated JPEG payload",
        "label_studio_modified": False,
        "benchmark_policy": {
            "split_unit": "video_clip",
            "same_match_in_train_and_test": True,
            "independent_match_generalization_test": False,
            "excluded_frames_are_not_scored": True,
            "test_is_not_validation_or_training": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    images = staging / "test/images"
    labels = staging / "test/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for row in rows:
        if row["status"] == "excluded":
            continue
        os.link(root / row["source_image"], images / f"{row['item_id']}.jpg")
        content = "".join(
            _yolo_line(box, row["width"], row["height"])
            for box in row["boxes_pixels"]
        )
        (labels / f"{row['item_id']}.txt").write_text(content)
    validation = audit(staging, target_size=640)
    if validation["issues"]:
        raise ValueError(f"Invalid dataset; preserved staging at {staging}: {validation['issues']}")
    # Null train/val deliberately prevent accidental use of test as validation.
    (staging / "dataset.yaml").write_text(
        f"path: {json.dumps(str(output))}\ntrain: null\nval: null\n"
        "test: test/images\nnames:\n  0: ball\n"
    )
    _save_json(staging / "review_snapshot.json", tasks)
    _save_json(staging / "luna_proposals.json", proposals)
    _save_json(staging / "review_manifest.json", rows)
    _save_json(staging / "build_report.json", report)
    validation["root"] = str(output)
    _save_json(staging / "audit.json", validation)
    (staging / "README.md").write_text(
        "# Lou Mistrau — test v001\n\n"
        f"Vidéo courte du 30 septembre 2026 : {len(rows)} images, une frame sur quatre.\n"
        f"Après revue : **{report['retained_images']} images retenues**, dont "
        f"{counts['positive']} avec ballon et {counts['negative']} sans ballon. "
        f"{counts['excluded']} exclusions ne participent pas aux métriques.\n\n"
        "Toutes les images restantes sont conservées, sans seuil de confiance. "
        "Les corrections soumises ont priorité. Les propositions non soumises sont "
        "acceptées inchangées sur confirmation explicite de l'utilisateur après revue. "
        "Le statut Submit dans Label Studio n'a pas été modifié.\n\n"
        "La longue vidéo du même match est destinée au train : ce benchmark teste "
        "un autre extrait, pas la généralisation à un match indépendant. "
        "Les frames voisines sont corrélées : ce ne sont pas des observations indépendantes. "
        "Ne pas utiliser ce test pour entraîner ou choisir les hyperparamètres.\n\n"
        "## Évaluer\n\n"
        "Appeler `model.val(data=str(dataset_yaml), split=\"test\", ...)`. "
        "Les entrées train et val sont volontairement nulles. "
        "La vidéo entière peut être visualisée avec les prédictions, mais seuls "
        "les frames retenues et annotées participent au score.\n\n"
        "## Contenu\n\n"
        "- `test/images/` : liens physiques vers les JPEG revus, sans recopier leur contenu.\n"
        "- `test/labels/` : une annotation YOLO par image ; fichier vide = sans ballon.\n"
        "- `dataset.yaml` : accès test uniquement pour Ultralytics.\n"
        "- `review_manifest.json` : décisions et provenance par image, exclusions comprises.\n"
        "- `review_snapshot.json` : export des décisions soumises au moment de la construction.\n"
        "- `luna_proposals.json` : propositions initiales, notamment celles acceptées inchangées.\n"
        "- `build_report.json` : compteurs, politique de sélection et empreintes des sources.\n"
        "- `audit.json` : validation des fichiers et dimensions des boîtes.\n\n"
        "Export figé : reconstruire dans une nouvelle version, ne pas modifier ce dossier "
        "ni les JPEG partagés en place. Les sources et le projet de revue sont conservés.\n"
    )
    if output.exists():
        raise FileExistsError(f"Output appeared during build; staging preserved: {staging}")
    staging.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-reviewed-unchanged", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "data/processed/lou_mistrau_test_v001").resolve()
    print(json.dumps(build(root, output, confirm_unchanged=args.confirm_reviewed_unchanged),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
