"""Import all personal-video proposals into separate train/test review projects.

Filtering only changes the review view, never labels or human decisions.
Existing review tasks, drafts, annotations, and edited views are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from pitchlens.data.label_studio_review import (
    _label_studio_request,
    build_tasks,
    download_local_export,
)
from pitchlens.data.luna_batch import _atomic_json

BASE = "http://127.0.0.1:8080"
TITLES = {
    "train": "PitchLens — Lou Mistrau — TRAIN",
    "test": "PitchLens — Lou Mistrau — TEST (30 septembre)",
}


def prepare(root: Path):
    import cv2

    source = root / "data/interim/gpt6_luna_lou_mistrau_20261008"
    shared = root / "data/interim/gpt6_luna_label_studio_review"
    output = shared / "lou_mistrau_20261008"
    frames = output / "frames"
    prediction_path = source / "predictions.json"
    predictions = json.loads(prediction_path.read_text())["predictions"]
    index = {p["item_id"]: p for p in predictions}
    if len(index) != len(predictions) or len(predictions) != 1403:
        raise ValueError("Expected all 1403 unique predictions")
    signature = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    signature_path = output / "input_signature.json"
    if (
        signature_path.exists()
        and json.loads(signature_path.read_text())["sha256"] != signature
    ):
        raise FileExistsError("Predictions changed; use another review directory")
    frames.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["video_id"]].append(row)
    tasks = {"train": [], "test": []}
    extracted = 0
    for video_id, rows in grouped.items():
        plan = json.loads((source / video_id / "plan.json").read_text())
        missing = {}
        for row in rows:
            destination = frames / f"{row['item_id']}.jpg"
            if destination.exists():
                continue
            if row["image_path"]:
                original = root / row["image_path"]
                if not original.is_file():
                    raise FileNotFoundError(original)
                os.link(original, destination)
            else:
                missing[int(row["source_frame"])] = row
        if missing:
            video = root / rows[0]["source_video"]
            digest = hashlib.sha256()
            with video.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != rows[0]["source_video_sha256"]:
                raise ValueError(f"Source changed: {video}")
            capture = cv2.VideoCapture(
                str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
            )
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {video}")
            try:
                for frame in range(max(missing) + 1):
                    ok, pixels = capture.read()
                    if not ok:
                        raise RuntimeError(f"Decode failed: {video}, frame {frame}")
                    if frame not in missing:
                        continue
                    row = missing[frame]
                    if pixels.shape[:2] != (row["height"], row["width"]):
                        raise ValueError(
                            "Coordinate dimensions differ from decoded pixels"
                        )
                    if shutil.disk_usage(output).free < 1024**3:
                        raise RuntimeError(
                            "Less than 1 GiB free: stop extraction safely"
                        )
                    destination = frames / f"{row['item_id']}.jpg"
                    if not cv2.imwrite(
                        str(destination), pixels, [cv2.IMWRITE_JPEG_QUALITY, 95]
                    ):
                        raise RuntimeError(f"Cannot save {destination}")
                    extracted += 1
            finally:
                capture.release()
        for row in rows:
            task_plan = {
                "model": "gpt-6-luna",
                "video": plan["video"],
                "cases": [
                    {"item_id": row["item_id"], "source_frame": row["source_frame"]}
                ],
            }
            task = build_tasks(task_plan, index)[0]
            role = row["intended_split"]
            if role not in tasks:
                raise ValueError(f"Unknown intended split: {role}")
            task["data"].update(
                {
                    "image": f"/data/local-files/?d=lou_mistrau_20261008/frames/{row['item_id']}.jpg",
                    "video_id": video_id,
                    "source_video": row["source_video"],
                    "image_path": str(
                        (frames / f"{row['item_id']}.jpg").relative_to(root)
                    ),
                    "width": row["width"],
                    "height": row["height"],
                    "timestamp_seconds": row["timestamp_seconds"],
                    "match_id": row["match_id"],
                    "intended_split": role,
                    "ball_visible": row["ball_in_play_visible"],
                    "frame_info": f"{role.upper()} · {video_id} · frame {row['source_frame']} · {row['timestamp_seconds']:.2f} s",
                }
            )
            tasks[role].append(task)
    for role, items in tasks.items():
        path = output / f"tasks_{role}.json"
        if path.exists() and json.loads(path.read_text()) != items:
            raise FileExistsError(f"Proposals changed: {path}")
        if not path.exists():
            _atomic_json(path, items)
    _atomic_json(signature_path, {"sha256": signature})
    print(
        f"Prepared train={len(tasks['train'])}, test={len(tasks['test'])}; {extracted} new images, other images linked",
        flush=True,
    )
    return output, tasks


def view_specs(role):
    def score_filter(operator):
        return {
            "filter": "filter:tasks:predictions_score",
            "operator": operator,
            "type": "Number",
            "value": 0.94,
        }

    if role == "test":
        return [("Test: fix everything (no threshold)", [])]
    return [
        ("Train: all images", []),
        (
            "Train: adjustable threshold",
            [
                score_filter("greater_or_equal"),
                {
                    "filter": "filter:tasks:data.luna_status",
                    "operator": "equal",
                    "type": "String",
                    "value": "boîte proposée",
                },
            ],
        ),
        ("Train: hard cases (< 0.94)", [score_filter("less")]),
    ]


def import_projects(root, output, tasks):
    shared = output.parent
    lines = (shared / "local_login.txt").read_text().splitlines()
    token = next(
        line.partition(": ")[2] for line in lines if line.startswith("API token:")
    )

    def request(path, method="GET", payload=None):
        return _label_studio_request(BASE + path, token, method=method, payload=payload)

    inventory = request("/api/projects/?page_size=100")
    if inventory.get("next"):
        raise ValueError("Project inventory paginated; stop to avoid duplicates")
    old_projects = inventory["results"]
    label_config = (root / "configs/label_studio_v003.xml").read_text()
    reports = {}
    for role, title in TITLES.items():
        aliases = {title}
        if role == "train":
            aliases.add("PitchLens — Lou Mistrau — TRAIN (7 octobre)")
        matches = [p for p in old_projects if p["title"] in aliases]
        if len(matches) > 1:
            raise ValueError("Ambiguous existing project title")
        project = (
            matches[0]
            if matches
            else request(
                "/api/projects/",
                "POST",
                {
                    "title": title,
                    "label_config": label_config,
                    "description": (
                        "Train: adjustable confidence filter, hard cases and negatives available."
                        if role == "train"
                        else "Test: ALL 380 frames. No confidence filter. Correct every proposal before metrics."
                    ),
                    "sampling": "Sequential sampling",
                    "model_version": "gpt-6-luna",
                    "show_collab_predictions": True,
                    "enable_empty_annotation": True,
                    "show_annotation_history": True,
                },
            )
        )
        project_id = int(project["id"])
        if project_id in {1, 2}:
            raise ValueError("Refusing to change older review projects")
        storages = request(f"/api/storages/localfiles/?project={project_id}")
        if not any(s["path"] == str(output / "frames") for s in storages):
            request(
                "/api/storages/localfiles/",
                "POST",
                {
                    "project": project_id,
                    "path": str(output / "frames"),
                    "title": "Lou Mistrau review images",
                    "regex_filter": "^$",
                    "use_blob_urls": True,
                    "recursive_scan": False,
                },
            )
        existing = download_local_export(token=token, project_id=project_id)
        expected = {t["data"]["item_id"] for t in tasks[role]}
        existing_ids = [t["data"]["item_id"] for t in existing]
        if (
            len(existing_ids) != len(set(existing_ids))
            or not set(existing_ids) <= expected
        ):
            raise ValueError("Unexpected or duplicate tasks; no overwrite")
        pending = [
            t for t in tasks[role] if t["data"]["item_id"] not in set(existing_ids)
        ]
        for start in range(0, len(pending), 150):
            request(
                f"/api/projects/{project_id}/import",
                "POST",
                pending[start : start + 150],
            )
        verified = download_local_export(token=token, project_id=project_id)
        if (
            len(verified) != len(expected)
            or {t["data"]["item_id"] for t in verified} != expected
        ):
            raise ValueError("Imported tasks do not match expected inventory")
        views = request(f"/api/dm/views/?project={project_id}")
        columns = request(f"/api/dm/columns/?project={project_id}")["columns"]
        visible = {
            "tasks:data.image",
            "tasks:data.video_id",
            "tasks:data.timestamp_seconds",
            "tasks:data.source_frame",
            "tasks:predictions_score",
            "tasks:total_annotations",
            "tasks:draft_exists",
        }
        hidden = [
            "tasks:" + (c["parent"] + "." if c.get("parent") else "") + c["id"]
            for c in columns
        ]
        for order, (name, filters) in enumerate(view_specs(role)):
            if any(v["data"].get("title") == name for v in views):
                continue  # Preserve any threshold or sorting the user changed.
            view = request(
                "/api/dm/views/",
                "POST",
                {
                    "project": project_id,
                    "order": order,
                    "data": {
                        "title": name,
                        "type": "list",
                        "target": "tasks",
                        "filters": {"conjunction": "and", "items": filters},
                        "ordering": ["tasks:id"],
                        "hiddenColumns": {
                            "explore": [c for c in hidden if c not in visible],
                            "labeling": [
                                c
                                for c in hidden
                                if c
                                not in {
                                    "tasks:data.image",
                                    "tasks:data.frame_info",
                                    "tasks:data.luna_context",
                                }
                            ],
                        },
                        "columnsDisplayType": {},
                        "columnsWidth": {},
                        "gridWidth": 4,
                        "gridFitImagesToWidth": False,
                    },
                },
            )
            views.append(view)
        reports[role] = {
            "project_id": project_id,
            "url": f"{BASE}/projects/{project_id}/data",
            "tasks": len(verified),
            "imported_now": len(pending),
            "annotations_preserved": sum(
                len(t.get("annotations", [])) for t in verified
            ),
            "views": [{"id": v["id"], "title": v["data"].get("title")} for v in views],
        }
    _atomic_json(output / "review_projects.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "import"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, tasks = prepare(root)
    if args.action == "import":
        import_projects(root, output, tasks)


if __name__ == "__main__":
    main()
