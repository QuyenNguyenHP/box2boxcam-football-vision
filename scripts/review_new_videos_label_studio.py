"""Import Luna proposals into a separate, resumable local review project."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from pitchlens.data.label_studio_review import (
    _label_studio_request,
    build_tasks,
    download_local_export,
)
from pitchlens.data.luna_batch import _atomic_json

TITLE = "PitchLens — Luna max — 782 nouvelles images"
BASE = "http://127.0.0.1:8080"
DATASETS = {
    "shorts": ("gpt6_luna_new_videos_20260928", "new_videos_20260928", TITLE),
    "closeups": (
        "gpt6_luna_personal_closeups_20261011", "personal_closeups_20261011",
        "PitchLens — Gros plans — Luna max (11 octobre)",
    ),
}


def prepare(root: Path, dataset: str = "shorts") -> tuple[Path, list[dict]]:
    source_name, review_name, _ = DATASETS[dataset]
    source = root / "data/interim" / source_name
    shared = root / "data/interim/gpt6_luna_label_studio_review"
    output = shared / review_name
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    predictions = json.loads((source / "predictions.json").read_text())["predictions"]
    index = {p["item_id"]: p for p in predictions}
    if len(index) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    with (source / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if set(index) != {r["item_id"] for r in rows} or len(rows) != len(index):
        raise ValueError("Predictions and extracted images do not match")
    tasks = []
    for row in rows:
        item_id = row["item_id"]
        original = root / row["image_path"]
        target = frames / f"{item_id}.jpg"
        # Hard links share the original bytes: no second 409 MiB image cache.
        if target.exists():
            if not os.path.samefile(original, target):
                raise FileExistsError(f"A different review image exists: {target}")
        else:
            os.link(original, target)
        if row.get("fps"):
            fps = float(row["fps"])
        else:
            video_plan = json.loads((source / row["video_id"] / "plan.json").read_text())
            fps = float(video_plan["video"]["fps"])
        plan = {
            "model": "gpt-6-luna",
            "video": {"width": int(row["width"]), "height": int(row["height"]),
                      "fps": fps},
            "cases": [{"item_id": item_id, "source_frame": int(row["source_frame"])}],
        }
        task = build_tasks(plan, index)[0]
        task["data"].update({
            "image": f"/data/local-files/?d={review_name}/frames/{item_id}.jpg",
            "video_id": row["video_id"], "uploader": row["uploader"],
            "source_url": row["source_url"], "source_video": row["source_video"],
            "image_path": row["image_path"],
            "width": int(row["width"]), "height": int(row["height"]),
            "timestamp_seconds": float(row["timestamp_seconds"]),
            "frame_info": (f"{row['uploader']} · {row['video_id']} · "
                           f"frame {row['source_frame']} · "
                           f"{float(row['timestamp_seconds']):.2f} s"),
        })
        if dataset == "closeups":
            for field in ("match_id", "role", "intended_split", "source_video_sha256"):
                task["data"][field] = row[field]
        tasks.append(task)
    task_path = output / "tasks.json"
    if task_path.exists() and json.loads(task_path.read_text()) != tasks:
        raise FileExistsError("Review task proposals have changed; refusing replacement")
    if not task_path.exists():
        _atomic_json(task_path, tasks)
    return output, tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "import"))
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="shorts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, tasks = prepare(root, args.dataset)
    title = DATASETS[args.dataset][2]
    if args.action == "prepare":
        print(f"{len(tasks)} proposals prepared, zero image bytes copied: {output}")
        return
    credentials = (output.parent / "local_login.txt").read_text().splitlines()
    token = next(line.partition(": ")[2] for line in credentials
                 if line.startswith("API token:"))
    response = _label_studio_request(f"{BASE}/api/projects/?page_size=100", token)
    matches = [p for p in response["results"] if p["title"] == title]
    if response.get("next") or len(matches) > 1:
        raise ValueError("Ambiguous project inventory")
    if matches:
        project = matches[0]
    else:
        # Reuse the proven thin-box UI, not any old annotations or tasks.
        original = _label_studio_request(f"{BASE}/api/projects/1/", token)
        label_config = original["label_config"]
        if args.dataset == "closeups":
            label_config = label_config.replace(
                "vérification du ballon en jeu", "vérification du ballon visible — gros plans"
            )
        project = _label_studio_request(
            f"{BASE}/api/projects/", token, method="POST", payload={
                "title": title, "label_config": label_config,
                "description": "Six private videos. Luna max proposals, human review pending.",
                "sampling": "Sequential sampling", "model_version": "gpt-6-luna",
                "show_collab_predictions": True, "enable_empty_annotation": True,
                "show_annotation_history": True,
            },
        )
    project_id = int(project["id"])
    protected_ids = {1, 2, 3, 4} if args.dataset == "closeups" else {1, 3, 4}
    if project_id in protected_ids:
        raise ValueError("Refusing to alter the previously reviewed project")
    storages = _label_studio_request(
        f"{BASE}/api/storages/localfiles/?project={project_id}", token
    )
    storage_path = str(output / "frames")
    if not any(storage["path"] == storage_path for storage in storages):
        # Authorize only this project's image directory. Never sync: tasks are
        # already imported with predictions, so scanning would create duplicates.
        _label_studio_request(f"{BASE}/api/storages/localfiles/", token,
                             method="POST", payload={
                                 "project": project_id, "path": storage_path,
                                 "title": "New video review images", "regex_filter": "^$",
                                 "use_blob_urls": True, "recursive_scan": False,
                             })
    existing = download_local_export(token=token, project_id=project_id)
    existing_ids = [t["data"]["item_id"] for t in existing]
    expected = {t["data"]["item_id"] for t in tasks}
    if len(existing_ids) != len(set(existing_ids)) or not set(existing_ids) <= expected:
        raise ValueError("Existing project contains unexpected/duplicate tasks")
    pending = [t for t in tasks if t["data"]["item_id"] not in set(existing_ids)]
    for start in range(0, len(pending), 200):
        _label_studio_request(f"{BASE}/api/projects/{project_id}/import", token,
                             method="POST", payload=pending[start:start + 200])
    verified = download_local_export(token=token, project_id=project_id)
    ids = [t["data"]["item_id"] for t in verified]
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError(f"Imported inventory does not match all {len(tasks)} proposals")
    report = {
        "project_id": project_id, "project_url": f"{BASE}/projects/{project_id}/data",
        "tasks": len(ids), "tasks_imported_now": len(pending),
        "predictions": sum(len(t.get("predictions", [])) for t in verified),
        "submitted_annotations": sum(len(t.get("annotations", [])) for t in verified),
        "image_bytes_copied": 0, "old_project_untouched": True,
    }
    _atomic_json(output / "review_project.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
