"""Import this session into its own local project, preserving all human edits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pitchlens.data.label_studio_review import (
    _label_studio_request,
    download_local_export,
    export_review,
)
from pitchlens.data.luna_batch import _atomic_json

BASE = "http://127.0.0.1:8080"
TITLE = "PitchLens — Match iPhone — Tous les ballons"


def import_review(root: Path, source: Path):
    shared = root / "data/interim/gpt6_luna_label_studio_review"
    target = shared / "field_all_balls_20261014"
    frames = target / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((source / "review/tasks.json").read_text())
    for task in tasks:
        item_id = task["data"]["item_id"]
        image = source / "review/frames" / f"{item_id}.jpg"
        destination = frames / image.name
        if destination.exists() and not os.path.samefile(image, destination):
            raise FileExistsError("Different image already exposed for review")
        if not destination.exists():
            os.link(image, destination)
        task["data"]["image"] = f"/data/local-files/?d=field_all_balls_20261014/frames/{item_id}.jpg"
    (target / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    credentials = (shared / "local_login.txt").read_text().splitlines()
    token = next(line.partition(": ")[2] for line in credentials if line.startswith("API token:"))

    def request(path, method="GET", payload=None):
        return _label_studio_request(BASE + path, token, method=method, payload=payload)

    inventory = request("/api/projects/?page_size=100")
    matches = [p for p in inventory["results"] if p["title"] == TITLE]
    if inventory.get("next") or len(matches) > 1:
        raise RuntimeError("Ambiguous project inventory")
    config = (root / "configs/label_studio_all_balls.xml").read_text()
    # Label Studio's Prediction score averages only predictions of the
    # project's active model. Until Luna has answered every planned frame,
    # use the available detector instead of showing empty Luna scores.
    active_model = "gpt-6-luna" if all(
        any(p["model_version"] == "gpt-6-luna" for p in task["predictions"])
        for task in tasks
    ) else "YOLO26n v003 960px proposals"
    project = matches[0] if matches else request("/api/projects/", "POST", {
        "title": TITLE, "label_config": config,
        "description": "22 private clips, one session. All visible balls including stationary sideline balls. Independent YOLO v003 and Luna proposals; human review required before training.",
        "sampling": "Sequential sampling", "model_version": active_model,
        "show_collab_predictions": True, "enable_empty_annotation": True, "show_annotation_history": True,
    })
    project_id = int(project["id"])
    protected_ids = {int(p["id"]) for p in inventory["results"] if p["title"] != TITLE}
    if project_id in protected_ids or project["label_config"].strip() != config.strip():
        raise RuntimeError("Refusing to modify a pre-existing or differently configured project")
    if project.get("model_version") != active_model:
        request(f"/api/projects/{project_id}/", "PATCH", {"model_version": active_model})
    storages = request(f"/api/storages/localfiles/?project={project_id}")
    if not any(s["path"] == str(frames) for s in storages):
        request("/api/storages/localfiles/", "POST", {
            "project": project_id, "path": str(frames), "title": "Match iPhone all-balls images",
            "regex_filter": "^$", "use_blob_urls": True, "recursive_scan": False,
        })
    desired = {task["data"]["item_id"]: task for task in tasks}
    if len(desired) != len(tasks):
        raise RuntimeError("Duplicate desired task")
    existing = download_local_export(token=token, project_id=project_id)
    by_id = {task["data"]["item_id"]: task for task in existing}
    if len(by_id) != len(existing) or not set(by_id) <= set(desired):
        raise RuntimeError("Unexpected existing tasks; no overwrite")
    pending = [t for t in tasks if t["data"]["item_id"] not in by_id]
    for start in range(0, len(pending), 100):
        request(f"/api/projects/{project_id}/import", "POST", pending[start:start + 100])
    added_luna, preserved = 0, 0
    for item_id, old in by_id.items():
        new = desired[item_id]
        if not any(p["model_version"] == "gpt-6-luna" for p in new["predictions"]):
            continue
        detail = request(f"/api/tasks/{old['id']}/")
        if detail.get("annotations") or detail.get("drafts"):
            preserved += 1
            continue
        if any(p["model_version"] == "gpt-6-luna" for p in detail.get("predictions", [])):
            continue
        prediction = next(p for p in new["predictions"] if p["model_version"] == "gpt-6-luna")
        request("/api/predictions/", "POST", {"task": old["id"], **prediction})
        request(f"/api/tasks/{old['id']}/", "PATCH", {"data": new["data"]})
        added_luna += 1
    verified = download_local_export(token=token, project_id=project_id)
    if len(verified) != len(desired) or {t["data"]["item_id"] for t in verified} != set(desired):
        raise RuntimeError("Imported task inventory differs")
    report = {"project_id": project_id, "project_url": f"{BASE}/projects/{project_id}/data", "tasks": len(verified),
              "prediction_score_model": active_model,
              "new_tasks": len(pending), "new_luna_predictions": added_luna,
              "human_reviews_preserved": preserved, "annotation_scope": "all_visible"}
    _atomic_json(source / "review/label_studio_project.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--export-dir", type=Path, help="Export only submitted human annotations to a NEW directory")
    args = parser.parse_args()
    root, source = Path(__file__).resolve().parents[1], args.source.resolve()
    if args.export_dir:
        report = json.loads((source / "review/label_studio_project.json").read_text())
        shared = root / "data/interim/gpt6_luna_label_studio_review"
        token = next(line.partition(": ")[2] for line in (shared / "local_login.txt").read_text().splitlines() if line.startswith("API token:"))
        tasks = download_local_export(token=token, project_id=report["project_id"])
        expected = {task["data"]["item_id"] for task in json.loads((source / "review/tasks.json").read_text())}
        if {task["data"]["item_id"] for task in tasks} != expected:
            raise RuntimeError("Review export inventory differs from this session")
        destination = source / "review/human_export_snapshot.json"
        destination.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
        summary = export_review(export_path=destination, output_root=args.export_dir, expected_ids=expected)
        _atomic_json(args.export_dir / "summary.json", {**summary, "annotation_scope": "all_visible", "labels_are_human_reviewed": True,
                     "source_run": str(source), "match_group": "personal_field_session_20261012"})
        print(json.dumps(summary, indent=2))
    else:
        import_review(root, source)


if __name__ == "__main__":
    main()
