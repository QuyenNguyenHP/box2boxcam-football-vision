"""Append a completed personal-video run without replacing reviewed tasks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from box2boxcam.data.label_studio_review import (
    _label_studio_request,
    build_tasks,
    download_local_export,
)
from box2boxcam.data.luna_batch import _atomic_json

BASE = "http://127.0.0.1:8080"


def append_review(
    root: Path, source: Path, project_id: int, *, allow_partial: bool = False,
) -> dict:
    if project_id not in {3, 4}:
        raise ValueError("Append only to the existing club match train/test projects")
    intended_split = "train" if project_id == 3 else "test"
    summary = json.loads((source / "summary.json").read_text())
    if summary["valid_predictions"] != summary["planned_cases"] and not allow_partial:
        raise ValueError("Incomplete annotation run")
    predictions = json.loads((source / "predictions.json").read_text())["predictions"]
    if len({p["video_id"] for p in predictions}) != 1:
        raise ValueError("Expected one additional video")
    if any(
        p["intended_split"] != intended_split
        or p["match_id"] != "club_match_2026-09-30"
        for p in predictions
    ):
        raise ValueError("Different split or match: stop before importing")
    video_id = predictions[0]["video_id"]
    plan = json.loads((source / video_id / "plan.json").read_text())
    if len(predictions) != plan["case_count"] and not allow_partial:
        raise ValueError("Incomplete results: keep failures visible before import")
    if len(predictions) != summary["valid_predictions"]:
        raise ValueError("Prediction inventory differs from collected summary")
    planned_ids = {
        f"{video_id}_frame_{int(case['source_frame']):08d}"
        for case in plan["cases"]
    }
    prediction_ids = [p["item_id"] for p in predictions]
    if len(prediction_ids) != len(set(prediction_ids)) or not set(prediction_ids) <= planned_ids:
        raise ValueError("Duplicate or unknown proposals")
    missing_ids = sorted(planned_ids - set(prediction_ids))
    if len(missing_ids) != summary["missing_or_failed_predictions"]:
        raise ValueError("Missing predictions not accounted for in summary")
    shared = root / "data/interim/gpt6_luna_label_studio_review"
    output = shared / source.name
    lines = (shared / "local_login.txt").read_text().splitlines()
    token = next(s.partition(": ")[2] for s in lines if s.startswith("API token:"))

    def request(path, method="GET", payload=None):
        return _label_studio_request(BASE + path, token, method=method, payload=payload)

    project = request(f"/api/projects/{project_id}/")
    allowed_titles = (
        {
            "PitchLens — club match — TRAIN",
            "PitchLens — club match — TRAIN (7 octobre)",
        }
        if project_id == 3
        else {"PitchLens — club match — TEST (30 septembre)"}
    )
    if project["title"] not in allowed_titles:
        raise ValueError("Unexpected destination project")
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    # Expose ALL cases for review, including low-confidence train proposals.
    missing = {}
    for row in predictions:
        destination = frames / f"{row['item_id']}.jpg"
        if destination.exists():
            continue
        if row["image_path"]:
            os.link(root / row["image_path"], destination)
        else:
            missing[int(row["source_frame"])] = row
    if missing:
        import cv2
        from annotate_personal_matches_luna import sha256

        video = root / predictions[0]["source_video"]
        if sha256(video) != predictions[0]["source_video_sha256"]:
            raise ValueError("Source video changed")
        capture = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
        )
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        try:
            for frame in range(max(missing) + 1):
                ok, pixels = capture.read()
                if not ok:
                    raise RuntimeError(f"Decode failed at frame {frame}")
                if frame not in missing:
                    continue
                row = missing[frame]
                if pixels.shape[:2] != (row["height"], row["width"]):
                    raise ValueError(
                        "Decoded dimensions differ from annotation dimensions"
                    )
                if shutil.disk_usage(output).free < 1024**3:
                    raise RuntimeError("Less than 1 GiB free: stop safely")
                if not cv2.imwrite(
                    str(frames / f"{row['item_id']}.jpg"),
                    pixels,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError("Cannot save review image")
        finally:
            capture.release()
    index = {p["item_id"]: p for p in predictions}
    tasks = []
    for row in predictions:
        destination = frames / f"{row['item_id']}.jpg"
        if not destination.is_file():
            raise FileNotFoundError(destination)
        task_plan = {
            "model": "gpt-6-luna",
            "video": plan["video"],
            "cases": [{"item_id": row["item_id"], "source_frame": row["source_frame"]}],
        }
        task = build_tasks(task_plan, index)[0]
        task["data"].update(
            {
                "image": f"/data/local-files/?d={output.name}/frames/{row['item_id']}.jpg",
                "video_id": video_id,
                "source_video": row["source_video"],
                "image_path": str(destination.relative_to(root)),
                "width": row["width"],
                "height": row["height"],
                "timestamp_seconds": row["timestamp_seconds"],
                "match_id": row["match_id"],
                "intended_split": intended_split,
                "ball_visible": row["ball_in_play_visible"],
                "frame_info": f"{intended_split.upper()} · {video_id} · frame {row['source_frame']} · {row['timestamp_seconds']:.2f} s",
            }
        )
        tasks.append(task)
    task_path = output / "tasks.json"
    if task_path.exists() and json.loads(task_path.read_text()) != tasks:
        raise FileExistsError("Proposals changed: no overwrite")
    if not task_path.exists():
        _atomic_json(task_path, tasks)
    existing = download_local_export(token=token, project_id=project_id)
    existing_ids = [t["data"]["item_id"] for t in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("Duplicate existing tasks")
    if any(t["data"].get("intended_split") != intended_split for t in existing):
        raise ValueError("Existing project tasks belong to another split")
    expected = {t["data"]["item_id"] for t in tasks}
    pending = [t for t in tasks if t["data"]["item_id"] not in set(existing_ids)]
    storages = request(f"/api/storages/localfiles/?project={project_id}")
    if not any(s["path"] == str(frames) for s in storages):
        request(
            "/api/storages/localfiles/",
            "POST",
            {
                "project": project_id,
                "path": str(frames),
                "regex_filter": "^$",
                "title": "club match additional clip",
                "use_blob_urls": True,
                "recursive_scan": False,
            },
        )
    for start in range(0, len(pending), 150):
        request(
            f"/api/projects/{project_id}/import", "POST", pending[start : start + 150]
        )
    verified = download_local_export(token=token, project_id=project_id)
    ids = [t["data"]["item_id"] for t in verified]
    if len(ids) != len(set(ids)) or set(ids) != set(existing_ids) | expected:
        raise ValueError("Appended inventory mismatch")
    if project_id == 3 and project["title"] != "PitchLens — club match — TRAIN":
        request(
            f"/api/projects/{project_id}/",
            "PATCH",
            {"title": "PitchLens — club match — TRAIN"},
        )
    report = {
        "project_id": project_id,
        "new_tasks": len(pending),
        "total_tasks": len(ids),
        "project_url": f"{BASE}/projects/{project_id}/data?tab={5 if project_id == 3 else 7}",
        "intended_split": intended_split,
        "existing_tasks_not_modified": True,
        "images_linked_not_copied": True,
        "planned_cases": plan["case_count"],
        "available_predictions": len(predictions),
        "missing_predictions_not_imported": len(missing_ids),
        "missing_item_ids": missing_ids,
    }
    _atomic_json(output / "review_project.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source if args.source.is_absolute() else root / args.source
    append_review(root, source, args.project_id, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
