"""Bridge between Luna's video predictions and a local Label Studio review."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Merge predictions in order; a retry replaces the corresponding first pass."""
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in _read_json(path)["predictions"]:
            merged[str(row["item_id"])] = row
    return merged


def _rectangle(bbox: list[int], width: int, height: int) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid box: {bbox}")
    return {
        "id": "luna-ball",
        "from_name": "bbox",
        "to_name": "image",
        "type": "rectangle",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": 100 * x0 / width,
            "y": 100 * y0 / height,
            "width": 100 * (x1 - x0) / width,
            "height": 100 * (y1 - y0) / height,
            "rotation": 0,
        },
    }


def build_tasks(
    plan: dict[str, Any], predictions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep every planned frame, including failed API responses."""
    width = int(plan["video"]["width"])
    height = int(plan["video"]["height"])
    fps = float(plan["video"]["fps"])
    tasks = []
    for case in plan["cases"]:
        item_id = str(case["item_id"])
        frame = int(case["source_frame"])
        row = predictions.get(item_id)
        if row:
            box = row.get("bbox_current_pixels")
            if "balls" in row:
                result = []
                for index, ball in enumerate(row["balls"]):
                    rectangle = _rectangle(ball["bbox_current_pixels"], width, height)
                    rectangle["id"] = f"luna-ball-{index}"
                    result.append(rectangle)
            else:
                result = [_rectangle(box, width, height)] if box is not None else []
            confidence = float(row["confidence"])
            state = (f"{len(result)} ball(s) proposed" if result else "no ball proposed") if "balls" in row else ("box proposed" if box is not None else "no ball proposed")
            context = (
                f"Luna confidence: {confidence:.0%} · {state}\n"
                f"Temporal cue: {row.get('temporal_evidence') or '—'}\n"
                f"Ambiguity: {row.get('ambiguity') or '—'}"
            )
            if "balls" in row:
                context += "\n" + "\n".join(
                    f"Ball {index + 1}: {ball['confidence']:.0%} · {ball.get('location', 'unknown')} · {ball.get('visibility', '—')} · {ball.get('temporal_evidence') or '—'}"
                    for index, ball in enumerate(row["balls"])
                )
        else:
            result = []
            confidence = 0.0
            # Persisted status value in existing tasks ("Luna response missing"); keep as-is.
            state = "réponse Luna manquante"
            context = "No usable response: check this image by hand."
        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d=frames/{item_id}.jpg",
                    "item_id": item_id,
                    "source_frame": frame,
                    "timestamp_seconds": round(frame / fps, 3),
                    "luna_status": state,
                    "luna_confidence": confidence,
                    "frame_info": f"{item_id} · {frame / fps:.1f} s · {state}",
                    "luna_context": context,
                },
                "predictions": [
                    {
                        "model_version": str(plan["model"]),
                        "score": confidence,
                        "result": result,
                    }
                ],
            }
        )
    return tasks


def extract_review_frames(
    video_path: Path, frames_dir: Path, frame_numbers: list[int], *, quality: int = 88
) -> tuple[int, int]:
    """Decode only the central frames, retaining full video resolution."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV manque : pip install -e '.[annotation]'") from error

    frames_dir.mkdir(parents=True, exist_ok=True)
    needed = set(frame_numbers)
    missing = {
        frame
        for frame in needed
        if not (frames_dir / f"frame_{frame:08d}.jpg").is_file()
    }
    if not missing:
        return 0, len(needed)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unreadable video: {video_path}")
    created = 0
    try:
        for number in range(max(missing) + 1):
            if number in missing:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot read frame {number}")
                path = frames_dir / f"frame_{number:08d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise RuntimeError(f"Cannot write: {path}")
                created += 1
            elif not capture.grab():
                raise RuntimeError(f"Cannot read frame {number}")
    finally:
        capture.release()
    return created, len(needed)


def prepare_review(
    *, project_root: Path, first_run: Path, retry_run: Path, output_root: Path
) -> dict[str, Any]:
    """Create one import file and a resumable JPEG cache for Label Studio."""
    plan = _read_json(first_run / "plan.json")
    video = (project_root / str(plan["source_video"])).resolve()
    predictions = prediction_index(
        [first_run / "predictions.json", retry_run / "predictions.json"]
    )
    cases = plan["cases"]
    frames = [int(case["source_frame"]) for case in cases]
    if len(frames) != len(set(frames)):
        raise ValueError("The plan contains duplicate images")
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(plan, predictions)
    target = output_root / "tasks.json"
    if target.exists():
        old = _read_json(target)
        if [row["data"]["item_id"] for row in old] != [
            row["data"]["item_id"] for row in tasks
        ]:
            raise RuntimeError("tasks.json exists with a different inventory")
    target.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    created, total = extract_review_frames(video, output_root / "frames", frames)
    return {
        "tasks": len(tasks),
        "with_prediction": sum(
            bool(row["predictions"][0]["result"]) for row in tasks
        ),
        "without_model_answer": len(tasks) - len(predictions),
        "frames_created_now": created,
        "frames_in_cache": total,
        "cache_size_mb": round(
            sum(path.stat().st_size for path in (output_root / "frames").glob("*.jpg"))
            / 1_000_000,
            1,
        ),
        "free_disk_gb": round(shutil.disk_usage(output_root).free / 1_000_000_000, 1),
        "task_file": str(target),
    }


def _label_studio_request(
    url: str, token: str, *, method: str = "GET", payload: dict[str, Any] | None = None
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def sync_local_project(
    *, tasks_path: Path, token: str, project_id: int = 1,
    base_url: str = "http://127.0.0.1:8080",
) -> dict[str, int]:
    """Fill missing Luna predictions without reimporting tasks or touching reviews."""
    if base_url not in {"http://127.0.0.1:8080", "http://localhost:8080"}:
        raise ValueError("Synchronization is restricted to the local Label Studio server")
    desired = {
        row["data"]["item_id"]: row
        for row in json.loads(tasks_path.read_text(encoding="utf-8"))
    }
    listed: list[dict[str, Any]] = []
    page = 1
    while True:
        response = _label_studio_request(
            f"{base_url}/api/tasks?project={project_id}&page={page}&page_size=200",
            token,
        )
        listed.extend(response["tasks"])
        if len(listed) >= response["total"]:
            break
        page += 1
    if len(listed) != len(desired):
        raise ValueError(
            f"Projet local : {len(listed)} tâches ; plan : {len(desired)}. "
            "Synchronization cancelled to avoid duplicates."
        )
    ids = [row["data"]["item_id"] for row in listed]
    if len(set(ids)) != len(ids) or set(ids) != set(desired):
        raise ValueError("The images of the local project do not match the plan")
    updated = 0
    already_reviewed = 0
    still_missing = 0
    for summary in listed:
        item_id = summary["data"]["item_id"]
        target = desired[item_id]
        if target["data"]["luna_status"] == "réponse Luna manquante":
            still_missing += 1
            continue
        if summary["data"]["luna_status"] != "réponse Luna manquante":
            continue
        task_id = int(summary["id"])
        current = _label_studio_request(f"{base_url}/api/tasks/{task_id}", token)
        if current["project"] != project_id or current["data"]["item_id"] != item_id:
            raise ValueError(f"Tâche inattendue : {task_id}")
        if current["annotations"] or current["drafts"]:
            already_reviewed += 1
            continue
        predictions = current["predictions"]
        if len(predictions) != 1:
            raise ValueError(f"{item_id}: {len(predictions)} local predictions")
        _label_studio_request(
            f"{base_url}/api/predictions/{predictions[0]['id']}/",
            token,
            method="PATCH",
            payload=target["predictions"][0],
        )
        _label_studio_request(
            f"{base_url}/api/tasks/{task_id}/",
            token,
            method="PATCH",
            payload={"data": target["data"]},
        )
        updated += 1
    return {
        "tasks": len(listed),
        "predictions_added": updated,
        "reviewed_or_drafted_preserved": already_reviewed,
        "still_without_model_answer": still_missing,
    }


def download_local_export(
    *, token: str, project_id: int = 1,
    base_url: str = "http://127.0.0.1:8080",
) -> list[dict[str, Any]]:
    """Download *all* tasks, including untouched ones, for a truthful audit."""
    if base_url not in {"http://127.0.0.1:8080", "http://localhost:8080"}:
        raise ValueError("Automatic export is restricted to the local server")
    tasks = _label_studio_request(
        f"{base_url}/api/projects/{project_id}/export"
        "?exportType=JSON&download_all_tasks=true",
        token,
    )
    if not isinstance(tasks, list):
        raise TypeError("Label Studio did not return a list of tasks")
    return tasks


def _box_from_result(result: dict[str, Any], width: int, height: int) -> list[int]:
    value = result["value"]
    if float(value.get("rotation", 0)) != 0:
        raise ValueError("Rotated boxes are not supported")
    x0 = round(float(value["x"]) * width / 100)
    y0 = round(float(value["y"]) * height / 100)
    x1 = round((float(value["x"]) + float(value["width"])) * width / 100)
    y1 = round((float(value["y"]) + float(value["height"])) * height / 100)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid exported box: {[x0, y0, x1, y1]}")
    return [x0, y0, x1, y1]


def _yolo_line(box: list[int], width: int, height: int) -> str:
    x0, y0, x1, y1 = box
    return (
        "0 "
        f"{(x0 + x1) / (2 * width):.8f} "
        f"{(y0 + y1) / (2 * height):.8f} "
        f"{(x1 - x0) / width:.8f} "
        f"{(y1 - y0) / height:.8f}\n"
    )


def export_review(
    *, export_path: Path, output_root: Path, width: int | None = None, height: int | None = None,
    expected_ids: set[str] | None = None,
) -> dict[str, int]:
    """Export only submitted human annotations; unreviewed tasks stay pending."""
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise TypeError("The Label Studio export must be a JSON list of tasks")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Export folder already populated: {output_root}")
    rows = []
    labels_to_write: dict[str, str] = {}
    seen: set[str] = set()
    for task in tasks:
        data = task["data"]
        item_id = str(data["item_id"])
        if expected_ids is not None and item_id not in expected_ids:
            raise ValueError(f"Image outside the review plan: {item_id}")
        if item_id in seen:
            raise ValueError(f"Image present several times in the export: {item_id}")
        seen.add(item_id)
        annotations = [
            row for row in task.get("annotations", []) if not row.get("was_cancelled")
        ]
        if not annotations:
            continue
        image_width, image_height = data.get("width", width), data.get("height", height)
        if not isinstance(image_width, int) or not isinstance(image_height, int) or image_width <= 0 or image_height <= 0:
            raise ValueError(f"Image dimensions missing or invalid: {item_id}")
        annotation = max(
            annotations,
            key=lambda row: (str(row.get("updated_at") or ""), int(row.get("id") or 0)),
        )
        results = annotation.get("result") or []
        excluded = any(
            result.get("from_name") == "decision"
            and "Exclure image" in result.get("value", {}).get("choices", [])
            for result in results
        )
        boxes = [
            _box_from_result(result, image_width, image_height)
            for result in results
            if result.get("from_name") == "bbox"
            and result.get("type") == "rectangle"
        ]
        status = "excluded" if excluded else "positive" if boxes else "negative"
        if not excluded:
            labels_to_write[item_id] = "".join(
                _yolo_line(box, image_width, image_height) for box in boxes
            )
        rows.append(
            {
                "item_id": item_id,
                "source_frame": data["source_frame"],
                "status": status,
                "boxes": 0 if excluded else len(boxes),
                "annotation_id": annotation.get("id"),
                "luna_confidence": data.get("luna_confidence"),
            }
        )
    labels = output_root / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for item_id, content in labels_to_write.items():
        (labels / f"{item_id}.txt").write_text(content)
    with (output_root / "review_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "item_id", "source_frame", "status", "boxes",
                "annotation_id", "luna_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "reviewed": len(rows),
        "positive": sum(row["status"] == "positive" for row in rows),
        "negative": sum(row["status"] == "negative" for row in rows),
        "excluded": sum(row["status"] == "excluded" for row in rows),
        "unreviewed": (len(expected_ids) if expected_ids is not None else len(tasks))
        - len(rows),
    }
