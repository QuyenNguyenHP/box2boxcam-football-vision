"""Dedicated field-validation review; predictions never become ground truth automatically."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

from pitchlens.data.build_mobile_dataset import sha256
from pitchlens.data.field_validation import (
    SUBMIT_PROTOCOL,
    parse_reviewed_task,
    proposed_role,
)
from pitchlens.data.label_studio_review import (
    _label_studio_request,
    build_tasks,
    download_local_export,
)
from pitchlens.data.luna_batch import _atomic_json, _yolo_line

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8080"
TITLE = "PitchLens — Validation terrain — 14 octobre 2026"
EXPOSED = "field_validation_20261021"
SHARED = ROOT / "data/interim/gpt6_luna_label_studio_review"


def get_run(config_path):
    spec = importlib.util.spec_from_file_location(
        "personal_runner", ROOT / "scripts/annotate_personal_matches_luna.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    raw, output, configs = runner.prepare(ROOT, config_path)
    if not raw.get("session_policy", {}).get("exclude_from_training") or any(
        v["role"] != "test_primary" for v, _, _ in configs
    ):
        raise ValueError("Only reserved field-validation clips are accepted")
    return raw, output, configs


def api_client():
    credentials = (SHARED / "local_login.txt").read_text().splitlines()
    token = next(
        line.partition(": ")[2] for line in credentials if line.startswith("API token:")
    )

    def request(path, method="GET", payload=None):
        return _label_studio_request(BASE + path, token, method=method, payload=payload)

    return request, token


def link_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(source, target) and sha256(source) != sha256(target):
            raise FileExistsError(f"Different exposed image: {target}")
    else:
        os.link(source, target)


def make_tasks(config_path):
    _, output, configs = get_run(config_path)
    frames = SHARED / EXPOSED / "frames"
    tasks = []
    for video, config, plan in configs:
        manifest = json.loads(
            (config.output_root / "frames_manifest.json").read_text()
        )["frames"]
        predpath = config.output_root / "predictions.json"
        predictions = (
            {p["item_id"]: p for p in json.loads(predpath.read_text())["predictions"]}
            if predpath.exists()
            else {}
        )
        built = build_tasks(plan, predictions)
        for case, task in zip(plan["cases"], built, strict=True):
            urls = {}
            for key in ["current_frame", "previous_frame", "next_frame"]:
                number = case[key]
                name = f"{config.output_root.name}_frame_{number:08d}.jpg"
                link_exact(
                    config.output_root / "frames" / f"frame_{number:08d}.jpg",
                    frames / name,
                )
                urls[key] = f"/data/local-files/?d={EXPOSED}/frames/{name}"
            frame = int(case["source_frame"])
            timestamp = manifest[str(frame)]["timestamp_seconds"]
            item = f"{config.output_root.name}_{case['item_id']}"
            task["data"].update(
                image=urls["current_frame"],
                image_previous=urls["previous_frame"],
                image_next=urls["next_frame"],
                item_id=item,
                source_frame=frame,
                timestamp_seconds=timestamp,
                match_id=video["match_id"],
                video_id=config.output_root.name,
                source_video=video["source_video"],
                source_video_sha256=video["sha256"],
                width=plan["video"]["width"],
                height=plan["video"]["height"],
                image_sha256=manifest[str(frame)]["sha256"],
                intended_use="development_validation",
                intended_split="val",
                exclude_from_training=True,
                annotation_scope="all_visible",
                review_protocol=SUBMIT_PROTOCOL,
                frame_info=f"{config.output_root.name} · {timestamp:.3f} s · validation terrain · environ 2 images/s",
                image_path=str((frames / f"{item}.jpg").relative_to(ROOT)),
            )
            row = predictions.get(case["item_id"])
            if not row:
                task["predictions"] = []
                task["data"]["luna_context"] = (
                    "Luna proposals pending. An empty task does not mean an image without a ball."
                )
            else:
                prediction = task["predictions"][0]
                for i, ball in enumerate(row["balls"]):
                    prediction["result"].append(
                        {
                            "id": f"luna-ball-{i}",
                            "from_name": "ball_role",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": [proposed_role(ball)]},
                        }
                    )
                state = (
                    "Actif visible"
                    if sum(proposed_role(b) == "Actif" for b in row["balls"]) == 1
                    else "Actif indéterminé"
                )
                prediction["result"].append(
                    {
                        "from_name": "active_state",
                        "to_name": "image",
                        "type": "choices",
                        "value": {"choices": [state]},
                    }
                )
                # Never prefill review_status: only a human Submit confirms boxes.
            tasks.append(task)
    path = SHARED / EXPOSED / "tasks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    _atomic_json(
        output / "review_plan.json",
        {
            "tasks": len(tasks),
            "task_file": str(path),
            "match_id": "personal_field_session_20261014",
            "luna_answered_frames": sum(bool(t["predictions"]) for t in tasks),
            "human_review_required": True,
            "evaluated_models_used_for_prelabels": False,
            "sample_interval_seconds_approx": 0.5,
        },
    )
    return output, frames, tasks


def project(request, *, create=False):
    inventory = request("/api/projects/?page_size=100")
    matches = [p for p in inventory["results"] if p["title"] == TITLE]
    if inventory.get("next") or len(matches) > 1:
        raise RuntimeError("Ambiguous project inventory")
    schema = (ROOT / "configs/label_studio_field_validation.xml").read_text()
    if not matches:
        if not create:
            raise RuntimeError("Run sync to create the validation review project first")
        return request(
            "/api/projects/",
            "POST",
            {
                "title": TITLE,
                "label_config": schema,
                "description": "One held-out iPhone session, 14 October. Uniform 2 Hz across both complete recordings. All visible balls + active identity. Luna drafts only; submitted human review required. Never use this match for training.",
                "sampling": "Sequential sampling",
                "model_version": "gpt-6-luna",
                "show_collab_predictions": True,
                "enable_empty_annotation": True,
                "show_annotation_history": True,
            },
        )
    if matches[0]["label_config"].strip() != schema.strip():
        raise RuntimeError("Refusing to change an existing review schema")
    return matches[0]


def sync(config_path):
    output, frames, tasks = make_tasks(config_path)
    request, token = api_client()
    proj = project(request, create=True)
    pid = int(proj["id"])
    storages = request(f"/api/storages/localfiles/?project={pid}")
    if not any(s["path"] == str(frames) for s in storages):
        request(
            "/api/storages/localfiles/",
            "POST",
            {
                "project": pid,
                "path": str(frames),
                "title": "Field validation frames",
                "regex_filter": "^$",
                "use_blob_urls": True,
                "recursive_scan": False,
            },
        )
    existing = download_local_export(token=token, project_id=pid)
    desired = {t["data"]["item_id"]: t for t in tasks}
    old = {t["data"]["item_id"]: t for t in existing}
    if (
        len(desired) != len(tasks)
        or len(old) != len(existing)
        or not set(old) <= set(desired)
    ):
        raise RuntimeError("Unexpected task inventory; nothing overwritten")
    missing = [t for t in tasks if t["data"]["item_id"] not in old]
    for i in range(0, len(missing), 100):
        request(f"/api/projects/{pid}/import", "POST", missing[i : i + 100])
    added, preserved = 0, 0
    for item, previous in old.items():
        new = desired[item]
        detail = request(f"/api/tasks/{previous['id']}/")
        if detail["data"].get("review_protocol") != SUBMIT_PROTOCOL:
            # Only add task metadata. Never modify predictions, corrections or
            # autosaved drafts while simplifying the review interface.
            data = {**detail["data"], "review_protocol": SUBMIT_PROTOCOL}
            data["legacy_active_annotation_ids"] = [
                a["id"]
                for a in detail.get("annotations", [])
                if not a.get("was_cancelled")
            ]
            request(f"/api/tasks/{previous['id']}/", "PATCH", {"data": data})
        if not new["predictions"]:
            continue
        if detail.get("annotations") or detail.get("drafts"):
            preserved += 1
            continue
        if detail.get("predictions"):
            continue
        request(
            "/api/predictions/",
            "POST",
            {"task": previous["id"], **new["predictions"][0]},
        )
        request(f"/api/tasks/{previous['id']}/", "PATCH", {"data": new["data"]})
        added += 1
    actual = download_local_export(token=token, project_id=pid)
    if {t["data"]["item_id"] for t in actual} != set(desired):
        raise RuntimeError("Imported inventory differs from frozen sample plan")
    report = {
        "project_id": pid,
        "project_url": f"{BASE}/projects/{pid}/data",
        "tasks": len(actual),
        "new_tasks": len(missing),
        "new_luna_predictions": added,
        "preserved_human_tasks": preserved,
        "luna_answered_frames": sum(bool(t.get("predictions")) for t in actual),
        "human_review_required": True,
    }
    _atomic_json(output / "review_project.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def reviewed(config_path):
    _, output, _ = get_run(config_path)
    request, token = api_client()
    proj = project(request)
    tasks = download_local_export(token=token, project_id=proj["id"])
    expected = json.loads((SHARED / EXPOSED / "tasks.json").read_text())
    if len(tasks) != len(expected) or {t["data"]["item_id"] for t in tasks} != {
        t["data"]["item_id"] for t in expected
    }:
        raise RuntimeError("Review inventory differs from reserved sample plan")
    records, pending, invalid = [], [], []
    expected_by_id = {t["data"]["item_id"]: t["data"] for t in expected}
    fixed_fields = (
        "match_id",
        "video_id",
        "source_video",
        "source_video_sha256",
        "source_frame",
        "timestamp_seconds",
        "width",
        "height",
        "image_sha256",
        "annotation_scope",
        "review_protocol",
        "intended_split",
        "exclude_from_training",
        "image_path",
    )
    for task in tasks:
        try:
            original = expected_by_id[task["data"]["item_id"]]
            if any(task["data"].get(key) != original[key] for key in fixed_fields):
                raise ValueError(
                    "Task provenance differs from the reserved sample plan"
                )
            record = parse_reviewed_task(task)
            if record is None:
                pending.append(task["data"]["item_id"])
            else:
                records.append(record)
        except (ValueError, KeyError, TypeError) as error:
            invalid.append({"item_id": task["data"]["item_id"], "error": str(error)})
    report = {
        "tasks": len(tasks),
        "submitted_valid": len(records),
        "pending": len(pending),
        "invalid": invalid,
        "statuses": dict(Counter(r["status"] for r in records)),
        "ready_for_metrics": not pending and not invalid,
        "project_url": f"{BASE}/projects/{proj['id']}/data",
    }
    _atomic_json(output / "human_review_status.json", report)
    return records, report


def freeze(config_path, destination):
    records, report = reviewed(config_path)
    if not report["ready_for_metrics"]:
        raise RuntimeError(
            f"Human reference incomplete: {report['pending']} pending, {len(report['invalid'])} invalid; no labels frozen"
        )
    if destination.exists():
        raise FileExistsError("Immutable benchmark already exists; use a new version")
    expected = {
        t["data"]["item_id"]: t
        for t in json.loads((SHARED / EXPOSED / "tasks.json").read_text())
    }
    # Preflight all images before creating any benchmark output.
    for r in records:
        image = ROOT / expected[r["item_id"]]["data"]["image_path"]
        if sha256(image) != r["image_sha256"]:
            raise RuntimeError(f"Reviewed image changed: {r['item_id']}")
    (destination / "images").mkdir(parents=True)
    (destination / "labels").mkdir()
    for r in records:
        if r["status"] == "excluded":
            continue
        image = ROOT / expected[r["item_id"]]["data"]["image_path"]
        os.link(image, destination / "images" / (r["item_id"] + ".jpg"))
        lines = "".join(
            _yolo_line(b, r["width"], r["height"]) + "\n" for b in r["boxes_pixels"]
        )
        (destination / "labels" / (r["item_id"] + ".txt")).write_text(lines)
    records.sort(key=lambda r: (r["video_id"], r["source_frame"]))
    _atomic_json(destination / "review_manifest.json", records)
    _atomic_json(
        destination / "build_report.json",
        {
            **report,
            "purpose": "development_validation",
            "exclude_from_training": True,
            "annotation_scope": "all_visible",
            "labels_are_human_reviewed": True,
            "match_groups": 1,
            "sample_interval_seconds_approx": 0.5,
            "temporal_metric_resolution_caveat": "Sub-500ms losses cannot be resolved by this sampled reference",
        },
    )
    (destination / "dataset.yaml").write_text(
        "path: "
        + json.dumps(str(destination.resolve()))
        + "\ntrain: null\nval: images\ntest: null\nnames:\n  0: sports ball\n"
    )
    print("Frozen human field validation:", destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["sync", "status", "freeze"])
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/field_validation_20261021.json"
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/field_validation_v001"
    )
    a = p.parse_args()
    if a.action == "sync":
        sync(a.config)
    elif a.action == "status":
        print(json.dumps(reviewed(a.config)[1], ensure_ascii=False, indent=2))
    else:
        freeze(a.config, a.output)


if __name__ == "__main__":
    main()
