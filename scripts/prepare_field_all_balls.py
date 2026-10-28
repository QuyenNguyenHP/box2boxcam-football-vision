"""Prepare reproducible all-ball inputs, detector proposals and review tasks.

The detector and Luna see identical full-resolution JPEGs. Nothing here writes
to a training/validation/test dataset or approves an annotation automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

from pitchlens.data.label_studio_review import _rectangle, build_tasks
from pitchlens.data.luna_batch import _atomic_json, _yolo_line, prediction_boxes
from pitchlens.data.luna_inline_batch import prepare_inline


def load_run(root, path):
    spec = importlib.util.spec_from_file_location("personal_runner", root / "scripts/annotate_personal_matches_luna.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner.prepare(root, path)


def prepare(configs):
    for index, (_, config, _) in enumerate(configs, 1):
        state = prepare_inline(config)
        print(f"Prepared {index}/{len(configs)}: {config.output_root.name}, {state['planned_cases']} cases, {len(state['shards'])} shard(s)", flush=True)


def teacher(root, raw, output, configs):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    checkpoint = root / raw["teacher_checkpoint"]
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    metadata = {"checkpoint": raw["teacher_checkpoint"], "sha256": digest,
                "imgsz": raw["teacher_imgsz"], "confidence": raw["teacher_confidence"],
                "iou": raw["teacher_iou"], "device": device, "scope": "all_visible",
                "input_policy": "Identical full-resolution JPEGs as Luna; no crop or trajectory filtering",
                "human_review_required": True}
    model = YOLO(str(checkpoint))
    if len(model.names) != 1:
        raise ValueError("Expected a single sports-ball class")
    for index, (_, config, plan) in enumerate(configs, 1):
        destination = config.output_root / "teacher_predictions.json"
        if destination.exists():
            previous = json.loads(destination.read_text())
            if previous["metadata"] != metadata:
                raise FileExistsError("Teacher settings changed: use another run")
            continue
        paths = [str(config.output_root / "frames" / f"{case['item_id']}.jpg") for case in plan["cases"]]
        # A Python list of filenames makes Ultralytics eagerly decode every
        # image as PIL/NumPy, even with stream=True. A text manifest selects
        # the lazy file loader and keeps memory bounded on this laptop.
        input_manifest = config.output_root / "teacher_input_files.txt"
        input_manifest.write_text("\n".join(paths) + "\n")
        records, labels = [], config.output_root / "teacher_labels"
        labels.mkdir(exist_ok=True)
        for case, result in zip(plan["cases"], model.predict(
                source=str(input_manifest), stream=True, device=device, batch=1,
                imgsz=raw["teacher_imgsz"], conf=raw["teacher_confidence"], iou=raw["teacher_iou"],
                classes=[0], verbose=False, half=False), strict=True):
            boxes = []
            for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True):
                x0, y0, x1, y1 = [round(v) for v in box]
                x0, x1 = max(0, x0), min(plan["video"]["width"], x1)
                y0, y1 = max(0, y0), min(plan["video"]["height"], y1)
                if x0 < x1 and y0 < y1:
                    boxes.append({"bbox_current_pixels": [x0, y0, x1, y1], "confidence": score})
            records.append({"item_id": case["item_id"], "source_frame": case["source_frame"], "balls": boxes})
            (labels / f"{case['item_id']}.txt").write_text("".join(
                _yolo_line(b["bbox_current_pixels"], plan["video"]["width"], plan["video"]["height"]) + "\n" for b in boxes))
        _atomic_json(destination, {"metadata": metadata, "predictions": records})
        print(f"Teacher {index}/{len(configs)}: {config.output_root.name}, {sum(len(r['balls']) for r in records)} proposals", flush=True)
    _atomic_json(output / "teacher_manifest.json", metadata)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def compare_proposals(teacher_boxes, luna_boxes, threshold=0.35):
    """Greedy overlap matching for review priority, never an accuracy metric."""
    pairs = sorted(((box_iou(a, b), i, j) for i, a in enumerate(teacher_boxes) for j, b in enumerate(luna_boxes)), reverse=True)
    matched_teacher, matched_luna, matched = set(), set(), []
    for score, i, j in pairs:
        if score < threshold or i in matched_teacher or j in matched_luna:
            continue
        matched_teacher.add(i)
        matched_luna.add(j)
        matched.append({"teacher": i, "luna": j, "iou": score})
    return {"matched": matched, "teacher_only": [i for i in range(len(teacher_boxes)) if i not in matched_teacher],
            "luna_only": [j for j in range(len(luna_boxes)) if j not in matched_luna]}


def review(root, output, configs):
    review_root = output / "review"
    frames = review_root / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    tasks, comparisons, counts = [], [], Counter()
    for video, config, plan in configs:
        detector = json.loads((config.output_root / "teacher_predictions.json").read_text())
        teacher_by_id = {p["item_id"]: p for p in detector["predictions"]}
        luna_file = config.output_root / "predictions.json"
        luna_by_id = {p["item_id"]: p for p in json.loads(luna_file.read_text())["predictions"]} if luna_file.exists() else {}
        frame_records = json.loads((config.output_root / "frames_manifest.json").read_text())["frames"]
        built = build_tasks(plan, luna_by_id)
        for case, task in zip(plan["cases"], built, strict=True):
            item_id = case["item_id"]
            global_id = f"{config.output_root.name}_{item_id}"
            target = frames / f"{global_id}.jpg"
            source = config.output_root / "frames" / f"{item_id}.jpg"
            if not target.exists():
                os.link(source, target)
            teacher_row = teacher_by_id[item_id]
            teacher_boxes = prediction_boxes(teacher_row)
            luna_row = luna_by_id.get(item_id)
            comparison = compare_proposals(teacher_boxes, prediction_boxes(luna_row)) if luna_row else None
            if comparison is None:
                reason = "luna_pending_or_api_error"
            elif comparison["teacher_only"] or comparison["luna_only"]:
                reason = "disagreement"
            elif not teacher_boxes:
                reason = "both_empty_check_for_missed_balls"
            else:
                reason = "overlapping_proposals_still_require_review"
            counts[reason] += 1
            comparisons.append({"item_id": global_id, "match_id": video["match_id"], "reason": reason,
                                "comparison": comparison, "teacher_boxes": len(teacher_boxes),
                                "luna_boxes": len(prediction_boxes(luna_row)) if luna_row else None,
                                "human_review_required": True})
            rectangles = []
            for number, ball in enumerate(teacher_row["balls"]):
                rect = _rectangle(ball["bbox_current_pixels"], plan["video"]["width"], plan["video"]["height"])
                rect["id"] = f"teacher-ball-{number}"
                rectangles.append(rect)
            teacher_prediction = {"model_version": "YOLO26n v003 960px — propositions",
                                  "score": max((b["confidence"] for b in teacher_row["balls"]), default=0), "result": rectangles}
            # Luna is the default proposal once available; detector remains independently selectable.
            task["predictions"] = ([task["predictions"][0]] if luna_row else []) + [teacher_prediction]
            timestamp = frame_records[str(case["source_frame"])]["timestamp_seconds"]
            task["data"].update({"image": f"/data/local-files/?d={target.relative_to(root)}",
                                 "item_id": global_id, "source_video": video["source_video"],
                                 "video_id": config.output_root.name, "width": plan["video"]["width"],
                                 "height": plan["video"]["height"], "image_path": str(target.relative_to(root)),
                                 "intended_split": "train", "role": video["role"],
                                 "match_id": video["match_id"], "timestamp_seconds": timestamp,
                                 "frame_info": f"{global_id} · {timestamp:.3f} s · TOUS les ballons visibles",
                                 "annotation_scope": "all_visible", "review_reason": reason,
                                 "teacher_context": f"YOLO v003 : {len(teacher_boxes)} proposition(s), seuil 10 %. Vérifier aussi les ballons immobiles et sur les bords."})
            tasks.append(task)
    (review_root / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    _atomic_json(review_root / "comparisons.json", {"iou_matching_threshold": 0.35, "not_ground_truth": True, "items": comparisons})
    summary = {"videos": len(configs), "frames": len(tasks), "teacher_boxes": sum(r["teacher_boxes"] for r in comparisons),
               "luna_answered_frames": sum(r["luna_boxes"] is not None for r in comparisons),
               "review_priority": dict(counts), "annotation_scope": "all_visible", "human_review_required": True,
               "dataset_modified": False, "label_studio_document_root": str(root)}
    _atomic_json(review_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "teacher", "review", "all"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw, output, configs = load_run(root, args.config)
    if args.action in {"prepare", "all"}:
        prepare(configs)
    if args.action in {"teacher", "all"}:
        teacher(root, raw, output, configs)
    if args.action in {"review", "all"}:
        review(root, output, configs)


if __name__ == "__main__":
    main()
