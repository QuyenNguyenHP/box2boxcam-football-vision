"""Create a reviewable YOLO pseudo-label dataset from a source video."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

MANIFEST_FIELDS = [
    "sequence",
    "frame",
    "split",
    "status",
    "source_image",
    "source_label",
    "source_video",
    "source_frame",
    "timestamp_seconds",
    "teacher_model",
    "teacher_confidence",
    "teacher_boxes",
]

PREDICTION_FIELDS = [
    "item_id",
    "teacher_model",
    "source_frame",
    "timestamp_seconds",
    "confidence",
    "x_center",
    "y_center",
    "width",
    "height",
]


def sampled_frame_numbers(
    total_frames: int,
    stride: int,
    max_samples: int | None = None,
) -> list[int]:
    """Return stride-aligned frames, optionally spread across the whole video."""
    if total_frames <= 0:
        raise ValueError("La vidéo ne contient aucune image")
    if stride <= 0:
        raise ValueError("stride doit être strictement positif")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples doit être strictement positif ou None")

    eligible = list(range(0, total_frames, stride))
    if max_samples is None or max_samples >= len(eligible):
        return eligible
    if max_samples == 1:
        return [eligible[len(eligible) // 2]]

    # Select evenly across the eligible sequence rather than taking only the
    # beginning of the video. Rounding can only duplicate positions when more
    # samples than eligible frames are requested, handled above.
    positions = [
        round(index * (len(eligible) - 1) / (max_samples - 1))
        for index in range(max_samples)
    ]
    return [eligible[position] for position in positions]


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError as error:
        raise ValueError(f"Le fichier doit être dans le projet : {path}") from error


def reusable_video_extraction(
    output_root: Path,
    selected_frames: list[int],
) -> list[tuple[int, int, Path, Path]] | None:
    """Reuse a complete extraction left by an interrupted inference.

    A reviewable dataset is only complete once its manifest and report exist.
    Before that point, ``images/`` and ``teacher_labels/`` are generated cache:
    inference can safely overwrite every provisional label and finish the run.
    Any other content is treated as potentially valuable and is never touched.
    """
    if not output_root.exists() or not any(output_root.iterdir()):
        return None

    allowed_entries = {"images", "teacher_labels"}
    actual_entries = {path.name for path in output_root.iterdir()}
    unexpected = actual_entries - allowed_entries
    if unexpected:
        raise FileExistsError(
            "Sortie déjà présente avec des fichiers non provisoires, aucun "
            f"écrasement automatique : {output_root} ({sorted(unexpected)})"
        )

    images_dir = output_root / "images"
    labels_dir = output_root / "teacher_labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileExistsError(
            f"Extraction interrompue et incomplète : {output_root}. "
            "Les dossiers images/ et teacher_labels/ sont tous deux requis."
        )

    expected_images = {f"frame_{frame:08d}.jpg" for frame in selected_frames}
    expected_labels = {f"frame_{frame:08d}.txt" for frame in selected_frames}
    actual_images = {path.name for path in images_dir.iterdir() if path.is_file()}
    actual_labels = {path.name for path in labels_dir.iterdir() if path.is_file()}
    if actual_images != expected_images or actual_labels != expected_labels:
        raise FileExistsError(
            "Extraction interrompue mais non réutilisable : "
            f"{len(actual_images)}/{len(expected_images)} images et "
            f"{len(actual_labels)}/{len(expected_labels)} labels dans {output_root}."
        )

    return [
        (
            sample_index,
            source_frame,
            images_dir / f"frame_{source_frame:08d}.jpg",
            labels_dir / f"frame_{source_frame:08d}.txt",
        )
        for sample_index, source_frame in enumerate(selected_frames)
    ]


def prepare_video_pseudo_labels(
    *,
    project_root: Path,
    video_path: Path,
    output_root: Path,
    teacher_model: str | Path,
    frame_stride: int = 4,
    max_samples: int | None = 100,
    image_size: int = 960,
    confidence: float = 0.10,
    max_detections: int = 5,
    source_class_id: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Extract video frames, pseudo-label them and create a review manifest.

    ``source_class_id=32`` is COCO's ``sports ball`` class. Every retained box
    is remapped to PitchLens class 0 in the generated YOLO labels.
    """
    import cv2
    from ultralytics import YOLO

    project_root = project_root.resolve()
    video_path = video_path.resolve()
    output_root = output_root.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not 0 < confidence <= 1:
        raise ValueError("confidence doit être dans ]0, 1]")
    if image_size <= 0 or max_detections <= 0:
        raise ValueError("image_size et max_detections doivent être positifs")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV ne peut pas ouvrir : {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    selected = sampled_frame_numbers(total_frames, frame_stride, max_samples)
    selected_set = set(selected)

    extracted = reusable_video_extraction(output_root, selected)
    if extracted is None:
        images_dir = output_root / "images"
        labels_dir = output_root / "teacher_labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir()

        extracted = []
        source_frame = 0
        sample_index = 0
        try:
            while sample_index < len(selected):
                ok, image = capture.read()
                if not ok:
                    break
                if source_frame in selected_set:
                    image_path = images_dir / f"frame_{source_frame:08d}.jpg"
                    label_path = labels_dir / f"frame_{source_frame:08d}.txt"
                    if not cv2.imwrite(
                        str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    ):
                        raise RuntimeError(f"Échec d'écriture : {image_path}")
                    label_path.write_text("")
                    extracted.append(
                        (sample_index, source_frame, image_path, label_path)
                    )
                    sample_index += 1
                source_frame += 1
        finally:
            capture.release()
    else:
        capture.release()

    if len(extracted) != len(selected):
        raise RuntimeError(
            f"Extraction incomplète : {len(extracted)} / {len(selected)} images"
        )

    started_at = time.perf_counter()
    model = YOLO(str(teacher_model))
    class_name = model.names[source_class_id]
    if str(class_name).lower() not in {"sports ball", "ball"}:
        raise ValueError(
            f"La classe {source_class_id} du teacher est {class_name!r}, "
            "pas 'sports ball'. Pour un modèle PitchLens à une classe, utiliser 0."
        )

    image_paths = [str(row[2]) for row in extracted]
    results = model.predict(
        source=image_paths,
        stream=True,
        batch=1,
        imgsz=image_size,
        conf=confidence,
        max_det=max_detections,
        classes=[source_class_id],
        device=device,
        verbose=False,
    )

    manifest_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    sequence = video_path.stem.replace(" ", "_")
    detected_frames = 0
    total_boxes = 0

    for extracted_row, result in zip(extracted, results, strict=True):
        sample_index, source_frame, image_path, label_path = extracted_row
        boxes = result.boxes
        label_lines: list[str] = []
        if boxes is not None and len(boxes):
            detected_frames += 1
            total_boxes += len(boxes)
            xywhn = boxes.xywhn.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            for values, box_confidence in zip(xywhn, confidences, strict=True):
                x_center, y_center, box_width, box_height = values
                label_lines.append(
                    "0 " + " ".join(f"{value:.8f}" for value in values)
                )
                prediction_rows.append(
                    {
                        "item_id": f"{sequence}_{sample_index:06d}",
                        "teacher_model": Path(str(teacher_model)).name,
                        "source_frame": source_frame,
                        "timestamp_seconds": f"{source_frame / fps:.6f}",
                        "confidence": f"{box_confidence:.8f}",
                        "x_center": f"{x_center:.8f}",
                        "y_center": f"{y_center:.8f}",
                        "width": f"{box_width:.8f}",
                        "height": f"{box_height:.8f}",
                    }
                )
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        manifest_rows.append(
            {
                "sequence": sequence,
                # Consecutive review index: ReviewStore can show neighbours even
                # though the original video frame numbers advance by four.
                "frame": sample_index,
                "split": "train",
                "status": "teacher_positive" if label_lines else "teacher_negative",
                "source_image": _relative_to_project(image_path, project_root),
                "source_label": _relative_to_project(label_path, project_root),
                "source_video": _relative_to_project(video_path, project_root),
                "source_frame": source_frame,
                "timestamp_seconds": f"{source_frame / fps:.6f}",
                "teacher_model": Path(str(teacher_model)).name,
                "teacher_confidence": f"{confidence:.4f}",
                "teacher_boxes": len(label_lines),
            }
        )

    elapsed = time.perf_counter() - started_at
    _write_csv_atomic(output_root / "manifest.csv", manifest_rows, MANIFEST_FIELDS)
    _write_csv_atomic(
        output_root / "teacher_predictions.csv",
        prediction_rows,
        PREDICTION_FIELDS,
    )

    summary = {
        "dataset_id": output_root.name,
        "source_video": _relative_to_project(video_path, project_root),
        "source_video_frames": total_frames,
        "source_video_fps": fps,
        "source_video_width": width,
        "source_video_height": height,
        "frame_stride": frame_stride,
        "max_samples": max_samples,
        "sampled_images": len(manifest_rows),
        "first_source_frame": selected[0],
        "last_source_frame": selected[-1],
        "teacher_model": str(teacher_model),
        "teacher_source_class_id": source_class_id,
        "teacher_source_class_name": str(class_name),
        "teacher_image_size": image_size,
        "teacher_confidence": confidence,
        "teacher_max_detections": max_detections,
        "device": device,
        "frames_with_teacher_box": detected_frames,
        "frames_without_teacher_box": len(manifest_rows) - detected_frames,
        "teacher_boxes": total_boxes,
        "inference_seconds": elapsed,
        "inference_images_per_second": len(manifest_rows) / elapsed,
        "review_policy": (
            "Teacher boxes are proposals only. Every image must be reviewed; "
            "human decisions replace the pseudo-label when v003 is built."
        ),
    }
    (output_root / "pseudo_label_report.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def expected_sample_count(total_frames: int, stride: int) -> int:
    """Count frames selected by ``range(0, total_frames, stride)``."""
    if total_frames <= 0 or stride <= 0:
        raise ValueError("total_frames et stride doivent être positifs")
    return math.ceil(total_frames / stride)


def box_iou_xywh(first: dict[str, Any], second: dict[str, Any]) -> float:
    """IoU between two normalized center-format boxes."""
    first_x = float(first["x_center"])
    first_y = float(first["y_center"])
    first_width = float(first["width"])
    first_height = float(first["height"])
    second_x = float(second["x_center"])
    second_y = float(second["y_center"])
    second_width = float(second["width"])
    second_height = float(second["height"])
    first_edges = (
        first_x - first_width / 2,
        first_y - first_height / 2,
        first_x + first_width / 2,
        first_y + first_height / 2,
    )
    second_edges = (
        second_x - second_width / 2,
        second_y - second_height / 2,
        second_x + second_width / 2,
        second_y + second_height / 2,
    )
    intersection_width = max(
        0.0, min(first_edges[2], second_edges[2]) - max(first_edges[0], second_edges[0])
    )
    intersection_height = max(
        0.0, min(first_edges[3], second_edges[3]) - max(first_edges[1], second_edges[1])
    )
    intersection = intersection_width * intersection_height
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_teacher_boxes(
    predictions: list[dict[str, Any]],
    *,
    teacher_priority: list[str],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Keep priority-teacher boxes and add only non-overlapping complements."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold doit être dans ]0, 1]")
    priority = {name: index for index, name in enumerate(teacher_priority)}
    ordered = sorted(
        predictions,
        key=lambda row: (
            priority.get(str(row["teacher_model"]), len(priority)),
            -float(row["confidence"]),
        ),
    )
    kept: list[dict[str, Any]] = []
    for prediction in ordered:
        if all(box_iou_xywh(prediction, previous) < iou_threshold for previous in kept):
            kept.append(prediction)
    return kept


def combine_teacher_datasets(
    *,
    project_root: Path,
    source_roots: list[Path],
    output_root: Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Combine complementary teacher proposals without copying source images."""
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    source_roots = [path.resolve() for path in source_roots]
    if len(source_roots) < 2:
        raise ValueError("Au moins deux datasets teachers sont nécessaires")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Sortie déjà présente, aucun écrasement automatique : {output_root}"
        )

    manifests: list[list[dict[str, str]]] = []
    predictions_by_item: dict[str, list[dict[str, Any]]] = {}
    teacher_names: list[str] = []
    for source_root in source_roots:
        with (source_root / "manifest.csv").open(newline="") as stream:
            manifest = list(csv.DictReader(stream))
        with (source_root / "teacher_predictions.csv").open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        report = json.loads((source_root / "pseudo_label_report.json").read_text())
        # Two fine-tuned checkpoints are commonly both named ``best.pt``.
        # The dataset id stays unique and therefore preserves teacher priority
        # and per-teacher contribution counts in that situation.
        teacher_name = str(report.get("dataset_id", source_root.name))
        teacher_names.append(teacher_name)
        manifests.append(manifest)
        for prediction in predictions:
            prediction = {**prediction, "teacher_model": teacher_name}
            predictions_by_item.setdefault(prediction["item_id"], []).append(prediction)

    reference_ids = [
        f"{row['sequence']}_{int(row['frame']):06d}" for row in manifests[0]
    ]
    for manifest in manifests[1:]:
        item_ids = [f"{row['sequence']}_{int(row['frame']):06d}" for row in manifest]
        if item_ids != reference_ids:
            raise ValueError("Les datasets teachers ne couvrent pas les mêmes images")

    labels_dir = output_root / "teacher_labels"
    labels_dir.mkdir(parents=True)
    output_manifest: list[dict[str, Any]] = []
    output_predictions: list[dict[str, Any]] = []
    per_teacher_kept = {name: 0 for name in teacher_names}
    for row, item_id in zip(manifests[0], reference_ids, strict=True):
        kept = deduplicate_teacher_boxes(
            predictions_by_item.get(item_id, []),
            teacher_priority=teacher_names,
            iou_threshold=iou_threshold,
        )
        label_path = labels_dir / f"frame_{int(row['source_frame']):08d}.txt"
        lines = []
        for prediction in kept:
            values = [
                float(prediction[key])
                for key in ("x_center", "y_center", "width", "height")
            ]
            lines.append("0 " + " ".join(f"{value:.8f}" for value in values))
            output_predictions.append(
                {field: prediction.get(field, "") for field in PREDICTION_FIELDS}
            )
            per_teacher_kept[str(prediction["teacher_model"])] += 1
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        output_manifest.append(
            {
                **row,
                "status": "teacher_positive" if kept else "teacher_negative",
                "source_label": _relative_to_project(label_path, project_root),
                "teacher_model": "+".join(teacher_names),
                "teacher_confidence": "mixed; see teacher_predictions.csv",
                "teacher_boxes": len(kept),
            }
        )

    _write_csv_atomic(output_root / "manifest.csv", output_manifest, MANIFEST_FIELDS)
    _write_csv_atomic(
        output_root / "teacher_predictions.csv",
        output_predictions,
        PREDICTION_FIELDS,
    )
    report = {
        "dataset_id": output_root.name,
        "source_datasets": [path.name for path in source_roots],
        "teacher_priority": teacher_names,
        "iou_deduplication_threshold": iou_threshold,
        "sampled_images": len(output_manifest),
        "frames_with_teacher_box": sum(
            int(row["teacher_boxes"]) > 0 for row in output_manifest
        ),
        "frames_without_teacher_box": sum(
            int(row["teacher_boxes"]) == 0 for row in output_manifest
        ),
        "teacher_boxes": len(output_predictions),
        "kept_boxes_by_teacher": per_teacher_kept,
        "images_are_referenced_not_copied": True,
        "review_policy": (
            "The first teacher has priority for overlapping boxes; complementary "
            "boxes from later teachers are retained. Every image requires review."
        ),
    }
    (output_root / "pseudo_label_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
