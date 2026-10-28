"""Run the 960 px video-pilot teacher inference on a Hugging Face GPU."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO

MODEL_REPO = os.environ["PITCHLENS_MODEL_REPO"]
INPUT_ROOT = Path(os.environ.get("PITCHLENS_INPUT_ROOT", "/input"))
OUTPUT_ROOT = Path("/tmp/pitchlens_video_pilot")
REMOTE_OUTPUT = "video_pilots/highres960_pilot100"
DATASET_ID = "video_teacher_inference_input_yt_pilot100_yolo26x_highres960"
LOCAL_DATASET_ROOT = f"data/interim/{DATASET_ID}"
SOURCE_VIDEO = "data/external/videos/inference_input_yt.mp4"
SEQUENCE = "inference_input_yt"
FPS = 29.97002997002997
IMAGE_SIZE = 960
CONFIDENCE = 0.10
MAX_DETECTIONS = 5

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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This inference job requires a CUDA GPU")
    image_paths = sorted(INPUT_ROOT.glob("frame_*.jpg"))
    if len(image_paths) != 100:
        raise ValueError(f"Expected 100 pilot images, found {len(image_paths)}")

    labels_dir = OUTPUT_ROOT / "teacher_labels"
    labels_dir.mkdir(parents=True)
    weights = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename="highres_960/best.pt",
        )
    )
    model = YOLO(str(weights))
    started_at = time.perf_counter()
    results = model.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        batch=4,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        max_det=MAX_DETECTIONS,
        classes=[0],
        device=0,
        verbose=False,
    )

    manifest_rows = []
    prediction_rows = []
    detected_frames = 0
    total_boxes = 0
    teacher_name = "yolo26x_v002_highres960_best.pt"
    for sample_index, (image_path, result) in enumerate(
        zip(image_paths, results, strict=True)
    ):
        source_frame = int(image_path.stem.removeprefix("frame_"))
        label_path = labels_dir / f"frame_{source_frame:08d}.txt"
        lines = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            detected_frames += 1
            total_boxes += len(boxes)
            for values, confidence in zip(
                boxes.xywhn.cpu().tolist(), boxes.conf.cpu().tolist(), strict=True
            ):
                lines.append("0 " + " ".join(f"{value:.8f}" for value in values))
                prediction_rows.append(
                    {
                        "item_id": f"{SEQUENCE}_{sample_index:06d}",
                        "teacher_model": teacher_name,
                        "source_frame": source_frame,
                        "timestamp_seconds": f"{source_frame / FPS:.6f}",
                        "confidence": f"{confidence:.8f}",
                        "x_center": f"{values[0]:.8f}",
                        "y_center": f"{values[1]:.8f}",
                        "width": f"{values[2]:.8f}",
                        "height": f"{values[3]:.8f}",
                    }
                )
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        manifest_rows.append(
            {
                "sequence": SEQUENCE,
                "frame": sample_index,
                "split": "train",
                "status": "teacher_positive" if lines else "teacher_negative",
                "source_image": f"{LOCAL_DATASET_ROOT}/images/{image_path.name}",
                "source_label": (
                    f"{LOCAL_DATASET_ROOT}/teacher_labels/{label_path.name}"
                ),
                "source_video": SOURCE_VIDEO,
                "source_frame": source_frame,
                "timestamp_seconds": f"{source_frame / FPS:.6f}",
                "teacher_model": teacher_name,
                "teacher_confidence": f"{CONFIDENCE:.4f}",
                "teacher_boxes": len(lines),
            }
        )

    elapsed = time.perf_counter() - started_at
    write_csv(OUTPUT_ROOT / "manifest.csv", manifest_rows, MANIFEST_FIELDS)
    write_csv(
        OUTPUT_ROOT / "teacher_predictions.csv",
        prediction_rows,
        PREDICTION_FIELDS,
    )
    report = {
        "dataset_id": DATASET_ID,
        "source_video": SOURCE_VIDEO,
        "source_video_frames": 4926,
        "source_video_fps": FPS,
        "source_video_width": 1920,
        "source_video_height": 1080,
        "frame_stride": 4,
        "max_samples": 100,
        "sampled_images": len(manifest_rows),
        "first_source_frame": int(image_paths[0].stem.removeprefix("frame_")),
        "last_source_frame": int(image_paths[-1].stem.removeprefix("frame_")),
        "teacher_model": "highres_960/best.pt",
        "teacher_source_class_id": 0,
        "teacher_source_class_name": "ball",
        "teacher_image_size": IMAGE_SIZE,
        "teacher_confidence": CONFIDENCE,
        "teacher_max_detections": MAX_DETECTIONS,
        "device": torch.cuda.get_device_name(0),
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
    (OUTPUT_ROOT / "pseudo_label_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    HfApi().upload_folder(
        repo_id=MODEL_REPO,
        repo_type="model",
        folder_path=OUTPUT_ROOT,
        path_in_repo=REMOTE_OUTPUT,
        commit_message="Run high-resolution teacher on 100-frame video pilot",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
