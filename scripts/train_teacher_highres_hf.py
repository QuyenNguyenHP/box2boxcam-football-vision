"""Continue fine-tuning the box2boxcam teacher at 960 px on HF Jobs."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
import shutil
import time
from pathlib import Path

import torch
import ultralytics
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from ultralytics import YOLO

DATASET_REPO = os.environ["BOX2BOXCAM_DATASET_REPO"]
MODEL_REPO = os.environ["BOX2BOXCAM_MODEL_REPO"]
WORK_ROOT = Path("/tmp/box2boxcam_teacher_960")
DATA_ROOT = WORK_ROOT / "dataset"
RUN_ROOT = WORK_ROOT / "runs"
ARTIFACT_ROOT = WORK_ROOT / "artifacts"
TRAIN_IMAGE_SIZE = 960
EPOCHS = 12


def load_benchmark_helpers():
    """Load the exact same validator used by the inference-resolution benchmark."""
    helper_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="benchmark_teacher_resolution_hf.py",
    )
    spec = importlib.util.spec_from_file_location(
        "box2boxcam_resolution_helpers", helper_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import benchmark helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_with_oom_fallback(base_weights: Path, dataset_yaml: Path):
    """Prefer batch 2, then retry at batch 1 only for a CUDA OOM."""
    common = {
        "data": str(dataset_yaml),
        "epochs": EPOCHS,
        "imgsz": TRAIN_IMAGE_SIZE,
        "device": 0,
        "workers": 4,
        "seed": 42,
        "deterministic": True,
        "optimizer": "AdamW",
        # This is a continuation from a trained detector, so use a lower LR
        # than the initial 640 px fine-tuning (0.001).
        "lr0": 0.0003,
        "amp": True,
        "cache": False,
        "mosaic": 0.0,
        "mixup": 0.0,
        "close_mosaic": 0,
        "patience": 5,
        "val": True,
        "plots": True,
        "save": True,
        "save_period": -1,
        "project": str(RUN_ROOT),
        "exist_ok": True,
    }
    attempts = []
    for batch in (2, 1):
        config = {**common, "batch": batch, "name": f"train_b{batch}"}
        print(f"Training at 960 px with batch={batch}", flush=True)
        started_at = time.perf_counter()
        try:
            model = YOLO(str(base_weights))
            result = model.train(**config)
            return result, config, time.perf_counter() - started_at, attempts
        except RuntimeError as error:
            elapsed = time.perf_counter() - started_at
            attempts.append(
                {"batch": batch, "seconds": elapsed, "error": str(error)}
            )
            if batch == 1 or "out of memory" not in str(error).lower():
                raise
            print("Batch 2 exceeded CUDA memory; retrying with batch 1", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError("No training attempt completed")


def copy_training_artifacts(train_dir: Path) -> None:
    destination = ARTIFACT_ROOT / "train"
    destination.mkdir(parents=True, exist_ok=True)
    patterns = (
        "results.csv",
        "results.png",
        "confusion_matrix*.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "PR_curve.png",
        "labels*.jpg",
        "train_batch*.jpg",
        "val_batch*.jpg",
        "args.yaml",
    )
    for pattern in patterns:
        for source in train_dir.glob(pattern):
            shutil.copy2(source, destination / source.name)


def write_readme(report: dict) -> None:
    result = report["validation_after"]["960"]
    fixed = result["fixed"]
    (ARTIFACT_ROOT / "README.md").write_text(
        f"""# box2boxcam YOLO26x teacher — 960 px continuation

This annotation teacher continues from the v002 640 px `best.pt` and is
fine-tuned at 960 px. It is an offline dataset-construction model, not the
mobile deployment model.

- training resolution: 960 px
- epochs requested: {EPOCHS}
- mAP50 at 960 px: {result['map50']:.4f}
- mAP50-95 at 960 px: {result['map50_95']:.4f}
- precision at confidence 0.25: {fixed['precision']:.4f}
- recall at confidence 0.25: {fixed['recall']:.4f}
- F1 at confidence 0.25: {fixed['f1']:.4f}
- false positives on negative images: {fixed['fp_on_negatives']}

The test split was not used. See `training_report.json`, `train/`, and
`resolution_after/` for the complete experiment.
"""
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Downloading dataset: {DATASET_REPO}", flush=True)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=DATA_ROOT,
    )
    base_weights = Path(
        hf_hub_download(repo_id=MODEL_REPO, filename="best.pt")
    )
    dataset_yaml = WORK_ROOT / "dataset.yaml"
    dataset_yaml.write_text(
        "path: "
        + json.dumps(str(DATA_ROOT))
        + "\ntrain: train/images\nval: val/images\ntest: test/images\n"
        + "names:\n  0: ball\n"
    )

    training_result, training_config, training_seconds, failed_attempts = (
        train_with_oom_fallback(base_weights, dataset_yaml)
    )
    train_dir = Path(training_result.save_dir)
    best_weights = train_dir / "weights/best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(best_weights)

    helpers = load_benchmark_helpers()
    model = YOLO(str(best_weights))
    validation_results = {}
    validation_root = WORK_ROOT / "validation"
    for resolution in (640, 960, 1280):
        print(f"Validating high-resolution teacher at {resolution}px", flush=True)
        started_at = time.perf_counter()
        metrics = model.val(
            validator=helpers.BallValidator,
            data=str(dataset_yaml),
            split="val",
            imgsz=resolution,
            batch=1,
            device=0,
            workers=4,
            conf=0.001,
            iou=0.7,
            max_det=300,
            plots=False,
            verbose=False,
            project=str(validation_root),
            name=f"imgsz_{resolution}",
            exist_ok=True,
        )
        validation_results[str(resolution)] = helpers.summarize(
            metrics,
            resolution=resolution,
            wall_seconds=time.perf_counter() - started_at,
        )

    resolution_root = ARTIFACT_ROOT / "resolution_after"
    resolution_root.mkdir()
    resolution_report = {
        "experiment": "highres_teacher_inference_resolution",
        "weights": "highres_960/best.pt",
        "training_resolution": TRAIN_IMAGE_SIZE,
        "split": "val",
        "test_split_used": False,
        "images": 670,
        "fixed_confidence": 0.25,
        "fixed_matching_iou": 0.5,
        "batch": 1,
        "results": validation_results,
    }
    (resolution_root / "comparison.json").write_text(
        json.dumps(resolution_report, indent=2) + "\n"
    )
    helpers.ARTIFACT_ROOT = resolution_root
    helpers.write_csv(validation_results)
    helpers.write_plot(validation_results)

    report = {
        "dataset_repo": DATASET_REPO,
        "model_repo": MODEL_REPO,
        "base_model": "best.pt (YOLO26x v002 fine-tuned at 640 px)",
        "training": training_config,
        "training_seconds": training_seconds,
        "failed_attempts": failed_attempts,
        "validation_after": validation_results,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "test_split_used": False,
    }
    (ARTIFACT_ROOT / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    shutil.copy2(best_weights, ARTIFACT_ROOT / "best.pt")
    copy_training_artifacts(train_dir)
    write_readme(report)

    print(f"Uploading high-resolution artifacts to {MODEL_REPO}/highres_960")
    HfApi().upload_folder(
        repo_id=MODEL_REPO,
        repo_type="model",
        folder_path=ARTIFACT_ROOT,
        path_in_repo="highres_960",
        commit_message="Fine-tune YOLO26x teacher at 960 px",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
