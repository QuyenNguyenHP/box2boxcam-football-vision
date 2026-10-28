"""Fine-tune and evaluate the large PitchLens annotation teacher on HF Jobs."""

from __future__ import annotations

import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import torch
import ultralytics
from huggingface_hub import HfApi, snapshot_download
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix

DATASET_REPO = os.environ["PITCHLENS_DATASET_REPO"]
MODEL_REPO = os.environ["PITCHLENS_MODEL_REPO"]
WORK_ROOT = Path("/tmp/pitchlens_teacher")
DATA_ROOT = WORK_ROOT / "dataset"
RUN_ROOT = WORK_ROOT / "run"
ARTIFACT_ROOT = WORK_ROOT / "artifacts"


class BallValidator(DetectionValidator):
    """Evaluate only sports-ball predictions against the one-class labels."""

    def init_metrics(self, model):
        names = model.names
        candidates = [
            int(key) for key, name in names.items() if name in {"sports ball", "ball"}
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one ball class, got {names}")
        self.ball_class = candidates[0]
        super().init_metrics(model)
        self.names = {0: "ball"}
        self.nc = 1
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(names=self.names)
        self.fixed = Counter()

    def postprocess(self, predictions):
        results = super().postprocess(predictions)
        for index, prediction in enumerate(results):
            selected = prediction["cls"] == self.ball_class
            results[index] = {
                key: value[selected] for key, value in prediction.items()
            }
            results[index]["cls"].zero_()
        return results

    def update_metrics(self, predictions, batch):
        for index, prediction in enumerate(predictions):
            target = self._prepare_batch(index, batch)
            selected = prediction["conf"] >= 0.25
            filtered = {key: value[selected] for key, value in prediction.items()}
            true_positives = int(
                self._process_batch(filtered, target)["tp"][:, 0].sum()
            )
            predicted = len(filtered["cls"])
            actual = len(target["cls"])
            self.fixed.update(
                tp=true_positives,
                fp=predicted - true_positives,
                fn=actual - true_positives,
                images=1,
            )
            if actual == 0:
                self.fixed.update(negative_images=1, fp_on_negatives=predicted)
        super().update_metrics(predictions, batch)

    def finalize_metrics(self):
        super().finalize_metrics()
        counts = dict(self.fixed)
        true_positives = self.fixed["tp"]
        false_positives = self.fixed["fp"]
        false_negatives = self.fixed["fn"]
        counts.update(
            precision=(
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives
                else 0.0
            ),
            recall=(
                true_positives / (true_positives + false_negatives)
                if true_positives + false_negatives
                else 0.0
            ),
            f1=(
                2
                * true_positives
                / (2 * true_positives + false_positives + false_negatives)
                if 2 * true_positives + false_positives + false_negatives
                else 0.0
            ),
            confidence=0.25,
            matching_iou=0.5,
        )
        self.metrics.fixed_threshold = counts


def evaluation_summary(result) -> dict:
    return {
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "fixed": result.fixed_threshold,
        "speed_ms_per_image": dict(result.speed),
    }


def write_model_card(report: dict) -> None:
    after = report["validation_after"]
    fixed = after["fixed"]
    card = f"""---
license: cc-by-nc-4.0
library_name: ultralytics
tags:
- object-detection
- football
- ball-detection
- pitchlens
---

# PitchLens YOLO26x v002 teacher

Non-commercial annotation teacher fine-tuned on PitchLens Infactory v002. It
is intended for dataset construction, not mobile deployment.

## Controlled validation at 640 px

- mAP50: {after['map50']:.4f}
- mAP50-95: {after['map50_95']:.4f}
- precision at confidence 0.25: {fixed['precision']:.4f}
- recall at confidence 0.25: {fixed['recall']:.4f}
- F1 at confidence 0.25: {fixed['f1']:.4f}
- false positives on negative images: {fixed['fp_on_negatives']}

See `training_report.json` and `train/results.csv` for complete details.
"""
    (ARTIFACT_ROOT / "README.md").write_text(card)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Downloading dataset: {DATASET_REPO}", flush=True)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=DATA_ROOT,
    )
    dataset_yaml = WORK_ROOT / "dataset.yaml"
    dataset_yaml.write_text(
        "path: "
        + json.dumps(str(DATA_ROOT))
        + "\ntrain: train/images\nval: val/images\ntest: test/images\n"
        + "names:\n  0: ball\n"
    )

    training_config = {
        "data": str(dataset_yaml),
        "epochs": 10,
        "imgsz": 640,
        "batch": 4,
        "device": 0,
        "workers": 4,
        "seed": 42,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "amp": True,
        "cache": False,
        "mosaic": 0.0,
        "mixup": 0.0,
        "close_mosaic": 0,
        "patience": 4,
        "val": True,
        "plots": True,
        "save": True,
        # Keep only best.pt and last.pt during the ephemeral Job. The final
        # upload exposes one top-level best.pt and avoids multi-gigabyte copies.
        "save_period": -1,
        "project": str(RUN_ROOT),
        "name": "train",
        "exist_ok": True,
    }
    started_at = time.perf_counter()
    model = YOLO("yolo26x.pt")
    training_result = model.train(**training_config)
    training_seconds = time.perf_counter() - started_at
    best_weights = Path(training_result.save_dir) / "weights/best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(best_weights)

    best_model = YOLO(str(best_weights))
    validation = best_model.val(
        validator=BallValidator,
        data=str(dataset_yaml),
        split="val",
        imgsz=640,
        batch=4,
        device=0,
        workers=4,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=True,
        verbose=False,
        project=str(RUN_ROOT),
        name="validation_after",
        exist_ok=True,
    )
    report = {
        "dataset_repo": DATASET_REPO,
        "model_repo": MODEL_REPO,
        "base_model": "yolo26x.pt",
        "training": training_config,
        "training_seconds": training_seconds,
        "validation_after": evaluation_summary(validation),
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
    write_model_card(report)

    # Keep the full Ultralytics run structure plus a convenient top-level copy.
    import shutil

    shutil.copytree(RUN_ROOT / "train", ARTIFACT_ROOT / "train")
    shutil.copy2(best_weights, ARTIFACT_ROOT / "best.pt")
    print(f"Uploading model artifacts to {MODEL_REPO}", flush=True)
    HfApi().upload_folder(
        repo_id=MODEL_REPO,
        repo_type="model",
        folder_path=ARTIFACT_ROOT,
        commit_message="Train YOLO26x teacher on PitchLens v002",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
