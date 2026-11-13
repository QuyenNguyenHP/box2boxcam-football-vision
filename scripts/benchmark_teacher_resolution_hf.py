"""Benchmark one fine-tuned teacher at several inference resolutions on HF Jobs."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import ultralytics
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix, box_iou

DATASET_REPO = os.environ["PITCHLENS_DATASET_REPO"]
MODEL_REPO = os.environ["PITCHLENS_MODEL_REPO"]
WORK_ROOT = Path("/tmp/pitchlens_resolution_benchmark")
DATA_ROOT = WORK_ROOT / "dataset"
RUN_ROOT = WORK_ROOT / "runs"
ARTIFACT_ROOT = WORK_ROOT / "artifacts"
RESOLUTIONS = (640, 960, 1280)
REFERENCE_SIZE = 640


def size_bin(box: torch.Tensor, image_shape: tuple[int, int]) -> str:
    """Classify a target by its longest side, normalized to a 640 px input."""
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    pixels_at_640 = max(width, height) * REFERENCE_SIZE / max(image_shape)
    if pixels_at_640 < 8:
        return "tiny_<8px"
    if pixels_at_640 < 16:
        return "small_8-16px"
    if pixels_at_640 < 32:
        return "medium_16-32px"
    return "large_>=32px"


class BallValidator(DetectionValidator):
    """Add fixed-threshold and target-size recall to Ultralytics validation."""

    def init_metrics(self, model):
        candidates = [
            int(key)
            for key, name in model.names.items()
            if name in {"sports ball", "ball"}
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one ball class, got {model.names}")
        self.ball_class = candidates[0]
        super().init_metrics(model)
        self.names = {0: "ball"}
        self.nc = 1
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(names=self.names)
        self.fixed = Counter()
        self.by_size = Counter()

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

            matched_targets: set[int] = set()
            if actual and predicted:
                overlaps = box_iou(target["bboxes"], filtered["bboxes"])
                candidates = []
                for target_index, prediction_index in torch.nonzero(
                    overlaps >= 0.5, as_tuple=False
                ).tolist():
                    candidates.append(
                        (
                            float(overlaps[target_index, prediction_index]),
                            target_index,
                            prediction_index,
                        )
                    )
                used_predictions: set[int] = set()
                for _, target_index, prediction_index in sorted(
                    candidates, reverse=True
                ):
                    if (
                        target_index not in matched_targets
                        and prediction_index not in used_predictions
                    ):
                        matched_targets.add(target_index)
                        used_predictions.add(prediction_index)

            for target_index, box in enumerate(target["bboxes"]):
                category = size_bin(box, target["imgsz"])
                self.by_size[f"actual::{category}"] += 1
                if target_index in matched_targets:
                    self.by_size[f"tp::{category}"] += 1
        super().update_metrics(predictions, batch)

    def finalize_metrics(self):
        super().finalize_metrics()
        true_positives = self.fixed["tp"]
        false_positives = self.fixed["fp"]
        false_negatives = self.fixed["fn"]
        fixed = dict(self.fixed)
        fixed.update(
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
        categories = (
            "tiny_<8px",
            "small_8-16px",
            "medium_16-32px",
            "large_>=32px",
        )
        recall_by_size = {}
        for category in categories:
            actual = self.by_size[f"actual::{category}"]
            matched = self.by_size[f"tp::{category}"]
            recall_by_size[category] = {
                "targets": actual,
                "matched": matched,
                "recall": matched / actual if actual else None,
            }
        self.metrics.fixed_threshold = fixed
        self.metrics.recall_by_size = recall_by_size


def summarize(result, *, resolution: int, wall_seconds: float) -> dict:
    return {
        "resolution": resolution,
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "fixed": result.fixed_threshold,
        "recall_by_size": result.recall_by_size,
        "speed_ms_per_image": dict(result.speed),
        "wall_seconds": wall_seconds,
    }


def write_csv(results: dict[str, dict]) -> None:
    fields = [
        "resolution",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "fp_on_negatives",
        "tiny_recall",
        "small_recall",
        "medium_recall",
        "large_recall",
        "inference_ms_per_image",
        "wall_seconds",
    ]
    with (ARTIFACT_ROOT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results.values():
            fixed = result["fixed"]
            by_size = result["recall_by_size"]
            writer.writerow(
                {
                    "resolution": result["resolution"],
                    "map50": result["map50"],
                    "map50_95": result["map50_95"],
                    "precision": fixed["precision"],
                    "recall": fixed["recall"],
                    "f1": fixed["f1"],
                    "tp": fixed["tp"],
                    "fp": fixed["fp"],
                    "fn": fixed["fn"],
                    "fp_on_negatives": fixed["fp_on_negatives"],
                    "tiny_recall": by_size["tiny_<8px"]["recall"],
                    "small_recall": by_size["small_8-16px"]["recall"],
                    "medium_recall": by_size["medium_16-32px"]["recall"],
                    "large_recall": by_size["large_>=32px"]["recall"],
                    "inference_ms_per_image": result["speed_ms_per_image"][
                        "inference"
                    ],
                    "wall_seconds": result["wall_seconds"],
                }
            )


def write_plot(results: dict[str, dict]) -> None:
    resolutions = list(results)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    metric_series = {
        "mAP50": [results[key]["map50"] for key in resolutions],
        "mAP50–95": [results[key]["map50_95"] for key in resolutions],
        "Precision": [results[key]["fixed"]["precision"] for key in resolutions],
        "Rappel": [results[key]["fixed"]["recall"] for key in resolutions],
        "F1": [results[key]["fixed"]["f1"] for key in resolutions],
    }
    for label, values in metric_series.items():
        axes[0].plot(resolutions, values, marker="o", label=label)
        for resolution, value in zip(resolutions, values, strict=True):
            axes[0].annotate(f"{value:.1%}", (resolution, value), fontsize=8)
    axes[0].set_ylim(0, 0.85)
    axes[0].set_xlabel("Inference resolution")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall quality: v002 validation")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    size_labels = {
        "tiny_<8px": "Tiny <8 px",
        "small_8-16px": "Small 8–16 px",
        "medium_16-32px": "Medium 16–32 px",
        "large_>=32px": "Large >=32 px",
    }
    for category, label in size_labels.items():
        values = [
            results[key]["recall_by_size"][category]["recall"]
            for key in resolutions
        ]
        axes[1].plot(resolutions, values, marker="o", label=label)
        for resolution, value in zip(resolutions, values, strict=True):
            if value is not None:
                axes[1].annotate(f"{value:.1%}", (resolution, value), fontsize=8)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel("Inference resolution")
    axes[1].set_ylabel("Recall at confidence 0.25 / IoU 0.50")
    axes[1].set_title("Recall by ball size at 640 px")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.suptitle("Same fine-tuned YOLO26x, only the resolution changes")
    figure.tight_layout()
    figure.savefig(ARTIFACT_ROOT / "comparison.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Downloading dataset: {DATASET_REPO}", flush=True)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=DATA_ROOT,
    )
    weights = Path(
        hf_hub_download(repo_id=MODEL_REPO, filename="best.pt")
    )
    dataset_yaml = WORK_ROOT / "dataset.yaml"
    dataset_yaml.write_text(
        "path: "
        + json.dumps(str(DATA_ROOT))
        + "\ntrain: train/images\nval: val/images\ntest: test/images\n"
        + "names:\n  0: ball\n"
    )

    model = YOLO(str(weights))
    results = {}
    for resolution in RESOLUTIONS:
        print(f"Evaluating at {resolution}px", flush=True)
        started_at = time.perf_counter()
        metrics = model.val(
            validator=BallValidator,
            data=str(dataset_yaml),
            split="val",
            imgsz=resolution,
            batch=1,
            device=0,
            workers=4,
            conf=0.001,
            iou=0.7,
            max_det=300,
            plots=True,
            verbose=False,
            project=str(RUN_ROOT),
            name=f"imgsz_{resolution}",
            exist_ok=True,
        )
        results[str(resolution)] = summarize(
            metrics,
            resolution=resolution,
            wall_seconds=time.perf_counter() - started_at,
        )
        print(json.dumps(results[str(resolution)], indent=2), flush=True)

    report = {
        "experiment": "teacher_inference_resolution",
        "dataset_repo": DATASET_REPO,
        "model_repo": MODEL_REPO,
        "weights": "best.pt (YOLO26x fine-tuned at 640 px on v002)",
        "split": "val",
        "test_split_used": False,
        "images": 670,
        "fixed_confidence": 0.25,
        "fixed_matching_iou": 0.5,
        "size_bins_reference_resolution": REFERENCE_SIZE,
        "batch": 1,
        "results": results,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    (ARTIFACT_ROOT / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    write_csv(results)
    write_plot(results)
    HfApi().upload_folder(
        repo_id=MODEL_REPO,
        repo_type="model",
        folder_path=ARTIFACT_ROOT,
        path_in_repo="resolution_benchmark",
        commit_message="Benchmark fine-tuned teacher at 640, 960 and 1280 px",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
