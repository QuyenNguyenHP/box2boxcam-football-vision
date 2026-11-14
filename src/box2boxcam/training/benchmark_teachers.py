"""Benchmark annotation-teacher candidates on the locked v002 validation split."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from box2boxcam.training.diagnostics import quiet_ultralytics_progress
from box2boxcam.training.evaluate import BallValidator


def _summary(result: Any, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "fixed": result.fixed_threshold,
        "speed_ms_per_image": dict(result.speed),
        "wall_seconds": elapsed_seconds,
    }


def evaluate_teacher(
    *,
    weights: Path,
    dataset_yaml: Path,
    output_root: Path,
    name: str,
    image_size: int = 640,
    batch: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate one model with the same labels, resolution and thresholds."""
    result_path = output_root / f"{name}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    model = YOLO(str(weights))
    started_at = time.perf_counter()
    with quiet_ultralytics_progress():
        metrics = model.val(
            validator=BallValidator,
            data=str(dataset_yaml),
            split="val",
            imgsz=image_size,
            batch=batch,
            device=device,
            workers=0,
            conf=0.001,
            iou=0.7,
            max_det=300,
            plots=True,
            verbose=False,
            project=str(output_root),
            name=name,
            exist_ok=True,
        )
    summary = _summary(metrics, time.perf_counter() - started_at)
    summary.update(
        {
            "name": name,
            "weights": str(weights.resolve()),
            "dataset_yaml": str(dataset_yaml.resolve()),
            "split": "val",
            "image_size": image_size,
            "batch": batch,
            "device": device,
            "images": 670,
            "test_split_used": False,
        }
    )
    result_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_comparison(
    output_root: Path, results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparison = {
        "protocol": {
            "dataset": "box2boxcam_infactory_v002",
            "split": "val",
            "images": 670,
            "image_size": 640,
            "fixed_confidence": 0.25,
            "fixed_matching_iou": 0.5,
            "model_selection_on_test": False,
        },
        "models": results,
    }
    (output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )

    fields = [
        "model",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "fp_on_negatives",
        "inference_ms_per_image",
        "device",
        "wall_seconds",
    ]
    with (output_root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_name, result in results.items():
            fixed = result["fixed"]
            writer.writerow(
                {
                    "model": model_name,
                    "map50": result["map50"],
                    "map50_95": result["map50_95"],
                    "precision": fixed["precision"],
                    "recall": fixed["recall"],
                    "f1": fixed["f1"],
                    "tp": fixed["tp"],
                    "fp": fixed["fp"],
                    "fn": fixed["fn"],
                    "fp_on_negatives": fixed["fp_on_negatives"],
                    "inference_ms_per_image": result["speed_ms_per_image"][
                        "inference"
                    ],
                    "device": result["device"],
                    "wall_seconds": result["wall_seconds"],
                }
            )

    import matplotlib.pyplot as plt

    display_names = {
        "yolo26n_v002_finetuned": "Nano fine-tuned",
        "yolo26x_coco_raw": "X raw COCO",
        "yolo26x_v002_finetuned": "X fine-tuned",
    }
    model_names = list(results)
    accuracy_names = ["mAP50", "mAP50–95", "Precision", "Recall", "F1"]
    accuracy_values = {
        name: [
            results[name]["map50"],
            results[name]["map50_95"],
            results[name]["fixed"]["precision"],
            results[name]["fixed"]["recall"],
            results[name]["fixed"]["f1"],
        ]
        for name in model_names
    }
    error_names = ["Correct detections", "False detections", "Missed balls"]
    error_values = {
        name: [
            results[name]["fixed"]["tp"],
            results[name]["fixed"]["fp"],
            results[name]["fixed"]["fn"],
        ]
        for name in model_names
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#3478bf", "#d95f59", "#4ca66b"]
    width = 0.8 / len(model_names)
    for model_index, name in enumerate(model_names):
        offset = (model_index - (len(model_names) - 1) / 2) * width
        positions = [index + offset for index in range(5)]
        bars = axes[0].bar(
            positions,
            accuracy_values[name],
            width,
            label=display_names.get(name, name),
            color=colors[model_index],
        )
        axes[0].bar_label(bars, labels=[f"{value:.1%}" for value in accuracy_values[name]], fontsize=8)
    axes[0].set_xticks(range(5), accuracy_names)
    axes[0].set_ylim(0, 0.75)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Quality on v002 validation")
    axes[0].legend()

    for model_index, name in enumerate(model_names):
        offset = (model_index - (len(model_names) - 1) / 2) * width
        positions = [index + offset for index in range(3)]
        bars = axes[1].bar(
            positions,
            error_values[name],
            width,
            label=display_names.get(name, name),
            color=colors[model_index],
        )
        axes[1].bar_label(bars, fontsize=8)
    axes[1].set_xticks(range(3), error_names)
    axes[1].set_ylabel("Number of boxes")
    axes[1].set_title("Point fixe : confiance 0,25 / IoU 0,50")
    axes[1].legend()
    figure.suptitle("Teachers compared at identical resolution (640 px, 670 images)")
    figure.tight_layout()
    figure.savefig(output_root / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return comparison


def run(project_root: Path) -> dict[str, Any]:
    """Run or resume the identical-resolution teacher benchmark."""
    project_root = project_root.resolve()
    output_root = project_root / "models/reports/teacher_benchmark_v002_val"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_yaml = output_root / "dataset.yaml"
    dataset_root = project_root / "data/processed/v002"
    dataset_yaml.write_text(
        "path: "
        + json.dumps(str(dataset_root))
        + "\ntrain: train/images\nval: val/images\ntest: test/images\n"
        + "names:\n  0: ball\n"
    )

    torch.set_num_threads(6)
    models = {
        "yolo26n_v002_finetuned": project_root
        / "models/checkpoints/baseline_runs/yolo26n_v002_local_20260918_002618"
        / "train/weights/best.pt",
        "yolo26x_coco_raw": project_root / "models/checkpoints/teachers/yolo26x.pt",
    }
    results = {}
    for name, weights in models.items():
        print(f"Evaluation: {name}", flush=True)
        results[name] = evaluate_teacher(
            weights=weights,
            dataset_yaml=dataset_yaml,
            output_root=output_root,
            name=name,
            image_size=640,
            batch=1,
            device="cpu",
        )
        print(json.dumps(results[name], indent=2), flush=True)

    hf_report_path = (
        project_root
        / "models/checkpoints/teachers/yolo26x_v002_hf/training_report.json"
    )
    if hf_report_path.exists():
        hf_report = json.loads(hf_report_path.read_text())
        hf_summary = hf_report["validation_after"]
        results["yolo26x_v002_finetuned"] = {
            **hf_summary,
            "name": "yolo26x_v002_finetuned",
            "weights": str(
                project_root
                / "models/checkpoints/teachers/yolo26x_v002_hf/best.pt"
            ),
            "dataset_yaml": str(dataset_yaml),
            "split": "val",
            "image_size": 640,
            "batch": 4,
            "device": "NVIDIA A10G",
            "images": 670,
            "wall_seconds": hf_report["training_seconds"],
            "test_split_used": False,
        }
    return write_comparison(output_root, results)


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
