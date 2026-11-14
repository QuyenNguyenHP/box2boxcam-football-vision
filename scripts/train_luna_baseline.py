"""Local nano baseline on the new Luna dataset, without using Infactory to train."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import torch
import ultralytics
from ultralytics import YOLO, settings

from box2boxcam.data.audit_yolo import audit
from box2boxcam.training.diagnostics import quiet_ultralytics_progress
from box2boxcam.training.evaluate import BallValidator


def summary(result):
    return {
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "fixed": result.fixed_threshold,
    }


def evaluate(weights: Path, dataset: Path, run: Path, split: str, name: str, size: int):
    model = YOLO(str(weights))
    with quiet_ultralytics_progress():
        metrics = model.val(
            validator=BallValidator, data=str(dataset), split=split,
            imgsz=size, batch=1, device="cpu", workers=0,
            conf=.001, iou=.7, max_det=100, plots=True, verbose=False,
            project=str(run / "evaluation"), name=name, exist_ok=False,
        )
    result = summary(metrics)
    del model, metrics
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS indisponible : demander explicitement --device cpu")
    torch.set_num_threads(4)
    settings.update({"sync": False, "wandb": False, "mlflow": False, "clearml": False})
    data_root = ROOT / "data/processed/luna_v001"
    dataset = data_root / "dataset.yaml"
    build = json.loads((data_root / "build_report.json").read_text())
    checked = audit(data_root, target_size=args.imgsz)
    if checked["issues_count"] or checked["images"] != build["images"]:
        raise ValueError("Dataset audit failed")
    run = args.output or ROOT / "models/checkpoints/baseline_runs" / (
        "yolo26n_luna_v001_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    )
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    pretrained = ROOT / "models/checkpoints/yolo26n.pt"
    previous = ROOT / "models/checkpoints/baseline_runs/yolo26n_v002_local_20260918_002618/train/weights/best.pt"
    for weight in (pretrained, previous):
        if not weight.is_file():
            raise FileNotFoundError(weight)
    training = {
        "data": str(dataset), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "device": args.device, "workers": 0, "seed": 42, "deterministic": True,
        "optimizer": "AdamW", "lr0": .001, "amp": False, "cache": False,
        "mosaic": 0., "mixup": 0., "close_mosaic": 0, "patience": 5, "time": .3,
        "plots": True, "save": True, "save_period": -1, "verbose": False,
        "project": str(run), "name": "train", "exist_ok": False,
    }
    report = {
        "status": "running", "run_root": str(run), "dataset": build,
        "training": training,
        "initialization": "COCO pretrained nano, not the Infactory checkpoint",
        "pretrained_sha256": hashlib.sha256(pretrained.read_bytes()).hexdigest(),
        "previous_weights": str(previous),
        "previous_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__,
        "metric_warning": "Only two inferred matches in validation and two in test; preliminary generalization estimate. Some empty labels were accepted without a Luna response.",
        "validation": {}, "test": {},
    }

    def save_report():
        (run / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    save_report()
    try:
        print(f"Run : {run}\nDataset : {build['images']} images, 10 groupes de matchs", flush=True)
        for label, weights in (("pretrained", pretrained), ("v002", previous)):
            print(f"Pre-training validation: {label}", flush=True)
            report["validation"][label] = evaluate(weights, dataset, run, "val", "val_" + label, args.imgsz)
            save_report()
        model = YOLO(str(pretrained))

        def progress(trainer):
            print(
                f"Epoch {trainer.epoch + 1}/{args.epochs} — "
                f"mAP50 val={trainer.metrics.get('metrics/mAP50(B)', 0):.3f}",
                flush=True,
            )

        model.add_callback("on_fit_epoch_end", progress)
        started = time.perf_counter()
        with quiet_ultralytics_progress():
            trained = model.train(**training)
        report["training_seconds"] = time.perf_counter() - started
        best = Path(trained.save_dir) / "weights/best.pt"
        report["best_weights"] = str(best)
        del model, trained
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        report["validation"]["luna"] = evaluate(best, dataset, run, "val", "val_luna", args.imgsz)
        save_report()
        # Test is accessed only after fitting/selection of best.pt is finished.
        for label, weights in (("pretrained", pretrained), ("v002", previous), ("luna", best)):
            print(f"Common final test: {label}", flush=True)
            report["test"][label] = evaluate(weights, dataset, run, "test", "test_" + label, args.imgsz)
            save_report()
        with (run / "train/results.csv").open() as stream:
            report["epochs_logged"] = len(list(csv.DictReader(stream)))
        report["epoch_count_note"] = (
            "Timed training may stop the last logged epoch mid-pass. "
            "CSV rows are not necessarily full dataset passes."
        )
        report["status"] = "complete"
        save_report()
        print(json.dumps({"status": report["status"], "test": report["test"]}, indent=2), flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save_report()
        raise


if __name__ == "__main__":
    main()
