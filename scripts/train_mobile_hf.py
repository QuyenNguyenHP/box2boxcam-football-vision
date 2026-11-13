"""HF GPU worker: train the configured mobile model and compare on validation only."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import tarfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch
import ultralytics
from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO, settings


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    if ultralytics.__version__ != "8.4.153":
        raise RuntimeError(f"Unexpected Ultralytics version: {ultralytics.__version__}")
    settings.update({"sync": False, "wandb": False, "mlflow": False, "clearml": False})
    torch.set_num_threads(4)
    api = HfApi()
    model_repo = os.environ["PITCHLENS_MODEL_REPO"]
    dataset_repo = os.environ["PITCHLENS_DATASET_REPO"]
    code_revision = os.environ["PITCHLENS_CODE_REVISION"]
    dataset_revision = os.environ["PITCHLENS_DATASET_REVISION"]
    root = Path("/tmp/pitchlens_mobile")
    root.mkdir(parents=True, exist_ok=True)
    artifacts = root / "artifacts"
    artifacts.mkdir()

    def model_file(filename):
        return Path(hf_hub_download(repo_id=model_repo, filename=filename, revision=code_revision))

    config = json.loads(model_file("code/config.json").read_text())
    transfer_module = None
    continuing = config.get("initialization") == "previous_checkpoint"
    if config.get("sports_ball_transfer") in {"retain_full_classifier", "continue_retained_classifier"}:
        transfer_helper = model_file("code/sports_ball_transfer.py")
        transfer_spec = importlib.util.spec_from_file_location("sports_ball_transfer", transfer_helper)
        transfer_module = importlib.util.module_from_spec(transfer_spec)
        transfer_spec.loader.exec_module(transfer_module)
    helper = model_file("code/evaluate.py")
    spec = importlib.util.spec_from_file_location("ball_validator", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validator = module.BallValidator
    archive = hf_hub_download(repo_id=dataset_repo, repo_type="dataset",
                              revision=dataset_revision, filename="dataset.tar")
    expected_sha = os.environ["PITCHLENS_ARCHIVE_SHA256"]
    if sha256(archive) != expected_sha:
        raise ValueError("Dataset transport checksum mismatch")
    dataset_root = root / "dataset"
    dataset_root.mkdir()
    with tarfile.open(archive) as bundle:
        # No symlinks, hardlinks or path traversal from a transport archive.
        for member in bundle.getmembers():
            target = (dataset_root / member.name).resolve()
            if not target.is_relative_to(dataset_root.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        bundle.extractall(dataset_root, filter="data")
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    for row in manifest:
        for folder, suffix, key in (("images", ".jpg", "image_sha256"), ("labels", ".txt", "label_sha256")):
            path = dataset_root / row["split"] / folder / (row["file_id"] + suffix)
            if sha256(path) != row[key]:
                raise ValueError(f"Manifest checksum mismatch: {path}")
    dataset_yaml = dataset_root / "dataset.yaml"
    class_name = config.get("class_name", "ball")
    if transfer_module and class_name != "sports ball":
        raise ValueError("Full transfer requires the named sports ball class")
    dataset_yaml.write_text("path: " + json.dumps(str(dataset_root))
                            + "\ntrain: train/images\nval: val/images\ntest: null\nnames:\n  0: "
                            + json.dumps(class_name) + "\n")
    build = json.loads((dataset_root / "build_report.json").read_text())
    base_model = config.get("base_model", "yolo26n.pt")
    pretrained = model_file(f"inputs/{base_model}")
    reference_key = config.get("reference_key", "previous_luna")
    previous = model_file(config.get("reference_filename", "inputs/previous_luna_best.pt"))
    initial_weights = previous if continuing else pretrained
    if continuing and config.get("sports_ball_transfer") != "continue_retained_classifier":
        raise ValueError("Continuation requires the retained one-class graph")
    report = {
        "status": "running", "config": config, "dataset": build,
        "dataset_repo": dataset_repo, "dataset_revision": dataset_revision,
        "archive_sha256": expected_sha, "code_revision": code_revision,
        "initialization": f"COCO pretrained {base_model}, not previous fine-tuned model",
        "input_weights_sha256": {"pretrained": sha256(pretrained), reference_key: sha256(previous)},
        "test_split_used": False,
        "validation": {},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0)},
    }
    if transfer_module:
        report["initialization"] = (
            f"Continue {reference_key} best.pt; retain its one-class classifier and all learned weights"
            if continuing else f"COCO pretrained {base_model}; full sports ball classifier retained; row 32 -> 0")
        report["class_name"] = class_name
    report["initial_weights_sha256"] = sha256(initial_weights)

    def publish(message, train_dir=None):
        if train_dir:
            target = artifacts / "train"
            target.mkdir(exist_ok=True)
            for path in train_dir.iterdir():
                if path.is_file() and path.suffix in {".csv", ".yaml", ".png", ".jpg"}:
                    shutil.copy2(path, target / path.name)
            best = train_dir / "weights/best.pt"
            if best.is_file():
                shutil.copy2(best, artifacts / "best.pt")
        (artifacts / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        api.upload_folder(repo_id=model_repo, folder_path=artifacts, path_in_repo="artifacts", commit_message=message)

    def evaluate(weights, name):
        result = YOLO(str(weights)).val(
            validator=validator, data=str(dataset_yaml), split="val", imgsz=config["imgsz"],
            batch=config["batch"], device=0, workers=2, conf=.001, iou=.7,
            max_det=100, plots=True, verbose=False, project=str(artifacts / "evaluation"),
            name=name, exist_ok=False,
        )
        # This matrix uses exactly the operating point in the metric table.
        # There is no count of true negatives for object-detection background.
        import matplotlib.pyplot as plt
        import numpy as np
        fixed = result.fixed_threshold
        values = np.array([[fixed.get("tp", 0), fixed.get("fp", 0)],
                           [fixed.get("fn", 0), np.nan]])
        fig, axis = plt.subplots(figsize=(5, 4))
        axis.imshow(values, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, "—" if np.isnan(values[i, j]) else str(int(values[i, j])), ha="center", va="center")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real ball", "No matching object"],
                 yticklabels=["Predicted ball", "Missed ball"], xlabel="Annotation", ylabel="Prediction",
                 title=f"{name} — conf >= 0.25; IoU >= 0.50")
        fig.tight_layout()
        fig.savefig(artifacts / "evaluation" / name / "fixed_confusion_matrix.png", dpi=130)
        plt.close(fig)
        return {"map50": float(result.box.map50), "map50_95": float(result.box.map),
                "fixed": result.fixed_threshold, "speed_ms_per_image": dict(result.speed)}

    def negative_diagnostic(weights, name):
        """Fixed-threshold FP counts on the added TRAIN frames; not an independent test."""
        selected = [row for row in manifest
                    if row["source_dataset"] == config["negative_diagnostic_source"]]
        if not selected or any(row["split"] != "train" or row["status"] != "negative" for row in selected):
            raise ValueError("Expected explicitly negative training frames for this diagnostic")
        paths = [dataset_root / "train/images" / (row["file_id"] + ".jpg") for row in selected]
        listing = root / f"negative_diagnostic_{name}.txt"
        listing.write_text("\n".join(str(path) for path in paths) + "\n")
        detector = YOLO(str(weights))
        image_count, fp, images_with_fp = 0, 0, 0
        for prediction in detector.predict(source=str(listing), stream=True, batch=config["batch"],
                                            imgsz=config["imgsz"], conf=config["fixed_confidence"],
                                            max_det=100, device=0, verbose=False, save=False):
            count = len(prediction.boxes)
            image_count += 1
            fp += count
            images_with_fp += count > 0
        if image_count != len(paths):
            raise ValueError("Incomplete negative diagnostic")
        return {"images": image_count, "fp": fp, "images_with_fp": images_with_fp,
                "fp_per_image": fp / image_count, "fraction_images_with_fp": images_with_fp / image_count,
                "confidence": config["fixed_confidence"], "imgsz": config["imgsz"],
                "source_dataset": config["negative_diagnostic_source"],
                "split": "train", "independent_generalization_test": False,
                "caveat": "These frames are used in training; reduction demonstrates fitting, not generalization"}

    print(f"Dataset: {build['splits']}; GPU: {torch.cuda.get_device_name(0)}; imgsz={config['imgsz']}", flush=True)
    try:
        for name, weights in (("pretrained", pretrained), (reference_key, previous)):
            print(f"Validation before training: {name} @ {config['imgsz']}", flush=True)
            report["validation"][name] = evaluate(weights, name)
        if config.get("negative_diagnostic_source"):
            report["negative_training_diagnostic"] = {reference_key: negative_diagnostic(previous, reference_key)}
        publish("Validation baselines at identical resolution")
        model = YOLO(str(initial_weights))
        training = {
            "data": str(dataset_yaml), "epochs": config["epochs"], "imgsz": config["imgsz"],
            "batch": config["batch"], "patience": config["patience"], "seed": config["seed"],
            "device": 0, "workers": 2, "deterministic": True, "optimizer": config["optimizer"],
            "lr0": config["lr0"], "cos_lr": True, "amp": True, "cache": False,
            "mosaic": 0., "mixup": 0., "close_mosaic": 0, "scale": .3,
            "plots": True, "save": True, "save_period": -1, "verbose": False,
            "project": str(root / "runs"), "name": "train", "exist_ok": False,
        }
        report["training"] = training

        def record_transfer(trainer):
            if transfer_module:
                transfer = trainer.sports_ball_transfer_report
                expected_class, expected_count = (0, 1) if continuing else (32, 80)
                if (transfer["source_class_id"] != expected_class
                        or transfer["source_classes"] != expected_count
                        or transfer["classification_hidden_widths"] != config.get("classification_hidden_widths", [80] * 6)
                        or not transfer["all_weight_checks_passed"]
                        or not transfer["dense_prediction_check"]["passed"]):
                    raise ValueError("COCO sports ball transfer verification failed")
                report["sports_ball_transfer"] = transfer
                print(f"Sports ball graph verified: full classifier, {expected_class} -> 0; "
                      f"{transfer['target_parameters']} parameters", flush=True)
                publish("Verified full sports-ball graph before optimization")

        def progress(trainer):
            epoch = trainer.epoch + 1
            report["epochs_logged"] = epoch
            report["latest_validation"] = {key: float(value) for key, value in trainer.metrics.items()}
            print(f"Epoch {epoch}/{config['epochs']} — mAP50={trainer.metrics.get('metrics/mAP50(B)', 0):.4f}", flush=True)
            # Save remotely during training, so a timeout cannot lose every weight.
            if epoch % config.get("publish_every_epochs", 5) == 0:
                publish(f"Training progress — epoch {epoch}", Path(trainer.save_dir))

        model.add_callback("on_model_save", progress)
        model.add_callback("on_pretrain_routine_end", record_transfer)
        started = time.perf_counter()
        trainer_options = {"trainer": transfer_module.SportsBallTransferTrainer} if transfer_module else {}
        trained = model.train(**training, **trainer_options)
        report["training_seconds"] = time.perf_counter() - started
        train_dir = Path(trained.save_dir)
        best = train_dir / "weights/best.pt"
        report["validation"]["mobile"] = evaluate(best, "mobile")
        if config.get("negative_diagnostic_source"):
            report["negative_training_diagnostic"]["mobile"] = negative_diagnostic(best, "mobile")
        with (train_dir / "results.csv").open() as stream:
            report["epochs_logged"] = len(list(csv.DictReader(stream)))
        report["status"] = "complete"
        publish("Complete mobile experiment — validation and learning curves", train_dir)
        print(json.dumps({"status": "complete", "validation": report["validation"]}, indent=2), flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        publish("Mobile experiment failure diagnostics")
        raise


if __name__ == "__main__":
    main()
