"""Final personal test at fixed settings, then a streamed annotated video."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import torch
from ultralytics import YOLO, settings

from box2boxcam.data.audit_yolo import audit
from box2boxcam.data.build_mobile_dataset import sha256
from box2boxcam.training.camera_benchmark import benchmark_targets
from box2boxcam.training.diagnostics import quiet_ultralytics_progress
from box2boxcam.training.evaluate import BallValidator

ROOT = Path(__file__).resolve().parents[1]


def draw_matrix(fixed, path, title):
    import matplotlib.pyplot as plt
    values = np.array([[fixed.get("tp", 0), fixed.get("fp", 0)],
                       [fixed.get("fn", 0), np.nan]])
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(values, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, "—" if np.isnan(values[i, j]) else str(int(values[i, j])), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real ball", "Background"],
             yticklabels=["Predicted ball", "Missed"], title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_video(weights, source, output, confidence=.25, size=960):
    """One decoded frame and one prediction at a time; no large image cache."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg required for browser-compatible H.264 video")
    if output.exists():
        raise FileExistsError(output)
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    header_height = 40
    model = YOLO(str(weights))
    predicted_frames, processed = 0, 0
    predictions = []
    started = time.perf_counter()
    with tempfile.TemporaryFile() as error_log:
        encoder = subprocess.Popen([
            ffmpeg, "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height + header_height}", "-r", str(fps), "-i", "pipe:0",
            # iPhone MOV can include an unsupported extra APAC spatial-audio
            # stream. Keep its first (AAC) audio track, not every audio stream.
            "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest",
            "-movflags", "+faststart", str(output),
        ], stdin=subprocess.PIPE, stderr=error_log)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(frame, imgsz=size, conf=confidence, iou=.7,
                                       max_det=100, device="cpu", rect=False, verbose=False)[0]
                header = f"Nouveau nano | 960 px | seuil {confidence:.2f} | aucune detection"
                selected = None
                if len(result.boxes):
                    index = int(result.boxes.conf.argmax())
                    box = result.boxes.xyxy[index].cpu().tolist()
                    score = float(result.boxes.conf[index])
                    x1, y1, x2, y2 = [round(value) for value in box]
                    cv2.rectangle(frame, (max(0, x1 - 2), max(0, y1 - 2)),
                                  (min(width - 1, x2 + 2), min(height - 1, y2 + 2)), (0, 255, 0), 2)
                    header = f"Nouveau nano | 960 px | seuil {confidence:.2f} | confiance {score:.2f}"
                    selected = {"box_pixels": box, "confidence": score}
                    predicted_frames += 1
                # Text lives above the image, never over the small ball.
                canvas = np.zeros((height + header_height, width, 3), dtype=np.uint8)
                canvas[header_height:] = frame
                cv2.putText(canvas, header, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
                encoder.stdin.write(canvas.tobytes())
                predictions.append({"source_frame": processed, "selected": selected})
                processed += 1
                if processed % 150 == 0:
                    print(f"Video: {processed}/{total} images", flush=True)
            if processed != total:
                raise RuntimeError(f"Incomplete video decode: {processed}/{total}")
        except BrokenPipeError as error:
            encoder.wait()
            error_log.seek(0)
            raise RuntimeError(error_log.read().decode(errors="replace")) from error
        finally:
            capture.release()
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
            code = encoder.wait()
        if code:
            error_log.seek(0)
            raise RuntimeError(error_log.read().decode(errors="replace"))
    elapsed = time.perf_counter() - started
    (output.parent / "video_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    del model
    gc.collect()
    return {"path": str(output), "frames": processed, "frames_with_target": predicted_frames,
            "source_fps": fps, "duration_seconds": processed / fps, "processing_seconds": elapsed,
            "device": "cpu", "resolution": size, "confidence": confidence,
            "selection": "highest confidence only; no tracking or temporal smoothing",
            "full_clip_includes_unscored_and_excluded_frames": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/reports/mobile_personal_test_20261010")
    parser.add_argument("--video-only", action="store_true", help="Resume encoding without rerunning or tuning the final test")
    args = parser.parse_args()
    output = args.output
    if output.exists() and not args.video_only:
        raise FileExistsError("Immutable evaluation already exists; use another output path")
    config = json.loads((ROOT / "configs/mobile_nano_960.json").read_text())
    run = ROOT / "models/checkpoints/baseline_runs" / config["experiment_id"] / "artifacts"
    training_report = json.loads((run / "training_report.json").read_text())
    if training_report["status"] != "complete" or training_report["test_split_used"]:
        raise RuntimeError("Training must be complete and selected on validation only")
    dataset = ROOT / "data/processed/club_match_test_v001"
    build = json.loads((dataset / "build_report.json").read_text())
    source = ROOT / build["source_video"]
    if sha256(source) != build["source_video_sha256"]:
        raise ValueError("Source video changed since annotation")
    checked = audit(dataset, target_size=960)
    if checked["issues_count"] or checked["images"] != 292:
        raise ValueError("Personal test audit failed")
    weights = {"previous_luna": ROOT / config["previous_weights"], "mobile": run / "best.pt"}
    for key, path in weights.items():
        if key == "previous_luna" and sha256(path) != training_report["input_weights_sha256"][key]:
            raise ValueError("Previous-model baseline changed")
    if args.video_only:
        report = json.loads((output / "report.json").read_text())
        if report.get("metrics_status") != "complete":
            raise RuntimeError("Final metrics must already be complete")
        if report["weights_sha256"] != {key: sha256(path) for key, path in weights.items()}:
            raise ValueError("Weights changed since the frozen final evaluation")
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        report["video"] = make_video(weights["mobile"], source, output / "club_match_nano960_ball.mp4")
        report["status"] = "complete"
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"Video complete; test metrics unchanged: {output}", flush=True)
        return
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    settings.update({"sync": False, "wandb": False, "mlflow": False, "clearml": False})
    manifest = json.loads((dataset / "review_manifest.json").read_text())
    records = [row for row in manifest if row["split"] == "test"]
    rows = [{"item_id": row["item_id"], "match_id": row["match_id"],
             "source_frame": row["source_frame"], "width": row["width"], "height": row["height"],
             "ground_truth": [[x1 / row["width"], y1 / row["height"], x2 / row["width"], y2 / row["height"]]
                              for x1, y1, x2, y2 in row["boxes_pixels"]]} for row in records]
    report = {"status": "running", "images": 292, "instances": build["boxes"], "image_size": 960,
              "confidence": .25, "matching_iou": .5, "device": "cpu", "test_used_for_selection": False,
              "weights_sha256": {key: sha256(path) for key, path in weights.items()},
              "test_manifest_sha256": sha256(dataset / "review_manifest.json"),
              "caveats": ["Train and test share September 30 match; clip-level benchmark",
                          "11 multi-box images: any annotated ball may match; no active-ball identity",
                          "No tracking; 88 excluded sampled frames do not enter metrics"],
              "detection": {}, "targeting": {}}

    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    for key, path in weights.items():
        print(f"Detector test: {key} @960, batch 1 CPU", flush=True)
        model = YOLO(str(path))
        started = time.perf_counter()
        with quiet_ultralytics_progress():
            result = model.val(validator=BallValidator, data=str(dataset / "dataset.yaml"), split="test",
                               imgsz=960, batch=1, device="cpu", workers=0, conf=.001, iou=.7,
                               max_det=100, plots=True, verbose=False, project=str(output), name=key, exist_ok=False)
        report["detection"][key] = {"map50": float(result.box.map50), "map50_95": float(result.box.map),
                                    "fixed": result.fixed_threshold, "seconds": time.perf_counter() - started}
        draw_matrix(result.fixed_threshold, output / key / "fixed_confusion_matrix.png", key + " — conf 0.25; IoU 0.50")
        print(json.dumps(report["detection"][key], indent=2), flush=True)
        save()
        # Cache per-image predictions once for the looser camera-target metric.
        for index, row in enumerate(rows):
            image = dataset / "test/images" / (row["item_id"] + ".jpg")
            prediction = model.predict(str(image), imgsz=960, conf=.25, iou=.7, max_det=100,
                                       device="cpu", rect=False, verbose=False)[0]
            row[key] = prediction.boxes.xyxyn.cpu().tolist()
            row[key + "_confidence"] = prediction.boxes.conf.cpu().tolist()
            if (index + 1) % 100 == 0:
                print(f"Cache ciblage {key} : {index + 1}/292", flush=True)
        for overlap in (.1, .2):
            report["targeting"].setdefault(str(overlap), {})[key] = benchmark_targets(
                rows, key, confidence=.25, matching_iou=overlap, cached_min_confidence=.25)
        del model, result
        gc.collect()
        save()
    (output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
    report["metrics_status"] = "complete"
    save()
    print("Creating the full video, one frame at a time", flush=True)
    report["video"] = make_video(weights["mobile"], source, output / "club_match_nano960_ball.mp4")
    report["status"] = "complete"
    save()
    print(f"Done: {output}", flush=True)


if __name__ == "__main__":
    main()
