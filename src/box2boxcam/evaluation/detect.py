"""Run an Ultralytics detector on evaluation frames and cache the raw detections."""

import json
from pathlib import Path

import numpy as np


def run_detector(weights, image_paths, imgsz=960, conf=0.05, device=None, classes=None):
    """Detections ``[(boxes_xyxy (m, 4), confidences (m,)), ...]`` in image pixels.

    Images are read at full resolution and resized by the model to ``imgsz``;
    ``conf`` is the lowest confidence kept (all thresholds are applied afterwards).
    ``classes`` restricts the output to those class ids (e.g. ``[32]`` keeps only the
    COCO "sports ball" of a stock model).
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"cannot read {path}")
        result = model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            device=device,
            classes=classes,
            verbose=False,
        )[0].boxes
        out.append((result.xyxy.cpu().numpy(), result.conf.cpu().numpy()))
    return out


def save_detections(path, ids, detections):
    """Write detections as JSON (``{frame id: {"boxes": ..., "conf": ...}}``)."""
    data = {
        i: {"boxes": np.asarray(b).tolist(), "conf": np.asarray(c).tolist()}
        for i, (b, c) in zip(ids, detections)
    }
    Path(path).write_text(json.dumps(data))


def load_detections(path, ids):
    data = json.loads(Path(path).read_text())
    return [
        (
            np.asarray(data[i]["boxes"], dtype=float).reshape(-1, 4),
            np.asarray(data[i]["conf"], dtype=float),
        )
        for i in ids
    ]
