"""Ground-truth loading from a frames directory and a labels file or directory.

Two label layouts are supported; both give one list of ground-truth boxes per image,
in **pixels** (``x_min, y_min, x_max, y_max``). An image without boxes is a valid
negative frame and must still be listed (or simply have no label entry).

1. YOLO text files: a directory with ``<image stem>.txt`` files, one line per ball,
   ``class cx cy w h`` with coordinates normalised to the image size. The image size
   is read from the image itself. A missing file means no ball.
2. JSON, either
   * an object ``{"<image stem>": [[x0, y0, x1, y1], ...], ...}``, or
   * a list of records ``{"item_id": "<image stem>", "boxes_pixels": [[...], ...]}``
     (optionally filtered with ``--labels-status`` style keys by the caller).

Images are the ``*.jpg``, ``*.jpeg`` and ``*.png`` files of ``frames_dir``
(non-recursive, sorted by name).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Frame:
    """One evaluation image and its ground-truth boxes (``(n, 4)`` float, pixels)."""

    id: str
    path: Path
    boxes: np.ndarray


def list_images(frames_dir):
    return sorted(
        p for p in Path(frames_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_yolo_label(text, width, height):
    """Convert YOLO ``class cx cy w h`` lines (normalised) to pixel ``xyxy`` boxes."""
    boxes = []
    for number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"line {number}: expected 'class cx cy w h', got {line!r}")
        cx, cy, w, h = (float(v) for v in fields[1:5])
        boxes.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def parse_json_labels(data):
    """Normalise the two JSON layouts to ``{stem: (n, 4) array}``."""
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = ((r["item_id"], r["boxes_pixels"]) for r in data)
    else:
        raise TypeError("labels JSON must be an object or a list of records")
    out = {}
    for stem, boxes in items:
        arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if (arr[:, 2] <= arr[:, 0]).any() or (arr[:, 3] <= arr[:, 1]).any():
            raise ValueError(f"{stem}: boxes must be x_min, y_min, x_max, y_max")
        out[str(stem)] = arr
    return out


def load_ground_truth(frames_dir, labels):
    """Return the list of ``Frame`` for every image of ``frames_dir``."""
    images = list_images(frames_dir)
    if not images:
        raise FileNotFoundError(f"no images in {frames_dir}")
    labels = Path(labels)
    frames = []
    if labels.is_dir():
        from PIL import Image

        for path in images:
            label = labels / f"{path.stem}.txt"
            if label.exists():
                with Image.open(path) as image:
                    width, height = image.size
                boxes = parse_yolo_label(label.read_text(), width, height)
            else:
                boxes = np.empty((0, 4))
            frames.append(Frame(path.stem, path, boxes))
        return frames
    table = parse_json_labels(json.loads(labels.read_text()))
    unknown = set(table) - {p.stem for p in images}
    if unknown:
        raise ValueError(f"labels without an image: {sorted(unknown)[:5]}")
    for path in images:
        frames.append(Frame(path.stem, path, table.get(path.stem, np.empty((0, 4)))))
    return frames
