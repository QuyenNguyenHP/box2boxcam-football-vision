"""Tiled person detector restricted to the pitch polygon."""

from time import perf_counter

import cv2
import numpy as np


class PeopleDetector:
    def __init__(self, weights, field, size, device):
        if hasattr(weights, "predict"):  # an already loaded model is shared
            self.model = weights
        else:
            from ultralytics import YOLO

            self.model = YOLO(str(weights))
        self.person_class = next(
            index for index, name in self.model.names.items() if name == "person"
        )
        self.field = np.asarray(field, dtype=np.float32)
        self.size = np.asarray(size)
        self.device = device
        self.tile_width = 960
        self.calls = []

    def __call__(self, frame, origin=(0, 0)):
        height, width = frame.shape[:2]
        # Only the geometric field bounds select pixels, independent of the action labels.
        lower = np.maximum(
            0, np.floor(self.field.min(0) * self.size - origin - [0, 100])
        ).astype(int)
        upper = np.minimum(
            [width, height], np.ceil(self.field.max(0) * self.size - origin + [0, 10])
        ).astype(int)
        x0, y0 = lower
        x1, y1 = upper
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 5))
        stride = int(self.tile_width * 0.8)
        starts = list(range(x0, max(x0 + 1, x1 - self.tile_width + 1), stride))
        starts.append(max(x0, x1 - self.tile_width))
        starts = sorted(set(starts))
        tiles = [frame[y0:y1, x : min(x + self.tile_width, x1)] for x in starts]
        started = perf_counter()
        results = self.model.predict(
            tiles,
            imgsz=960,
            device=self.device,
            classes=[self.person_class],
            conf=0.25,
            iou=0.55,
            max_det=80,
            verbose=False,
        )
        self.calls.append((perf_counter() - started) * 1000)
        proposals = []
        for x, result in zip(starts, results):
            boxes = result.boxes.xyxy.cpu().numpy()
            boxes += np.array([x + origin[0], y0 + origin[1]] * 2)
            for box, confidence in zip(boxes, result.boxes.conf.cpu().numpy()):
                foot = np.array([(box[0] + box[2]) / 2, box[3]]) / self.size
                if cv2.pointPolygonTest(self.field, tuple(foot), False) >= 0:
                    proposals.append([*box, float(confidence)])
        if not proposals:
            return np.empty((0, 5))
        proposals = np.asarray(proposals)
        xywh = proposals[:, :4].copy()
        xywh[:, 2:] -= xywh[:, :2]
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), proposals[:, 4].tolist(), 0.25, 0.45)
        result = proposals[np.asarray(keep).reshape(-1)]
        result[:, :4] /= np.tile(self.size, 2)
        return result
