from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pitchlens.data.audit_yolo import audit


class YoloAuditTests(unittest.TestCase):
    def test_class_metrics_and_negative_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "train" / "images"
            labels = root / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)

            Image.new("RGB", (100, 50)).save(images / "positive.jpg")
            (labels / "positive.txt").write_text("0 0.5 0.5 0.1 0.2\n")
            Image.new("RGB", (100, 50)).save(images / "negative.jpg")

            report = audit(root, missing_label_is_negative=True, target_size=640)

            self.assertEqual(report["images"], 2)
            self.assertEqual(report["negative_images"], 1)
            self.assertEqual(report["class_metrics"][0]["images_with_class"], 1)
            self.assertEqual(report["class_metrics"][0]["images_without_class"], 1)
            self.assertEqual(report["class_metrics"][0]["multi_box_images_count"], 0)
            self.assertEqual(report["class_metrics"][0]["width_px"]["median"], 10)
            self.assertEqual(report["class_metrics"][0]["height_px"]["median"], 10)
            self.assertEqual(
                report["class_metrics"][0]["letterbox_640_px"]["width"]["median"],
                64,
            )
            self.assertEqual(report["issues_count"], 0)


if __name__ == "__main__":
    unittest.main()
