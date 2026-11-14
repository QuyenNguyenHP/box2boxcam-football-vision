from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from box2boxcam.data.audit_roboflow_splits import audit_splits


class RoboflowSplitAuditTests(unittest.TestCase):
    def test_detects_parent_name_in_multiple_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train = root / "train" / "images"
            valid = root / "valid" / "images"
            train.mkdir(parents=True)
            valid.mkdir(parents=True)

            Image.new("RGB", (10, 10), "red").save(
                train / "video1_frame1_png.rf.aaaa.jpg"
            )
            Image.new("RGB", (10, 10), "blue").save(
                valid / "video1_frame1_png.rf.bbbb.jpg"
            )

            report = audit_splits(root)

            self.assertEqual(report["images"], 2)
            self.assertEqual(report["unique_parent_names"], 1)
            self.assertEqual(report["parent_names_across_splits"], 1)
            self.assertEqual(report["source_prefixes_across_splits"], 1)
            self.assertEqual(report["exact_file_hashes_across_splits"], 0)


if __name__ == "__main__":
    unittest.main()
