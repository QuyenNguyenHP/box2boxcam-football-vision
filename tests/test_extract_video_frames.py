import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from box2boxcam.data.extract_video_frames import extract_video_frames


class ExtractionTests(unittest.TestCase):
    def test_stride_provenance_and_safe_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "fixture.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(9):
                writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
            writer.release()
            source = root / "download_manifest.json"
            source.write_text(json.dumps({"videos": [{
                "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "resolved_url": "https://example.org/video", "uploader": "fixture",
            }]}))
            args = {"project_root": root, "download_manifest": source,
                    "output_root": root / "output", "frame_stride": 4}
            report = extract_video_frames(**args)
            self.assertEqual(report["images"], 3)
            with (root / "output/manifest.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(r["source_frame"]) for r in rows], [0, 4, 8])
            self.assertTrue(all(r["status"] == "unannotated" for r in rows))
            self.assertTrue(all(r["split"] == "unassigned" for r in rows))
            self.assertTrue(all(r["match_id"] == "" for r in rows))
            self.assertFalse(list((root / "output").rglob("*.txt")))
            timestamps = [float(r["timestamp_seconds"]) for r in rows]
            self.assertLess(timestamps[0], timestamps[1])
            self.assertLess(timestamps[1], timestamps[2])
            self.assertEqual(report, extract_video_frames(**args))
            with self.assertRaises(FileExistsError):
                extract_video_frames(**(args | {"frame_stride": 2}))
