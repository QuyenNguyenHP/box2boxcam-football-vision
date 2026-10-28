import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_hard_negatives.py"
spec = importlib.util.spec_from_file_location("hard_negative_preparation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NegativePreparationTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "downloads/fixture.avi"
        source.parent.mkdir()
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
        self.assertTrue(writer.isOpened())
        for frame in range(13):
            writer.write(np.full((48, 64, 3), frame * 15, dtype=np.uint8))
        writer.release()
        return {
            "source_files": [str(source)], "video_directory": "data/external/videos/negative",
            "extraction_directory": "data/interim/negative", "dataset": "data/processed/negative",
            "capture_session": "session", "frame_stride": 6, "jpeg_quality": 95,
        }

    def test_video_copy_and_explicit_empty_labels_require_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            module.prepare(root, config)
            self.assertTrue(Path(config["source_files"][0]).is_file())
            self.assertFalse(list((root / config["extraction_directory"]).rglob("*.txt")))
            with self.assertRaises(ValueError):
                module.freeze(root, config, confirm_negative=False)
            module.freeze(root, config, confirm_negative=True)
            output = root / config["dataset"]
            labels = list((output / "labels").glob("*.txt"))
            self.assertEqual(len(labels), 3)
            self.assertTrue(all(p.read_text() == "" for p in labels))
            rows = json.loads((output / "review_manifest.json").read_text())
            self.assertTrue(all(r["status"] == "negative" and r["split"] == "unassigned" for r in rows))
            self.assertTrue(all(r["label_source"] == "user_confirmed_negative_video" for r in rows))
            with self.assertRaises(FileExistsError):
                module.freeze(root, config, confirm_negative=True)

    def test_different_existing_video_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            destination = root / config["video_directory"] / "fixture.avi"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"existing unrelated video")
            with self.assertRaises(FileExistsError):
                module.prepare(root, config)
            self.assertEqual(destination.read_bytes(), b"existing unrelated video")


if __name__ == "__main__":
    unittest.main()
