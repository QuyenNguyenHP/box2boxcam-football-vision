from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pitchlens.data.audit_sequences import audit_sequences


class SequenceAuditTests(unittest.TestCase):
    def test_groups_frames_and_flags_invalid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            root.mkdir(exist_ok=True)

            for name in ("clip_a_000000", "clip_a_000001", "clip_b_000000"):
                Image.new("RGB", (100, 50)).save(root / f"{name}.jpg")
            (root / "clip_a_000000.txt").write_text("0 0.5 0.5 0.1 0.2\n")
            (root / "clip_a_000001.txt").write_text("0 0.5 0.5 0.0 0.2\n")

            report = audit_sequences(root)

            self.assertEqual(report["sequences"], 2)
            self.assertEqual(report["parsed_frames"], 3)
            self.assertEqual(report["contiguous_sequences"], 2)
            self.assertEqual(report["all_negative_sequences"], 1)
            self.assertEqual(report["sequences_with_invalid_annotations"], 1)


if __name__ == "__main__":
    unittest.main()
