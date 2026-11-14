from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from box2boxcam.data.pseudo_label_video import (
    box_iou_xywh,
    deduplicate_teacher_boxes,
    expected_sample_count,
    reusable_video_extraction,
    sampled_frame_numbers,
)


class VideoSamplingTests(unittest.TestCase):
    def test_full_stride_sampling(self) -> None:
        self.assertEqual(sampled_frame_numbers(10, 4), [0, 4, 8])
        self.assertEqual(expected_sample_count(4926, 4), 1232)

    def test_pilot_is_spread_across_entire_video(self) -> None:
        frames = sampled_frame_numbers(4926, 4, max_samples=100)
        self.assertEqual(len(frames), 100)
        self.assertEqual(frames[0], 0)
        self.assertEqual(frames[-1], 4924)
        self.assertTrue(all(frame % 4 == 0 for frame in frames))
        self.assertEqual(len(frames), len(set(frames)))

    def test_single_pilot_frame_comes_from_middle(self) -> None:
        self.assertEqual(sampled_frame_numbers(20, 4, max_samples=1), [8])

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            sampled_frame_numbers(0, 4)
        with self.assertRaises(ValueError):
            sampled_frame_numbers(10, 0)
        with self.assertRaises(ValueError):
            sampled_frame_numbers(10, 4, max_samples=0)

    def test_teacher_boxes_are_deduplicated_by_priority(self) -> None:
        nano = {
            "teacher_model": "nano.pt", "confidence": "0.30",
            "x_center": "0.50", "y_center": "0.50",
            "width": "0.10", "height": "0.10",
        }
        large_overlap = {
            "teacher_model": "large.pt", "confidence": "0.90",
            "x_center": "0.505", "y_center": "0.505",
            "width": "0.10", "height": "0.10",
        }
        large_complement = {
            "teacher_model": "large.pt", "confidence": "0.20",
            "x_center": "0.80", "y_center": "0.20",
            "width": "0.05", "height": "0.05",
        }
        self.assertGreater(box_iou_xywh(nano, large_overlap), 0.5)
        kept = deduplicate_teacher_boxes(
            [large_overlap, large_complement, nano],
            teacher_priority=["nano.pt", "large.pt"],
        )
        self.assertEqual(kept, [nano, large_complement])

    def test_complete_interrupted_extraction_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "pilot"
            images = output_root / "images"
            labels = output_root / "teacher_labels"
            images.mkdir(parents=True)
            labels.mkdir()
            for frame in (0, 4, 8):
                (images / f"frame_{frame:08d}.jpg").write_bytes(b"image")
                (labels / f"frame_{frame:08d}.txt").write_text("")

            extracted = reusable_video_extraction(output_root, [0, 4, 8])

            self.assertIsNotNone(extracted)
            assert extracted is not None
            self.assertEqual([row[1] for row in extracted], [0, 4, 8])

    def test_review_files_prevent_automatic_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "pilot"
            output_root.mkdir()
            (output_root / "review_decisions.csv").write_text("decision\n")

            with self.assertRaises(FileExistsError):
                reusable_video_extraction(output_root, [0])


if __name__ == "__main__":
    unittest.main()
