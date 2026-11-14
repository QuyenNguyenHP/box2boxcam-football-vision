import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from box2boxcam.data.build_auto_judged_field import build, judge_frame, mask_rectangle
from box2boxcam.data.build_mobile_dataset import sha256


class AutoJudgedPolicyTests(unittest.TestCase):
    def frame(self, *boxes_and_verdicts):
        frame = {"item_id": "frame", "path": "image.jpg", "sha256": "abc", "image_size": [200, 100],
                 "cells": list(range(1, len(boxes_and_verdicts) + 1))}
        candidates = {str(cell): {"frame_path": "image.jpg", "frame_sha256": "abc", "box_xyxy_pixels": box,
                                  "sources": ["detector_full"], "detector_confidence": 0.5, "luna_confidence": None}
                      for cell, (box, _) in enumerate(boxes_and_verdicts, 1)}
        verdicts = {cell: verdict for cell, (_, verdict) in enumerate(boxes_and_verdicts, 1)}
        return judge_frame(frame, candidates, verdicts)

    def test_any_candidate_over_64_px_drops_the_frame(self):
        row = self.frame(([10, 10, 20, 20], "ball"), ([100, 10, 165, 40], "not_ball"))
        self.assertEqual(row["split"], "excluded")
        self.assertEqual(row["drop_reason"], "candidate_side_over_64px")
        self.assertEqual(self.frame(([100, 10, 164, 40], "not_ball"))["split"], "unassigned")

    def test_mask_is_enlarged_by_half_with_minimum_side_and_clamped(self):
        self.assertEqual(mask_rectangle([80, 40, 120, 60], 200, 100), [70, 35, 130, 65])
        self.assertEqual(mask_rectangle([95, 45, 105, 55], 200, 100), [88, 38, 112, 62])
        self.assertEqual(mask_rectangle([0, 0, 4, 4], 200, 100), [0, 0, 14, 14])

    def test_ball_overlapping_an_unsure_mask_drops_the_frame(self):
        row = self.frame(([95, 45, 105, 55], "unsure"), ([110, 45, 120, 55], "ball"))
        self.assertEqual(row["drop_reason"], "ball_overlaps_unsure_mask")
        row = self.frame(([95, 45, 105, 55], "unsure"), ([112, 45, 120, 55], "ball"))
        self.assertEqual(row["status"], "positive")
        self.assertEqual(row["masks_pixels"], [[88, 38, 112, 62]])
        self.assertTrue(row["masked"])

    def test_not_ball_is_background_and_ball_is_clamped(self):
        row = self.frame(([10, 10, 20, 20], "not_ball"))
        self.assertEqual((row["status"], row["boxes_pixels"], row["masks_pixels"]), ("negative", [], []))
        row = self.frame(([190, 90, 205, 104], "ball"))
        self.assertEqual(row["boxes_pixels"], [[190, 90, 200, 100]])

    def test_missing_verdict_is_rejected(self):
        with self.assertRaises(ValueError):
            self.frame(([10, 10, 20, 20], "maybe"))


class AutoJudgedBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        (self.inputs / "manifest").mkdir(parents=True)
        frames_dir = self.root / "data/interim/frames"
        frames_dir.mkdir(parents=True)
        specs = {"a": [([20, 20, 30, 30], "ball"), ([100, 40, 110, 50], "unsure")],
                 "b": [([20, 20, 30, 30], "not_ball")],
                 "c": [([20, 20, 30, 30], "ball"), ([0, 0, 90, 30], "not_ball")]}
        frames, candidates, verdicts, cell = [], {}, [], 0
        for index, (name, specs_for_frame) in enumerate(specs.items()):
            path = frames_dir / f"{name}.jpg"
            Image.new("RGB", (200, 100), (0, 128 + index, 0)).save(path, quality=95)
            cells = []
            for box, verdict in specs_for_frame:
                cell += 1
                cells.append(cell)
                candidates[str(cell)] = {"frame_path": str(path), "frame_sha256": sha256(path), "box_xyxy_pixels": box,
                                         "sources": ["detector_tile"], "detector_confidence": 0.3, "luna_confidence": None}
                verdicts.append({"cell": cell, "verdict": verdict})
            frames.append({"item_id": name, "path": str(path), "sha256": sha256(path), "clip_id": "clip",
                           "match_id": "session", "frame_index": index, "timestamp_seconds": index / 30,
                           "image_size": [200, 100], "priority_class": 1, "luna_status": "no_ball",
                           "luna_frame_confidence": 0.9, "cells": cells})
        (self.inputs / "manifest/frames.json").write_text(json.dumps(frames))
        (self.inputs / "manifest/candidates.json").write_text(json.dumps(candidates))
        (self.inputs / "verdicts_part1.json").write_text(json.dumps(verdicts))
        self.videos = {"clip": {"source_video": "data/external/videos/clip.mov", "source_video_sha256": "video"}}
        self.output = self.root / "data/processed/auto"

    def test_build_masks_copies_and_hardlinks_unmasked_frames(self):
        report = build(self.root, self.inputs, self.output, self.videos)
        self.assertEqual((report["retained_images"], report["positive_images"], report["negative_images"]), (2, 1, 1))
        self.assertEqual(report["dropped_frames"], {"candidate_side_over_64px": 1})
        self.assertEqual(report["kept_not_ball_detector_conf_ge_0_25"], 1)
        rows = {row["item_id"]: row for row in json.loads((self.output / "review_manifest.json").read_text())}
        self.assertEqual(rows["c"]["split"], "excluded")
        self.assertFalse(rows["a"]["human_reviewed"])
        masked = self.output / "images/a.jpg"
        self.assertNotEqual(masked.stat().st_ino, (self.root / rows["a"]["source_image"]).stat().st_ino)
        self.assertEqual((self.output / "images/b.jpg").stat().st_ino,
                         (self.root / rows["b"]["source_image"]).stat().st_ino)
        with Image.open(masked) as pixels:
            self.assertTrue(all(abs(c - 114) <= 3 for c in pixels.getpixel((105, 45))))
            self.assertGreater(pixels.getpixel((50, 80))[1], 100)
        self.assertEqual((self.output / "labels/b.txt").read_text(), "")
        self.assertEqual(len((self.output / "labels/a.txt").read_text().splitlines()), 1)
        self.assertEqual(rows["a"]["image_sha256"], sha256(masked))
        with self.assertRaises(FileExistsError):
            build(self.root, self.inputs, self.output, self.videos)

    def test_reserved_session_is_refused_before_writing(self):
        (self.root / "configs").mkdir()
        (self.root / "configs/reserved_sessions.json").write_text(json.dumps({
            "schema_version": 1, "sessions": [{"match_id": "other",
            "videos": [{"source_video_sha256": "video", "source_video": "x.mov"}]}]}))
        with self.assertRaisesRegex(ValueError, "Reserved field session"):
            build(self.root, self.inputs, self.output, self.videos)
        self.assertFalse(self.output.parent.exists())


if __name__ == "__main__":
    unittest.main()
