import unittest

from pitchlens.data.build_personal_test import select_review


class PersonalTestExportTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "item_id": "frame_0", "source_frame": 0, "timestamp_seconds": 0,
            "video_id": "clip", "source_video": "clip.MOV", "match_id": "match",
            "width": 1920, "height": 1080, "image_path": "image.jpg",
            "luna_confidence": 0.01,
        }
        self.rectangle = {
            "from_name": "bbox", "type": "rectangle",
            "value": {"x": 10, "y": 20, "width": 5, "height": 10},
        }
        self.proposal = {
            "data": self.data, "predictions": [{"result": [self.rectangle]}],
        }

    def select(self, annotations, confirm=True):
        return select_review(
            {"data": self.data, "annotations": annotations}, self.proposal,
            confirm_unchanged=confirm,
        )

    def test_human_empty_negative_overrides_positive_proposal(self):
        row = self.select([{"id": 1, "result": []}])
        self.assertEqual(row["status"], "negative")
        self.assertEqual(row["label_source"], "submitted_human")
        self.assertEqual(row["boxes_pixels"], [])

    def test_exclusion_wins_over_existing_box(self):
        exclusion = {"from_name": "decision", "value": {"choices": ["Exclure image"]}}
        row = self.select([{"result": [self.rectangle, exclusion]}])
        self.assertEqual(row["status"], "excluded")
        self.assertEqual(row["split"], "excluded")

    def test_low_confidence_untouched_is_kept_after_confirmation(self):
        row = self.select([])
        self.assertEqual(row["status"], "positive")
        self.assertEqual(row["boxes_pixels"], [[192, 216, 288, 324]])
        self.assertEqual(row["label_source"], "user_confirmed_unchanged_luna")
        with self.assertRaises(ValueError):
            self.select([], confirm=False)

    def test_latest_non_cancelled_annotation_wins(self):
        row = self.select([
            {"id": 1, "updated_at": "2026-10-08", "result": [self.rectangle]},
            {"id": 2, "updated_at": "2026-10-09", "result": []},
            {"id": 3, "updated_at": "2026-10-10", "was_cancelled": True,
             "result": [self.rectangle]},
        ])
        self.assertEqual(row["annotation_id"], 2)
        self.assertEqual(row["status"], "negative")

    def test_invalid_box_is_rejected(self):
        self.rectangle["value"]["x"] = 99
        with self.assertRaises(ValueError):
            self.select([{"result": [self.rectangle]}])

    def test_multiple_human_boxes_are_preserved(self):
        row = self.select([{"result": [self.rectangle, self.rectangle]}])
        self.assertEqual(len(row["boxes_pixels"]), 2)

    def test_180_rotation_preserves_physical_rectangle(self):
        self.rectangle["value"]["rotation"] = 180
        row = self.select([{"result": [self.rectangle]}])
        self.assertEqual(row["boxes_pixels"], [[96, 108, 192, 216]])
        self.assertEqual(row["rotations_canonicalized"], [180])
        # Conversion never changes the saved human result itself.
        self.assertEqual(self.rectangle["value"]["x"], 10)

    def test_non_axis_rotation_is_not_silently_enlarged(self):
        self.rectangle["value"]["rotation"] = 45
        with self.assertRaises(ValueError):
            self.select([{"result": [self.rectangle]}])
