import copy
import unittest

from box2boxcam.data.build_reviewed_closeups import select_closeup


class CloseupSelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "item_id": "clip_frame_00000000", "source_frame": 0,
            "timestamp_seconds": 0, "video_id": "clip", "source_video": "clip.MOV",
            "width": 100, "height": 200, "image_path": "image.jpg",
            "match_id": "closeup_session", "source_video_sha256": "abc",
            "luna_confidence": .1,
        }
        self.box = {"from_name": "bbox", "type": "rectangle",
                    "value": {"x": 10, "y": 20, "width": 5, "height": 10}}
        self.proposal = {"data": self.data, "predictions": [{"result": [self.box]}]}

    def select(self, results=None, drafts=None, confirmed=True):
        task = {"data": self.data, "annotations": [] if results is None else [
            {"id": 1, "result": results}], "drafts": drafts or []}
        return select_closeup(task, self.proposal, confirm_reviewed=confirmed)

    def test_low_confidence_and_unsubmitted_are_kept_when_all_reviewed(self):
        row = self.select()
        self.assertEqual(row["boxes_pixels"], [[10, 40, 15, 60]])
        self.assertEqual(row["split"], "unassigned")
        self.assertEqual(row["label_source"], "user_confirmed_unchanged_luna")

    def test_submitted_correction_has_priority(self):
        box = copy.deepcopy(self.box)
        box["value"]["x"] = 30
        self.assertEqual(self.select([box])["boxes_pixels"], [[30, 40, 35, 60]])

    def test_deleted_box_is_a_reviewed_negative(self):
        self.assertEqual(self.select([])["status"], "negative")

    def test_exclusion_is_not_a_negative(self):
        row = self.select([self.box, {"from_name": "decision",
                                    "value": {"choices": ["Exclure image"]}}])
        self.assertEqual(row["split"], "excluded")
        self.assertEqual(row["boxes_pixels"], [])

    def test_drafts_and_missing_confirmation_cannot_be_ignored(self):
        with self.assertRaises(ValueError):
            self.select(drafts=[{"result": []}])
        with self.assertRaises(ValueError):
            self.select(confirmed=False)


if __name__ == "__main__":
    unittest.main()
