import copy
import unittest

from pitchlens.data.build_reviewed_shorts import select_short


class ReviewedShortsTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "item_id": "clip_frame_00000000", "source_frame": 0,
            "timestamp_seconds": 0, "video_id": "clip", "source_video": "clip.mp4",
            "width": 1080, "height": 1920, "image_path": "image.jpg",
            "source_url": "https://example.com/video", "uploader": "source",
            "luna_confidence": .86,
        }
        self.rectangle = {"from_name": "bbox", "type": "rectangle",
                          "value": {"x": 10, "y": 20, "width": 5, "height": 10}}
        self.proposal = {"data": self.data, "predictions": [{"result": [self.rectangle]}]}

    def select(self, results=None, drafts=None):
        annotations = [] if results is None else [{"id": 1, "result": results}]
        return select_short(
            {"data": self.data, "annotations": annotations, "drafts": drafts or []},
            self.proposal, threshold=.85,
        )

    def test_confirmed_unchanged_above_threshold_is_retained_unsplit(self):
        row = self.select()
        self.assertEqual(row["label_source"], "user_confirmed_unchanged_luna")
        self.assertEqual(row["split"], "unassigned")
        self.assertEqual(row["match_id"], "")
        self.assertEqual(row["boxes_pixels"], [[108, 384, 162, 576]])

    def test_threshold_is_strict(self):
        self.data["luna_confidence"] = .85
        self.assertEqual(self.select()["split"], "not_selected")
        self.data["luna_confidence"] = .849
        self.assertEqual(self.select()["split"], "not_selected")

    def test_submitted_correction_below_threshold_is_retained(self):
        self.data["luna_confidence"] = .4
        self.assertEqual(self.select([self.rectangle])["label_source"], "submitted_human")
        self.assertEqual(self.select([self.rectangle])["split"], "unassigned")

    def test_human_empty_negative_overrides_model_box(self):
        row = self.select([])
        self.assertEqual(row["status"], "negative")
        self.assertEqual(row["boxes_pixels"], [])

    def test_reviewed_unchanged_negative_is_kept(self):
        self.proposal["predictions"][0]["result"] = []
        self.assertEqual(self.select()["status"], "negative")

    def test_exclusion_overrides_box_at_any_confidence(self):
        self.data["luna_confidence"] = .1
        exclusion = {"from_name": "decision", "value": {"choices": ["Exclure image"]}}
        self.assertEqual(self.select([self.rectangle, exclusion])["split"], "excluded")

    def test_draft_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            self.select(drafts=[{"result": []}])

    def test_cancelled_annotation_does_not_bypass_threshold(self):
        self.data["luna_confidence"] = .3
        row = select_short(
            {"data": self.data, "annotations": [{"was_cancelled": True, "result": [self.rectangle]}]},
            self.proposal, threshold=.85,
        )
        self.assertEqual(row["split"], "not_selected")

    def test_original_metadata_and_proposals_are_unchanged(self):
        before = copy.deepcopy(self.proposal)
        self.select([self.rectangle])
        self.assertEqual(self.proposal, before)
        self.assertNotIn("match_id", self.data)


if __name__ == "__main__":
    unittest.main()
