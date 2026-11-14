import copy
import unittest

from box2boxcam.data.build_personal_train import select_train


class PersonalTrainSelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "item_id": "frame_0", "source_frame": 0, "timestamp_seconds": 0,
            "video_id": "clip", "source_video": "clip.MOV", "match_id": "match",
            "width": 1920, "height": 1080, "image_path": "image.jpg", "luna_confidence": .86,
        }
        self.rectangle = {"from_name": "bbox", "type": "rectangle",
                          "value": {"x": 10, "y": 20, "width": 5, "height": 10}}
        self.proposal = {"data": self.data, "predictions": [{"result": [self.rectangle]}]}

    def select(self, results=None):
        annotations = [] if results is None else [{"id": 1, "result": results}]
        return select_train({"data": self.data, "annotations": annotations}, self.proposal, .86)

    def test_threshold_is_inclusive(self):
        self.assertEqual(self.select()["split"], "train")
        self.data["luna_confidence"] = .859
        self.assertEqual(self.select()["status"], "not_selected_below_threshold")

    def test_low_confidence_human_correction_is_retained(self):
        self.data["luna_confidence"] = .3
        row = self.select([self.rectangle])
        self.assertEqual(row["split"], "train")
        self.assertEqual(row["label_source"], "submitted_human")

    def test_exclusion_wins_even_at_high_confidence(self):
        self.data["luna_confidence"] = .99
        row = self.select([self.rectangle, {"from_name": "decision",
                                           "value": {"choices": ["Exclure image"]}}])
        self.assertEqual(row["split"], "excluded")
        self.assertEqual(row["boxes_pixels"], [])

    def test_submitted_negative_overrides_luna_box(self):
        row = self.select([])
        self.assertEqual(row["status"], "negative")
        self.assertEqual(row["split"], "train")

    def test_reviewed_high_confidence_negative_is_retained(self):
        self.proposal["predictions"][0]["result"] = []
        self.assertEqual(self.select()["status"], "negative")

    def test_no_in_place_changes_to_human_or_model_results(self):
        original = copy.deepcopy(self.proposal)
        self.select([self.rectangle])
        self.assertEqual(self.proposal, original)
