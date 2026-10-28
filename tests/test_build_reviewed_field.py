import copy
import unittest

from pitchlens.data.build_reviewed_field import SESSION, select_submitted


class FieldSelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(item_id="frame", source_frame=0, video_id="IMG_7030", source_video="clip.MOV",
                         match_id=SESSION, width=200, height=100, image_path="image.jpg",
                         annotation_scope="all_visible", intended_split="train", timestamp_seconds=0, luna_confidence=0)
        self.box = {"from_name": "bbox", "type": "rectangle", "value": {"x": 10, "y": 20, "width": 10, "height": 10}}
        self.proposal = {"data": self.data, "predictions": [{"score": 1, "result": [self.box]}]}

    def task(self, results):
        return {"data": dict(self.data), "annotations": [{"id": 1, "result": results}]}

    def test_unsubmitted_high_confidence_is_never_training_ground_truth(self):
        self.assertIsNone(select_submitted(self.proposal, self.proposal))

    def test_human_empty_label_is_negative_even_when_detector_proposed_a_ball(self):
        row = select_submitted(self.task([]), self.proposal)
        self.assertEqual(row["status"], "negative")
        self.assertEqual(row["label_source"], "submitted_human")

    def test_multiple_balls_are_kept_without_a_confidence_filter(self):
        second = copy.deepcopy(self.box)
        second["value"]["x"] = 50
        row = select_submitted(self.task([self.box, second]), self.proposal)
        self.assertEqual(row["boxes_pixels"], [[20, 20, 40, 30], [100, 20, 120, 30]])
        self.assertEqual(row["split"], "unassigned")

    def test_pending_draft_does_not_train_an_outdated_submitted_label(self):
        task = self.task([self.box])
        task["drafts"] = [{"result": []}]
        self.assertIsNone(select_submitted(task, self.proposal))

    def test_excluded_image_is_not_a_negative(self):
        task = self.task([{"from_name": "decision", "value": {"choices": ["Exclure image"]}}])
        self.assertEqual(select_submitted(task, self.proposal)["split"], "excluded")

    def test_changed_source_dimensions_and_reserved_split_are_rejected(self):
        task = self.task([self.box])
        task["data"]["width"] = 100
        with self.assertRaises(ValueError):
            select_submitted(task, self.proposal)
        task = self.task([self.box])
        task["data"]["intended_split"] = "test"
        with self.assertRaises(ValueError):
            select_submitted(task, self.proposal)
