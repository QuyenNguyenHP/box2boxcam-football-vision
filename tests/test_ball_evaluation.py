import unittest
from collections import Counter
from unittest.mock import patch

import torch
from ultralytics.models.yolo.detect import DetectionValidator

from box2boxcam.training.diagnostics import box_iou, change_category, detection_counts
from box2boxcam.training.evaluate import BallValidator


class BallEvaluationTests(unittest.TestCase):
    def test_only_sports_ball_is_mapped_to_zero(self):
        evaluator = BallValidator.__new__(BallValidator)
        evaluator.ball_class = 32
        output = {"cls": torch.tensor([0., 32., 5.]),
                  "conf": torch.tensor([.9, .8, .7]),
                  "bboxes": torch.zeros((3, 4)), "extra": torch.zeros((3, 0))}
        with patch.object(DetectionValidator, "postprocess", return_value=[output]):
            result = evaluator.postprocess(None)[0]
        self.assertEqual(result["cls"].tolist(), [0.])
        self.assertAlmostEqual(result["conf"].item(), .8)

    def test_fixed_threshold_includes_false_positives_on_negatives(self):
        evaluator = BallValidator.__new__(BallValidator)
        evaluator.fixed = Counter()
        evaluator.niou = 10
        evaluator.iouv = torch.linspace(.5, .95, 10)
        targets = {"cls": torch.empty(0), "bboxes": torch.empty((0, 4))}
        pred = {"cls": torch.tensor([0., 0.]), "conf": torch.tensor([.9, .1]),
                "bboxes": torch.zeros((2, 4)), "extra": torch.zeros((2, 0))}
        with patch.object(evaluator, "_prepare_batch", return_value=targets), patch.object(DetectionValidator, "update_metrics"):
            evaluator.update_metrics([pred], {})
        self.assertEqual(evaluator.fixed["fp"], 1)
        self.assertEqual(evaluator.fixed["fp_on_negatives"], 1)
        self.assertEqual(evaluator.fixed["tp"], 0)

    def test_detection_counts_match_each_box_only_once(self):
        truth = [[0, 0, 10, 10]]
        predictions = [[0, 0, 10, 10], [1, 1, 9, 9]]
        self.assertEqual(detection_counts(truth, predictions), {"tp": 1, "fp": 1, "fn": 0})

    def test_empty_box_iou_has_expected_shape(self):
        self.assertEqual(box_iou([], [[0, 0, 1, 1]]).shape, (0, 1))

    def test_change_categories(self):
        self.assertEqual(change_category({"tp": 0, "fp": 1, "fn": 1}, {"tp": 1, "fp": 0, "fn": 0}), "corrected")
        self.assertEqual(change_category({"tp": 1, "fp": 0, "fn": 0}, {"tp": 0, "fp": 0, "fn": 1}), "regressed")
