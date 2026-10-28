import unittest

from pitchlens.training.camera_benchmark import benchmark_targets, score_target


class CameraBenchmarkTests(unittest.TestCase):
    def test_secondary_false_box_does_not_penalize_correct_top_target(self):
        result = score_target([[0, 0, 1, 1]], [[0, 0, 1, 1], [2, 2, 3, 3]], [.9, .8])
        self.assertEqual(result["outcome"], "correct_target")
        self.assertEqual(result["selected_index"], 0)

    def test_no_oracle_selecting_less_confident_correct_box(self):
        result = score_target([[0, 0, 1, 1]], [[0, 0, 1, 1], [2, 2, 3, 3]], [.8, .9])
        self.assertEqual(result["outcome"], "wrong_target")

    def test_confidence_gate_and_inclusive_boundary(self):
        box = [[0, 0, 1, 1]]
        self.assertEqual(score_target(box, box, [.24])["outcome"], "missed")
        self.assertEqual(score_target(box, box, [.25])["outcome"], "correct_target")

    def test_10_and_20_percent_iou_are_distinct(self):
        # Intersection .3; union 1.7 -> IoU ~17.65%, not 30%.
        args = ([[0, 0, 1, 1]], [[.7, 0, 1.7, 1]], [.9])
        self.assertEqual(score_target(*args, matching_iou=.1)["outcome"], "correct_target")
        self.assertEqual(score_target(*args, matching_iou=.2)["outcome"], "wrong_target")

    def test_negative_frames(self):
        self.assertEqual(score_target([], [], [])["outcome"], "correct_silence")
        self.assertEqual(score_target([], [[0, 0, 1, 1]], [.8])["outcome"], "false_alarm")

    def test_multiple_true_balls_and_stable_confidence_tie(self):
        boxes = [[0, 0, 1, 1], [2, 2, 3, 3]]
        result = score_target(boxes, boxes[::-1], [.8, .8])
        self.assertEqual(result["outcome"], "correct_target")
        self.assertEqual(result["selected_index"], 0)

    def test_counts_wrong_target_as_fp_and_fn_but_one_frame_error(self):
        rows = [{"item_id": "a", "match_id": "match", "ground_truth": [[0, 0, 1, 1]],
                 "after": [[2, 2, 3, 3]], "after_confidence": [.8]},
                {"item_id": "b", "match_id": "match", "ground_truth": [],
                 "after": [], "after_confidence": []}]
        result = benchmark_targets(rows, "after")["metrics"]
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (0, 1, 1, 1))
        self.assertEqual(result["correct_decision_rate"], .5)
        with self.assertRaises(ValueError):
            benchmark_targets(rows, "after", confidence=.1)


if __name__ == "__main__":
    unittest.main()
