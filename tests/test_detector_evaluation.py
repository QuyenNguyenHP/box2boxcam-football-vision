import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from box2boxcam.evaluation.bootstrap import paired_bootstrap, stat_functions
from box2boxcam.evaluation.ground_truth import (
    load_ground_truth,
    parse_json_labels,
    parse_yolo_label,
)
from box2boxcam.evaluation.matching import (
    count_small_tp,
    count_tp_fp,
    iou,
    is_small,
    match_all,
    match_frame,
    max_tp_under_fp_budget,
    recovery,
    small_balls,
)
from box2boxcam.evaluation.report import evaluate


def square(x, y, side):
    return [x, y, x + side, y + side]


class MatchingTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_small_is_longest_side_at_most_16(self):
        self.assertTrue(is_small([0, 0, 16, 5]))
        self.assertFalse(is_small([0, 0, 17, 5]))

    def test_greedy_matching_uses_each_ground_truth_once(self):
        gt = np.array([square(0, 0, 10)], float)
        boxes = np.array([square(0, 0, 10), square(1, 0, 10)], float)
        matches = match_frame(gt, boxes, np.array([0.4, 0.9]))
        # The most confident detection takes the ball, the other is a false positive.
        self.assertEqual(matches, [(0.9, 0), (0.4, -1)])

    def test_iou_threshold_is_inclusive(self):
        gt = np.array([[0, 0, 10, 10]], float)
        half = np.array([[0, 0, 10, 5]], float)  # IoU exactly 0.5
        self.assertEqual(match_frame(gt, half, np.array([0.8]))[0][1], 0)
        self.assertEqual(match_frame(gt, half, np.array([0.8]), 0.51)[0][1], -1)

    def test_counts_and_thresholds(self):
        gts = [np.array([square(0, 0, 10)], float), np.empty((0, 4))]
        dets = [
            (
                np.array([square(0, 0, 10), square(50, 50, 10)], float),
                np.array([0.6, 0.3]),
            ),
            (np.array([square(5, 5, 10)], float), np.array([0.5])),
        ]
        matches = match_all(gts, dets)
        self.assertEqual(count_tp_fp(matches, 0.25)[:2], (1, 2))
        self.assertEqual(count_tp_fp(matches, 0.45)[:2], (1, 1))
        self.assertEqual(count_tp_fp(matches, 0.7)[:2], (0, 0))
        self.assertEqual(count_tp_fp(matches, 0.25)[2], {(0, 0)})

    def test_resampled_frames_count_with_multiplicity(self):
        matches = [[(0.9, 0)], [(0.9, -1)]]
        self.assertEqual(count_tp_fp(matches, 0.5, [0, 0, 1])[:2], (2, 1))

    def test_budget_sweeps_threshold_down(self):
        # Sorted by confidence: TP, FP, TP, FP, TP.
        matches = [[(0.9, 0), (0.8, -1), (0.7, 1), (0.6, -1), (0.5, 2)]]
        self.assertEqual(max_tp_under_fp_budget(matches, 0), 1)
        self.assertEqual(max_tp_under_fp_budget(matches, 1), 2)
        self.assertEqual(max_tp_under_fp_budget(matches, 2), 3)

    def test_small_true_positives(self):
        gts = [np.array([square(0, 0, 10), square(40, 0, 30)], float)]
        small = small_balls(gts)
        self.assertEqual(small, {(0, 0)})
        matches = [[(0.9, 0), (0.8, 1)]]
        self.assertEqual(count_small_tp(matches, small, 0.5), 1)

    def test_recovery(self):
        gts = [np.array([square(0, 0, 10), square(40, 0, 10)], float)]
        a = [[(0.9, 0), (0.9, 1)]]
        b = [[(0.9, 0)]]
        r = recovery(a, b, gts, 0.25, small_balls(gts))
        self.assertEqual(
            (r["b_misses"], r["a_recovers"], r["a_loses"], r["a_recovers_small"]),
            (1, 1, 0, 1),
        )


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        # 30 frames; model A finds the ball in every frame, model B in none.
        self.a = [[(0.9, 0)] for _ in range(30)]
        self.b = [[(0.9, -1)] for _ in range(30)]
        self.stats = stat_functions((0.25,), (5,), set(), 0.25)

    def test_difference_and_interval(self):
        out = paired_bootstrap(self.a, self.b, self.stats, resamples=200, seed=1)
        tp = out["TP@0.25"]
        self.assertEqual(tp["observed"], 30)
        self.assertLessEqual(tp["low"], 30)
        self.assertGreaterEqual(tp["low"], 20)
        self.assertEqual(tp["p_le_0"], 0.0)
        self.assertEqual(out["FP@0.25"]["observed"], -30)

    def test_same_seed_is_reproducible(self):
        one = paired_bootstrap(self.a, self.b, self.stats, resamples=50, seed=3)
        two = paired_bootstrap(self.a, self.b, self.stats, resamples=50, seed=3)
        self.assertEqual(one, two)

    def test_identical_models_have_zero_difference(self):
        out = paired_bootstrap(self.a, self.a, self.stats, resamples=50)
        self.assertEqual(out["TP@0.25"]["observed"], 0)
        self.assertEqual((out["TP@0.25"]["low"], out["TP@0.25"]["high"]), (0.0, 0.0))

    def test_frame_count_must_match(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.a, self.b[:5], self.stats, resamples=5)


class GroundTruthTest(unittest.TestCase):
    def test_yolo_label_conversion(self):
        boxes = parse_yolo_label("0 0.5 0.5 0.1 0.2\n\n", 200, 100)
        np.testing.assert_allclose(boxes, [[90, 40, 110, 60]])

    def test_bad_yolo_line(self):
        with self.assertRaises(ValueError):
            parse_yolo_label("0 0.5 0.5", 10, 10)

    def test_json_layouts_agree(self):
        as_dict = parse_json_labels({"a": [[0, 0, 4, 4]]})
        as_list = parse_json_labels([{"item_id": "a", "boxes_pixels": [[0, 0, 4, 4]]}])
        np.testing.assert_array_equal(as_dict["a"], as_list["a"])
        with self.assertRaises(ValueError):
            parse_json_labels({"a": [[4, 4, 0, 0]]})

    def test_load_from_directory_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            for name in ("f1", "f2"):
                Image.new("RGB", (100, 50)).save(root / "images" / f"{name}.jpg")
            (root / "labels").mkdir()
            (root / "labels" / "f1.txt").write_text("0 0.5 0.5 0.1 0.2\n")
            frames = load_ground_truth(root / "images", root / "labels")
            self.assertEqual([f.id for f in frames], ["f1", "f2"])
            np.testing.assert_allclose(frames[0].boxes, [[45, 20, 55, 30]])
            self.assertEqual(len(frames[1].boxes), 0)
            labels = root / "labels.json"
            labels.write_text(json.dumps({"f2": [[1, 1, 5, 5]]}))
            frames = load_ground_truth(root / "images", labels)
            self.assertEqual((len(frames[0].boxes), len(frames[1].boxes)), (0, 1))
            labels.write_text(json.dumps({"missing": [[1, 1, 5, 5]]}))
            with self.assertRaises(ValueError):
                load_ground_truth(root / "images", labels)


class ReportTest(unittest.TestCase):
    def test_two_model_report(self):
        gts = [np.array([square(0, 0, 10)], float)] * 4
        good = [(np.array([square(0, 0, 10)], float), np.array([0.9]))] * 4
        bad = [(np.empty((0, 4)), np.empty(0))] * 4
        report = evaluate(
            gts,
            {"good": match_all(gts, good), "bad": match_all(gts, bad)},
            thresholds=(0.25, 0.5),
            budgets=(0, 5),
            resamples=20,
        )
        rows = report["models"]["good"]["thresholds"]
        self.assertEqual((rows[0]["tp"], rows[0]["fp"], rows[0]["small_tp"]), (4, 0, 4))
        self.assertEqual(report["models"]["good"]["max_tp_under_fp_budget"]["0"], 4)
        self.assertEqual(report["bootstrap"]["stats"]["TP@0.25"]["observed"], 4)
        self.assertIn("| good |", report["table"])
        recovered = report["recovery_at_first_threshold"]
        self.assertEqual((recovered["b_misses"], recovered["a_recovers"]), (4, 4))


if __name__ == "__main__":
    unittest.main()
