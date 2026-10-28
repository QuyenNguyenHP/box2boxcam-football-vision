import importlib.util
import unittest
from pathlib import Path


def load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/annotate_personal_matches_luna.py"
    )
    spec = importlib.util.spec_from_file_location("personal_match_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PersonalSelectionTests(unittest.TestCase):
    def test_review_all_keeps_negatives_low_confidence_and_api_failures(self):
        runner = load_runner()
        for prediction in (
            None,
            {"ball_in_play_visible": False, "confidence": 0.99},
            {"ball_in_play_visible": True, "confidence": 0.1},
        ):
            self.assertTrue(runner.keep_frame(
                "train_candidate", prediction, 0.94, retain_all=True,
            ))

    def test_test_reserve_is_not_filtered_by_confidence_or_api_failure(self):
        runner = load_runner()
        for role in ("test_primary", "test_reserve"):
            for prediction in (
                None,
                {"ball_in_play_visible": False, "confidence": 0.01},
                {"ball_in_play_visible": True, "confidence": 0.1},
            ):
                self.assertTrue(runner.keep_frame(role, prediction, 0.94))

    def test_train_only_selects_positive_predictions_at_threshold(self):
        runner = load_runner()
        self.assertFalse(runner.keep_frame("train_candidate", None, 0.88))
        self.assertFalse(
            runner.keep_frame(
                "train_candidate",
                {"ball_in_play_visible": False, "confidence": 0.99},
                0.88,
            )
        )
        self.assertFalse(
            runner.keep_frame(
                "train_candidate",
                {"ball_in_play_visible": True, "confidence": 0.87},
                0.88,
            )
        )
        self.assertTrue(
            runner.keep_frame(
                "train_candidate",
                {"ball_in_play_visible": True, "confidence": 0.88},
                0.88,
            )
        )
