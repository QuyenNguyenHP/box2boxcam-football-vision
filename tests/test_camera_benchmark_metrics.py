import itertools
import json
import unittest
from pathlib import Path

import numpy as np

from box2boxcam.benchmark.aggregate import aggregate, bootstrap_mean, segments
from box2boxcam.benchmark.metrics import FPS, WARM, lag_of, score, transitions
from box2boxcam.benchmark.variants import VARIANTS
from box2boxcam.camera_sim.planner import DEFAULTS, Planner, make_attention
from box2boxcam.camera_sim.rig import Rig

SEGMENTS = (
    Path(__file__).resolve().parents[1] / "configs/camera_benchmark_segments.json"
)


def ground_truth(n, feet, action):
    """Constant ground truth: the same players and action centre for ``n`` decisions."""
    return {
        "t": np.arange(n) / FPS,
        "feet": [np.asarray(feet, float)] * n,
        "ac": np.tile(action, (n, 1)),
        "events": [],
    }


def rows_at(center, view, n):
    return [
        {"center": list(center), "view": list(view), "measured": False}
        for _ in range(n)
    ]


class TransitionTest(unittest.TestCase):
    def test_fast_move_is_detected_once(self):
        ac = np.r_[np.full(20, 0.2), np.linspace(0.2, 0.8, 10), np.full(20, 0.8)]
        found = transitions(ac, view_w=0.4)
        self.assertEqual(len(found), 1)
        start, end, sign = found[0]
        self.assertEqual(start, 0)
        self.assertEqual(sign, 1)
        self.assertEqual(end, 29)

    def test_slow_drift_is_not_a_transition(self):
        self.assertEqual(transitions(np.linspace(0, 0.3, 100), view_w=0.4), [])

    def test_lag_is_time_between_half_crossings(self):
        ac = np.r_[np.zeros(10), np.ones(30)]
        cx = np.r_[np.zeros(15), np.ones(25)]  # camera follows 0.5 s late
        lag, reached = lag_of(cx, ac, 9, 10, 1)
        self.assertTrue(reached)
        self.assertAlmostEqual(lag, 0.5)

    def test_lag_when_camera_never_arrives(self):
        ac = np.r_[np.zeros(10), np.ones(30)]
        lag, reached = lag_of(np.zeros(40), ac, 9, 10, 1, horizon=8.0)
        self.assertFalse(reached)
        self.assertEqual(lag, 8.0)


class ScoreTest(unittest.TestCase):
    def test_camera_on_action_scores_perfectly(self):
        n = 100
        feet = [[0.48, 0.5], [0.5, 0.5], [0.52, 0.5], [0.9, 0.5]]
        g = ground_truth(n, feet, [0.5, 0.5])
        m = score(rows_at([0.5, 0.5], [0.25, 0.25], n), g)
        self.assertEqual(m["n_in"], 3.0)  # the far player at 0.9 is out of frame
        self.assertEqual(m["frac_in"], 0.75)
        self.assertEqual(m["lt3"], 0.0)
        self.assertEqual(m["ac_dx"], 0.0)
        self.assertEqual(m["ac_in"], 1.0)
        self.assertEqual(m["n_trans"], 0)
        self.assertEqual(m["reversals_pm"], 0.0)
        self.assertEqual(m["dur"], n / FPS - WARM)

    def test_camera_away_from_action(self):
        n = 100
        g = ground_truth(n, [[0.5, 0.5]] * 3, [0.5, 0.5])
        m = score(rows_at([0.1, 0.5], [0.2, 0.2], n), g)
        self.assertEqual(m["n_in"], 0.0)
        self.assertEqual(m["lt3"], 1.0)
        self.assertAlmostEqual(m["ac_dx"], 2.0)  # 0.4 away, 0.2 wide
        self.assertEqual(m["ac_in"], 0.0)

    def test_reversals_are_counted_per_minute(self):
        n = 200
        # Triangle wave: the pan direction flips every 10 decisions (1 s).
        wave = np.abs((np.arange(n) % 20) - 10) * 0.01
        rows = [
            {"center": [0.5 + x, 0.5], "view": [0.2, 0.2], "measured": False}
            for x in wave
        ]
        g = ground_truth(n, [[0.5, 0.5]] * 3, [0.5, 0.5])
        m = score(rows, g)
        # Reversals happen at decisions 40, 50, ..., 190 (16 flips) inside the scored window.
        minutes = (n - WARM * FPS) / FPS / 60
        self.assertAlmostEqual(m["reversals_pm"], 16 / minutes)


class AggregateTest(unittest.TestCase):
    def test_bootstrap_mean_is_seeded_and_brackets_mean(self):
        values = [1.0, 2.0, 3.0, 4.0]
        a = bootstrap_mean(values, np.random.default_rng(0), resamples=100)
        b = bootstrap_mean(values, np.random.default_rng(0), resamples=100)
        self.assertEqual(a, b)
        self.assertEqual(a[0], 2.5)
        self.assertLessEqual(a[1], 2.5)
        self.assertGreaterEqual(a[2], 2.5)

    def test_paired_difference_against_first_variant(self):
        def metrics(lt3):
            keys = [
                "n_in", "frac_in", "ac_dx", "ac_in", "ev_in", "ev_dx", "ev_m",
                "ev_ac_in", "lag", "speed", "reversals_pm", "ball_used",
            ]  # fmt: skip
            m = dict.fromkeys(keys, 1.0)
            m.update(lt3=lt3, lags=[1.0, 2.0])
            return m

        per_seg = [
            {"id": str(i), "a": metrics(0.5), "b": metrics(0.3)} for i in range(4)
        ]
        agg = aggregate(per_seg, ["a", "b"])
        self.assertAlmostEqual(agg["b"]["lt3"][0], 0.3)
        self.assertAlmostEqual(agg["b"]["paired_vs_a"]["lt3"][0], -0.2)
        self.assertEqual(agg["a"]["n_trans"], 8)
        self.assertAlmostEqual(agg["a"]["lag_pooled"], 1.5)


class SegmentFileTest(unittest.TestCase):
    def test_split_sizes_and_overlaps(self):
        sets = {s: segments(SEGMENTS, s) for s in ("TUNE", "TEST", "FINAL")}
        self.assertEqual(
            {k: len(v) for k, v in sets.items()}, {"TUNE": 7, "TEST": 5, "FINAL": 4}
        )
        self.assertEqual(len(segments(SEGMENTS, "ALL")), 16)
        self.assertEqual(len(segments(SEGMENTS, "REF")), 1)
        total = sum(s["t1"] - s["t0"] for s in segments(SEGMENTS, "ALL"))
        self.assertEqual(total, 577)
        ids = [s["id"] for s in segments(SEGMENTS, "ALL")]
        self.assertEqual(len(ids), len(set(ids)))
        by_match = {}
        for s in segments(SEGMENTS, "ALL"):
            by_match.setdefault(s["match"], []).append((s["t0"], s["t1"]))
        for windows in by_match.values():
            windows.sort()
            for (_, end), (start, _) in itertools.pairwise(windows):
                self.assertLessEqual(end, start)

    def test_file_has_no_local_paths(self):
        self.assertNotIn("/Users", SEGMENTS.read_text())
        self.assertIn("segments", json.loads(SEGMENTS.read_text()))


class VariantTest(unittest.TestCase):
    def test_variant_keys_are_known(self):
        for name, cfg in VARIANTS.items():
            for key in cfg:
                self.assertTrue(
                    key in DEFAULTS
                    or key in ("no_ball", "static", "oracle", "max_speed"),
                    (name, key),
                )

    def test_main_variants_exist(self):
        for name in (
            "current",
            "guard",
            "centroid",
            "centroid_motion",
            "static",
            "oracle_ac",
        ):
            self.assertIn(name, VARIANTS)
        self.assertEqual(VARIANTS["current"], {})
        self.assertTrue(VARIANTS["guard"]["guard"])

    def test_make_attention_filters_rig_only_keys(self):
        planner = make_attention(
            VARIANTS["guard_v5"],
            [100, 50],
            np.array([0.4, 0.2]),
            [[0, 0], [1, 0], [1, 1]],
        )
        self.assertIsInstance(planner, Planner)
        self.assertTrue(planner.p["guard"])
        self.assertEqual(planner.p["vel_lead"], 0.5)


class RigSmokeTest(unittest.TestCase):
    """The rig is deterministic and steers toward a visible group of players."""

    def run_rig(self, variant):
        size = np.array([3840, 1906])
        field = [[0.05, 0.3], [0.95, 0.3], [0.98, 0.9], [0.02, 0.9]]
        rig = Rig(VARIANTS[variant], size, field, np.array([0.3, 0.55]))
        feet = np.array(
            [[0.36, 0.6], [0.38, 0.62], [0.4, 0.58], [0.37, 0.65], [0.39, 0.6]]
        )
        people = np.c_[
            feet[:, 0] - 0.004,
            feet[:, 1] - 0.05,
            feet[:, 0] + 0.004,
            feet[:, 1],
            np.full(5, 0.9),
        ]
        det = {
            "people": people,
            "pose": np.full((5, 17, 3), np.nan),
            "ball": np.empty((0, 5)),
        }
        for k in range(120):
            rig.step(k, det, k / FPS)
        return rig

    def test_centroid_variant_moves_toward_players_and_repeats(self):
        first = self.run_rig("centroid_motion")
        again = self.run_rig("centroid_motion")
        self.assertEqual(len(first.rows), 120)
        np.testing.assert_array_equal(
            first.rows[-1]["center"], again.rows[-1]["center"]
        )
        self.assertGreater(first.rows[-1]["center"][0], 0.32)
        self.assertEqual(first.rows[-1]["mode"].split(":")[0], "players")
        self.assertEqual(first.rows[-1]["n"], 5)

    def test_guard_variant_runs_without_orientation_data(self):
        rig = self.run_rig("guard")
        self.assertTrue(np.isfinite(rig.rows[-1]["center"]).all())


if __name__ == "__main__":
    unittest.main()
