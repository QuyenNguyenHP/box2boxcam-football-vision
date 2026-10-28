"""Only complete human reviews can become field-validation labels."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pitchlens.data.field_validation import SUBMIT_PROTOCOL, parse_reviewed_task


def chosen(name, value, region=None):
    result = {"from_name": name, "type": "choices", "value": {"choices": [value]}}
    if region:
        result["id"] = region
    return result


def box(region, x=10):
    return {
        "id": region,
        "from_name": "bbox",
        "type": "rectangle",
        "value": {"x": x, "y": 20, "width": 5, "height": 10},
    }


class HumanReferenceTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "data": {
                "item_id": "clip_frame_0",
                "match_id": "held_out",
                "video_id": "clip",
                "source_video": "clip.mov",
                "source_video_sha256": "abc",
                "source_frame": 0,
                "timestamp_seconds": 0.0,
                "width": 1920,
                "height": 1080,
                "image_sha256": "def",
            }
        }

    def submit(self, results):
        self.task["annotations"] = [{"id": 5, "result": results}]
        return parse_reviewed_task(self.task)

    def test_predictions_drafts_and_cancelled_reviews_are_not_ground_truth(self):
        self.task.update(
            predictions=[{"result": [box("ai")]}],
            drafts=[{"result": [box("draft")]}],
            annotations=[{"id": 1, "was_cancelled": True, "result": []}],
        )
        self.assertIsNone(parse_reviewed_task(self.task))

    def test_missing_completeness_decision_is_not_a_negative(self):
        with self.assertRaisesRegex(ValueError, "completeness"):
            self.submit([chosen("active_state", "Actif hors champ")])

    def test_single_submit_confirms_boxes_without_extra_choices(self):
        self.task["data"]["review_protocol"] = SUBMIT_PROTOCOL
        record = self.submit([box("new-human-box")])
        self.assertEqual(record["status"], "positive")
        self.assertEqual(record["boxes_pixels"], [[192, 216, 288, 324]])
        self.assertEqual(record["roles"], ["Rôle indéterminé"])
        self.assertEqual(record["review_confirmation"], "submit")
        self.assertFalse(record["active_identity_reviewed"])

    def test_single_submit_still_never_auto_approves_predictions_or_drafts(self):
        self.task["data"]["review_protocol"] = SUBMIT_PROTOCOL
        self.task.update(
            predictions=[{"result": [box("ai")]}],
            drafts=[{"result": [box("human-draft")]}],
        )
        self.assertIsNone(parse_reviewed_task(self.task))

    def test_single_submit_after_deleting_ai_box_is_a_real_negative(self):
        self.task["data"]["review_protocol"] = SUBMIT_PROTOCOL
        # A deleted region can leave stale optional AI metadata. It must not
        # block detection review or become active-ball ground truth.
        record = self.submit(
            [
                chosen("active_state", "Actif visible"),
                chosen("ball_role", "Actif", "deleted-box"),
            ]
        )
        self.assertEqual(record["status"], "negative")
        self.assertEqual(record["active_state"], "Actif indéterminé")
        self.assertIsNone(record["active_box_pixels"])

    def test_single_submit_does_not_validate_luna_active_identity_implicitly(self):
        self.task["data"]["review_protocol"] = SUBMIT_PROTOCOL
        record = self.submit(
            [
                box("ai"),
                chosen("ball_role", "Actif", "ai"),
                chosen("active_state", "Actif visible"),
            ]
        )
        self.assertEqual(record["status"], "positive")
        self.assertFalse(record["active_identity_reviewed"])
        self.assertIsNone(record["active_box_pixels"])

    def test_optional_active_identity_review_is_preserved_when_confirmed(self):
        self.task["data"]["review_protocol"] = SUBMIT_PROTOCOL
        record = self.submit(
            [
                box("ai"),
                chosen("ball_role", "Actif", "ai"),
                chosen("active_state", "Actif visible"),
                chosen("active_review_status", "Identité vérifiée"),
            ]
        )
        self.assertTrue(record["active_identity_reviewed"])
        self.assertEqual(record["active_box_pixels"], [192, 216, 288, 324])

    def test_single_submit_can_still_exclude_an_unusable_image(self):
        self.task["data"]["review_protocol"] = SUBMIT_PROTOCOL
        record = self.submit([chosen("review_status", "Image inexploitable")])
        self.assertEqual(record["status"], "excluded")

    def test_migration_preserves_previously_submitted_active_identity(self):
        self.task["data"].update(
            review_protocol=SUBMIT_PROTOCOL, legacy_active_annotation_ids=[5]
        )
        record = self.submit(
            [
                chosen("review_status", "Vérifié"),
                chosen("active_state", "Actif visible"),
                box("old"),
                chosen("ball_role", "Actif", "old"),
            ]
        )
        self.assertTrue(record["active_identity_reviewed"])
        self.assertEqual(record["active_box_pixels"], [192, 216, 288, 324])
        self.assertEqual(record["review_confirmation"], "review_status")

    def test_verified_empty_frame_is_a_real_negative(self):
        record = self.submit(
            [
                chosen("review_status", "Vérifié"),
                chosen("active_state", "Actif hors champ"),
            ]
        )
        self.assertEqual(record["status"], "negative")
        self.assertEqual(record["boxes_pixels"], [])
        self.assertEqual(record["label_source"], "submitted_human")

    def test_all_balls_kept_but_only_one_is_active(self):
        record = self.submit(
            [
                chosen("review_status", "Vérifié"),
                chosen("active_state", "Actif visible"),
                box("a"),
                chosen("ball_role", "Actif", "a"),
                box("b", 50),
                chosen("ball_role", "Secondaire", "b"),
            ]
        )
        self.assertEqual(
            record["boxes_pixels"], [[192, 216, 288, 324], [960, 216, 1056, 324]]
        )
        self.assertEqual(record["active_box_pixels"], record["boxes_pixels"][0])
        self.assertEqual(record["roles"], ["Actif", "Secondaire"])

    def test_inconsistent_active_visibility_is_rejected(self):
        for state, roles in [
            ("Actif visible", []),
            ("Actif visible", ["Actif", "Actif"]),
            ("Actif masqué", ["Actif"]),
        ]:
            with self.subTest(state=state, roles=roles):
                results = [
                    chosen("review_status", "Vérifié"),
                    chosen("active_state", state),
                ]
                for i, role in enumerate(roles):
                    results.extend([box(str(i)), chosen("ball_role", role, str(i))])
                with self.assertRaisesRegex(ValueError, "disagree"):
                    self.submit(results)

    def test_boxes_need_roles_and_roles_need_boxes(self):
        base = [
            chosen("review_status", "Vérifié"),
            chosen("active_state", "Actif indéterminé"),
        ]
        for extra in [[box("a")], [chosen("ball_role", "Secondaire", "missing")]]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.submit(base + extra)

    def test_out_of_frame_box_is_rejected(self):
        with self.assertRaises(ValueError):
            self.submit(
                [
                    chosen("review_status", "Vérifié"),
                    chosen("active_state", "Actif indéterminé"),
                    box("a", 99),
                    chosen("ball_role", "Rôle indéterminé", "a"),
                ]
            )

    def test_unusable_image_is_excluded_not_negative(self):
        record = self.submit([chosen("review_status", "Image inexploitable")])
        self.assertEqual(record["status"], "excluded")

    def test_multiple_human_annotations_require_adjudication(self):
        self.task["annotations"] = [{"id": 1}, {"id": 2}]
        with self.assertRaisesRegex(ValueError, "adjudication"):
            parse_reviewed_task(self.task)

    def test_freeze_refuses_pending_reference_before_creating_output(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts/review_field_validation.py"
        )
        spec = importlib.util.spec_from_file_location("field_review", script)
        review = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(review)
        report = {"ready_for_metrics": False, "pending": 568, "invalid": []}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(review, "reviewed", return_value=([], report)),
        ):
            output = Path(directory) / "benchmark"
            with self.assertRaisesRegex(RuntimeError, "no labels frozen"):
                review.freeze(Path("config.json"), output)
            self.assertFalse(output.exists())

    def test_freeze_preserves_active_identity_and_excludes_unusable_images(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts/review_field_validation.py"
        )
        spec = importlib.util.spec_from_file_location("field_review_export", script)
        review = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(review)
        positive = self.submit(
            [
                chosen("review_status", "Vérifié"),
                chosen("active_state", "Actif visible"),
                box("a"),
                chosen("ball_role", "Actif", "a"),
            ]
        )
        excluded = dict(
            positive,
            item_id="unusable_frame",
            status="excluded",
            boxes_pixels=[],
            roles=[],
            active_box_pixels=None,
        )
        records = [positive, excluded]
        report = {"ready_for_metrics": True, "pending": 0, "invalid": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "shared"
            exposed = shared / review.EXPOSED
            exposed.mkdir(parents=True)
            image = root / "input.jpg"
            image.write_bytes(b"immutable image bytes for export test")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            for record in records:
                record["image_sha256"] = digest
            (exposed / "tasks.json").write_text(
                json.dumps(
                    [
                        {"data": {"item_id": r["item_id"], "image_path": "input.jpg"}}
                        for r in records
                    ]
                )
            )
            output = root / "benchmark"
            with (
                patch.object(review, "ROOT", root),
                patch.object(review, "SHARED", shared),
                patch.object(review, "reviewed", return_value=(records, report)),
            ):
                review.freeze(Path("config.json"), output)
                with self.assertRaises(FileExistsError):
                    review.freeze(Path("config.json"), output)
            self.assertEqual(
                sorted(p.stem for p in (output / "images").iterdir()),
                [positive["item_id"]],
            )
            label = (output / "labels" / (positive["item_id"] + ".txt")).read_text()
            self.assertEqual(
                label.strip(), "0 0.12500000 0.25000000 0.05000000 0.10000000"
            )
            manifest = json.loads((output / "review_manifest.json").read_text())
            kept = next(r for r in manifest if r["status"] == "positive")
            self.assertEqual(kept["active_box_pixels"], [192, 216, 288, 324])
            self.assertIn("train: null", (output / "dataset.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
