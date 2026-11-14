import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from box2boxcam.data.label_studio_review import build_tasks, export_review
from box2boxcam.data.luna_batch import (
    ALL_BALLS_SCHEMA,
    _write_prediction_artifacts,
    parse_batch_output,
    prediction_request_body,
    validate_all_balls_prediction,
)
from box2boxcam.data.luna_inline_batch import collect_inline_batch, write_shards


def ball(box, location="field"):
    return {"bbox_current_pixels": box, "confidence": 0.9,
            "visibility": "clear", "location": location, "temporal_evidence": "immobile"}


class AllVisibleTests(unittest.TestCase):
    def plan(self):
        return {"annotation_scope": "all_visible", "model": "gpt-6-luna",
                "reasoning_effort": "max", "image_detail": "original", "max_output_tokens": 32768,
                "source_video": "test.MOV", "video": {"width": 100, "height": 80, "fps": 30},
                "cases": [{"item_id": "frame_00000000", "source_frame": 0,
                           "previous_frame": 0, "current_frame": 0, "next_frame": 3}]}

    def test_multiple_static_and_sideline_balls_survive_export_and_review(self):
        payload = {"balls": [ball([10, 20, 30, 40]), ball([70, 50, 80, 60], "sideline")],
                   "confidence": 0.8, "ambiguity": ""}
        validate_all_balls_prediction(payload, width=100, height=80)
        prediction = {"item_id": "frame_00000000", "source_frame": 0, **payload}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(project_root=root, output_root=root)
            _write_prediction_artifacts(config=config, plan=self.plan(), predictions=[prediction], failures=[])
            lines = (root / "labels/frame_00000000.txt").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads((root / "summary.json").read_text())["proposed_boxes"], 2)
        tasks = build_tasks(self.plan(), {prediction["item_id"]: prediction})
        rectangles = tasks[0]["predictions"][0]["result"]
        self.assertEqual(len(rectangles), 2)
        self.assertNotEqual(rectangles[0]["id"], rectangles[1]["id"])

    def test_empty_valid_inventory_is_negative_but_missing_api_answer_is_not(self):
        plan = self.plan()
        plan["cases"].append({"item_id": "frame_00000006", "source_frame": 6})
        prediction = {"item_id": "frame_00000000", "source_frame": 0,
                      "balls": [], "confidence": 0.8, "ambiguity": ""}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_prediction_artifacts(config=SimpleNamespace(project_root=root, output_root=root),
                                        plan=plan, predictions=[prediction], failures=[{"item_id": "frame_00000006"}])
            self.assertEqual((root / "labels/frame_00000000.txt").read_text(), "")
            self.assertFalse((root / "labels/frame_00000006.txt").exists())

    def test_invalid_or_duplicated_box_is_rejected(self):
        for balls in ([ball([90, 0, 110, 20])], [ball([1, 1, 4, 4]), ball([1, 1, 4, 4])]):
            with self.assertRaises(ValueError):
                validate_all_balls_prediction({"balls": balls, "confidence": 0.9, "ambiguity": ""}, width=100, height=80)

    def test_request_uses_multibox_schema_and_inline_images(self):
        plan = self.plan()
        body = prediction_request_body(case=plan["cases"][0], file_ids={"0": "data:image/jpeg;base64,YQ==", "3": "file-next"}, plan=plan)
        self.assertEqual(body["text"]["format"]["schema"], ALL_BALLS_SCHEMA)
        self.assertIn("ALL", body["input"][0]["content"])
        images = [i for i in body["input"][1]["content"] if i["type"] == "input_image"]
        self.assertIn("image_url", images[0])
        self.assertNotIn("file_id", images[0])
        self.assertEqual(images[-1]["file_id"], "file-next")

    def test_parser_retains_multibox_response(self):
        payload = {"balls": [ball([10, 20, 30, 40]), ball([70, 50, 80, 60], "sideline")], "confidence": 0.8, "ambiguity": ""}
        row = {"custom_id": "frame_00000000", "response": {"status_code": 200, "body": {"output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]}]}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.jsonl"
            path.write_text(json.dumps(row) + "\n")
            predictions, failures = parse_batch_output(output_path=path, plan=self.plan())
            self.assertEqual(len(predictions[0]["balls"]), 2)
            self.assertFalse(failures)

    def test_shards_bound_bytes_and_never_drop_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = write_shards(Path(directory), [(str(n), b"abc\n") for n in range(5)], limit=9)
            self.assertEqual([s["bytes"] for s in shards], [8, 8, 4])
            self.assertEqual([i for s in shards for i in s["item_ids"]], [str(n) for n in range(5)])
            with self.assertRaises(ValueError):
                write_shards(Path(directory), [("too-big", b"a" * 10)], limit=9)

    def test_unfinished_inline_batch_is_not_collected_or_resubmitted(self):
        with patch("box2boxcam.data.luna_inline_batch.refresh_inline_status", return_value={"batch_status": "in_progress"}), patch("box2boxcam.data.luna_inline_batch.openai_client") as client:
            with self.assertRaises(RuntimeError):
                collect_inline_batch(SimpleNamespace())
            client.assert_not_called()

    def test_review_priority_matches_multiple_balls_including_sideline(self):
        path = Path(__file__).resolve().parents[1] / "scripts/prepare_field_all_balls.py"
        spec = importlib.util.spec_from_file_location("field_review", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        comparison = module.compare_proposals(
            [[1, 1, 5, 5], [60, 60, 70, 70], [20, 20, 30, 30]],
            [[60, 60, 70, 70], [1, 1, 5, 5], [80, 60, 90, 70]],
        )
        self.assertEqual(len(comparison["matched"]), 2)
        self.assertEqual(comparison["teacher_only"], [2])
        self.assertEqual(comparison["luna_only"], [2])

    def test_human_export_keeps_multiple_boxes_and_per_image_orientation(self):
        rectangle = {"from_name": "bbox", "type": "rectangle", "value": {"x": 10, "y": 20, "width": 10, "height": 10, "rotation": 0}}
        tasks = [
            {"data": {"item_id": "landscape", "source_frame": 0, "width": 1920, "height": 1080},
             "annotations": [{"id": 1, "result": [rectangle, rectangle]}]},
            {"data": {"item_id": "portrait", "source_frame": 0, "width": 1080, "height": 1920},
             "annotations": [{"id": 2, "result": [rectangle]}]},
            {"data": {"item_id": "pending", "source_frame": 1}, "annotations": []},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "human.json"
            path.write_text(json.dumps(tasks))
            summary = export_review(export_path=path, output_root=root / "export")
            self.assertEqual(summary["reviewed"], 2)
            self.assertEqual(summary["unreviewed"], 1)
            self.assertEqual(len((root / "export/labels/landscape.txt").read_text().splitlines()), 2)
            self.assertEqual((root / "export/labels/portrait.txt").read_text().strip(), "0 0.15000000 0.25000000 0.10000000 0.10000000")
            self.assertFalse((root / "export/labels/pending.txt").exists())


if __name__ == "__main__":
    unittest.main()
