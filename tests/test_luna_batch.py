from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pitchlens.data.luna_batch import (
    OUTPUT_SCHEMA,
    collect_batch,
    parse_batch_output,
    prediction_request_body,
    temporal_cases,
    validate_prediction,
)


class LunaBatchTests(unittest.TestCase):
    def test_closeup_prompt_override_is_local_to_request_plan(self):
        case = temporal_cases(total_frames=20, stride=4, temporal_offset=3)[0]
        plan = {
            "video": {"width": 1080, "height": 1920},
            "model": "gpt-6-luna", "reasoning_effort": "max",
            "image_detail": "original", "max_output_tokens": 32768,
            "annotation_prompt": "Annotate the visible ball, including held or static balls.",
            "target_instruction": "Locate the visible ball in CURRENT.",
        }
        body = prediction_request_body(
            case=case, file_ids={"0": "file-0", "3": "file-3"}, plan=plan,
        )
        self.assertEqual(body["input"][0]["content"], plan["annotation_prompt"])
        self.assertIn(plan["target_instruction"], body["input"][1]["content"][0]["text"])
        self.assertEqual(body["text"]["format"]["schema"], OUTPUT_SCHEMA)
        from pitchlens.data.luna_batch import DEVELOPER_PROMPT
        del plan["annotation_prompt"]
        del plan["target_instruction"]
        default = prediction_request_body(
            case=case, file_ids={"0": "file-0", "3": "file-3"}, plan=plan,
        )
        self.assertEqual(default["input"][0]["content"], DEVELOPER_PROMPT)
        self.assertIn("ball in play", default["input"][1]["content"][0]["text"])

    def test_expired_collection_requires_opt_in(self):
        with patch("pitchlens.data.luna_batch.build_plan", return_value={}), patch(
            "pitchlens.data.luna_batch.refresh_status", return_value={"batch_status": "expired"}
        ), patch("pitchlens.data.luna_batch.openai_client") as client:
            with self.assertRaises(RuntimeError):
                collect_batch(SimpleNamespace())
            client.assert_not_called()

    def test_partial_collection_preserves_expiry_without_negative_label(self):
        plan = {
            "model": "gpt-6-luna", "reasoning_effort": "max", "source_video": "video.MOV",
            "video": {"width": 100, "height": 80, "fps": 30},
            "cases": [{"item_id": "frame_00000004", "source_frame": 4},
                      {"item_id": "frame_00000008", "source_frame": 8}],
        }
        state = {"batch_status": "expired", "output_file_id": "output", "error_file_id": "errors"}
        payload = {"ball_in_play_visible": True, "bbox_current_pixels": [10, 20, 30, 40],
                   "confidence": 0.9, "temporal_evidence": "", "ambiguity": ""}
        output = {"custom_id": "frame_00000004", "response": {
            "status_code": 200, "body": {"id": "resp-4", "output": [{
                "type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]
            }]}}}
        error = {"custom_id": "frame_00000008", "response": None,
                 "error": {"code": "batch_expired", "message": "Completion window expired"}}

        def download(client, file_id, destination):
            destination.write_text(json.dumps(output if file_id == "output" else error) + "\n")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(output_root=root, project_root=root)
            with patch("pitchlens.data.luna_batch.build_plan", return_value=plan), patch(
                "pitchlens.data.luna_batch.refresh_status", return_value=state
            ), patch("pitchlens.data.luna_batch.openai_client"), patch(
                "pitchlens.data.luna_batch._download_file", side_effect=download
            ):
                summary = collect_batch(config, allow_partial=True)
            self.assertEqual(summary["valid_predictions"], 1)
            self.assertEqual(summary["failed_predictions"], 1)
            self.assertTrue(summary["partial_collection"])
            self.assertEqual(summary["unresolved_item_ids"], ["frame_00000008"])
            self.assertFalse((root / "labels/frame_00000008.txt").exists())
            failures = json.loads((root / "failures.json").read_text())["failures"]
            self.assertIn("batch_expired", failures[0]["error"])

    def test_full_video_plan_has_expected_cases_and_temporal_frames(self) -> None:
        cases = temporal_cases(
            total_frames=4926,
            stride=4,
            temporal_offset=3,
        )
        self.assertEqual(len(cases), 1232)
        self.assertEqual(cases[0]["previous_frame"], 0)
        self.assertEqual(cases[0]["current_frame"], 0)
        self.assertEqual(cases[0]["next_frame"], 3)
        self.assertEqual(cases[-1]["current_frame"], 4924)
        self.assertEqual(cases[-1]["next_frame"], 4925)

    def test_request_labels_three_images_and_uses_strict_schema(self) -> None:
        case = temporal_cases(
            total_frames=100,
            stride=4,
            temporal_offset=3,
            center_frames=[20],
        )[0]
        plan = {
            "video": {"width": 1920, "height": 1080},
            "model": "gpt-6-luna",
            "reasoning_effort": "max",
            "image_detail": "original",
            "max_output_tokens": 8192,
        }
        ids = {"17": "file-prev", "20": "file-current", "23": "file-next"}
        body = prediction_request_body(case=case, file_ids=ids, plan=plan)
        content = body["input"][1]["content"]
        images = [item for item in content if item["type"] == "input_image"]
        self.assertEqual([item["file_id"] for item in images], list(ids.values()))
        self.assertEqual(body["reasoning"]["effort"], "max")
        self.assertEqual(body["max_output_tokens"], 8192)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(body["text"]["format"]["schema"], OUTPUT_SCHEMA)

    def test_validation_accepts_positive_and_negative(self) -> None:
        positive = validate_prediction(
            {
                "ball_in_play_visible": True,
                "bbox_current_pixels": [10, 20, 30, 40],
                "confidence": 0.9,
                "temporal_evidence": "mouvement",
                "ambiguity": "aucune",
            },
            width=100,
            height=80,
        )
        self.assertEqual(positive["center_current_pixels"], [20, 30])
        negative = validate_prediction(
            {
                "ball_in_play_visible": False,
                "bbox_current_pixels": None,
                "confidence": 0.8,
                "temporal_evidence": "absent",
                "ambiguity": "hors champ",
            },
            width=100,
            height=80,
        )
        self.assertIsNone(negative["center_current_pixels"])

    def test_validation_rejects_out_of_bounds_box(self) -> None:
        with self.assertRaises(ValueError):
            validate_prediction(
                {
                    "ball_in_play_visible": True,
                    "bbox_current_pixels": [10, 20, 130, 40],
                    "confidence": 0.9,
                    "temporal_evidence": "",
                    "ambiguity": "",
                },
                width=100,
                height=80,
            )

    def test_batch_parser_restores_source_frame_and_ignores_output_order(self) -> None:
        plan = {
            "video": {"width": 100, "height": 80},
            "cases": [
                {"item_id": "frame_00000004", "source_frame": 4},
                {"item_id": "frame_00000008", "source_frame": 8},
            ],
        }

        def line(item_id: str, visible: bool) -> str:
            payload = {
                "ball_in_play_visible": visible,
                "bbox_current_pixels": [10, 20, 30, 40] if visible else None,
                "confidence": 0.9,
                "temporal_evidence": "test",
                "ambiguity": "aucune",
            }
            return json.dumps(
                {
                    "custom_id": item_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": f"resp-{item_id}",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": json.dumps(payload),
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.jsonl"
            path.write_text(
                line("frame_00000008", False)
                + "\n"
                + line("frame_00000004", True)
                + "\n"
            )
            predictions, failures = parse_batch_output(output_path=path, plan=plan)
        self.assertFalse(failures)
        self.assertEqual([row["source_frame"] for row in predictions], [4, 8])


if __name__ == "__main__":
    unittest.main()
