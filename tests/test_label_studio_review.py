from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchlens.data.label_studio_review import (
    build_tasks,
    export_review,
    prediction_index,
)


class LabelStudioReviewTests(unittest.TestCase):
    def test_retry_replaces_missing_prediction_and_boxes_use_percentages(self) -> None:
        plan = {
            "video": {"width": 200, "height": 100, "fps": 25},
            "model": "gpt-6-luna",
            "cases": [
                {"item_id": "frame_00000000", "source_frame": 0},
                {"item_id": "frame_00000004", "source_frame": 4},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            retry = Path(directory) / "retry.json"
            first.write_text(json.dumps({"predictions": []}))
            retry.write_text(
                json.dumps(
                    {
                        "predictions": [
                            {
                                "item_id": "frame_00000004",
                                "confidence": 0.72,
                                "bbox_current_pixels": [20, 10, 40, 30],
                            }
                        ]
                    }
                )
            )
            tasks = build_tasks(plan, prediction_index([first, retry]))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["predictions"][0]["score"], 0.0)
        box = tasks[1]["predictions"][0]["result"][0]
        self.assertEqual(box["type"], "rectangle")
        self.assertEqual(
            [box["value"][key] for key in ("x", "y", "width", "height")],
            [10.0, 10.0, 10.0, 20.0],
        )

    def test_export_distinguishes_positive_negative_excluded_and_pending(self) -> None:
        box = {
            "from_name": "bbox",
            "type": "rectangle",
            "value": {"x": 10, "y": 20, "width": 10, "height": 10, "rotation": 0},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = root / "export.json"
            exported.write_text(
                json.dumps(
                    [
                        {
                            "data": {"item_id": "frame_00000000", "source_frame": 0},
                            "annotations": [{"id": 1, "result": [box]}],
                        },
                        {
                            "data": {"item_id": "frame_00000004", "source_frame": 4},
                            "annotations": [{"id": 2, "result": []}],
                        },
                        {
                            "data": {"item_id": "frame_00000008", "source_frame": 8},
                            "annotations": [
                                {
                                    "id": 3,
                                    "result": [
                                        box,
                                        {
                                            "from_name": "decision",
                                            "value": {"choices": ["Exclure image"]},
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "data": {"item_id": "frame_00000012", "source_frame": 12},
                            "annotations": [],
                        },
                    ]
                )
            )
            report = export_review(
                export_path=exported,
                output_root=root / "labels_out",
                width=200,
                height=100,
            )
            self.assertEqual(
                report,
                {"reviewed": 3, "positive": 1, "negative": 1,
                 "excluded": 1, "unreviewed": 1},
            )
            with (root / "labels_out/review_manifest.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["status"] for row in rows], [
                "positive", "negative", "excluded"
            ])
            self.assertEqual(
                (root / "labels_out/labels/frame_00000000.txt").read_text(),
                "0 0.15000000 0.25000000 0.10000000 0.10000000\n",
            )
            self.assertEqual(
                (root / "labels_out/labels/frame_00000004.txt").read_text(), ""
            )
            self.assertFalse(
                (root / "labels_out/labels/frame_00000008.txt").exists()
            )
