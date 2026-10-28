from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pitchlens.data.review import ReviewStore, build_master_queue


class ReviewStoreTests(unittest.TestCase):
    def test_rectangle_visible_after_navigation_and_zoom(self) -> None:
        import matplotlib

        matplotlib.use("module://ipympl.backend_nbagg")
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import MouseEvent
        from PIL import Image

        from pitchlens.data.review import ReviewWidget

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "frame.jpg"
            Image.new("RGB", (640, 480)).save(image_path)
            rows = [{
                "item_id": str(i), "frame": str(i), "split": "test", "sequence": "a",
                "original_status": "negative", "human_finding": "",
                "review_reasons": "single_negative_gap",
                "source_image": str(image_path), "source_label": "",
            } for i in (1, 2)]
            store = Mock()
            store.filtered_items.return_value = rows
            store.progress.return_value = {
                "total": 2, "reviewed": 0, "test_reviewed": 0, "test_total": 2,
                "val_reviewed": 0, "val_total": 0,
                "train_reviewed": 0, "train_total": 0,
            }
            store.decision_for.return_value = None
            store.neighbor.return_value = None
            widget = ReviewWidget(store)
            try:
                for frame in (0, 1):
                    if frame:
                        widget._move(1)
                    widget.figure.canvas.toolbar.zoom()
                    widget._activate_draw(None)
                    widget.figure.canvas.draw()
                    axis = widget.axes[1]
                    for event_name, point in (
                        ("button_press_event", (100, 100)),
                        ("motion_notify_event", (180, 180)),
                        ("button_release_event", (180, 180)),
                    ):
                        x, y = axis.transData.transform(point)
                        event = MouseEvent(
                            event_name, widget.figure.canvas, x, y, button=1
                        )
                        widget.figure.canvas.callbacks.process(event_name, event)
                    self.assertIsNotNone(widget.drawn_bbox)
                    for artist in widget.selector.artists:
                        self.assertIn(artist, axis.get_children())
                        self.assertTrue(artist.get_visible())
                    for actual, expected in zip(
                        widget.drawn_bbox, (140/640, 140/480, 80/640, 80/480)
                    ):
                        self.assertAlmostEqual(actual, expected)
                store.save.assert_not_called()
                widget._jump_sequence(0)
                self.assertEqual(widget.index, 0)
                widget._jump_sequence(-1)
                self.assertEqual(widget.index, 1)
                widget.frame_position.value = 1
                self.assertEqual(widget.index, 0)
                widget.frame_position.value = 2
                self.assertEqual(widget.index, 1)
                # Saving must leave the item in the list and Previous must restore
                # its correction; validating it again must preserve that box.
                decisions = {}
                store.decision_for.side_effect = decisions.get

                def save(item_id, decision, *, bbox=None, note=""):
                    decisions[item_id] = {
                        "decision": decision, "note": note,
                        **dict(zip(("x_center", "y_center", "width", "height"), bbox or ("",) * 4)),
                    }

                store.save.side_effect = save
                widget.index = 0
                widget._render()
                correction = (0.5, 0.5, 0.1, 0.1)
                widget.drawn_bbox = correction
                widget._save("bbox")
                self.assertEqual(widget.index, 1)
                self.assertEqual(len(widget.items), 2)
                widget._move(-1)
                self.assertEqual(widget.items[widget.index]["item_id"], "1")
                self.assertEqual(widget.drawn_bbox, correction)
                self.assertIn("boîte corrigée", widget.status_label.value)
                widget._validate_current(None)
                self.assertEqual(decisions["1"]["decision"], "bbox")
                widget._move(-1)
                widget._next_pending()
                self.assertEqual(widget.index, 1)
                widget._save("negative")
                widget._move(-1)
                widget._next_pending()
                self.assertIn("Toutes les images", widget.message.value)
            finally:
                widget.selector.disconnect_events()
                plt.close(widget.figure)

    def test_merges_queues_and_persists_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            processed = project / "data" / "processed" / "v001"
            processed.mkdir(parents=True)
            image = project / "frame.jpg"
            image.write_bytes(b"image")

            self._write_csv(
                processed / "manifest.csv",
                [
                    {
                        "sequence": "sequence-a",
                        "frame": "1",
                        "split": "test",
                        "status": "positive_single",
                        "source_image": "frame.jpg",
                        "source_label": "",
                    }
                ],
            )
            self._write_csv(
                processed / "qa_review_queue.csv",
                [
                    {
                        "sequence": "sequence-a",
                        "frame": "1",
                        "current_status": "positive_single",
                        "review_reasons": "large_bbox",
                        "human_finding": "",
                    }
                ],
            )
            self._write_csv(processed / "review_queue.csv", [])

            queue_path = build_master_queue(processed)
            self.assertTrue(queue_path.exists())

            store = ReviewStore(processed)
            self.assertEqual(store.progress()["pending"], 1)
            store.save("sequence-a_000001", "negative", note="ball absent")

            reopened = ReviewStore(processed)
            self.assertEqual(reopened.progress()["reviewed"], 1)
            self.assertEqual(
                reopened.decision_for("sequence-a_000001")["decision"], "negative"
            )
            # Expanding the source manifest must expose unflagged positive AND
            # negative frames without rewriting previously saved decisions.
            decision_bytes = reopened.decisions_path.read_bytes()
            with (processed / "manifest.csv").open(newline="") as stream:
                original = next(iter(csv.DictReader(stream)))
            self._write_csv(processed / "manifest.csv", [
                original,
                {**original, "frame": "2", "status": "positive_single"},
                {**original, "frame": "3", "status": "negative"},
            ])
            expanded = ReviewStore(processed)
            self.assertEqual(expanded.progress()["total"], 3)
            self.assertEqual(expanded.progress()["reviewed"], 1)
            self.assertEqual(expanded.progress()["pending"], 2)
            self.assertEqual(
                [row["frame"] for row in expanded.filtered_items()], ["2", "3"]
            )
            self.assertEqual(expanded.decisions_path.read_bytes(), decision_bytes)
            old_decision = expanded.decision_for("sequence-a_000001").copy()
            self.assertEqual(expanded.exclude_sequence_remaining("sequence-a"), 2)
            self.assertEqual(expanded.decision_for("sequence-a_000001"), old_decision)
            self.assertEqual(expanded.progress()["sequence_excluded"], 2)
            # Explicitly corrected frames survive undo, including after reopening.
            expanded.save("sequence-a_000002", "bbox", bbox=(0.5, 0.5, 0.1, 0.1))
            again = ReviewStore(processed)
            self.assertEqual(again.exclude_sequence_remaining("sequence-a", restore=True), 1)
            self.assertEqual(again.decision_for("sequence-a_000002")["decision"], "bbox")
            self.assertEqual(again.decision_for("sequence-a_000001"), old_decision)
            self.assertIsNone(again.decision_for("sequence-a_000003"))
            before_full = {key: value.copy() for key, value in again.decisions.items()}
            self.assertEqual(again.exclude_entire_sequence("sequence-a"), 3)
            self.assertTrue(all(row["decision"] == "exclude" for row in again.decisions.values()))
            self.assertEqual(again.progress()["sequence_excluded"], 3)
            with self.assertRaises(ValueError):
                again.exclude_entire_sequence("sequence-a")
            reopened = ReviewStore(processed)
            self.assertEqual(reopened.exclude_entire_sequence("sequence-a", restore=True), 3)
            self.assertEqual(reopened.decisions, before_full)
            self.assertFalse(reopened.sequence_backup_path("sequence-a").exists())
            # Re-reviewing one image after batch exclusion must survive undo.
            reopened.exclude_entire_sequence("sequence-a")
            reopened.save("sequence-a_000003", "bbox", bbox=(0.5, 0.5, 0.2, 0.2))
            self.assertEqual(reopened.exclude_entire_sequence("sequence-a", restore=True), 2)
            self.assertEqual(reopened.decision_for("sequence-a_000003")["decision"], "bbox")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        if not rows:
            path.write_text("")
            return
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
