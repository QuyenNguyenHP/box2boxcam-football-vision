"""Persistent notebook interface for reviewing box2boxcam annotations."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

MASTER_QUEUE_FIELDS = [
    "item_id",
    "priority",
    "sequence",
    "frame",
    "split",
    "original_status",
    "review_reasons",
    "human_finding",
    "source_image",
    "source_label",
]

DECISION_FIELDS = [
    "item_id",
    "decision",
    "x_center",
    "y_center",
    "width",
    "height",
    "note",
    "reviewed_at_utc",
]

VALID_DECISIONS = {"keep", "negative", "bbox", "exclude"}
SEQUENCE_EXCLUSION_NOTE = "[sequence_exclusion] Not reviewed: remainder dropped with the sequence."
FULL_SEQUENCE_NOTE = "[full_sequence_exclusion] Entire sequence dropped."


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv_atomic(
    path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    """Replace a CSV atomically so an interrupted save cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _priority(split: str, reasons: str) -> int:
    if "visual_sample_" in reasons:
        return 0
    return {"test": 1, "val": 2, "train": 3}.get(split, 4)


def build_master_queue(processed_root: Path) -> Path:
    """Include every source frame, enriched with automatic review signals.

    The result is deterministic and may be rebuilt without touching decisions.
    """
    processed_root = processed_root.resolve()
    project_root = processed_root.parents[2]
    manifest_rows = _read_csv(processed_root / "manifest.csv")
    automatic_rows = _read_csv(processed_root / "qa_review_queue.csv")
    excluded_rows = _read_csv(processed_root / "review_queue.csv")

    if not manifest_rows:
        raise FileNotFoundError(f"Manifest missing or empty: {processed_root / 'manifest.csv'}")

    manifest_by_key = {
        (row["sequence"], row["frame"]): row for row in manifest_rows
    }
    merged: dict[str, dict[str, str]] = {}

    def add_candidate(
        row: dict[str, str], *, status_field: str, reasons: str, finding: str = ""
    ) -> None:
        key = (row["sequence"], row["frame"])
        manifest = manifest_by_key[key]
        item_id = f"{row['sequence']}_{int(row['frame']):06d}"
        existing = merged.get(item_id)
        combined_reasons = set(reasons.split(";")) if reasons else set()
        if existing:
            combined_reasons.update(existing["review_reasons"].split(";"))
        combined_reasons.discard("")
        source_image = manifest["source_image"]
        source_label = manifest["source_label"]
        merged[item_id] = {
            "item_id": item_id,
            "priority": str(_priority(manifest["split"], ";".join(combined_reasons))),
            "sequence": row["sequence"],
            "frame": str(int(row["frame"])),
            "split": manifest["split"],
            "original_status": row[status_field],
            "review_reasons": ";".join(sorted(combined_reasons)),
            "human_finding": finding or (existing or {}).get("human_finding", ""),
            "source_image": str(project_root / source_image),
            "source_label": str(project_root / source_label) if source_label else "",
        }

    # A frame without an automatic warning still requires human review.
    for row in manifest_rows:
        add_candidate(row, status_field="status", reasons="")
    for row in automatic_rows:
        add_candidate(
            row,
            status_field="current_status",
            reasons=row["review_reasons"],
            finding=row.get("human_finding", ""),
        )
    for row in excluded_rows:
        add_candidate(row, status_field="status", reasons=row["status"])

    queue_rows = sorted(
        merged.values(),
        key=lambda row: (
            {"test": 0, "val": 1, "train": 2}.get(row["split"], 3),
            row["sequence"],
            int(row["frame"]),
        ),
    )
    queue_path = processed_root / "master_review_queue.csv"
    _write_csv_atomic(queue_path, queue_rows, MASTER_QUEUE_FIELDS)
    return queue_path


def parse_yolo_boxes(label_path: str | Path) -> list[tuple[float, float, float, float]]:
    """Return only syntactically valid class-0 YOLO boxes."""
    path = Path(label_path) if label_path else None
    if path is None or not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            continue
        if (
            class_id == 0
            and 0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            boxes.append((x_center, y_center, width, height))
    return boxes


class ReviewStore:
    """Queue and decisions with resumable, atomic persistence."""

    def __init__(self, processed_root: str | Path):
        self.processed_root = Path(processed_root).resolve()
        self.project_root = self.processed_root.parents[2]
        self.queue_path = build_master_queue(self.processed_root)
        self.decisions_path = self.processed_root / "qa_review_decisions.csv"
        self.queue = _read_csv(self.queue_path)
        self.decisions = {
            row["item_id"]: row for row in _read_csv(self.decisions_path)
        }
        self._manifest = _read_csv(self.processed_root / "manifest.csv")
        self._frames = {
            (row["sequence"], int(row["frame"])): row for row in self._manifest
        }

    def filtered_items(
        self, *, split: str = "all", state: str = "pending"
    ) -> list[dict[str, str]]:
        rows = self.queue
        if split != "all":
            rows = [row for row in rows if row["split"] == split]
        if state == "pending":
            rows = [row for row in rows if row["item_id"] not in self.decisions]
        elif state == "reviewed":
            rows = [row for row in rows if row["item_id"] in self.decisions]
        elif state != "all":
            raise ValueError(f"Unknown state: {state}")
        return rows

    def progress(self) -> dict[str, int]:
        counts = Counter(row["split"] for row in self.queue)
        reviewed = Counter(
            row["split"]
            for row in self.queue
            if row["item_id"] in self.decisions
        )
        return {
            "total": len(self.queue),
            "sequence_excluded": sum(
                row["decision"] == "exclude" and row["note"] in {SEQUENCE_EXCLUSION_NOTE, FULL_SEQUENCE_NOTE}
                for row in self.decisions.values()
            ),
            "reviewed": len(self.decisions),
            "pending": len(self.queue) - len(self.decisions),
            **{f"{split}_total": counts[split] for split in ("train", "val", "test")},
            **{
                f"{split}_reviewed": reviewed[split]
                for split in ("train", "val", "test")
            },
        }

    def neighbor(self, item: dict[str, str], offset: int) -> dict[str, str] | None:
        return self._frames.get((item["sequence"], int(item["frame"]) + offset))

    def decision_for(self, item_id: str) -> dict[str, str] | None:
        return self.decisions.get(item_id)

    def exclude_sequence_remaining(self, sequence: str, *, restore: bool = False) -> int:
        """Exclude only undecided frames; undo only untouched batch exclusions."""
        targets = [row["item_id"] for row in self.queue if row["sequence"] == sequence]
        if not targets:
            raise ValueError("Unknown sequence")
        decisions = {row["item_id"]: row for row in _read_csv(self.decisions_path)}
        count = 0
        for item_id in targets:
            current = decisions.get(item_id)
            if restore:
                if current and current["decision"] == "exclude" and current["note"] == SEQUENCE_EXCLUSION_NOTE:
                    del decisions[item_id]
                    count += 1
            elif current is None:
                decisions[item_id] = {
                    "item_id": item_id, "decision": "exclude",
                    "x_center": "", "y_center": "", "width": "", "height": "",
                    "note": SEQUENCE_EXCLUSION_NOTE,
                    "reviewed_at_utc": datetime.now(UTC).isoformat(),
                }
                count += 1
        _write_csv_atomic(self.decisions_path, [decisions[key] for key in sorted(decisions)], DECISION_FIELDS)
        self.decisions = decisions
        return count

    def save(
        self,
        item_id: str,
        decision: str,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        note: str = "",
    ) -> None:
        if item_id not in {row["item_id"] for row in self.queue}:
            raise KeyError(f"Item missing from the queue: {item_id}")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Unknown decision: {decision}")
        if decision == "bbox":
            if bbox is None or not _valid_bbox(bbox):
                raise ValueError("A valid normalized box is required")
        else:
            bbox = None

        values = [f"{value:.8f}" for value in bbox] if bbox else ["", "", "", ""]
        self.decisions = {row["item_id"]: row for row in _read_csv(self.decisions_path)}
        # An explicit re-review is no longer an untouched batch exclusion.
        if note.strip() in {SEQUENCE_EXCLUSION_NOTE, FULL_SEQUENCE_NOTE}:
            note = "Exclusion confirmed individually."
        self.decisions[item_id] = {
            "item_id": item_id,
            "decision": decision,
            "x_center": values[0],
            "y_center": values[1],
            "width": values[2],
            "height": values[3],
            "note": note.strip(),
            "reviewed_at_utc": datetime.now(UTC).isoformat(),
        }
        ordered = [
            self.decisions[key] for key in sorted(self.decisions)
        ]
        _write_csv_atomic(self.decisions_path, ordered, DECISION_FIELDS)

    def sequence_backup_path(self, sequence: str) -> Path:
        if sequence not in {row["sequence"] for row in self.queue}:
            raise ValueError("Unknown sequence")
        return self.processed_root / "sequence_backups" / f"{sequence}.csv"

    def exclude_entire_sequence(self, sequence: str, *, restore: bool = False) -> int:
        """Save prior decisions before exclusion; restore without losing later edits."""
        backup_path = self.sequence_backup_path(sequence)
        decisions = {row["item_id"]: row for row in _read_csv(self.decisions_path)}
        targets = [row["item_id"] for row in self.queue if row["sequence"] == sequence]
        count = 0
        if restore:
            if not backup_path.exists():
                return 0
            for previous in _read_csv(backup_path):
                item_id = previous["item_id"]
                current = decisions.get(item_id, {})
                if current.get("note") != FULL_SEQUENCE_NOTE or current.get("decision") != "exclude":
                    continue
                if previous["decision"]:
                    decisions[item_id] = previous
                else:
                    decisions.pop(item_id, None)
                count += 1
        else:
            if backup_path.exists():
                raise ValueError("An exclusion already exists: restore it before starting again.")
            snapshot = [decisions.get(item_id, {
                **dict.fromkeys(DECISION_FIELDS, ""), "item_id": item_id,
            }) for item_id in targets]
            _write_csv_atomic(backup_path, snapshot, DECISION_FIELDS)
            for item_id in targets:
                decisions[item_id] = {
                    **dict.fromkeys(DECISION_FIELDS, ""),
                    "item_id": item_id, "decision": "exclude", "note": FULL_SEQUENCE_NOTE,
                    "reviewed_at_utc": datetime.now(UTC).isoformat(),
                }
                count += 1
        _write_csv_atomic(self.decisions_path, [decisions[key] for key in sorted(decisions)], DECISION_FIELDS)
        self.decisions = decisions
        if restore:
            backup_path.unlink()
        return count


def _valid_bbox(bbox: tuple[float, float, float, float]) -> bool:
    x_center, y_center, width, height = bbox
    return (
        0 <= x_center <= 1
        and 0 <= y_center <= 1
        and 0 < width <= 1
        and 0 < height <= 1
        and x_center - width / 2 >= 0
        and x_center + width / 2 <= 1
        and y_center - height / 2 >= 0
        and y_center + height / 2 <= 1
    )


class ReviewWidget:
    """ipywidgets + Matplotlib review interface, imported lazily."""

    def __init__(self, store: ReviewStore):
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import RectangleSelector

        self.widgets = widgets
        self.plt = plt
        self.Rectangle = Rectangle
        self.RectangleSelector = RectangleSelector
        self.store = store
        self.items: list[dict[str, str]] = []
        self.index = 0
        self.drawn_bbox: tuple[float, float, float, float] | None = None

        self.split = widgets.Dropdown(
            options=[("All images", "all"), ("Test", "test"), ("Validation", "val"), ("Train", "train")],
            value="all",
            description="Split",
        )
        self.progress_label = widgets.HTML()
        self.item_label = widgets.HTML()
        self.note = widgets.Textarea(
            placeholder="Note facultative", description="Note", layout=widgets.Layout(width="100%")
        )
        self.box_choice = widgets.Dropdown(
            options=[("No box", None)],
            description="Red box",
            disabled=True,
        )
        self.message = widgets.HTML()
        self.sequence_choice = widgets.Dropdown(description="Sequence", layout=widgets.Layout(width="95%"))
        self.sequence_info = widgets.HTML()
        self.skip_sequence_button = widgets.Button(description="Skip the rest", button_style="warning")
        self.restore_sequence_button = widgets.Button(description="Restore the skipped rest")
        self.exclude_all_button = widgets.Button(description="Exclude the whole sequence", button_style="danger")
        self.restore_all_button = widgets.Button(description="Annuler l'exclusion totale")
        self.first_button = widgets.Button(description="⏮ First")
        self.last_button = widgets.Button(description="Last ⏭")
        self.frame_position = widgets.IntSlider(
            description="Image", min=1, max=1, value=1,
            continuous_update=False, layout=widgets.Layout(width="95%"),
        )
        self._sync_sequence = False

        self.previous_button = widgets.Button(description="← Previous")
        self.next_button = widgets.Button(description="Suivante →")
        self.pending_button = widgets.Button(description="Next to review")
        self.keep_button = widgets.Button(description="✓ Confirm and advance", button_style="success")
        self.negative_button = widgets.Button(description="Aucun ballon", button_style="warning")
        self.draw_mode_button = widgets.Button(
            description="✏️ Enable drawing", button_style="info"
        )
        self.keep_selected_button = widgets.Button(
            description="Keep only this box",
            button_style="info",
            disabled=True,
        )
        self.bbox_button = widgets.Button(description="Save the drawn box", button_style="info")
        self.exclude_button = widgets.Button(description="Exclure l'image", button_style="danger")

        self.figure, self.axes = plt.subplots(
            1, 3, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 2.5, 1]}
        )
        self.selector = None

        self.split.observe(self._on_filter, names="value")
        self.previous_button.on_click(lambda _: self._move(-1))
        self.next_button.on_click(lambda _: self._move(1))
        self.pending_button.on_click(lambda _: self._next_pending())
        self.keep_button.on_click(self._validate_current)
        self.negative_button.on_click(lambda _: self._save("negative"))
        self.draw_mode_button.on_click(self._activate_draw)
        self.keep_selected_button.on_click(self._save_selected_box)
        self.bbox_button.on_click(lambda _: self._save("bbox"))
        self.exclude_button.on_click(lambda _: self._save("exclude"))
        self.sequence_choice.observe(self._choose_sequence, names="value")
        self.skip_sequence_button.on_click(lambda _: self._exclude_remaining())
        self.restore_sequence_button.on_click(lambda _: self._exclude_remaining(restore=True))
        self.exclude_all_button.on_click(lambda _: self._exclude_all())
        self.restore_all_button.on_click(lambda _: self._exclude_all(restore=True))
        self.first_button.on_click(lambda _: self._jump_sequence(0))
        self.last_button.on_click(lambda _: self._jump_sequence(-1))
        self.frame_position.observe(self._jump_slider, names="value")
        self._reload()

        wrap = widgets.Layout(display="flex", flex_flow="row wrap")
        for button in (
            self.previous_button, self.next_button, self.pending_button,
            self.keep_button, self.negative_button, self.draw_mode_button,
            self.keep_selected_button, self.bbox_button, self.exclude_button,
            self.skip_sequence_button, self.restore_sequence_button,
            self.exclude_all_button, self.restore_all_button, self.first_button, self.last_button,
        ):
            button.layout.width = "auto"
            button.layout.margin = "4px"
        correction = widgets.VBox([
            widgets.HTML(
                "Zoom if necessary, then enable drawing and draw the box "
                "on the large central image. Save to advance."
            ),
            widgets.HBox([self.draw_mode_button, self.bbox_button], layout=wrap),
            widgets.HTML("Several boxes? Choose the one to keep:"),
            widgets.HBox([self.box_choice, self.keep_selected_button], layout=wrap),
        ])
        self.details = widgets.Accordion(children=[correction, self.note, self.item_label])
        for index, title in enumerate(("Correct the boxes", "Add a note", "Origin and flags")):
            self.details.set_title(index, title)
        self.details.selected_index = None
        self.status_label = widgets.HTML()
        self.ui = widgets.VBox(
            [
                widgets.HBox([self.split, self.pending_button], layout=wrap),
                self.progress_label,
                self.sequence_choice,
                self.sequence_info,
                widgets.HBox([self.skip_sequence_button, self.restore_sequence_button], layout=wrap),
                widgets.HBox([self.exclude_all_button, self.restore_all_button], layout=wrap),
                widgets.HTML("Full exclusion: confirmations and corrections are dropped too, with a backup so it can be undone."),
                self.frame_position,
                widgets.HBox([self.first_button, self.last_button], layout=wrap),
                widgets.HBox([self.previous_button, self.next_button], layout=wrap),
                self.status_label,
                self.figure.canvas,
                widgets.HBox(
                    [self.keep_button, self.negative_button, self.exclude_button],
                    layout=wrap,
                ),
                self.message,
                self.details,
            ]
        )
        self._render()

    def display(self):
        from IPython.display import display

        display(self.ui)
        return self

    def _on_filter(self, _change) -> None:
        self.index = 0
        self._reload()

    def _reload(self) -> None:
        # Keep reviewed items in place so Previous always returns to them.
        self.items = self.store.filtered_items(split=self.split.value, state="all")
        self.index = next((
            i for i, item in enumerate(self.items)
            if self.store.decision_for(item["item_id"]) is None
        ), 0)
        self._render()

    def _next_pending(self) -> None:
        order = list(range(self.index + 1, len(self.items))) + list(range(self.index + 1))
        for index in order:
            if index < len(self.items) and self.store.decision_for(self.items[index]["item_id"]) is None:
                self.index = index
                self._render()
                return
        self.message.value = "All images in this group have a recorded decision."

    def _move(self, offset: int) -> None:
        if self.items:
            self.index = min(max(self.index + offset, 0), len(self.items) - 1)
            self._render()

    def _choose_sequence(self, change) -> None:
        if self._sync_sequence or change["new"] is None:
            return
        indices = [i for i, row in enumerate(self.items) if row["sequence"] == change["new"]]
        self.index = next((i for i in indices if self.store.decision_for(self.items[i]["item_id"]) is None), indices[0])
        self._render()

    def _jump_sequence(self, position: int) -> None:
        if not self.items:
            return
        sequence = self.items[self.index]["sequence"]
        indices = [i for i, row in enumerate(self.items) if row["sequence"] == sequence]
        self.index = indices[position]
        self._render()

    def _jump_slider(self, change) -> None:
        if not self._sync_sequence:
            self._jump_sequence(change["new"] - 1)

    def _exclude_all(self, *, restore: bool = False) -> None:
        if not self.items:
            return
        sequence = self.items[self.index]["sequence"]
        count = self.store.exclude_entire_sequence(sequence, restore=restore)
        # Stay on the sequence so its undo remains immediately accessible.
        self._render()
        self.message.value = (
            f"{count} images {'restored according to their previous decisions' if restore else 'excluded from the future v002'}. "
            "The image files remain available."
        )

    def _exclude_remaining(self, *, restore: bool = False) -> None:
        if not self.items:
            return
        sequence = self.items[self.index]["sequence"]
        count = self.store.exclude_sequence_remaining(sequence, restore=restore)
        if not restore:
            self._next_pending()
        self._render()
        self.message.value = (
            f"{count} images {'put back for review' if restore else 'dropped without individual review'} "
            f"in {sequence[:8]}. Existing corrections are kept."
        )

    def _render(self) -> None:
        progress = self.store.progress()
        self.progress_label.value = (
            f"<b>Progress:</b> {progress['reviewed'] - progress.get('sequence_excluded', 0)} individual decisions "
            f"+ {progress.get('sequence_excluded', 0)} images dropped by sequence / {progress['total']} "
            f"— test {progress['test_reviewed']}/{progress['test_total']} "
            f"— val {progress['val_reviewed']}/{progress['val_total']} "
            f"— train {progress['train_reviewed']}/{progress['train_total']}"
        )
        groups = {}
        for row in self.items:
            groups.setdefault(row["sequence"], []).append(row)
        options = []
        for sequence, rows in groups.items():
            remaining = sum(self.store.decision_for(row["item_id"]) is None for row in rows)
            options.append((f"{rows[0]['split']} · {sequence[:8]} · {remaining}/{len(rows)} to review", sequence))
        self._sync_sequence = True
        try:
            self.sequence_choice.options = options
            if self.items:
                self.sequence_choice.value = self.items[self.index]["sequence"]
                positions = [i for i, row in enumerate(self.items) if row["sequence"] == self.sequence_choice.value]
                self.frame_position.value = 1
                self.frame_position.max = len(positions)
                self.frame_position.value = positions.index(self.index) + 1
        finally:
            self._sync_sequence = False
        if self.items:
            rows = groups[self.items[self.index]["sequence"]]
            pending = sum(self.store.decision_for(row["item_id"]) is None for row in rows)
            excluded = sum(
                (self.store.decision_for(row["item_id"]) or {}).get("note") in {SEQUENCE_EXCLUSION_NOTE, FULL_SEQUENCE_NOTE}
                for row in rows
            )
            self.sequence_info.value = (
                f"This sequence: <b>{pending} images without a decision</b>, "
                f"{len(rows) - pending - excluded} individual decisions, {excluded} grouped exclusions. "
                "Skipping the rest excludes all its images without a decision, even those skipped earlier."
            )
            self.skip_sequence_button.description = f"Skip the remaining {pending}"
            self.skip_sequence_button.disabled = pending == 0
            self.restore_sequence_button.disabled = not any(
                (self.store.decision_for(row["item_id"]) or {}).get("note") == SEQUENCE_EXCLUSION_NOTE
                for row in rows
            )
            has_backup = self.store.sequence_backup_path(self.items[self.index]["sequence"]).exists()
            self.exclude_all_button.disabled = bool(has_backup)
            self.restore_all_button.disabled = not has_backup
        # clear() removes the selector's rectangle and handles as well as images.
        # Disconnect the old callbacks and rebuild its artists on the cleared axes.
        if self.selector is not None:
            self.selector.set_active(False)
            self.selector.disconnect_events()
        for axis in self.axes:
            axis.clear()
            axis.axis("off")
        self.selector = self.RectangleSelector(
            self.axes[1], self._on_rectangle,
            useblit=False, button=[1], interactive=True,
            props={"facecolor": "deepskyblue", "edgecolor": "deepskyblue",
                   "alpha": 0.35, "linewidth": 2},
        )
        self.selector.set_active(False)
        self.drawn_bbox = None
        self.current_original_boxes: list[tuple[float, float, float, float]] = []
        self.message.value = ""
        self.previous_button.disabled = not self.items or self.index == 0
        self.next_button.disabled = not self.items or self.index == len(self.items) - 1
        self.bbox_button.disabled = True
        self.keep_button.disabled = not self.items
        if not self.items:
            self.item_label.value = "<b>No image in this filter.</b>"
            if hasattr(self, "status_label"):
                self.status_label.value = "No image to display."
            self.figure.canvas.draw_idle()
            return

        item = self.items[self.index]
        finding = item["human_finding"] or "—"
        previous_decision = self.store.decision_for(item["item_id"])
        self.note.value = previous_decision["note"] if previous_decision else ""
        decision_text = previous_decision["decision"] if previous_decision else "none"
        labels = {"keep": "Confirmed", "negative": "Confirmed: no ball",
                  "bbox": "Confirmed: box corrected", "exclude": "Excluded", "none": "To review"}
        if hasattr(self, "status_label"):
            self.status_label.value = (
                f"<b>Image {self.index + 1} / {len(self.items)} — {labels[decision_text]}</b>"
                " · Any new confirmation replaces the previous one."
            )
        self.keep_button.disabled = (
            previous_decision is None
            and item["original_status"] not in {"positive_single", "negative"}
        )
        self.item_label.value = (
            f"<b>{self.index + 1}/{len(self.items)}</b> — {item['split']} — "
            f"frame {item['frame']} — statut : <code>{item['original_status']}</code><br>"
            f"Signaux : <code>{item['review_reasons'] or 'aucun signal automatique — revue requise'}</code><br>"
            f"Finding already confirmed: {finding}<br>"
            f"Recorded decision: <code>{decision_text}</code>"
        )

        frames = [self.store.neighbor(item, -1), item, self.store.neighbor(item, 1)]
        titles = ["previous frame", "frame to decide", "next frame"]
        for frame_index, (axis, frame, title) in enumerate(
            zip(self.axes, frames, titles)
        ):
            if frame is None:
                axis.set_title(f"{title}\nabsente")
                continue
            image_path = Path(frame["source_image"])
            if not image_path.is_absolute():
                image_path = self.store.project_root / image_path
            image = Image.open(image_path).convert("RGB")
            axis.imshow(image)
            label_path = frame.get("source_label", "")
            if label_path and not Path(label_path).is_absolute():
                label_path = str(self.store.project_root / label_path)
            original_boxes = parse_yolo_boxes(label_path)
            if frame_index == 1:
                self.current_original_boxes = original_boxes
            show_original = not (
                frame_index == 1
                and previous_decision
                and previous_decision["decision"] in {"negative", "bbox", "exclude"}
            )
            if show_original:
                for box in original_boxes:
                    self._draw_box(
                        axis,
                        box,
                        image.width,
                        image.height,
                        color="red",
                        original=True,
                    )
            if (
                frame_index == 1
                and previous_decision
                and previous_decision["decision"] == "bbox"
            ):
                corrected_box = tuple(
                    float(previous_decision[field])
                    for field in ("x_center", "y_center", "width", "height")
                )
                self._draw_box(
                    axis, corrected_box, image.width, image.height, color="deepskyblue"
                )
                self.drawn_bbox = corrected_box
            axis.set_title(f"{title}\n{image_path.stem[-6:]}")
            axis.axis("off")
        if self.current_original_boxes:
            self.box_choice.options = [
                (f"Box {index}", box)
                for index, box in enumerate(self.current_original_boxes, start=1)
            ]
            self.box_choice.disabled = False
            self.keep_selected_button.disabled = False
        else:
            self.box_choice.options = [("No box", None)]
            self.box_choice.disabled = True
            self.keep_selected_button.disabled = True
        self.figure.canvas.draw_idle()

    def _draw_box(
        self,
        axis,
        box,
        image_width: int,
        image_height: int,
        *,
        color: str,
        original: bool = False,
    ) -> None:
        x_center, y_center, width, height = box
        left = (x_center - width / 2) * image_width
        top = (y_center - height / 2) * image_height
        patch = self.Rectangle(
            (left, top),
            width * image_width,
            height * image_height,
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        if original:
            patch.set_gid("original_bbox")
        axis.add_patch(patch)

    def _on_rectangle(self, click, release) -> None:
        if not self.items or click.xdata is None or release.xdata is None:
            return
        item = self.items[self.index]
        with Image.open(item["source_image"]) as image:
            image_width, image_height = image.size
        left, right = sorted((click.xdata, release.xdata))
        top, bottom = sorted((click.ydata, release.ydata))
        left = min(max(left, 0), image_width)
        right = min(max(right, 0), image_width)
        top = min(max(top, 0), image_height)
        bottom = min(max(bottom, 0), image_height)
        self.drawn_bbox = (
            ((left + right) / 2) / image_width,
            ((top + bottom) / 2) / image_height,
            (right - left) / image_width,
            (bottom - top) / image_height,
        )
        self.bbox_button.disabled = not _valid_bbox(self.drawn_bbox)
        self.message.value = "<span style='color:#1565c0'>Box drawn, ready to be saved.</span>"

    def _activate_draw(self, _button) -> None:
        """Give drag events back to RectangleSelector after zooming or panning."""
        toolbar = self.figure.canvas.toolbar
        mode = str(toolbar.mode)
        if mode == "zoom rect":
            toolbar.zoom()
        elif mode == "pan/zoom":
            toolbar.pan()
        for patch in list(self.axes[1].patches):
            if patch.get_gid() == "original_bbox":
                patch.remove()
        self.selector.set_active(True)
        self.figure.canvas.draw_idle()
        self.message.value = (
            "<span style='color:#1565c0'><b>Mode dessin actif.</b> "
            "The old red boxes are hidden. Now drag around "
            "the ball in the central image.</span>"
        )

    def _save_selected_box(self, _button) -> None:
        selected_box = self.box_choice.value
        if selected_box is None:
            self.message.value = (
                "<span style='color:#b71c1c'>No red box selected.</span>"
            )
            return
        self.drawn_bbox = tuple(selected_box)
        self._save("bbox")

    def _validate_current(self, _button) -> None:
        if not self.items:
            return
        # Revalidating a corrected image must not restore its source label.
        if self.drawn_bbox is not None:
            self._save("bbox")
            return
        previous = self.store.decision_for(self.items[self.index]["item_id"])
        self._save(previous["decision"] if previous else "keep")

    def _save(self, decision: str) -> None:
        if not self.items:
            return
        item = self.items[self.index]
        if decision == "keep" and item["original_status"] not in {"positive_single", "negative"}:
            self.message.value = "<span style='color:#b71c1c'>This label was invalid or multiple: redraw a box, switch to negative or exclude the image.</span>"
            return
        try:
            self.store.save(
                item["item_id"], decision, bbox=self.drawn_bbox, note=self.note.value
            )
        except ValueError as error:
            self.message.value = f"<span style='color:#b71c1c'>{error}</span>"
            return
        self._move(1)
        self.message.value = (
            "<span style='color:#2e7d32'>Decision recorded. "
            "← Previous lets you change it.</span>"
        )
