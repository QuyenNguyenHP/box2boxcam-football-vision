"""Audit frame groups without mixing neighbouring video frames across splits."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from pitchlens.data.audit_yolo import IMAGE_SUFFIXES, label_path_for

FRAME_NAME = re.compile(r"^(?P<sequence>.+)_(?P<frame>[0-9]+)$")


def parse_frame_name(path: Path) -> tuple[str, int] | None:
    match = FRAME_NAME.fullmatch(path.stem)
    if match is None:
        return None
    return match.group("sequence"), int(match.group("frame"))


def label_state(image_path: Path, root: Path) -> tuple[str, int]:
    """Return the annotation state and number of valid YOLO boxes."""
    label_path = label_path_for(image_path, root)
    if not label_path.exists():
        return "negative", 0

    rows = [
        line.strip() for line in label_path.read_text().splitlines() if line.strip()
    ]
    if not rows:
        return "negative", 0

    valid_boxes = 0
    invalid_rows = 0
    for row in rows:
        fields = row.split()
        if len(fields) != 5:
            invalid_rows += 1
            continue
        try:
            int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            invalid_rows += 1
            continue
        if (
            not all(
                0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)
            )
            or width == 0.0
            or height == 0.0
        ):
            invalid_rows += 1
            continue
        valid_boxes += 1

    if invalid_rows:
        return "invalid", valid_boxes
    return "positive", valid_boxes


def audit_sequences(root: Path) -> dict[str, object]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    unparsed: list[str] = []

    for image_path in sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        parsed = parse_frame_name(image_path)
        if parsed is None:
            unparsed.append(str(image_path.relative_to(root)))
            continue
        sequence, frame_number = parsed
        groups[sequence].append((frame_number, image_path))

    sequence_rows: list[dict[str, object]] = []
    for sequence, frames in sorted(groups.items()):
        frame_numbers = sorted(frame_number for frame_number, _ in frames)
        states = defaultdict(int)
        boxes = 0
        for _, image_path in frames:
            state, valid_boxes = label_state(image_path, root)
            states[state] += 1
            boxes += valid_boxes

        expected_frames = frame_numbers[-1] - frame_numbers[0] + 1
        sequence_rows.append(
            {
                "sequence": sequence,
                "frames": len(frames),
                "first_frame": frame_numbers[0],
                "last_frame": frame_numbers[-1],
                "missing_frame_numbers": expected_frames - len(set(frame_numbers)),
                "positive_frames": states["positive"],
                "negative_frames": states["negative"],
                "invalid_annotation_frames": states["invalid"],
                "valid_boxes": boxes,
            }
        )

    lengths = [int(row["frames"]) for row in sequence_rows]
    return {
        "root": str(root),
        "sequences": len(sequence_rows),
        "parsed_frames": sum(lengths),
        "unparsed_images": unparsed,
        "sequence_length_frames": {
            "min": min(lengths) if lengths else None,
            "median": median(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
        "contiguous_sequences": sum(
            int(row["missing_frame_numbers"] == 0) for row in sequence_rows
        ),
        "all_negative_sequences": sum(
            int(row["negative_frames"] == row["frames"]) for row in sequence_rows
        ),
        "all_valid_positive_sequences": sum(
            int(row["positive_frames"] == row["frames"]) for row in sequence_rows
        ),
        "sequences_with_invalid_annotations": sum(
            int(row["invalid_annotation_frames"] > 0) for row in sequence_rows
        ),
        "sequence_details": sequence_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Dataset root")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"dataset root does not exist: {args.root}")

    report = audit_sequences(args.root)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
