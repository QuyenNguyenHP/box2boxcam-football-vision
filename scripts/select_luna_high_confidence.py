"""Prepare high-confidence candidates, not human-validated training labels.

No API calls, image copies, changes to Label Studio, or changes to existing splits.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data/interim/gpt6_luna_new_videos_20260928/predictions.json"
    output = root / "data/interim/luna_high_confidence95_20260928"
    threshold = 0.95
    predictions = json.loads(source.read_text())["predictions"]
    selected = [
        p
        for p in predictions
        if p["ball_in_play_visible"]
        and p["bbox_current_pixels"] is not None
        and float(p["confidence"]) >= threshold
    ]
    if len({p["item_id"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    for p in selected:
        w, h = int(p["width"]), int(p["height"])
        x0, y0, x1, y1 = p["bbox_current_pixels"]
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"Invalid box: {p['item_id']}")
        if not (root / p["image_path"]).is_file():
            raise FileNotFoundError(p["image_path"])
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    if output.exists():
        report = json.loads((output / "selection_report.json").read_text())
        if report["source_sha256"] != fingerprint:
            raise FileExistsError("Source changed: do not overwrite this selection")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    output.mkdir()
    (output / "labels").mkdir()
    rows = []
    for p in selected:
        w, h = int(p["width"]), int(p["height"])
        x0, y0, x1, y1 = p["bbox_current_pixels"]
        label = output / "labels" / f"{p['item_id']}.txt"
        label.write_text(
            f"0 {(x0 + x1) / 2 / w:.8f} {(y0 + y1) / 2 / h:.8f} "
            f"{(x1 - x0) / w:.8f} {(y1 - y0) / h:.8f}\n"
        )
        rows.append(
            {
                key: p[key]
                for key in (
                    "item_id",
                    "video_id",
                    "source_url",
                    "uploader",
                    "source_frame",
                    "timestamp_seconds",
                    "width",
                    "height",
                    "image_path",
                    "confidence",
                )
            }
            | {
                "source_label": str(label.relative_to(root)),
                "annotation_status": "unreviewed_luna_proposal",
                "split": "unassigned",
                "match_id": "",
                "scene_id": "",
            }
        )
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Four chronological quantiles per video: repeatable, not cherry-picked.
    sample = []
    for video in sorted({p["video_id"] for p in selected}):
        group = sorted(
            [p for p in selected if p["video_id"] == video],
            key=lambda p: int(p["source_frame"]),
        )
        sample.extend(group[round((len(group) - 1) * q)] for q in (0, 1 / 3, 2 / 3, 1))
    for page in range(2):
        sheet = Image.new("RGB", (1440, 1260), "white")
        draw = ImageDraw.Draw(sheet)
        for i, p in enumerate(sample[page * 12 : (page + 1) * 12]):
            left, top = (i % 4) * 360, (i // 4) * 420
            with Image.open(root / p["image_path"]) as original:
                original = original.convert("RGB")
                x0, y0, x1, y1 = p["bbox_current_pixels"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                side = max(120, 5 * max(x1 - x0, y1 - y0))
                crop_box = (
                    max(0, int(cx - side / 2)),
                    max(0, int(cy - side / 2)),
                    min(original.width, int(cx + side / 2)),
                    min(original.height, int(cy + side / 2)),
                )
                full = original.copy()
                full.thumbnail((150, 310))
                scale = full.width / original.width
                ImageDraw.Draw(full).rectangle(
                    tuple(v * scale for v in (x0, y0, x1, y1)), outline="red", width=1
                )
                crop = original.crop(crop_box)
                crop.thumbnail((200, 310))
                sx, sy = (
                    crop.width / (crop_box[2] - crop_box[0]),
                    crop.height / (crop_box[3] - crop_box[1]),
                )
                ImageDraw.Draw(crop).rectangle(
                    (
                        (x0 - crop_box[0]) * sx,
                        (y0 - crop_box[1]) * sy,
                        (x1 - crop_box[0]) * sx,
                        (y1 - crop_box[1]) * sy,
                    ),
                    outline="red",
                    width=1,
                )
                sheet.paste(full, (left, top + 65))
                sheet.paste(crop, (left + 155, top + 65))
            draw.text((left + 4, top + 5), p["video_id"], fill="black")
            draw.text(
                (left + 4, top + 23),
                f"Frame {p['source_frame']} | confiance {p['confidence']:.0%}",
                fill="black",
            )
            draw.text((left + 4, top + 41), "Contexte + zoom (cadre fin)", fill="black")
        sheet.save(output / f"inspection_{page + 1:02d}.jpg", quality=92)
    report = {
        "threshold": threshold,
        "comparison": ">=",
        "source_sha256": fingerprint,
        "total_predictions": len(predictions),
        "selected_positive_images": len(selected),
        "positive_images_before_filter": sum(
            p["ball_in_play_visible"] for p in predictions
        ),
        "selected_by_video": dict(Counter(p["video_id"] for p in selected)),
        "high_confidence_negative_images_not_selected": sum(
            not p["ball_in_play_visible"] and p["confidence"] >= threshold
            for p in predictions
        ),
        "sample_item_ids": [p["item_id"] for p in sample],
        "sample_policy": "Four chronological quantiles per source video; qualitative inspection only",
        "human_review_status": "Not merged with Label Studio submissions or drafts",
        "training_status": "Candidates only; existing train/val/test untouched",
        "images_copied": False,
    }
    (output / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    (output / "README.md").write_text(
        "# Luna selection >= 95%\n\n"
        "Positive proposals only: the confidence is self-reported by Luna, "
        "not a measured precision. The three negatives >= 95% are set aside.\n\n"
        "manifest.csv keeps the original image paths and their provenance. "
        "labels/ holds the YOLO proposals; no image is copied. "
        "inspection_01.jpg and inspection_02.jpg show 24 cases spread over the six videos.\n\n"
        "This batch remains a candidate, with no split. Before adding it to train, fetch "
        "the Label Studio corrections/exclusions and check the match groups. "
        "Do not filter validation/test by confidence: it would artificially make the evaluation easier.\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
