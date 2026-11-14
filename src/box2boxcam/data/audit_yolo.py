"""Audit a YOLO object-detection dataset before it enters box2boxcam."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def label_path_for(image_path: Path, root: Path) -> Path:
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


def split_name(image_path: Path, root: Path) -> str:
    for part in image_path.relative_to(root).parts:
        if part in {"train", "valid", "val", "test"}:
            return part
    return "unsplit"


def dimension_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": min(values),
        "p10": percentile(0.10),
        "median": median(values),
        "p90": percentile(0.90),
        "max": max(values),
    }


def audit(
    root: Path,
    missing_label_is_negative: bool = False,
    target_size: int | None = None,
) -> dict[str, object]:
    images = sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    issues: list[str] = []
    class_counts: Counter[int] = Counter()
    class_image_counts: Counter[int] = Counter()
    class_box_distributions: dict[int, Counter[int]] = defaultdict(Counter)
    class_multi_box_images: dict[int, list[str]] = defaultdict(list)
    class_widths: dict[int, list[float]] = defaultdict(list)
    class_heights: dict[int, list[float]] = defaultdict(list)
    class_areas: dict[int, list[float]] = defaultdict(list)
    resized_widths: dict[int, list[float]] = defaultdict(list)
    resized_heights: dict[int, list[float]] = defaultdict(list)
    resized_areas: dict[int, list[float]] = defaultdict(list)
    resolution_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    negative_images = 0
    labelled_images = 0

    for image_path in images:
        split = split_name(image_path, root)
        split_counts[split]["images"] += 1
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        resolution_counts[f"{image_width}x{image_height}"] += 1
        label_path = label_path_for(image_path, root)
        if not label_path.exists():
            if missing_label_is_negative:
                negative_images += 1
                split_counts[split]["negative_images"] += 1
            else:
                issues.append(f"missing label: {image_path.relative_to(root)}")
            continue

        rows = [
            line.strip() for line in label_path.read_text().splitlines() if line.strip()
        ]
        if not rows:
            negative_images += 1
            split_counts[split]["negative_images"] += 1
            continue

        labelled_images += 1
        split_counts[split]["labelled_images"] += 1
        classes_in_image: set[int] = set()
        boxes_in_image: Counter[int] = Counter()

        for line_number, row in enumerate(rows, start=1):
            fields = row.split()
            if len(fields) != 5:
                issues.append(
                    f"invalid field count: {label_path.relative_to(root)}:{line_number}"
                )
                continue
            try:
                class_id = int(fields[0])
                x_center, y_center, width, height = map(float, fields[1:])
            except ValueError:
                issues.append(
                    f"invalid number: {label_path.relative_to(root)}:{line_number}"
                )
                continue

            if not all(
                0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)
            ):
                issues.append(
                    f"coordinate outside [0,1]: {label_path.relative_to(root)}:{line_number}"
                )
                continue
            if width == 0.0 or height == 0.0:
                issues.append(
                    f"zero-area box: {label_path.relative_to(root)}:{line_number}"
                )
                continue

            class_counts[class_id] += 1
            classes_in_image.add(class_id)
            boxes_in_image[class_id] += 1
            split_counts[split]["boxes"] += 1
            pixel_width = width * image_width
            pixel_height = height * image_height
            widths.append(pixel_width)
            heights.append(pixel_height)
            areas.append(pixel_width * pixel_height)
            class_widths[class_id].append(pixel_width)
            class_heights[class_id].append(pixel_height)
            class_areas[class_id].append(pixel_width * pixel_height)
            if target_size is not None:
                scale = min(target_size / image_width, target_size / image_height)
                resized_widths[class_id].append(pixel_width * scale)
                resized_heights[class_id].append(pixel_height * scale)
                resized_areas[class_id].append(pixel_width * pixel_height * scale**2)

        class_image_counts.update(classes_in_image)
        for class_id, count in boxes_in_image.items():
            class_box_distributions[class_id][count] += 1
            if count > 1:
                class_multi_box_images[class_id].append(
                    str(image_path.relative_to(root))
                )

    boxes = sum(class_counts.values())
    class_metrics = {
        class_id: {
            "boxes": class_counts[class_id],
            "images_with_class": class_image_counts[class_id],
            "images_without_class": len(images) - class_image_counts[class_id],
            "boxes_per_positive_image": dict(
                sorted(class_box_distributions[class_id].items())
            ),
            "multi_box_images_count": len(class_multi_box_images[class_id]),
            "multi_box_image_examples": class_multi_box_images[class_id][:100],
            "width_px": dimension_summary(class_widths[class_id]),
            "height_px": dimension_summary(class_heights[class_id]),
            "area_px": dimension_summary(class_areas[class_id]),
            **(
                {
                    f"letterbox_{target_size}_px": {
                        "width": dimension_summary(resized_widths[class_id]),
                        "height": dimension_summary(resized_heights[class_id]),
                        "area": dimension_summary(resized_areas[class_id]),
                    }
                }
                if target_size is not None
                else {}
            ),
        }
        for class_id in sorted(class_counts)
    }
    return {
        "root": str(root),
        "images": len(images),
        "labelled_images": labelled_images,
        "negative_images": negative_images,
        "boxes": boxes,
        "class_counts": dict(sorted(class_counts.items())),
        "class_metrics": class_metrics,
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "split_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_counts.items())
        },
        "median_box_width_px": median(widths) if widths else None,
        "median_box_height_px": median(heights) if heights else None,
        "median_box_area_px": median(areas) if areas else None,
        "issues_count": len(issues),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Dataset root containing images/labels")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--missing-label-is-negative",
        action="store_true",
        help="Treat images without a .txt file as intentional negative samples",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        help="Also report box sizes after aspect-preserving square letterboxing",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"dataset root does not exist: {args.root}")

    report = audit(
        args.root,
        missing_label_is_negative=args.missing_label_is_negative,
        target_size=args.target_size,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
