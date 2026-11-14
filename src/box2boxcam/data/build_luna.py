"""Build a private experiment from reviewed Luna labels, grouped by match."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
from collections import Counter
from pathlib import Path


def choose_splits(groups: dict[str, Counter]) -> dict[str, str]:
    """Exhaustively balance images and positives; never split a match."""
    ids = sorted(groups)
    totals = sum((groups[key] for key in ids), Counter())
    best = None
    for val in itertools.combinations(ids, 2):
        remaining = [key for key in ids if key not in val]
        for test in itertools.combinations(remaining, 2):
            mapping = {
                key: "val" if key in val else "test" if key in test else "train"
                for key in ids
            }
            counts = {split: Counter() for split in ("train", "val", "test")}
            for key, split in mapping.items():
                counts[split].update(groups[key])
            score = sum(
                (counts[split][metric] / totals[metric] - target) ** 2
                for split, target in (("train", .7), ("val", .15), ("test", .15))
                for metric in ("images", "positive", "negative")
                if totals[metric]
            )
            if best is None or score < best[0]:
                best = (score, mapping)
    if best is None:
        raise ValueError("At least four groups are required")
    return best[1]


def build_dataset(root: Path) -> dict:
    review = root / "data/interim/gpt6_luna_label_studio_review"
    export = review / "review_export_002"
    config_path = root / "configs/luna_matches.json"
    config = json.loads(config_path.read_text())
    matches = config["matches"]
    cuts = [group["start"] for group in matches[1:]]
    buffer = config["transition_buffer_frames"]
    with (export / "review_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1232 or len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("The export must contain the 1232 unique submitted images")
    tasks = {
        row["data"]["item_id"]: row["data"]
        for row in json.loads((review / "tasks.json").read_text())
    }
    groups: dict[str, Counter] = {group["id"]: Counter() for group in matches}
    excluded = Counter()
    selected = []
    for row in rows:
        frame = int(row["source_frame"])
        if row["status"] not in {"positive", "negative", "excluded"}:
            raise ValueError(f"Decision not submitted for {row['item_id']}")
        if row["status"] == "excluded":
            excluded["human_excluded"] += 1
            continue
        if any(cut - buffer <= frame < cut + buffer for cut in cuts):
            excluded["transition_quarantine"] += 1
            continue
        owners = [group for group in matches if group["start"] <= frame < group["end"]]
        if len(owners) != 1:
            raise ValueError(f"Ambiguous group for {row['item_id']}")
        row = dict(row, match_id=owners[0]["id"])
        row["luna_status"] = tasks[row["item_id"]]["luna_status"]
        row["source_image"] = str(review / "frames" / (row["item_id"] + ".jpg"))
        groups[row["match_id"]].update(images=1, **{row["status"]: 1})
        selected.append(row)
    mapping = choose_splits(groups)
    for row in selected:
        row["split"] = mapping[row["match_id"]]
    destination = root / "data/processed/luna_v001"
    if destination.exists():
        raise FileExistsError(f"Immutable dataset already present: {destination}")
    # Validate all sources before creating any output.
    for row in selected:
        for path in (Path(row["source_image"]), export / "labels" / (row["item_id"] + ".txt")):
            if not path.is_file():
                raise FileNotFoundError(path)
    for split in ("train", "val", "test"):
        (destination / split / "images").mkdir(parents=True)
        (destination / split / "labels").mkdir()
    for row in selected:
        split = destination / row["split"]
        # Same disk: hard links add no second JPEG data copy. Fail rather than
        # silently duplicating hundreds of MB if linking is unavailable.
        os.link(row["source_image"], split / "images" / (row["item_id"] + ".jpg"))
        source_label = export / "labels" / (row["item_id"] + ".txt")
        (split / "labels" / source_label.name).write_text(source_label.read_text())
    with (destination / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    (destination / "dataset.yaml").write_text(
        "path: " + json.dumps(str(destination))
        + "\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: ball\n"
    )
    counts = {
        split: {
            "images": sum(row["split"] == split for row in selected),
            "positive": sum(row["split"] == split and row["status"] == "positive" for row in selected),
            "negative": sum(row["split"] == split and row["status"] == "negative" for row in selected),
            "matches": [key for key, value in mapping.items() if value == split],
        }
        for split in ("train", "val", "test")
    }
    report = {
        "dataset_id": "box2boxcam_luna_v001_private_pilot",
        "images": len(selected), "splits": counts, "excluded": dict(excluded),
        "match_groups": matches,
        "grouping_method": "manual visual inspection; inferred match identities",
        "match_leakage": 0,
        "images_without_luna_answer_submitted_empty": sum(
            # "réponse Luna manquante" = "Luna response missing" (persisted status value)
            row["luna_status"] == "réponse Luna manquante" and row["status"] == "negative"
            for row in selected
        ),
        "review_policy": "Existing submissions preserved; drafts submitted exactly; unchanged predictions accepted in bulk at user's explicit request.",
        "license_status": "private experimental use; redistribution rights not cleared",
        "image_storage": "hard links to existing review JPEGs; do not edit in place",
        "review_manifest_sha256": hashlib.sha256((export / "review_manifest.csv").read_bytes()).hexdigest(),
        "group_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (destination / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
