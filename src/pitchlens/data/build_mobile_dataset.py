"""Compose a frozen mobile experiment without changing the reviewed exports."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from pitchlens.data.audit_yolo import audit
from pitchlens.data.reserved_sessions import assert_no_reserved_training


def select_luna_rows(rows):
    """Use existing train/val groups; quarantine ambiguous empty responses."""
    selected, omitted = [], []
    for row in rows:
        if row["split"] not in {"train", "val"}:
            continue
        if row["status"] == "negative" and row["luna_status"] == "réponse Luna manquante":
            omitted.append(row)
        else:
            selected.append(row)
    return selected, omitted


def check_groups(rows):
    """Reject a match assigned to both fitting and checkpoint selection."""
    owners = {}
    for row in rows:
        group = (row["source_dataset"], row["match_id"])
        previous = owners.setdefault(group, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Match shared by train/val: {group}")


def select_tv_rows(rows):
    """Only the curated TV train; do not silently repurpose its val/test."""
    kept = [row for row in rows if row.get("output_image") and row.get("decision_applied") != "exclude"]
    train_sequences = {row["sequence"] for row in kept if row["split"] == "train"}
    reserved_sequences = {row["sequence"] for row in kept if row["split"] in {"val", "test"}}
    if train_sequences & reserved_sequences:
        raise ValueError("TV train shares sequences with reserved validation/test")
    return [row for row in kept if row["split"] == "train"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_reviewed_train(rows, source_name, candidates):
    """Add unsplit reviewed exports to train only; never resplit adjacent frames."""
    if Path(source_name).name != source_name or source_name in {"", ".", ".."}:
        raise ValueError("Expected a simple frozen dataset name")
    for row in candidates:
        if row["split"] == "excluded" or row["split"] == "not_selected":
            continue
        if row["split"] != "unassigned" or row["status"] not in {"positive", "negative"}:
            raise ValueError("Only reviewed, unassigned training candidates can be added")
        rows.append({
            key: row[key] for key in ("item_id", "source_frame", "status", "label_source")
        } | {
            "match_id": row["match_id"] or "unverified_video:" + row["video_id"],
            "match_id_verified": bool(row["match_id"]), "video_id": row["video_id"],
            "source_video_sha256": row["source_video_sha256"],
            "source_dataset": source_name, "source_layout": "unsplit", "split": "train",
        } | {key: row[key] for key in ("image_sha256", "label_sha256", "annotation_scope") if key in row})


def source_directory(root, row):
    source = root / "data/processed" / row["source_dataset"]
    return source if row.get("source_layout") == "unsplit" else source / row["split"]


def build(root: Path, output: Path | None = None, *, include_tv=False, extra_train_sources=(),
          baseline_dataset="mobile_v001"):
    root = root.resolve()
    dataset_name = "mobile_tv_v001" if include_tv else "mobile_v001"
    output = output or root / "data/processed" / dataset_name
    if extra_train_sources:
        if output.name in {"mobile_v001", baseline_dataset}:
            raise ValueError("Enrichment requires a new dataset version")
        dataset_name = output.name
    if output.exists():
        raise FileExistsError(f"Immutable dataset already exists: {output}")
    if include_tv and extra_train_sources:
        raise ValueError("Do not mix the TV control with modern-data enrichment")
    if Path(baseline_dataset).name != baseline_dataset or baseline_dataset in {"", ".", ".."}:
        raise ValueError("Expected a simple baseline dataset name")
    if baseline_dataset != "mobile_v001" and (include_tv or not extra_train_sources):
        raise ValueError("A different baseline is supported only for additive modern enrichment")
    luna = root / "data/processed/luna_v001"
    personal = root / "data/processed/lou_mistrau_train_v001"
    baseline = root / "data/processed" / baseline_dataset
    if include_tv:
        # Freeze exactly the first experiment, not a fresh selection or split.
        rows = json.loads((baseline / "manifest.json").read_text())
        omitted = json.loads((baseline / "quarantine.json").read_text())
        with (root / "data/processed/v002/manifest.csv").open(newline="") as stream:
            tv_rows = select_tv_rows(list(csv.DictReader(stream)))
        if len(tv_rows) != 3048:
            raise ValueError(f"Expected 3048 curated TV training images, got {len(tv_rows)}")
        for row in tv_rows:
            rows.append({"item_id": row["item_id"], "source_frame": int(row["frame"]),
                         "match_id": row["sequence"], "split": "train", "status": row["status"],
                         "source_dataset": "v002", "label_source": row["decision_applied"],
                         "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"]})
    elif extra_train_sources:
        # Reuse the frozen reference exactly, including its validation and quarantine.
        rows = json.loads((baseline / "manifest.json").read_text())
        omitted = json.loads((baseline / "quarantine.json").read_text())
        for name in extra_train_sources:
            candidates = json.loads((root / "data/processed" / name / "review_manifest.json").read_text())
            append_reviewed_train(rows, name, candidates)
    else:
        with (luna / "manifest.csv").open(newline="") as stream:
            selected, omitted = select_luna_rows(list(csv.DictReader(stream)))
        rows = [dict(row, source_dataset="luna_v001") for row in selected]
        for row in json.loads((personal / "review_manifest.json").read_text()):
            if row["split"] == "train":
                rows.append({key: row[key] for key in ("item_id", "match_id", "split", "status", "source_frame", "label_source")}
                            | {"source_dataset": "lou_mistrau_train_v001"})
    assert_no_reserved_training(root, rows)
    check_groups(rows)
    reserved_directories = [luna / "test/images", root / "data/processed/lou_mistrau_test_v001/test/images"]
    if include_tv:
        reserved_directories.extend(root / "data/processed/v002" / split / "images" for split in ("val", "test"))
    test_hashes = {
        sha256(path)
        for directory in reserved_directories
        for path in directory.glob("*.jpg")
    }
    observed = {}
    # Preflight every source and duplicate before writing or linking anything.
    for row in rows:
        source = source_directory(root, row)
        image = source / "images" / (row["item_id"] + ".jpg")
        label = source / "labels" / (row["item_id"] + ".txt")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing image or label: {image}")
        digest = sha256(image)
        label_digest = sha256(label)
        if (include_tv or extra_train_sources) and "image_sha256" in row and (
                row["image_sha256"] != digest or row["label_sha256"] != label_digest):
            raise ValueError(f"Frozen source changed since manifest: {image}")
        if digest in test_hashes or digest in observed:
            raise ValueError(f"Exact image duplicate or test overlap: {image}")
        observed[digest] = row["item_id"]
        row["image_sha256"] = digest
        row["label_sha256"] = label_digest
        row["file_id"] = row["source_dataset"] + "__" + row["item_id"]
    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir()
    for row in rows:
        source = source_directory(root, row)
        os.link(source / "images" / (row["item_id"] + ".jpg"),
                output / row["split"] / "images" / (row["file_id"] + ".jpg"))
        (output / row["split"] / "labels" / (row["file_id"] + ".txt")).write_bytes(
            (source / "labels" / (row["item_id"] + ".txt")).read_bytes())
    (output / "dataset.yaml").write_text(
        "path: " + json.dumps(str(output))
        + "\ntrain: train/images\nval: val/images\ntest: null\nnames:\n  0: ball\n")
    result = audit(output, target_size=960 if include_tv or extra_train_sources else 640)
    if result["issues_count"]:
        raise ValueError(f"New dataset failed audit: {result['issues_count']}")
    report = {
        "dataset_id": "pitchlens_" + dataset_name, "images": len(rows),
        # Original review statuses can be stale after human correction. Count the
        # exported labels, not statuses such as positive_single/review_multi.
        "splits": {split: {"images": result["split_counts"][split]["images"],
                            "positive": result["split_counts"][split]["labelled_images"],
                            "negative": result["split_counts"][split]["negative_images"]}
                   for split in ("train", "val")},
        "sources": dict(Counter(r["source_dataset"] for r in rows)),
        "quarantined_empty_missing_luna": len(omitted),
        "test_images_included": 0, "train_val_match_overlap": 0,
        "exact_image_duplicates": 0, "exact_overlap_with_reserved_tests": 0,
        "validation_policy": "Original visually inferred Luna match groups; no random frame split; no personal test selection",
        "personal_benchmark_caveat": "Long train and short test share the September 30 match, by user choice",
        "initialization": "COCO pretrained YOLO26n; no Infactory fine-tuned weights",
        "image_storage": "hardlinks to immutable reviewed exports",
        "visibility": "private experiment; no new public distribution",
        "source_manifests_sha256": {"luna_v001": sha256(luna / "manifest.csv"),
                                    "lou_mistrau_train_v001": sha256(personal / "review_manifest.json")},
    }
    if include_tv:
        base_rows = json.loads((baseline / "manifest.json").read_text())
        signature = lambda values: sorted((r["file_id"], r["image_sha256"], r["label_sha256"]) for r in values)
        if signature(r for r in rows if r["split"] == "val") != signature(r for r in base_rows if r["split"] == "val"):
            raise ValueError("Controlled experiment changed validation images or labels")
        report.update(
            baseline_dataset="mobile_v001", validation_identical_to_baseline=True,
            added_train_images=len(tv_rows), tv_val_test_images_included=0,
            comparison_caveat="Same recipe and maximum epochs; larger dataset means more optimizer steps per epoch and changes domain/negative balance, not label-quality-only causality",
        )
        report["source_manifests_sha256"].update(
            mobile_v001=sha256(baseline / "manifest.json"), v002=sha256(root / "data/processed/v002/manifest.csv"))
    if extra_train_sources:
        base_rows = json.loads((baseline / "manifest.json").read_text())
        signature = lambda values: sorted((r["file_id"], r["image_sha256"], r["label_sha256"]) for r in values)
        if signature(r for r in rows if r["split"] == "val") != signature(r for r in base_rows if r["split"] == "val"):
            raise ValueError("Enrichment changed validation images or labels")
        report.update(
            baseline_dataset=baseline_dataset, validation_identical_to_baseline=True,
            added_train_sources=list(extra_train_sources), added_train_images=len(rows) - len(base_rows),
            tv_images_included=0,
            added_source_group_caveat="Shorts match identities are unverified; all added clips are train only; exact overlap with reserved evaluation rejected, but edited/cropped scene overlap is not ruled out",
            closeup_validation_caveat="Unchanged validation has little close-up coverage; a separate capture session is needed to measure close-up generalization",
        )
        if baseline_dataset != "mobile_v001":
            report["initialization"] = "Configured independently in GPU run; additive dataset, no Infactory weights required"
        report["source_manifests_sha256"].update(
            json.loads((baseline / "build_report.json").read_text())["source_manifests_sha256"])
        report["source_manifests_sha256"][baseline_dataset] = sha256(baseline / "manifest.json")
        for name in extra_train_sources:
            report["source_manifests_sha256"][name] = sha256(root / "data/processed" / name / "review_manifest.json")
    for name, value in (("manifest.json", rows), ("quarantine.json", omitted),
                        ("build_report.json", report), ("audit.json", result)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        f"# PitchLens {dataset_name} — private training experiment\n\n"
        "Train: reviewed Luna compilation + Lou Mistrau personal train"
        + (" + the 3048 curated Infactory train images" if include_tv else "")
        + (" + reviewed extras: " + ", ".join(extra_train_sources) if extra_train_sources else "")
        + ". Validation: the original separate Luna match groups; no reserved test images.\n\n"
        "Ambiguous empty labels from missing Luna answers are quarantined, not silently used "
        "as negatives. Existing reviewed exports are unchanged. Human-corrected positive "
        "labels remain included even if the original Luna answer was missing.\n\n"
        "Images use hardlinks; do not edit them in place. `manifest.json` records origin, "
        "match, frame and image/label hashes; `build_report.json` has the counts.\n\n"
        "The personal final benchmark shares the September 30 match with a training clip. "
        "It is not an independent-match generalization test.\n")
    return report
