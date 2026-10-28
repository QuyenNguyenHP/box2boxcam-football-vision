"""Build an immutable dataset from source labels and a frozen review snapshot."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from pitchlens.data.audit_yolo import audit


def validate_label(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"Invalid ball label: {line}")
        x, y, width, height = map(float, fields[1:])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, width, height)):
            raise ValueError(f"Invalid coordinates: {line}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Empty box: {line}")
        count += 1
    return count


def resolve_label(source_label: str, decision: dict | None) -> tuple[str, str]:
    action = decision["decision"] if decision else "default_keep"
    if action in {"keep", "default_keep"}:
        text = source_label
    elif action in {"negative", "exclude"}:
        text = ""
    elif action == "bbox":
        values = [decision[key] for key in ("x_center", "y_center", "width", "height")]
        text = "0 " + " ".join(values) + "\n"
    else:
        raise ValueError(f"Unknown decision: {action}")
    validate_label(text)
    return action, text


def build(project_root: Path) -> dict:
    project_root = project_root.resolve()
    parent = project_root / "data/processed/v001"
    destination = project_root / "data/processed/v002"
    if destination.exists():
        raise FileExistsError(f"Version déjà présente, aucun écrasement : {destination}")
    manifest_bytes = (parent / "manifest.csv").read_bytes()
    decision_bytes = (parent / "qa_review_decisions.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode())))
    decision_rows = list(csv.DictReader(io.StringIO(decision_bytes.decode())))
    decisions = {row["item_id"]: row for row in decision_rows}
    keys = {f"{row['sequence']}_{int(row['frame']):06d}" for row in rows}
    if len(keys) != len(rows) or len(decisions) != len(decision_rows):
        raise ValueError("Duplicate manifest or decision identifiers")
    if set(decisions) - keys:
        raise ValueError("Decisions reference unknown frames")

    plan = []
    split_sequences = defaultdict(set)
    for row in rows:
        item_id = f"{row['sequence']}_{int(row['frame']):06d}"
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        split_sequences[row["sequence"]].add(row["split"])
        source_image = project_root / row["source_image"]
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        source_label = (project_root / row["source_label"]).read_text() if row["source_label"] else ""
        action, text = resolve_label(source_label, decisions.get(item_id))
        plan.append((row, item_id, action, text))
    if any(len(splits) != 1 for splits in split_sequences.values()):
        raise ValueError("Sequence shared across splits")

    staging = Path(tempfile.mkdtemp(prefix=".v002-build-", dir=destination.parent))
    try:
        for split in ("train", "val", "test"):
            (staging / split / "images").mkdir(parents=True)
            (staging / split / "labels").mkdir()
        (staging / "review_decisions_snapshot.csv").write_bytes(decision_bytes)
        (staging / "parent_manifest_snapshot.csv").write_bytes(manifest_bytes)
        actions = Counter()
        stats = defaultdict(Counter)
        hashes = defaultdict(set)
        output_rows = []
        kept_sequences = defaultdict(set)
        for row, item_id, action, label in plan:
            actions[action] += 1
            output = {**row, "decision_applied": action, "item_id": item_id,
                      "output_image": "", "output_label": "", "image_sha256": "", "label_sha256": ""}
            if action != "exclude":
                source = project_root / row["source_image"]
                image_rel = Path(row["split"]) / "images" / source.name
                label_rel = Path(row["split"]) / "labels" / f"{source.stem}.txt"
                shutil.copy2(source, staging / image_rel)
                (staging / label_rel).write_text(label)
                digest = hashlib.sha256((staging / image_rel).read_bytes()).hexdigest()
                if digest != hashlib.sha256(source.read_bytes()).hexdigest():
                    raise ValueError("Image copy verification failed")
                hashes[digest].add(row["split"])
                boxes = validate_label(label)
                output.update(
                    output_image=str(image_rel), output_label=str(label_rel),
                    image_sha256=digest, label_sha256=hashlib.sha256(label.encode()).hexdigest(),
                )
                stats[row["split"]]["images"] += 1
                stats[row["split"]]["positive_images" if boxes else "negative_images"] += 1
                stats[row["split"]]["boxes"] += boxes
                stats[row["split"]]["multi_box_images"] += boxes > 1
                kept_sequences[row["split"]].add(row["sequence"])
            output_rows.append(output)
        if any(len(splits) > 1 for splits in hashes.values()):
            raise ValueError("Exact image duplicate across splits")
        with (staging / "manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
        (staging / "dataset.yaml").write_text(
            "# Local root; adapt path when moving to Hugging Face.\n"
            f"path: {json.dumps(str(destination))}\n"
            "train: train/images\nval: val/images\ntest: test/images\nnames:\n  0: ball\n"
        )
        result = audit(staging, target_size=640)
        expected = len(rows) - actions["exclude"]
        if result["issues_count"] or result["images"] != expected:
            raise ValueError(f"Final audit failed: {result}")
        report = {
            "dataset_id": "pitchlens_infactory_v002",
            "source": json.loads((project_root / "data/manifests/infactory_ball_tracking.json").read_text()),
            "policy": "No decision keeps original labels; multiple real balls allowed; original sequence splits preserved.",
            "quality_status": "user_curated_with_original_labels_kept_by_default",
            "source_images": len(rows), "images": expected,
            "actions": dict(actions),
            "splits": {split: {**dict(counts), "sequences": len(kept_sequences[split])} for split, counts in stats.items()},
            "decisions_sha256": hashlib.sha256(decision_bytes).hexdigest(),
            "parent_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "cross_split_duplicate_groups": 0, "sequence_leakage": 0,
            "audit": result,
        }
        (staging / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    except BaseException:
        # Preserve partial artifacts for diagnosis instead of deleting source data.
        print(f"Construction interrompue ; fichiers partiels : {staging}")
        raise
    report_path = project_root / "models/reports/infactory_v002_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build(Path.cwd()), indent=2))
