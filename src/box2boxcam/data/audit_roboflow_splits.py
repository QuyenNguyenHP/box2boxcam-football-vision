"""Find parent-image leakage in a Roboflow-style exported dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from box2boxcam.data.audit_yolo import IMAGE_SUFFIXES, split_name

ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
ORIGINAL_EXTENSION = re.compile(r"_(?:jpg|jpeg|png)$", re.IGNORECASE)


def parent_name(path: Path) -> str:
    """Remove Roboflow's export hash and encoded original extension."""
    without_hash = ROBOFLOW_SUFFIX.sub("", path.stem)
    return ORIGINAL_EXTENSION.sub("", without_hash)


def source_prefix(path: Path) -> str:
    """Return the upstream prefix embedded before the first underscore."""
    return parent_name(path).split("_", maxsplit=1)[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_splits(root: Path) -> dict[str, object]:
    images = sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    split_counts: Counter[str] = Counter()
    parent_splits: dict[str, set[str]] = defaultdict(set)
    prefix_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    parent_counts: Counter[str] = Counter()

    for image_path in images:
        split = split_name(image_path, root)
        parent = parent_name(image_path)
        prefix = source_prefix(image_path)
        split_counts[split] += 1
        parent_counts[parent] += 1
        parent_splits[parent].add(split)
        prefix_splits[prefix].add(split)
        hash_splits[file_sha256(image_path)].add(split)

    cross_split_parents = {
        parent: sorted(splits)
        for parent, splits in parent_splits.items()
        if len(splits) > 1
    }
    cross_split_prefixes = {
        prefix: sorted(splits)
        for prefix, splits in prefix_splits.items()
        if len(splits) > 1
    }
    cross_split_exact_hashes = {
        digest: sorted(splits)
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    }

    return {
        "root": str(root),
        "images": len(images),
        "split_counts": dict(sorted(split_counts.items())),
        "unique_parent_names": len(parent_splits),
        "copies_per_parent": dict(sorted(Counter(parent_counts.values()).items())),
        "parent_names_across_splits": len(cross_split_parents),
        "parent_split_examples": dict(sorted(cross_split_parents.items())[:20]),
        "unique_source_prefixes": len(prefix_splits),
        "source_prefixes_across_splits": len(cross_split_prefixes),
        "source_prefix_split_details": dict(sorted(cross_split_prefixes.items())),
        "exact_file_hashes_across_splits": len(cross_split_exact_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Roboflow-style dataset root")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"dataset root does not exist: {args.root}")

    report = audit_splits(args.root)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
