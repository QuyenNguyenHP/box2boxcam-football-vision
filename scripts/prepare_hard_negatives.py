"""Import user-confirmed negative videos, inspect frames, then freeze empty labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from pitchlens.data.audit_yolo import audit
from pitchlens.data.build_mobile_dataset import sha256
from pitchlens.data.extract_video_frames import extract_video_frames
from pitchlens.data.luna_batch import _atomic_json

ROOT = Path(__file__).resolve().parents[1]


def gallery(root, extraction, rows, position):
    """Generated QA thumbnails only: original JPEGs stay untouched."""
    videos = sorted({r["video_id"] for r in rows})
    sheet = Image.new("RGB", (990, 450 * ((len(videos) + 2) // 3)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, video in enumerate(videos):
        samples = [r for r in rows if r["video_id"] == video]
        row = samples[0 if position == "first" else len(samples) // 2]
        x, y = (index % 3) * 330, (index // 3) * 450
        draw.text((x + 8, y + 8), f"{video} | frame {row['source_frame']} | {float(row['timestamp_seconds']):.2f}s",
                  fill="black")
        with Image.open(root / row["image_path"]) as pixels:
            thumb = ImageOps.contain(pixels.convert("RGB"), (320, 410))
            sheet.paste(thumb, (x + (330 - thumb.width) // 2, y + 30 + (410 - thumb.height) // 2))
    output = extraction / f"inspection_{position}.jpg"
    sheet.save(output, quality=90)
    return str(output.relative_to(root))


def prepare(root, config):
    video_dir = root / config["video_directory"]
    sources = [Path(p) for p in config["source_files"]]
    if not sources or len({p.name for p in sources}) != len(sources):
        raise ValueError("Expected unique source videos")
    videos = []
    # Resolve every input and existing destination before copying anything.
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256(source)
        destination = video_dir / source.name
        if destination.exists() and sha256(destination) != digest:
            raise FileExistsError(f"Different existing video: {destination}")
        videos.append({"file": source.name, "sha256": digest, "resolved_url": "",
                       "uploader": "personal", "original_file": str(source)})
    manifest = {"videos": videos, "capture_session": config["capture_session"]}
    source_manifest = video_dir / "source_manifest.json"
    if source_manifest.exists() and json.loads(source_manifest.read_text()) != manifest:
        raise ValueError("Immutable source manifest differs")
    video_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        destination = video_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        if sha256(destination) != next(v["sha256"] for v in videos if v["file"] == source.name):
            raise ValueError("Video copy checksum mismatch")
    if not source_manifest.exists():
        _atomic_json(source_manifest, manifest)
    extraction = root / config["extraction_directory"]
    report = extract_video_frames(project_root=root, download_manifest=source_manifest,
                                  output_root=extraction, frame_stride=config["frame_stride"],
                                  jpeg_quality=config["jpeg_quality"])
    with (extraction / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    inspection = [gallery(root, extraction, rows, position) for position in ("first", "middle")]
    print(json.dumps({"copied_videos": len(videos), "source_bytes": sum(p.stat().st_size for p in sources),
                      "sampled_images": report["images"], "frame_stride": config["frame_stride"],
                      "inspection_images": inspection, "labels_created": False}, indent=2))
    return rows


def freeze(root, config, *, confirm_negative):
    if not confirm_negative:
        raise ValueError("Explicit user confirmation of ball-free videos is required")
    output = root / config["dataset"]
    if output.exists():
        raise FileExistsError(f"Frozen dataset already exists: {output}")
    extraction = root / config["extraction_directory"]
    report = json.loads((extraction / "extraction_report.json").read_text())
    if report["signature"]["frame_stride"] != config["frame_stride"]:
        raise ValueError("Extraction parameters differ")
    with (extraction / "manifest.csv").open() as stream:
        extracted = list(csv.DictReader(stream))
    if len(extracted) != report["images"] or len({r["item_id"] for r in extracted}) != len(extracted):
        raise ValueError("Incomplete/duplicate extracted inventory")
    # Revalidate the exact video identity recorded when the frames were decoded.
    for video in json.loads((root / config["video_directory"] / "source_manifest.json").read_text())["videos"]:
        if sha256(root / config["video_directory"] / video["file"]) != video["sha256"]:
            raise ValueError("Source video changed after extraction")
    rows = []
    for source in extracted:
        image = (root / source["image_path"]).resolve()
        image.relative_to(extraction.resolve())
        with Image.open(image) as pixels:
            if pixels.size != (int(source["width"]), int(source["height"])):
                raise ValueError("Decoded image dimensions differ")
        rows.append({**source, "source_frame": int(source["source_frame"]),
                     "width": int(source["width"]), "height": int(source["height"]),
                     "source_image": source["image_path"], "status": "negative",
                     "match_id": config["capture_session"], "split": "unassigned",
                     "intended_split": "train", "boxes_pixels": [],
                     "label_source": "user_confirmed_negative_video", "luna_confidence": None,
                     "image_sha256": sha256(image)})
    if len({r["image_sha256"] for r in rows}) != len(rows):
        raise ValueError("Exact duplicate frames: revise sampling before freezing")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    (staging / "images").mkdir()
    (staging / "labels").mkdir()
    for row in rows:
        os.link(root / row["source_image"], staging / "images" / f"{row['item_id']}.jpg")
        (staging / "labels" / f"{row['item_id']}.txt").write_text("")
        row["label_sha256"] = sha256(staging / "labels" / f"{row['item_id']}.txt")
    validation = audit(staging, target_size=960)
    if validation["issues"] or validation["boxes"] or validation["negative_images"] != len(rows):
        raise ValueError(f"Negative export failed audit; staging retained: {staging}")
    validation["root"] = str(output)
    build_report = {
        "retained_images": len(rows), "positive_images": 0, "negative_images": len(rows),
        "videos": dict(Counter(r["video_id"] for r in rows)), "frame_stride": config["frame_stride"],
        "jpeg_quality": config["jpeg_quality"], "label_source": "user_confirmed_negative_video",
        "paid_annotation_calls": 0, "image_storage": "hardlinks",
        "split_policy": "All nine videos are training candidates in one home session; no independent negative holdout",
        "label_policy": "Explicit empty labels: no ball target; not missing or failed annotations",
        "confirmation": "User stated these videos are entirely negative; first/middle visual samples also inspected",
        "source_manifest_sha256": sha256(extraction / "manifest.csv"),
        "existing_datasets_modified": False,
    }
    for name, value in (("review_manifest.json", rows), ("build_report.json", build_report),
                        ("audit.json", validation), ("config.json", config)):
        _atomic_json(staging / name, value)
    (staging / "README.md").write_text(
        "# Négatifs difficiles — maison\n\n"
        f"{len(rows)} images sans ballon, issues des neuf clips IMG_7016–IMG_7024. "
        "Absence de ballon confirmée par l'utilisateur ; inspection de début/milieu "
        "de chaque clip. Une image sur six, JPEG 95, résolution et orientation préservées.\n\n"
        "labels/ contient explicitement des fichiers vides. Aucune classe négative "
        "ni annotation Luna. images/ partage les JPEG par liens physiques. "
        "Le manifeste conserve vidéo, frame, présentation temporelle et empreintes.\n\n"
        "Toutes ces candidates sont destinées au train. Une évaluation dessus décrit "
        "l'apprentissage, pas la généralisation à d'autres objets ou scènes. "
        "Sources et autres exports conservés. Dataset privé, figé.\n")
    if output.exists():
        raise FileExistsError("Output appeared during export")
    staging.rename(output)
    print(json.dumps(build_report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hard_negative_videos_20261012.json")
    parser.add_argument("--confirm-negative-videos", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.action == "prepare":
        prepare(ROOT, config)
    else:
        freeze(ROOT, config, confirm_negative=args.confirm_negative_videos)


if __name__ == "__main__":
    main()
