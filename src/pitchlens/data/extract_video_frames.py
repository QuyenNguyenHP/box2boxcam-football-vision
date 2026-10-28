"""Extract unannotated candidate frames from a local downloaded-video manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from pitchlens.data.pseudo_label_video import sampled_frame_numbers


def extract_video_frames(*, project_root: Path, download_manifest: Path,
                         output_root: Path, frame_stride: int = 4,
                         jpeg_quality: int = 95) -> dict:
    """Preserve provenance and timing; no models, APIs, labels or splits.

    A completed extraction is reusable only with the same inputs/parameters.
    Partial or different existing outputs are never silently overwritten.
    Source-video identity is not asserted to be an independent match identity.
    """
    import cv2

    if frame_stride <= 0 or not 1 <= jpeg_quality <= 100:
        raise ValueError("Stride must be positive; JPEG quality must be in [1,100].")
    root = project_root.resolve()
    source = download_manifest.resolve()
    output = output_root.resolve()
    downloaded = json.loads(source.read_text())
    videos = downloaded["videos"]
    if not videos or len({v["file"] for v in videos}) != len(videos):
        raise ValueError("The manifest must contain unique source videos.")
    signature = {
        "download_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frame_stride": frame_stride, "jpeg_quality": jpeg_quality,
    }
    # Verify all videos before creating output or trusting a cached extraction.
    for video in videos:
        path = source.parent / video["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != video["sha256"]:
            raise ValueError(f"Source video changed: {path}")
    report_path = output / "extraction_report.json"
    if output.exists():
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["signature"] == signature:
                with (output / "manifest.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) == report["images"] and all((root / row["image_path"]).is_file() for row in rows):
                    return report
        raise FileExistsError(f"Different or incomplete extraction: {output}. Use a new output path.")
    (output / "images").mkdir(parents=True)
    rows, summaries = [], []
    for video in videos:
        path = source.parent / video["file"]
        video_id = path.stem
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_N_THREADS, 2])
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        selected = sampled_frame_numbers(total, frame_stride)
        selected_set = set(selected)
        count = 0
        image_dir = output / "images" / video_id
        image_dir.mkdir()
        try:
            for frame_index in range(total):
                ok, pixels = capture.read()
                if not ok:
                    raise RuntimeError(f"Decode stopped at frame {frame_index}/{total} in {path}")
                if frame_index not in selected_set:
                    continue
                item_id = f"{video_id}_frame_{frame_index:08d}"
                image_path = image_dir / f"{item_id}.jpg"
                if not cv2.imwrite(str(image_path), pixels, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise RuntimeError(f"Cannot write {image_path}")
                height, width = pixels.shape[:2]
                # Decoder presentation time, not frame/fps (sources may be VFR).
                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                rows.append({
                    "item_id": item_id, "video_id": video_id,
                    "source_video": str(path.relative_to(root)),
                    "source_video_sha256": video["sha256"],
                    "source_url": video["resolved_url"], "uploader": video["uploader"],
                    "source_frame": frame_index, "timestamp_seconds": timestamp_ms / 1000,
                    "fps": fps, "width": width, "height": height,
                    "image_path": str(image_path.relative_to(root)),
                    "status": "unannotated", "match_id": "", "scene_id": "", "split": "unassigned",
                })
                count += 1
        finally:
            capture.release()
        if count != len(selected):
            raise RuntimeError(f"Missing sampled frames in {path}")
        summaries.append({"video_id": video_id, "source_frames": total,
                          "fps": fps, "images": count, "uploader": video["uploader"]})
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "signature": signature, "images": len(rows), "videos": summaries,
        "output_root": str(output), "status": "extracted_unannotated",
        "timestamp_method": "OpenCV decoded presentation time CAP_PROP_POS_MSEC",
        "image_bytes": sum((root / row["image_path"]).stat().st_size for row in rows),
        "split_policy": "Unassigned. Group clips/replays from the same match before splitting.",
        "label_policy": "No label files generated; missing annotations are not negatives.",
        "redistribution_rights": "not cleared; private experiment only",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
