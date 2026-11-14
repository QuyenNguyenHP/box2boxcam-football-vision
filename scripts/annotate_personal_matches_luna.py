"""Stream personal videos to Luna; reserve whole matches for evaluation.

Images are transient during uploads. After collection, retain only train
positives above the configured threshold and ALL sampled test frames.
The explicit retain_all_sampled_frames option keeps all proposals for review,
including negatives and API failures; it never approves training labels.
These are proposals, never human-validated reference annotations.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from box2boxcam.data.luna_batch import (
    TERMINAL_BATCH_STATUSES,
    _atomic_json,
    build_plan,
    collect_batch,
    load_config,
    refresh_status,
    submit_batch,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root, path):
    raw = json.loads(path.read_text())
    output = root / raw["output_root"]
    inventory = []
    configs = []
    groups = {}
    for video in raw["videos"]:
        intended = "train" if video["role"] == "train_candidate" else "test"
        if video["role"] not in {"train_candidate", "test_primary", "test_reserve"}:
            raise ValueError("Unknown video role")
        if video["match_id"] in groups and groups[video["match_id"]] != intended:
            raise ValueError("A match cannot be both train and test")
        groups[video["match_id"]] = intended
        source = root / "data/external/videos" / video["file"]
        inventory.append(
            {
                **video,
                "source_video": str(source.relative_to(root)),
                "sha256": sha256(source),
            }
        )
    snapshot = {"config": raw, "videos": inventory}
    snapshot_path = output / "source_manifest.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != snapshot:
        raise FileExistsError("Inputs changed; use a new output directory")
    _atomic_json(snapshot_path, snapshot)
    config_dir = output / "configs"
    config_dir.mkdir(exist_ok=True)
    for video in inventory:
        if not video["annotate"]:
            continue
        video_id = Path(video["file"]).stem
        config_path = config_dir / f"{video_id}.toml"
        values = {
            "video": video["source_video"],
            "output": str((output / video_id).relative_to(root)),
            **{
                key: raw[key]
                for key in (
                    "model",
                    "reasoning_effort",
                    "image_detail",
                    "jpeg_quality",
                    "frame_stride",
                    "temporal_offset",
                    "max_output_tokens",
                    "prompt_cache_key",
                )
            },
            "max_samples": 0,
        }
        for key in ("annotation_prompt", "target_instruction", "annotation_scope"):
            if raw.get(key):
                values[key] = raw[key]
        text = (
            "\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items())
            + "\n"
        )
        if config_path.exists() and config_path.read_text() != text:
            raise FileExistsError(config_path)
        if not config_path.exists():
            config_path.write_text(text)
        config = load_config(config_path, root)
        plan = build_plan(config)
        configs.append((video, config, plan))
    report = {
        "planned_cases": sum(p["case_count"] for _, _, p in configs),
        "videos": [
            {
                "file": v["file"],
                "role": v["role"],
                "cases": p["case_count"],
                "upload_images": p["unique_uploaded_frame_count"],
            }
            for v, _, p in configs
        ],
        "reserved_unannotated_videos": [
            v["file"] for v in inventory if not v["annotate"]
        ],
        "train_threshold": raw["train_confidence_threshold"],
        "test_policy": "No confidence filter; human correction required before metrics",
        "match_groups": groups,
        "cache_policy": "Stable prompt_cache_key and implicit cache; reuse not guaranteed",
        "image_policy": "Exact full-resolution JPEG cache shared by Luna and detector" if raw.get("inline_images") else "Transient during uploads; selective extraction only after collection",
        "annotation_scope": raw.get("annotation_scope", "in_play"),
        "retain_all_sampled_frames": raw.get("retain_all_sampled_frames", False),
    }
    _atomic_json(output / "run_plan.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return raw, output, configs


def keep_frame(role, prediction, threshold, *, retain_all=False):
    if retain_all or role in {"test_primary", "test_reserve"}:
        return True
    return bool(
        prediction
        and (bool(prediction["balls"]) if "balls" in prediction else prediction["ball_in_play_visible"])
        and prediction["confidence"] >= threshold
    )


def assemble(root, raw, output, configs):
    import cv2

    all_rows = []
    selected_rows = []
    merged = []
    for video, config, plan in configs:
        predictions = json.loads((config.output_root / "predictions.json").read_text())[
            "predictions"
        ]
        by_frame = {p["source_frame"]: p for p in predictions}
        video_id = config.video_path.stem
        rows = {}
        for case in plan["cases"]:
            frame = int(case["source_frame"])
            p = by_frame.get(frame)
            keep = keep_frame(
                video["role"], p, raw["train_confidence_threshold"],
                retain_all=raw.get("retain_all_sampled_frames", False),
            )
            item_id = f"{video_id}_frame_{frame:08d}"
            image = output / "images" / video_id / f"{item_id}.jpg"
            row = {
                "item_id": item_id,
                "video_id": video_id,
                "source_video": video["source_video"],
                "source_video_sha256": video["sha256"],
                "source_url": "",
                "uploader": "personal",
                "source_frame": frame,
                "timestamp_seconds": "",
                "width": plan["video"]["width"],
                "height": plan["video"]["height"],
                "match_id": video["match_id"],
                "role": video["role"],
                "intended_split": "test"
                if video["role"].startswith("test")
                else "train",
                "split": "unassigned",
                "confidence": p["confidence"] if p else "",
                "annotation_status": "luna_proposal" if p else "api_error",
                "human_review_required": True,
                "selected": keep,
                "image_path": str(image.relative_to(root)) if keep else "",
                "source_label": str(
                    (
                        config.output_root / "labels" / f"frame_{frame:08d}.txt"
                    ).relative_to(root)
                )
                if p
                else "",
            }
            rows[frame] = row
        # Exact decoded presentation time; never approximate VFR timing by frame/fps.
        capture = cv2.VideoCapture(
            str(config.video_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
        )
        if not capture.isOpened():
            raise RuntimeError(f"Cannot decode {config.video_path}")
        try:
            for frame in range(max(rows) + 1):
                ok, pixels = capture.read()
                if not ok:
                    raise RuntimeError(f"Decode failed at frame {frame}")
                if frame not in rows:
                    continue
                row = rows[frame]
                row["timestamp_seconds"] = (
                    float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                )
                if pixels.shape[:2] != (row["height"], row["width"]):
                    raise ValueError(
                        "Decoded image dimensions differ from annotation dimensions"
                    )
                if row["selected"]:
                    image = root / row["image_path"]
                    image.parent.mkdir(parents=True, exist_ok=True)
                    if not image.exists():
                        if shutil.disk_usage(output).free < 1024**3:
                            raise RuntimeError(
                                "Less than 1 GiB free; stop extraction safely"
                            )
                        if not cv2.imwrite(
                            str(image),
                            pixels,
                            [cv2.IMWRITE_JPEG_QUALITY, raw["jpeg_quality"]],
                        ):
                            raise RuntimeError(f"Cannot save {image}")
                    selected_rows.append(row)
                all_rows.append(row)
                if frame in by_frame:
                    merged.append({**by_frame[frame], **row})
        finally:
            capture.release()
    for filename, rows in (
        ("manifest.csv", all_rows),
        ("selected_manifest.csv", selected_rows),
    ):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _atomic_json(output / "predictions.json", {"predictions": merged})
    usage = [p.get("usage", {}) for p in merged]
    summary = {
        "planned_cases": len(all_rows),
        "valid_predictions": len(merged),
        "missing_or_failed_predictions": len(all_rows) - len(merged),
        "train_candidates_selected": sum(
            r["intended_split"] == "train" for r in selected_rows
        ),
        "test_frames_to_review": sum(
            r["intended_split"] == "test" for r in selected_rows
        ),
        "train_threshold": raw["train_confidence_threshold"],
        "test_confidence_filter": False,
        "human_review_required": True,
        "retain_all_sampled_frames": raw.get("retain_all_sampled_frames", False),
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "cached_tokens": sum(
            u.get("input_tokens_details", {}).get("cached_tokens", 0) for u in usage
        ),
        "cache_write_tokens": sum(
            u.get("input_tokens_details", {}).get("cache_write_tokens", 0)
            for u in usage
        ),
        "existing_datasets_modified": False,
    }
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status", "collect"))
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Collect available results of an expired batch; never resubmit missing requests",
    )
    args = parser.parse_args()
    if args.allow_partial and args.action != "collect":
        parser.error("--allow-partial is only supported with collect")
    root = Path(__file__).resolve().parents[1]
    raw, output, configs = prepare(root, args.config)
    if raw.get("inline_images"):
        from box2boxcam.data.luna_inline_batch import (
            collect_inline_batch,
            refresh_inline_status,
            submit_inline_batch,
        )
        submitter, refresher, collector = submit_inline_batch, refresh_inline_status, collect_inline_batch
    else:
        submitter, refresher, collector = submit_batch, refresh_status, collect_batch
    if args.action == "plan":
        return
    lock = (output / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            "Run already active; inspect state.json or run.log"
        ) from error
    if args.action == "run":
        with ThreadPoolExecutor(max_workers=raw["upload_workers"]) as pool:
            list(pool.map(submitter, [config for _, config, _ in configs]))
    pending = list(configs)
    while pending:
        for entry in list(pending):
            video, config, _ = entry
            state = refresher(config)
            if args.action == "status":
                pending.remove(entry)
                continue
            if state["batch_status"] == "completed" or (
                args.allow_partial and state["batch_status"] == "expired"
            ):
                if not (config.output_root / "summary.json").exists():
                    collector(config, allow_partial=args.allow_partial)
                pending.remove(entry)
            elif state["batch_status"] in TERMINAL_BATCH_STATUSES:
                raise RuntimeError(
                    f"Batch {video['file']} ended {state['batch_status']}; not resubmitted"
                )
        if args.action != "run":
            break
        if pending:
            time.sleep(raw["poll_seconds"])
    if args.action != "status" and not pending:
        assemble(root, raw, output, configs)


if __name__ == "__main__":
    main()
