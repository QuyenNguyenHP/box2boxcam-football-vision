"""Bounded Batch files with inline JPEGs, sharing Luna's schemas and artifacts.

This avoids thousands of individual vision uploads. Every shard stays below
the 200 MB API limit. Images and state remain local for reproducible review.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from pitchlens.data.luna_batch import (
    TERMINAL_BATCH_STATUSES,
    LunaBatchConfig,
    _atomic_json,
    _download_file,
    _write_prediction_artifacts,
    build_plan,
    openai_client,
    parse_batch_output,
    prediction_request_body,
)

SHARD_BYTES = 55 * 1024**2


def extract_inputs(config: LunaBatchConfig, plan: dict) -> dict:
    """Cache the exact full-resolution JPEGs supplied to both annotators."""
    import cv2

    cache = config.output_root / "frames"
    cache.mkdir(exist_ok=True)
    needed = set(plan["unique_uploaded_frames"])
    manifest_path = config.output_root / "frames_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = previous.get("frames", {})
    missing = {
        frame for frame in needed
        if not (cache / f"frame_{frame:08d}.jpg").is_file() or str(frame) not in records
    }
    if not missing:
        return previous
    cap = cv2.VideoCapture(str(config.video_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {config.video_path}")
    try:
        for frame in range(max(missing) + 1):
            ok, pixels = cap.read()
            if not ok:
                raise RuntimeError(f"Decode failed at frame {frame}")
            if frame not in missing:
                continue
            if list(pixels.shape[:2][::-1]) != [plan["video"]["width"], plan["video"]["height"]]:
                raise ValueError("Orientation/dimensions differ from annotation plan")
            ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            content = encoded.tobytes()
            path = cache / f"frame_{frame:08d}.jpg"
            path.write_bytes(content)
            records[str(frame)] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "timestamp_seconds": float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000,
                "bytes": len(content),
            }
    finally:
        cap.release()
    result = {"source_video": plan["source_video"], "jpeg_quality": config.jpeg_quality,
              "orientation_auto": True, "frames": records}
    _atomic_json(manifest_path, result)
    return result


def request_lines(config: LunaBatchConfig, plan: dict):
    for case in plan["cases"]:
        images = {}
        for key in ("previous_frame", "current_frame", "next_frame"):
            frame = case[key]
            pixels = (config.output_root / "frames" / f"frame_{frame:08d}.jpg").read_bytes()
            images[str(frame)] = "data:image/jpeg;base64," + base64.b64encode(pixels).decode("ascii")
        body = prediction_request_body(case=case, file_ids=images, plan=plan)
        row = {"custom_id": case["item_id"], "method": "POST", "url": "/v1/responses", "body": body}
        yield case["item_id"], (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")


def write_shards(output: Path, lines, *, limit: int = SHARD_BYTES) -> list[dict]:
    """Write atomically, bound bytes, and retain exact request membership."""
    output.mkdir(parents=True, exist_ok=True)
    shards, pending, size = [], [], 0

    def flush():
        nonlocal pending, size
        if not pending:
            return
        path = output / f"batch_input_{len(shards):03d}.jsonl"
        temporary = path.with_suffix(".tmp")
        digest = hashlib.sha256()
        with temporary.open("wb") as stream:
            for _, line in pending:
                stream.write(line)
                digest.update(line)
        temporary.replace(path)
        shards.append({"input_filename": path.name, "bytes": size,
                       "sha256": digest.hexdigest(), "item_ids": [item for item, _ in pending]})
        pending, size = [], 0

    for item_id, line in lines:
        if len(line) > limit:
            raise ValueError(f"Request {item_id} exceeds shard limit")
        if size + len(line) > limit:
            flush()
        pending.append((item_id, line))
        size += len(line)
    flush()
    return shards


def prepare_inline(config: LunaBatchConfig) -> dict:
    plan = build_plan(config)
    extract_inputs(config, plan)
    path = config.output_root / "inline_state.json"
    if path.exists():
        return json.loads(path.read_text())
    shards = write_shards(config.output_root, request_lines(config, plan))
    state = {"schema_version": 1, "transport": "inline_jpeg", "batch_status": "not_submitted",
             "shards": shards, "planned_cases": plan["case_count"]}
    _atomic_json(path, state)
    return state


def submit_inline_batch(config: LunaBatchConfig) -> dict:
    state = prepare_inline(config)
    client = openai_client(config)
    path = config.output_root / "inline_state.json"
    for shard in state["shards"]:
        if shard.get("batch_id"):
            continue
        file = config.output_root / shard["input_filename"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != shard["sha256"]:
            raise RuntimeError("Batch input changed after preparation")
        if not shard.get("input_file_id"):
            with file.open("rb") as stream:
                uploaded = client.files.create(file=stream, purpose="batch")
            shard["input_file_id"] = uploaded.id
            _atomic_json(path, state)
        batch = client.batches.create(
            input_file_id=shard["input_file_id"], endpoint="/v1/responses", completion_window="24h",
            metadata={"project": "pitchlens", "run": config.output_root.name, "scope": "all_visible"},
        )
        shard["batch_id"], shard["batch_status"] = batch.id, batch.status
        _atomic_json(path, state)
        print(f"{config.output_root.name}: submitted {batch.id}, {len(shard['item_ids'])} cases", flush=True)
    state["batch_status"] = "submitted"
    _atomic_json(path, state)
    return state


def refresh_inline_status(config: LunaBatchConfig) -> dict:
    path = config.output_root / "inline_state.json"
    state = json.loads(path.read_text())
    client = openai_client(config)
    for shard in state["shards"]:
        if not shard.get("batch_id"):
            shard["batch_status"] = "not_submitted"
            continue
        if shard.get("batch_status") in TERMINAL_BATCH_STATUSES:
            continue
        batch = client.batches.retrieve(shard["batch_id"])
        shard.update({"batch_status": batch.status, "output_file_id": batch.output_file_id,
                      "error_file_id": batch.error_file_id,
                      "request_counts": batch.request_counts.model_dump() if batch.request_counts else {}})
    statuses = [s.get("batch_status", "not_submitted") for s in state["shards"]]
    terminal_failure = next((s for s in statuses if s in TERMINAL_BATCH_STATUSES and s != "completed"), None)
    state["batch_status"] = terminal_failure or ("completed" if all(s == "completed" for s in statuses) else "in_progress")
    _atomic_json(path, state)
    print(f"{config.output_root.name}: {state['batch_status']}", flush=True)
    return state


def collect_inline_batch(config: LunaBatchConfig, *, allow_partial: bool = False) -> dict:
    state = refresh_inline_status(config)
    if state["batch_status"] != "completed":
        raise RuntimeError(f"Inline run {state['batch_status']}; no automatic resubmission")
    plan, client = build_plan(config), openai_client(config)
    combined = []
    for index, shard in enumerate(state["shards"]):
        if not shard.get("output_file_id"):
            raise RuntimeError("Completed shard has no output file")
        for kind in ("output", "error"):
            file_id = shard.get(f"{kind}_file_id")
            if file_id:
                destination = config.output_root / f"batch_{kind}_{index:03d}.jsonl"
                _download_file(client, file_id, destination)
                combined.extend(destination.read_text().splitlines())
    output = config.output_root / "batch_output.jsonl"
    output.write_text("\n".join(combined) + "\n")
    predictions, failures = parse_batch_output(output_path=output, plan=plan)
    expected = {case["item_id"] for case in plan["cases"]}
    ids = [row["item_id"] for row in predictions] + [row["item_id"] for row in failures]
    if len(ids) != len(set(ids)) or set(ids) - expected:
        raise RuntimeError("Duplicate or unknown batch result; preserve raw output for review")
    missing = sorted(expected - set(ids))
    failures.extend({"item_id": item, "error": "missing_batch_result"} for item in missing)
    _write_prediction_artifacts(config=config, plan=plan, predictions=predictions, failures=failures)
    summary_path = config.output_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update({"batch_status": state["batch_status"], "unresolved_item_ids": [r["item_id"] for r in failures],
                    "automatic_retry": False, "human_review_required": True})
    _atomic_json(summary_path, summary)
    return summary
