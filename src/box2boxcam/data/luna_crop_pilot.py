"""Budget-checked crop refinement experiment; never replaces reviewed labels."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from box2boxcam.data.luna_batch import (
    OUTPUT_SCHEMA,
    _atomic_json,
    _download_file,
    _response_output_text,
    validate_prediction,
)

PROMPT = """You are refining a football ball annotation.
You receive the full CURRENT image for context, then a wide zoom taken
from that same image. Locate the ball in play that is actually visible IN THE ZOOM.
Do not confuse logos, shoes, heads or spare balls with the ball.
Ignore any artificial guide line. If the ball is not visible in
the zoom, answer ball_in_play_visible=false and bbox_current_pixels=null.
Do not copy or guess a box from the description alone.

Give a tight box enclosing the visible outline of the ball, without arbitrary
margin and without including the shoe or the player. If the ball is blurred,
enclose the visible blurred silhouette, without looking for a perfect circle.
The coordinates of bbox_current_pixels are [left,top,right,bottom] IN PIXELS OF
THE PROVIDED ZOOM, origin at the top left of the zoom, white border included.
They are neither percentages nor coordinates of the full image. The conversion
back to the full image will be done by the code, not by you. Briefly describe your
cue in temporal_evidence and the uncertainties in ambiguity."""


def select_cases(predictions: list[dict], count: int) -> list[dict]:
    """Cover sources/confidence strata without using ground-truth geometry."""
    groups = defaultdict(list)
    for row in predictions:
        if row.get("ball_in_play_visible") and row.get("bbox_current_pixels"):
            groups[row["video_id"]].append(row)
    selected = []
    remaining = count
    for index, video_id in enumerate(sorted(groups)):
        rows = sorted(
            groups[video_id], key=lambda row: (row["confidence"], row["item_id"])
        )
        take = min(len(rows), math.ceil(remaining / (len(groups) - index)))
        positions = (
            [round(i * (len(rows) - 1) / (take - 1)) for i in range(take)]
            if take > 1
            else [0]
        )
        selected.extend(rows[pos] for pos in positions)
        remaining -= take
    if len(selected) != count or len({r["item_id"] for r in selected}) != count:
        raise ValueError("Cannot select the requested unique pilot images")
    return selected


def crop_geometry(bbox: list[int], width: int, height: int, config: dict) -> dict:
    """Clip the ROI at image edges; keep exact realized x/y resize factors."""
    x0, y0, x1, y1 = bbox
    side = min(
        config["crop_max_side"],
        max(
            config["crop_min_side"],
            config["crop_ball_margin_factor"] * max(x1 - x0, y1 - y0),
        ),
    )
    cw, ch = min(side, width), min(side, height)
    left = max(0, min(width - cw, round((x0 + x1 - cw) / 2)))
    top = max(0, min(height - ch, round((y0 + y1 - ch) / 2)))
    scale = config["zoom_side"] / max(cw, ch)
    zw, zh = round(cw * scale), round(ch * scale)
    border = config["coordinate_border"]
    return {
        "crop_box": [left, top, left + cw, top + ch],
        "zoom_content_size": [zw, zh],
        "border": border,
        "scale_x": zw / cw,
        "scale_y": zh / ch,
        "zoom_image_size": [zw + 2 * border, zh + 2 * border],
    }


def restore_bbox(bbox: list[int], geometry: dict) -> list[int]:
    """Map back without clamping invalid results into plausible-looking labels."""
    left, top, _, _ = geometry["crop_box"]
    border = geometry["border"]
    width, height = geometry["zoom_content_size"]
    x0, y0, x1, y1 = bbox
    if not (
        border <= x0 < x1 <= border + width and border <= y0 < y1 <= border + height
    ):
        raise ValueError("Returned box lies outside the actual crop content")
    mapped = [
        round(left + (x0 - border) / geometry["scale_x"]),
        round(top + (y0 - border) / geometry["scale_y"]),
        round(left + (x1 - border) / geometry["scale_x"]),
        round(top + (y1 - border) / geometry["scale_y"]),
    ]
    if mapped[0] >= mapped[2] or mapped[1] >= mapped[3]:
        raise ValueError("Returned box becomes empty in original pixels")
    return mapped


def make_zoom(image_path: Path, geometry: dict, output: Path, step: int):
    """Deterministic preprocessing only: no generative edits to dataset pixels."""
    width, height = geometry["zoom_content_size"]
    border = geometry["border"]
    with Image.open(image_path) as image:
        crop = image.convert("RGB").crop(geometry["crop_box"])
        crop = crop.resize((width, height), Image.Resampling.LANCZOS)
    clean = Image.new("RGB", tuple(geometry["zoom_image_size"]), "white")
    clean.paste(crop, (border, border))
    clean_path = output / "zoom.jpg"
    clean.save(clean_path, quality=95)
    marks = Image.new("RGBA", clean.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(marks)
    font = ImageFont.load_default(size=13)
    for x in range(border, border + width + 1, step):
        draw.line((x, border, x, border + height), fill=(255, 255, 255, 55), width=1)
        draw.text((x, border - 14), str(x), fill=(0, 0, 0, 255), font=font, anchor="mb")
        draw.text(
            (x, border + height + 6),
            str(x),
            fill=(0, 0, 0, 255),
            font=font,
            anchor="mt",
        )
    for y in range(border, border + height + 1, step):
        draw.line((border, y, border + width, y), fill=(255, 255, 255, 55), width=1)
        draw.text((border - 3, y), str(y), fill=(0, 0, 0, 255), font=font, anchor="rm")
        draw.text(
            (border + width + 3, y), str(y), fill=(0, 0, 0, 255), font=font, anchor="lm"
        )
    marked = Image.alpha_composite(clean.convert("RGBA"), marks).convert("RGB")
    grid_path = output / "zoom_grid.jpg"
    marked.save(grid_path, quality=95)
    return clean_path, grid_path


def prepare(root: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    source = root / config["source_predictions"]
    output = root / config["output_root"]
    signature = {
        "config": config,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["signature"] != signature:
            raise FileExistsError("Pilot inputs changed: use a new output directory")
        return plan
    rows = select_cases(json.loads(source.read_text())["predictions"], config["cases"])
    cases = []
    for row in rows:
        item_id = row["item_id"]
        case_dir = output / "inputs" / item_id
        case_dir.mkdir(parents=True, exist_ok=True)
        geometry = crop_geometry(
            row["bbox_current_pixels"], int(row["width"]), int(row["height"]), config
        )
        paths = make_zoom(
            root / row["image_path"], geometry, case_dir, config["grid_step"]
        )
        cases.append(
            {
                "item_id": item_id,
                "image_path": row["image_path"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "video_id": row["video_id"],
                "baseline_bbox": row["bbox_current_pixels"],
                "baseline_confidence": row["confidence"],
                "geometry": geometry,
                "zoom_paths": {
                    v: str(p.relative_to(root))
                    for v, p in zip(config["variants"], paths)
                },
            }
        )
    plan = {
        "signature": signature,
        "cases": cases,
        "requests": len(cases) * len(config["variants"]),
        "human_ground_truth": "not available: visual pilot, no accuracy claim",
    }
    _atomic_json(plan_path, plan)
    return plan


def request_body(case: dict, variant: str, files: dict, config: dict) -> dict:
    width, height = case["geometry"]["zoom_image_size"]
    border = case["geometry"]["border"]
    content = [
        {"type": "input_text", "text": "IMAGE ENTIÈRE CURRENT — contexte uniquement."},
        {
            "type": "input_image",
            "file_id": files[case["image_path"]],
            "detail": config["image_detail"],
        },
        {
            "type": "input_text",
            "text": (
                f"ZOOM CURRENT {width}x{height} pixels, white border {border}px. "
                "The requested coordinates relate exclusively to this zoom, border included. "
                "If numbers are present, they are zoom coordinates, not full-image coordinates."
            ),
        },
        {
            "type": "input_image",
            "file_id": files[case["zoom_paths"][variant]],
            "detail": config["image_detail"],
        },
    ]
    return {
        "model": config["model"],
        "reasoning": {"effort": config["reasoning_effort"]},
        "input": [
            {"role": "developer", "content": PROMPT},
            {"role": "user", "content": content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "crop_ball_annotation",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "max_output_tokens": config["max_output_tokens"],
        "store": False,
    }


def budget_bound(input_tokens: int, requests: int, config: dict) -> float:
    if input_tokens < 0 or requests <= 0:
        raise ValueError("Invalid budget inputs")
    # All input may incur cache-write pricing; assume no cached-token discount.
    input_rate = max(
        config["batch_input_per_million"], config["batch_cache_write_per_million"]
    )
    return (
        input_tokens * input_rate
        + requests * config["max_output_tokens"] * config["batch_output_per_million"]
    ) / 1_000_000


def submit(root: Path, config: dict, plan: dict, client: Any) -> dict:
    output = root / config["output_root"]
    state_path = output / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"files": {}, "input_counts": {}}
    )
    if state.get("batch_id"):
        print(f"Pilot already submitted: {state['batch_id']}", flush=True)
        return state
    paths = sorted(
        {c["image_path"] for c in plan["cases"]}
        | {p for c in plan["cases"] for p in c["zoom_paths"].values()}
    )
    for number, path in enumerate(paths, 1):
        if path not in state["files"]:
            with (root / path).open("rb") as stream:
                uploaded = client.files.create(file=stream, purpose="vision")
            state["files"][path] = uploaded.id
            _atomic_json(state_path, state)
        if number % 10 == 0:
            print(f"Pilot uploaded files: {number}/{len(paths)}", flush=True)
    requests = []
    for case in plan["cases"]:
        for variant in config["variants"]:
            custom_id = f"{case['item_id']}__{variant}"
            body = request_body(case, variant, state["files"], config)
            if custom_id not in state["input_counts"]:
                result = client.responses.input_tokens.count(
                    **{k: body[k] for k in ("model", "reasoning", "input", "text")}
                )
                state["input_counts"][custom_id] = int(result.input_tokens)
                _atomic_json(state_path, state)
            if state["input_counts"][custom_id] > 272000:
                raise ValueError("Long-context pricing is outside this budget policy")
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
            )
    upper = budget_bound(sum(state["input_counts"].values()), len(requests), config)
    state["budget_check"] = {
        "counted_input_tokens": sum(state["input_counts"].values()),
        "output_ceiling_tokens": len(requests) * config["max_output_tokens"],
        "estimated_upper_cost_usd_before_tax": upper,
        "approved_usd_before_tax": config["budget_usd_before_tax"],
    }
    _atomic_json(state_path, state)
    print(
        f"Budget upper estimate: ${upper:.4f} / approved ${config['budget_usd_before_tax']:.2f}",
        flush=True,
    )
    if upper > config["budget_usd_before_tax"]:
        raise ValueError("Budget exceeded: no paid annotation batch submitted")
    input_path = output / "batch_input.jsonl"
    with input_path.open("w") as stream:
        for request in requests:
            stream.write(json.dumps(request, ensure_ascii=False) + "\n")
    if not state.get("input_file_id"):
        with input_path.open("rb") as stream:
            state["input_file_id"] = client.files.create(
                file=stream, purpose="batch"
            ).id
        _atomic_json(state_path, state)
    # A timed-out create must not be repeated blindly: SDK retries are disabled.
    if state.get("submission_attempted"):
        raise RuntimeError(
            "Uncertain previous submission; inspect remote batches before retrying"
        )
    state["submission_attempted"] = True
    _atomic_json(state_path, state)
    batch = client.batches.create(
        input_file_id=state["input_file_id"],
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"project": "box2boxcam", "run": output.name},
    )
    state.update(batch_id=batch.id, batch_status=batch.status)
    _atomic_json(state_path, state)
    print(f"Pilot submitted: {batch.id} ({batch.status})", flush=True)
    return state


def collect(root: Path, config: dict, plan: dict, client: Any, state: dict):
    output = root / config["output_root"]
    result_file = output / "batch_output.jsonl"
    _download_file(client, state["output_file_id"], result_file)
    cases = {case["item_id"]: case for case in plan["cases"]}
    results, failures, usage_rows = [], [], []
    for line in result_file.read_text().splitlines():
        raw = json.loads(line)
        item_id, variant = raw["custom_id"].rsplit("__", 1)
        body = (raw.get("response") or {}).get("body") or {}
        usage_rows.append(body.get("usage", {}))
        try:
            if (raw.get("response") or {}).get("status_code") != 200 or body.get(
                "status"
            ) != "completed":
                raise ValueError(
                    f"Unusable response: {body.get('status')} {raw.get('error')}"
                )
            case = cases[item_id]
            if variant not in config["variants"]:
                raise ValueError("Unknown pilot variant")
            prediction = json.loads(_response_output_text(body))
            width, height = case["geometry"]["zoom_image_size"]
            validate_prediction(prediction, width=width, height=height)
            original_bbox = (
                restore_bbox(prediction["bbox_current_pixels"], case["geometry"])
                if prediction["ball_in_play_visible"]
                else None
            )
            results.append(
                {
                    "item_id": item_id,
                    "variant": variant,
                    "bbox_original_pixels": original_bbox,
                    "prediction": prediction,
                    "usage": body.get("usage", {}),
                }
            )
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"custom_id": raw["custom_id"], "error": str(error)})
    received = {r["item_id"] + "__" + r["variant"] for r in results} | {
        r["custom_id"] for r in failures
    }
    expected = {
        c["item_id"] + "__" + v for c in plan["cases"] for v in config["variants"]
    }
    for missing in sorted(expected - received):
        failures.append({"custom_id": missing, "error": "Missing batch response"})
    cost = 0.0
    for usage in usage_rows:
        details = usage.get("input_tokens_details", {})
        cached, writes = (
            details.get("cached_tokens", 0),
            details.get("cache_write_tokens", 0),
        )
        cost += (
            (usage.get("input_tokens", 0) - cached - writes)
            * config["batch_input_per_million"]
            + cached * 0.005
            + writes * config["batch_cache_write_per_million"]
            + usage.get("output_tokens", 0) * config["batch_output_per_million"]
        ) / 1_000_000
    summary = {
        "requests": plan["requests"],
        "valid_responses": len(results),
        "failed_responses": len(failures),
        "estimated_actual_usd_before_tax": cost,
        "review_required": True,
        "human_ground_truth": plan["human_ground_truth"],
        "old_annotations_replaced": False,
    }
    _atomic_json(output / "predictions.json", {"predictions": results})
    _atomic_json(output / "failures.json", {"failures": failures})
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return results
