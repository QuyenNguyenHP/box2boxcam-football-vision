"""Render side-by-side crop results offline; no API, label changes or scores."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait", action="store_true", help="Wait locally for collected results"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/luna_crop_pilot.json").read_text())
    output = root / config["output_root"]
    if args.wait:
        deadline = time.monotonic() + 86400
        while not (output / "summary.json").exists():
            state_path = output / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get("batch_status") in {"failed", "expired", "cancelled"}:
                    raise RuntimeError("Pilot terminated without collected results")
            if time.monotonic() >= deadline:
                raise RuntimeError("Results not collected within 24h; no paid retry")
            time.sleep(15)
    plan = json.loads((output / "plan.json").read_text())
    predictions = json.loads((output / "predictions.json").read_text())["predictions"]
    by_case = {(p["item_id"], p["variant"]): p for p in predictions}
    paths = []
    for start in range(0, len(plan["cases"]), 4):
        cases = plan["cases"][start : start + 4]
        fig, axes = plt.subplots(
            len(cases), 3, figsize=(12, 4 * len(cases)), squeeze=False
        )
        for row_index, case in enumerate(cases):
            geometry = case["geometry"]
            left, top, _, _ = geometry["crop_box"]
            border = geometry["border"]
            baseline = case["baseline_bbox"]
            baseline_zoom = [
                border + (baseline[0] - left) * geometry["scale_x"],
                border + (baseline[1] - top) * geometry["scale_y"],
                border + (baseline[2] - left) * geometry["scale_x"],
                border + (baseline[3] - top) * geometry["scale_y"],
            ]
            with Image.open(root / case["zoom_paths"]["zoom"]) as image:
                pixels = image.copy()
            options = [
                ("Proposition initiale", baseline_zoom, case["baseline_confidence"], "")
            ]
            for variant, title in (
                ("zoom", "Zoom seul"),
                ("zoom_grid", "Zoom + repères"),
            ):
                result = by_case.get((case["item_id"], variant))
                if result:
                    prediction = result["prediction"]
                    options.append(
                        (
                            title,
                            prediction["bbox_current_pixels"],
                            prediction["confidence"],
                            "aucun ballon"
                            if not prediction["ball_in_play_visible"]
                            else "",
                        )
                    )
                else:
                    options.append((title, None, None, "réponse non exploitable"))
            for column, (title, box, confidence, note) in enumerate(options):
                ax = axes[row_index][column]
                ax.imshow(pixels)
                if box:
                    x0, y0, x1, y1 = box
                    ax.add_patch(
                        Rectangle(
                            (x0, y0),
                            x1 - x0,
                            y1 - y0,
                            fill=False,
                            edgecolor="#e11d48",
                            linewidth=1.2,
                        )
                    )
                confidence_text = (
                    f" · confiance {confidence:.0%}" if confidence is not None else ""
                )
                ax.set_title(
                    title + confidence_text + (f"\n{note}" if note else ""), fontsize=10
                )
                ax.axis("off")
                if column == 0:
                    ax.text(
                        0,
                        -0.03,
                        f"{start + row_index + 1}/20 · {case['item_id']}",
                        transform=ax.transAxes,
                        fontsize=7,
                        va="top",
                    )
            pixels.close()
        fig.suptitle(
            "Même zoom propre, trois propositions — aucune référence humaine",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=3)
        path = output / f"comparison_page_{start // 4 + 1:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(str(path.relative_to(root)))
    (output / "comparison_pages.json").write_text(
        json.dumps({"pages": paths}, indent=2)
    )
    print("Offline comparisons rendered:", len(paths), "pages")


if __name__ == "__main__":
    main()
