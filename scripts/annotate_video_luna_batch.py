"""Command line entry point for GPT-6 Luna video annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitchlens.data.luna_batch import (
    build_plan,
    cleanup_remote_files,
    collect_batch,
    load_config,
    refresh_status,
    submit_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate video frame triplets with GPT-6 Luna through the Batch API."
    )
    parser.add_argument(
        "action", choices=("plan", "submit", "status", "collect", "cleanup")
    )
    parser.add_argument("config", type=Path, help="TOML run configuration")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow cleanup before a successful collection",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config, project_root)

    if args.action == "plan":
        plan = build_plan(config)
        print(
            json.dumps(
                {
                    "output": str(config.output_root),
                    "source_video": plan["source_video"],
                    "cases": plan["case_count"],
                    "unique_remote_images": plan["unique_uploaded_frame_count"],
                    "model": plan["model"],
                    "reasoning_effort": plan["reasoning_effort"],
                    "image_detail": plan["image_detail"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.action == "submit":
        submit_batch(config)
    elif args.action == "status":
        refresh_status(config)
    elif args.action == "collect":
        collect_batch(config)
    elif args.action == "cleanup":
        cleanup_remote_files(config, force=args.force)


if __name__ == "__main__":
    main()
