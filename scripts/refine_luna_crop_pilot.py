"""Prepare, run or inspect only the approved 20-image crop pilot."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from box2boxcam.data.luna_batch import _atomic_json
from box2boxcam.data.luna_crop_pilot import collect, prepare, submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    plan = prepare(root, args.config)
    output = root / config["output_root"]
    if args.action == "prepare":
        print(
            f"Prepared {len(plan['cases'])} images, {plan['requests']} requests; no API calls"
        )
        return
    load_dotenv(root / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI(timeout=60, max_retries=0)
    if args.action == "run":
        run_lock = (output / "run.lock").open("a")
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = submit(root, config, plan, client)
    else:
        state = json.loads((output / "state.json").read_text())
    if not state.get("batch_id"):
        print("Upload/counting in progress; not submitted yet")
        return
    while True:
        batch = client.batches.retrieve(state["batch_id"])
        state.update(
            batch_status=batch.status,
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
            request_counts=batch.request_counts.model_dump(),
        )
        _atomic_json(output / "state.json", state)
        print(batch.status, state["request_counts"], flush=True)
        if args.action == "status":
            return
        if batch.status == "completed":
            if not (output / "summary.json").exists():
                collect(root, config, plan, client, state)
            return
        if batch.status in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"Pilot ended {batch.status}; no automatic paid retry")
        time.sleep(30)


if __name__ == "__main__":
    main()
