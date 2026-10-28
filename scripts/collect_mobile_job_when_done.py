"""Wait for one already-submitted GPU job and collect its final artifacts.

This never submits or retries a paid job. It stops on failure/timeout and keeps
the remote progress checkpoints separate from completed local results.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    path = root / "models/reports" / f"{config['experiment_id']}_job.json"
    record = json.loads(path.read_text())
    if not record.get("job_id"):
        raise RuntimeError("No submitted job; collection never starts training")
    api = HfApi()
    deadline = time.monotonic() + 3 * 3600
    previous = None
    while time.monotonic() < deadline:
        job = api.inspect_job(job_id=record["job_id"])
        stage = job.status.stage
        if stage != previous:
            print(f"{record['job_id']}: {stage}", flush=True)
            previous = stage
        if stage == "COMPLETED":
            subprocess.run([str(root / ".venv/bin/python"), "-u", "scripts/mobile_training_hf.py",
                            "collect", "--config", str(args.config)], cwd=root, check=True)
            return
        if stage in {"ERROR", "FAILED", "CANCELED", "CANCELLED", "TIMEOUT", "EXPIRED"}:
            raise RuntimeError(f"GPU job ended {stage}: {job.status.message}; not resubmitted")
        time.sleep(60)
    raise RuntimeError("Local collector deadline reached; inspect the recorded GPU job, do not resubmit")


if __name__ == "__main__":
    main()
