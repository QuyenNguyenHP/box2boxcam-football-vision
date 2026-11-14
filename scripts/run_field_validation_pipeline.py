"""Complete one reserved-session annotation run, then sync only its draft proposals."""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from pathlib import Path

from box2boxcam.data.luna_batch import _atomic_json

root = Path(__file__).resolve().parents[1]
config = root / "configs/field_validation_20261021.json"
raw = json.loads(config.read_text())
output = root / raw["output_root"]
lock = (output / "pipeline.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
try:
    _atomic_json(
        output / "pipeline_status.json",
        {"phase": "luna_annotation", "human_review_required": True},
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(root / "scripts/annotate_personal_matches_luna.py"),
            "run",
            str(config),
        ],
        cwd=root,
        check=True,
    )
    _atomic_json(
        output / "pipeline_status.json",
        {"phase": "syncing_draft_proposals", "human_review_required": True},
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(root / "scripts/review_field_validation.py"),
            "sync",
            "--config",
            str(config),
        ],
        cwd=root,
        check=True,
    )
    _atomic_json(
        output / "pipeline_status.json",
        {
            "phase": "awaiting_human_review",
            "human_review_required": True,
            "training_started": False,
        },
    )
except BaseException as error:
    _atomic_json(
        output / "pipeline_status.json",
        {
            "phase": "failed",
            "error": f"{type(error).__name__}: {error}",
            "automatic_resubmission": False,
            "human_review_required": True,
        },
    )
    raise
