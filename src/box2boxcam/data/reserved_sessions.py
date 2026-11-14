"""Keep explicitly reserved recording sessions out of model fitting."""

from __future__ import annotations

import json
from pathlib import Path


def assert_no_reserved_training(root: Path, rows: list[dict]) -> None:
    path = root / "configs/reserved_sessions.json"
    if not path.exists():
        return
    registry = json.loads(path.read_text())
    if registry.get("schema_version") != 1 or not isinstance(
        registry.get("sessions"), list
    ):
        raise ValueError("Invalid reserved-session registry")
    for session in registry["sessions"]:
        if not session.get("match_id") or not session.get("videos"):
            raise ValueError("Incomplete reserved-session policy")
        hashes = {video["source_video_sha256"] for video in session["videos"]}
        paths = {video["source_video"] for video in session["videos"]}
        for row in rows:
            if row.get("split") not in session.get("forbidden_splits", ["train"]):
                continue
            if (
                row.get("match_id") == session["match_id"]
                or row.get("source_video_sha256") in hashes
                or row.get("source_video") in paths
            ):
                raise ValueError(
                    f"Reserved field session cannot enter {row['split']}: {session['match_id']}"
                )
