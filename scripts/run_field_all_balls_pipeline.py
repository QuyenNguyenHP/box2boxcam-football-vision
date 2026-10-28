"""Submit once, collect asynchronously, then refresh this session's review."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = str(root / ".venv/bin/python")
    subprocess.run([python, "-u", "scripts/annotate_personal_matches_luna.py", "run", str(args.config)], cwd=root, check=True)
    subprocess.run([python, "-u", "scripts/prepare_field_all_balls.py", "review", str(args.config)], cwd=root, check=True)
    # Import only after the complete detector pass exists. Human reviews are
    # preserved by the importer; missing API answers remain review items.
    import json
    raw = json.loads(args.config.read_text())
    subprocess.run([python, "-u", "scripts/import_field_all_balls_review.py", raw["output_root"]], cwd=root, check=True)


if __name__ == "__main__":
    main()
