"""Prepare or export the local Label Studio review of Luna's video labels."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from pitchlens.data.label_studio_review import (
    download_local_export,
    export_review,
    prepare_review,
    sync_local_project,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("prepare", help="Créer les images et les tâches Label Studio")
    actions.add_parser("sync", help="Ajouter les réponses Luna de rattrapage sans doublons")
    actions.add_parser("serve", help="Démarrer Label Studio sur localhost:8080")
    local_export = actions.add_parser(
        "export-local", help="Exporter toutes les tâches locales et convertir la revue en YOLO"
    )
    local_export.add_argument("output_dir", type=Path, help="Nouveau dossier de sortie")
    export = actions.add_parser("export", help="Convertir les corrections en YOLO")
    export.add_argument("json_file", type=Path, help="Export JSON de Label Studio")
    export.add_argument("output_dir", type=Path, help="Nouveau dossier de sortie")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    review_root = root / "data/interim/gpt6_luna_label_studio_review"
    if args.action == "serve":
        executable = root / ".venv-label-studio/bin/label-studio"
        if not executable.is_file():
            raise FileNotFoundError(f"Label Studio n'est pas installé : {executable}")
        env = os.environ.copy()
        env.update({
            "LABEL_STUDIO_DISABLE_UPDATES": "true",
            "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(review_root),
        })
        subprocess.run(
            [str(executable), "start", "--data-dir", str(review_root / "label_studio_state"),
             "--internal-host", "127.0.0.1", "--port", "8080", "--no-browser",
             "--log-level", "WARNING"],
            env=env,
            check=True,
        )
        return
    if args.action == "prepare":
        report = prepare_review(
            project_root=root,
            first_run=root / "data/interim/gpt6_luna_api_full_stride4",
            retry_run=root / "data/interim/gpt6_luna_api_full_stride4_retry",
            output_root=review_root,
        )
    elif args.action in {"sync", "export-local"}:
        credentials = (review_root / "local_login.txt").read_text().splitlines()
        token = next(
            line.partition(": ")[2] for line in credentials if line.startswith("API token:")
        )
        if args.action == "sync":
            report = sync_local_project(
                tasks_path=review_root / "tasks.json", token=token,
            )
        else:
            plan = json.loads(
                (root / "data/interim/gpt6_luna_api_full_stride4/plan.json").read_text()
            )
            expected_ids = {str(case["item_id"]) for case in plan["cases"]}
            tasks = download_local_export(token=token)
            if {str(task["data"]["item_id"]) for task in tasks} != expected_ids:
                raise ValueError("L'export local ne contient pas le plan complet")
            with tempfile.TemporaryDirectory(prefix="pitchlens-review-") as directory:
                temporary_json = Path(directory) / "all_tasks.json"
                temporary_json.write_text(json.dumps(tasks, ensure_ascii=False))
                report = export_review(
                    export_path=temporary_json,
                    output_root=args.output_dir,
                    width=int(plan["video"]["width"]),
                    height=int(plan["video"]["height"]),
                    expected_ids=expected_ids,
                )
    else:
        plan = json.loads(
            (root / "data/interim/gpt6_luna_api_full_stride4/plan.json").read_text()
        )
        report = export_review(
            export_path=args.json_file,
            output_root=args.output_dir,
            width=int(plan["video"]["width"]),
            height=int(plan["video"]["height"]),
            expected_ids={str(case["item_id"]) for case in plan["cases"]},
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
