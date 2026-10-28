"""Download a pinned Hugging Face dataset and record its provenance."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def download_dataset(
    repo_id: str,
    registry_id: str,
    destination_root: Path,
    revision: str | None = None,
    max_workers: int = 8,
    manifest_root: Path = Path("data/manifests"),
) -> Path:
    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, revision=revision)
    resolved_revision = info.sha
    destination = destination_root / registry_id
    destination.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=destination,
        max_workers=max_workers,
    )

    manifest = {
        "registry_id": registry_id,
        "provider": "huggingface",
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_url": f"https://huggingface.co/datasets/{repo_id}",
    }
    rendered_manifest = json.dumps(manifest, indent=2) + "\n"
    (destination / "pitchlens_source_manifest.json").write_text(rendered_manifest)

    # The copy beside the raw files travels with a local download. The second
    # copy is deliberately outside data/raw so Git can preserve provenance
    # without tracking the dataset itself.
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / f"{registry_id}.json").write_text(rendered_manifest)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Hugging Face dataset ID")
    parser.add_argument("registry_id", help="ID from configs/data_sources.toml")
    parser.add_argument("--destination-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--revision", help="Optional branch, tag, or commit hash")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel file downloads (default: 8; higher values may be rate-limited)",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/manifests"),
        help="Tracked provenance manifests (default: data/manifests)",
    )
    args = parser.parse_args()

    destination = download_dataset(
        repo_id=args.repo_id,
        registry_id=args.registry_id,
        destination_root=args.destination_root,
        revision=args.revision,
        max_workers=args.max_workers,
        manifest_root=args.manifest_root,
    )
    print(destination.resolve())


if __name__ == "__main__":
    main()
