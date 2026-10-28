# Data directory

The folders below are intentionally excluded from Git.

- `raw/`: immutable source downloads and original recordings.
- `interim/`: extracted frames, converted annotations, and pseudo-labels.
- `processed/<version>/`: immutable training-ready dataset versions.

Each downloaded source has two provenance records: a local manifest beside the
raw files and a small tracked copy in `manifests/`. The record contains the
registry ID, source URL, retrieval date, and an immutable revision or archive
checksum. License decisions remain in `configs/data_sources.toml`.

Never silently overwrite a processed dataset version. More importantly, split
by the highest known source group (match, video, clip, or parent image) before
creating train, validation, and test sets. Neighbouring frames or augmented
copies of one parent must never cross those boundaries.

Current local version:

- `processed/v001/`: conservative Infactory subset produced by
  `notebooks/02_dataset_preparation.ipynb`. On APFS it contains copy-on-write
  image clones, plus independent YOLO labels, a complete `manifest.csv`, and
  the excluded cases in `review_queue.csv`.
