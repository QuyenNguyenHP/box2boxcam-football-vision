---
license: cc-by-nc-4.0
pretty_name: box2boxcam Ball Detection v002
task_categories:
- object-detection
size_categories:
- 1K<n<10K
tags:
- football
- soccer
- ball-detection
- tiny-object-detection
- yolo
- box2boxcam
source_datasets:
- infactory-ai/ball-tracking
---

# box2boxcam Ball Detection v002

box2boxcam v002 is a curated football-ball detection dataset built for one practical question: how reliably can a small detector keep the ball in frame for a low-cost automatic match camera?

The release contains 4,549 images and 3,259 bounding boxes in YOLO format. It is derived from the [Infactory Soccer Ball Tracking dataset](https://huggingface.co/datasets/infactory-ai/ball-tracking), then audited and corrected for box2boxcam. Source sequences remain isolated between train, validation and test.

## Dataset summary

| Split | Images | Positive images | Negative images | Boxes | Sequences |
|---|---:|---:|---:|---:|---:|
| Train | 3,048 | 2,113 | 935 | 2,114 | 30 |
| Validation | 670 | 494 | 176 | 494 | 7 |
| Test | 831 | 639 | 192 | 651 | 8 |
| **Total** | **4,549** | **3,246** | **1,303** | **3,259** | **45** |

There is one class:

```text
0: ball
```

Empty label files are intentional negative examples.

## What changed from the source

The source revision was frozen at commit `75854a5837dd3302cbb1843355ecc1a6f124f818`. All 7,152 source frames were reviewed through a resumable annotation interface.

- 2,603 unusable or ambiguous frames were excluded;
- 246 bounding boxes were redrawn;
- 92 frames were explicitly marked as negative;
- 1,298 source annotations were manually confirmed;
- remaining undecided frames retained their source annotation;
- exact cross-split duplicates and sequence leakage were checked: none remain.

The repository includes the frozen review decisions, source manifest snapshot, output hashes and build report so that the curation history remains inspectable.

## Layout

```text
box2boxcam-v002/
├── train/{images,labels}
├── val/{images,labels}
├── test/{images,labels}
├── dataset.yaml
├── manifest.csv
├── parent_manifest_snapshot.csv
├── review_decisions_snapshot.csv
└── build_report.json
```

## Download and use

```python
from huggingface_hub import snapshot_download

dataset_root = snapshot_download(
    repo_id="Totorotibo84/box2boxcam-football-ball-detection-dataset",
    repo_type="dataset",
)
print(dataset_root)
```

For Ultralytics, either run from the downloaded repository or update the `path` field in `dataset.yaml` to the local snapshot directory.

## Intended use

This release is intended for non-commercial research on football-ball detection, tiny-object detection, tracking and automatic sports-camera prototypes. The validation split is suitable for model selection; keep the test split untouched until the final evaluation.

The images do not yet represent the final box2boxcam deployment domain. Most footage comes from existing match video rather than a fixed iPhone on an amateur touchline. Expect domain shift in camera height, compression, lighting, motion blur and ball size.

## License and attribution

The upstream dataset is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This derived release uses the same license and is not authorized for commercial use.

Please attribute both:

- [Infactory Soccer Ball Tracking](https://huggingface.co/datasets/infactory-ai/ball-tracking), the source dataset;
- box2boxcam v002, for the review, corrections, exclusions and sequence-safe release.

## Related model

The matching high-resolution annotation teacher is available at [Totorotibo84/box2boxcam-football-ball-detection](https://huggingface.co/Totorotibo84/box2boxcam-football-ball-detection).
