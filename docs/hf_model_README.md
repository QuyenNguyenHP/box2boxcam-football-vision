---
license: cc-by-nc-4.0
library_name: ultralytics
pipeline_tag: object-detection
datasets:
- Totorotibo84/pitchlens-v002
tags:
- yolo
- object-detection
- football
- soccer
- ball-detection
- tiny-object-detection
- pitchlens
---

# PitchLens YOLO26x Ball-Detection Teacher

This repository contains the large annotation teacher used by PitchLens to propose football bounding boxes before human review. It is deliberately optimized for recall and annotation quality rather than mobile deployment.

The model was fine-tuned on [PitchLens Ball Detection v002](https://huggingface.co/datasets/Totorotibo84/pitchlens-v002), a sequence-safe, human-curated derivative of the Infactory Soccer Ball Tracking dataset.

## Checkpoints

| Checkpoint | Training resolution | Recommended inference resolution | Role |
|---|---:|---:|---|
| `best.pt` | 640 px | 640 px | original teacher baseline |
| `highres_960/best.pt` | 960 px | 960 px | primary annotation teacher |

## Validation results

All rows use the same locked PitchLens v002 validation set and the same evaluator. Fixed-threshold precision, recall and F1 use confidence 0.25 and IoU 0.5.

| Experiment | mAP50 | mAP50–95 | Precision | Recall | F1 | TP | FP | FN | FP on negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 640-trained teacher at 640 | 69.2% | 34.3% | 71.8% | 66.4% | 69.0% | 328 | 129 | 166 | 12 |
| 640-trained teacher at 960 | 70.0% | 28.2% | 64.8% | 69.4% | 67.1% | 343 | 186 | 151 | 60 |
| **960-trained teacher at 960** | **75.2%** | **36.8%** | **82.5%** | **70.9%** | **76.3%** | **350** | **74** | **144** | **16** |

Continuing training at 960 px matters more than simply increasing inference resolution: compared with the 640-trained model evaluated at 960, the high-resolution checkpoint adds seven true positives and removes 112 false positives.

![Resolution comparison](highres_960/resolution_after/comparison.png)

## Usage

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(
    repo_id="Totorotibo84/pitchlens-yolo26x-v002-teacher",
    filename="highres_960/best.pt",
)
model = YOLO(weights)

results = model.predict(
    source="frame.jpg",
    imgsz=960,
    conf=0.25,
    max_det=5,
)
```

The checkpoint has one class: `ball`. For pseudo-label generation, PitchLens uses a lower confidence threshold and sends every proposal to human review.

## Repository contents

- `highres_960/train/`: training curves, batches and confusion matrices;
- `highres_960/resolution_after/`: controlled resolution comparison;
- `training_report.json` and `highres_960/training_report.json`: reproducible run metadata;
- training and inference scripts used for the Hugging Face GPU jobs;
- a 100-frame pseudo-label pilot under `video_pilots/`.

## Limitations

- Small, blurred and airborne balls remain difficult.
- Static round objects, heads, signage and spare balls can produce false positives.
- Multiple real balls may be visible even though only one belongs to the action.
- The training domain does not yet match a fixed amateur iPhone camera.
- This model is too large for the intended on-device application; it is a teacher for building better data.

Predictions should be treated as annotation proposals, not ground truth.

## License

The model is released under CC BY-NC 4.0 because its training dataset is derived from non-commercially licensed source data. Commercial use is not permitted.
