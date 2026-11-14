# YOLO26s with automatic field labels, first batch (intermediate experiment)

This experiment (internally `mobile_v005`) adds 196 iPhone images whose labels were judged automatically (no human check) by a vision model to the first field model's training set (YOLO26s). On the reserved match it produces fewer false boxes than the first field model but does not improve the detection ranking at equal recall, so the first field model stayed the reference. The follow-up with human click review is [box2boxcam v1.0](mobile_auto_field_v006_20261111.md).

## New source

`auto_field_judged_v001`: 208 unreviewed iPhone images (the 12 October match and `club_match` clips from 30 September and 7 October), absent from `mobile_v004` (the first field model's dataset), the validation set, the reserved test clip and the reserved 14 October session. Their 999 candidates (full-frame detector, tiled detector, a multimodal proposer) were judged on zoomed crops by a vision model: 179 `ball`, 650 `not_ball`, 170 `unsure`.

**No human checked these labels.** The `label_source` field is `auto_judged_claude_sonnet` in the manifests, and the `mobile_v005` README says so.

## Label policy

- Image dropped if any of its candidates exceeds 64 px on a side (the judge did not see it entirely).
- `ball`: one YOLO box, class 0, candidate box clipped to the image.
- `not_ball`: no label, the object stays background (hard negative).
- `unsure`: neither positive nor negative. Gray rectangle (114, 114, 114) over the box enlarged by 50% (at least 24 px per side), in a JPEG quality 95 copy. Images without `unsure` stay hard links to the original JPEG.
- Image dropped if a `ball` box overlaps a mask.
- Everything goes to train. The manifest keeps, per image, the source path and SHA-256, the match, the clip, the masking and, per candidate, verdict, sources and confidences.

## Calibration and limits

Blind against human labels, on detector candidates, the judge is right 98.7% of the time when confident: 1 wrong `not_ball` out of 272, 3 wrong `ball` out of 45. The 11 `ball` boxes proposed by the multimodal proposer alone fall outside this calibration scope.

A ball proposed by neither the detector nor the proposer stays unannotated and becomes background.

A quick visual check of the 84 `not_ball` kept with detector confidence >= 0.6 found candidate 150 (`2026-09-30_club_match_19h38m04_frame_00002412`, confidence 0.65) to be a real blurred ball at a player's foot, trained here as background. A few others (931, 451, 248) remain ambiguous. This rate is compatible with the calibration, but these errors land precisely on hard negatives.

## Numbers

| | Value |
|---|---:|
| Input images | 208 |
| Dropped: candidate > 64 px | 10 |
| Dropped: ball under a mask | 2 |
| Kept | 196 |
| Positive / negative | 119 / 77 |
| Boxes | 175 |
| Masked images / masks | 106 / 148 |
| `not_ball` kept with detector confidence >= 0.25 | 310 |

By match: 12 October session 138 images (87 positive, 125 boxes), `club_match` 7 October 40 (23 positive, 36 boxes), `club_match` 30 September 18 (9 positive, 14 boxes).

`mobile_v005` reuses `mobile_v004` unchanged and adds the source to train: 3,002 train images (2,534 positive, 468 negative) and the same 168 validation images. Image/label hashes of validation are identical to `mobile_v004`.

## Training setup

Configuration `configs/mobile_small_auto_field_960.json`: same recipe as the first field model (Small; YOLO26s COCO, `sports ball` transfer, 960 px, 30 epochs, patience 8, batch 8, AdamW lr0 0.001, seed 42, A10G GPU, 90 min limit, private repositories). The reference evaluated on the same validation is the first field model (Small). Only the train set changes, but these images are close in time to images already in train.

```sh
python -m box2boxcam.data.build_auto_judged_field <judge_output_dir>
python scripts/mobile_training_hf.py prepare --config configs/mobile_small_auto_field_960.json
python scripts/mobile_training_hf.py launch --config configs/mobile_small_auto_field_960.json
```

`prepare` only builds `mobile_v005` locally. `launch` creates the private repositories, uploads the archive and code, then immediately submits the paid GPU job (about 1 USD/h, 90 min cap).

## Results (11 November 2026)

30 epochs, 30.9 minutes of training.

Unchanged validation (168 images, 134 balls, threshold 0.25, IoU 0.50):

| Model | TP | FP | FN | FP on negative images | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| First field model (Small) | 101 | 21 | 33 | 8 | 49.4% |
| + automatic field labels, first batch | 100 | 19 | 34 | 3 | 52.3% |

Reserved 14 October match, 250 reviewed images of `recording_01` (94 balls, 60 small), evaluation only, full-image PyTorch at 960 px:

| Model | Thr. 0.25 TP / FP | Thr. 0.45 TP / FP | Thr. 0.55 TP / FP | FP per 30 TP | FP per 37 TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| First field model (Small) | 37 / 133 | 34 / 45 | 29 / 17 | 19 | 81 |
| + automatic field labels, first batch | 34 / 92 | 30 / 48 | 28 / 13 | 48 | 180 |

At threshold 0.25, the new model produces fewer false boxes (per-image bootstrap 95% interval: -66 to -16) but also finds fewer balls (-8 to +2). At equal recall it is not better: its scores are shifted downward and the ranking of detections does not improve. **It was not adopted; the first field model remained the reference.** No Core ML export was made.

Reading: 196 images added to 2,806 is a small dose, close to images already present, and the dominant false detections of the reserved match (floodlight, a patch on the grass, shoes at this stadium) are not in these images. One clip, one stadium.
