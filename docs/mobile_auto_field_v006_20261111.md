# box2boxcam v1.0: automatic field labels with human click review

box2boxcam v1.0 (YOLO26s, internally `mobile_v006`) is the released model. It trains on 477 automatically judged iPhone images after human click review of the judge's "ball" and "unsure" thumbnails. On the enlarged reserved match it makes clearly fewer false boxes than the first field model at equal recall; a recall gain is not demonstrated.

The headline comparison with stock YOLO26s is in the [README](../README.md#fine-tuning-vs-stock-yolo) (public test split, heads blurred). This page documents how the model was built and the intermediate steps.

Training of 11 November 2026, 30 epochs, same recipe as the first field model (`configs/mobile_small_auto_field_v006_960.json`). The only change: train adds `auto_field_judged_v004` (477 images, 360 boxes, 352 gray masks).

## Added source

`auto_field_judged_v004` is batch 1 (208 images) plus batch 2 (300 images), judged by a vision model (Claude Sonnet), then human click review of the "ball" and "unsure" thumbnails: 21 "ball" rejected (turned into masks), 63 "unsure" promoted to "ball". The "not a ball" verdicts were left as judged (section browsed, no correction made).

## Enlarged reserved test

The 14 October test grows from 250 to 396 images (219 balls, of which 149 are 16 px or smaller): 146 pending images reviewed by clicks on 11 November (132 others excluded). Boxes on the new images are those of the candidates (a multimodal proposer or the first field model), not hand-drawn; 23 of the 125 new balls were proposed only by the first field model.

## Results on the original (unblurred) frames

True / false positives, IoU >= 0.5, 960 px. These numbers come from the original frames of the reserved match, before head blurring. **The reference table is the one in the README, measured on the public blurred test split**; it differs from this one by a few boxes (for example 96 / 207 against 95 / 203 for the first field model at 0.25). This table is kept because it is the only one that includes the intermediate "+ automatic field labels, first batch" step ([page](mobile_auto_field_v005_20261111.md)).

| Threshold | First field model | + automatic field labels, first batch | box2boxcam v1.0 |
|---|---:|---:|---:|
| 0.25 | 95 / 203 | 97 / 136 | 100 / 103 |
| 0.35 | 93 / 113 | 93 / 101 | 93 / 60 |
| 0.45 | 89 / 64 | 88 / 61 | 86 / 30 |
| 0.55 | 81 / 23 | 82 / 16 | 81 / 3 |

Maximum TP for a given FP budget (same three columns): <= 25 -> 82 / 83 / 84; <= 60 -> 88 / 87 / 93; <= 100 -> 91 / 92 / 99; <= 200 -> 95 / 102 / 109. Per-image bootstrap, v1.0 minus first field model: at 0.25 dTP [-4, +5, +14], dFP [-132, -99, -69]; at 0.45 dTP [-11, -3, +4], dFP [-50, -34, -19].

Old validation (168 images, threshold 0.25): first field model 101 TP / 21 FP / 33 FN, mAP50-95 49.4%; v1.0 96 / 19 / 38, 48.3% (FP on negatives 8 -> 4).

Conclusion: v1.0 makes clearly fewer false boxes on the reserved match at equal recall; recall does not demonstrably improve. On small, far balls recall is still under 50 % (57 of 149 at 0.25 on the original frames, 55 on the blurred public split).
