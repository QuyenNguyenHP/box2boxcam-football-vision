# YOLO26s mobile v006: auto-judged candidates reviewed by clicks

Small v006 trains on 477 auto-judged iPhone images after human click review of the judge's "ball" and "unsure" thumbnails. On the enlarged reserved match it makes clearly fewer false boxes than Small v004 at equal recall; recall gain is not demonstrated.

Training of 11 November 2026, 30 epochs, same recipe as Small v004 (`configs/mobile_small_auto_field_v006_960.json`). The only change: train adds `auto_field_judged_v004` (477 images, 360 boxes, 352 gray masks).

## Added source

`auto_field_judged_v004` = batch 1 (208 images) + batch 2 (300 images) judged by Claude Sonnet, then human click review of the "ball" and "unsure" thumbnails: 21 "ball" rejected (turned into masks), 63 "unsure" promoted to "ball". The "not a ball" verdicts were left as judged (section browsed, no correction made).

## Enlarged reserved test

The 14 October test grows from 250 to 396 images (219 balls, of which 149 are 16 px or smaller): 146 pending images reviewed by clicks on 11 November (132 others excluded). Boxes on the new images are those of the candidates (Luna or the v004 detector), not hand-drawn; 23 of the 125 new balls were proposed only by the v004 detector.

## Results (TP / FP, IoU >= 0.5, imgsz 960)

| Threshold | Small v004 | v005 | v006 |
|---|---:|---:|---:|
| 0.25 | 95 / 203 | 97 / 136 | 100 / 103 |
| 0.35 | 93 / 113 | 93 / 101 | 93 / 60 |
| 0.45 | 89 / 64 | 88 / 61 | 86 / 30 |
| 0.55 | 81 / 23 | 82 / 16 | 81 / 3 |

Maximum TP for a given FP budget: <= 25 -> 81 / 83 / 84; <= 60 -> 88 / 87 / 93; <= 100 -> 91 / 92 / 99; <= 200 -> 95 / 102 / 109. Per-image bootstrap, v006 minus v004: at 0.25 dTP [-4, +5, +14], dFP [-132, -99, -69]; at 0.45 dTP [-11, -3, +4], dFP [-50, -34, -19].

Old validation (168 images, threshold 0.25): v004 101 TP / 21 FP / 33 FN, mAP50-95 49.4%; v006 96 / 19 / 38, 48.3% (FP on negatives 8 -> 4).

Conclusion: v006 makes clearly fewer false boxes on the reserved match at equal recall; recall does not demonstrably improve. No Core ML export at this stage.
