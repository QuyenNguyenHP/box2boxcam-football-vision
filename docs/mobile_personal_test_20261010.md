# Mobile Nano at 960 px: personal test

Comparison of an earlier Nano (trained on frames proposed by a multimodal model, at 640 px) and the first mobile Nano on 292 reviewed frames of a short clip. The new nano raises detection F1 from 12.9% to 64.2% and framing decisions from 17.5% to 68.5% correct, on a clip from the same match as part of its training data.

## Protocol

Both models, at 960 px, are evaluated on the 292 reviewed frames of the short 30 September clip. Models and threshold were frozen before the test. No training, label or threshold change was decided from these results.

- 264 positive images, 28 negatives, 279 boxes; 11 multi-box images.
- 88 sampled frames excluded by review are not evaluated.
- Confidence >= 0.25. Strict detection: IoU >= 0.50.
- CPU, batch 1 and sequential processing to limit memory use.
- **Limit: the long train clip and the short test clip come from the same match.** The test measures another clip of this situation, not generalization to an unknown match.

## Detection of all boxes

| Metric | Old nano | New nano |
|---|---:|---:|
| mAP50 | 5.5% | 65.4% |
| mAP50-95 | 1.3% | 33.1% |
| Precision | 11.0% | 62.9% |
| Recall | 15.8% | 65.6% |
| F1 | 12.9% | 64.2% |
| TP | 44 | 183 |
| FP | 357 | 108 |
| FN | 235 | 96 |
| FP on the 28 negative images | 32 | 15 |

The mAPs integrate several thresholds; precision/recall/F1 and TP/FP/FN use the fixed point above. The custom confusion matrix uses the same thresholds; its true-negative cell is empty because background is not a countable set of negative objects.

## Framing: a single target

The most confident box is chosen without looking at the annotation, then threshold 0.25 is applied. A wrong target on a positive frame counts as a selection error and a missed ball. A correct abstention on a negative frame counts as a correct decision.

| Per-frame decision, IoU >= 0.20 | Old nano | New nano |
|---|---:|---:|
| Correct target on positive frame | 40 | 182 |
| Wrong target on positive frame | 150 | 23 |
| No target on positive frame | 74 | 59 |
| False alarm on negative frame | 17 | 10 |
| Correct abstention on negative frame | 11 | 18 |
| Correct decisions over 292 frames | 17.5% | 68.5% |
| Targeting F1 | 17.0% | 76.0% |

The result is identical at IoU >= 0.10 for these predictions. The normalized targeting predictions use a square inference at 960, common to both models. Ultralytics detection validation uses its own validation preprocessing. These are two explicit protocols, not a mere recomputation of mAP at threshold 0.20.

With several annotations, **any annotated ball may match**. This benchmark cannot identify the active ball among those of other teams. It also does not measure a tracker or gimbal control.

## Video

The full video shows the new nano on all 1,517 frames of the clip, at the source's average frame rate. It includes unannotated or excluded frames; no extra metric is derived from them. A single target is drawn, with a thin green outline and a two-pixel visual margin; confidence is shown above the image, not on the ball. No tracker, smoothing or human correction of predictions.

## Conclusion

The gain is large on this clip, but 59 positive frames remain without a target, 23 target a wrong box and 10 negatives trigger an alert. A video from another match/date is needed to measure more independent generalization. No new GPU job or paid model call was launched for this evaluation.

Reproduction: `scripts/evaluate_mobile_personal.py --output <new_dir>` runs the evaluation into a new folder; `--video-only` regenerates only the video when metrics are already frozen, after checking the weights are unchanged. Results are presented in `notebooks/09_mobile_training.ipynb`.
