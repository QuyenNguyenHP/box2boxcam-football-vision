# iPhone field validation (14 October 2026 session)

Protocol for a frozen development-validation set of 568 frames from one iPhone-filmed match, excluded from training, with human review of all visible balls and a planned model comparison.

This session is reserved for **development validation**, outside training. It has already been looked at and compared qualitatively, so it is not a blind final test. The two recordings belong to **a single match**; they are not two independent matches. The iPhone training session is a different one (12 October), identified separately.

## Frozen sample

| Source recording | Identifier | Duration | Frames to review |
| --- | --- | --- | --- |
| `match 2.mov` | `recording_01` | 157.29 s | 315 |
| `match.mov` | `recording_02` | 126.39 s | 253 |

The **568 frames** cover both clips uniformly: one frame every 15 decoded frames, about 2 Hz. There is no confidence filtering or cherry-picking of favorable passages. Each task provides the center frame and context at about -0.2/+0.2 s. Files stay at 1920 x 1080, JPEG quality 95; SHA-256 hashes and decoded presentation times are kept.

This 2 Hz reference suits a first detection comparison. It cannot precisely measure tracking losses shorter than 500 ms; tracking and framing at 30 Hz need denser temporal annotation.

## Human review

A third-party multimodal model (called through a paid API, highest reasoning effort) proposes boxes from the three images. The evaluated Nano/Small models do not provide the pre-annotations. This does not remove the model's errors: **no proposal is ground truth**.

**Submit is the only confirmation click needed.** The large image is shown alone; context and active-ball options are collapsed. For each image:

1. Correct, add or delete boxes to cover **all visible true balls**, including spare balls. Do not guess hidden positions. A box tightly covers the visible part of the ball.
2. Click **Submit**. If the proposals are already correct, no other action is needed.

If the whole image cannot be annotated reliably, tick the unusable-image checkbox above the image, then click **Submit**. It is excluded from the denominator and never converted into a negative image.

Identifying the active ball is **optional**: in the options, assign the roles active, secondary or undetermined, choose the visibility state, and tick identity-verified only after that review. "Active visible" requires exactly one active box. Without this extra confirmation, the export keeps the verified boxes but classes the active identity as undetermined, with `active_identity_reviewed: false`. Role proposals stay in the raw annotations without being turned into tracking ground truth.

Images with no visible ball must also be explicitly verified. A missing API response, a draft or a skipped task stays pending. The `submit_boxes_v2` protocol only accepts an actually submitted annotation; no task is validated automatically.

## Commands

From the repository root, `scripts/run_field_validation_pipeline.py` waits for batches already submitted to the paid model, collects responses and synchronizes proposals into the review project. A failure never triggers a new paid submission.

```sh
# Review status: launches neither the paid model nor training.
python scripts/review_field_validation.py status

# Resume only the synchronization of already-collected responses.
# Never replaces an existing annotation or human draft.
python scripts/review_field_validation.py sync

# After the full review; refuses while any task is missing or inconsistent.
python scripts/review_field_validation.py freeze
```

The immutable export (`data/processed/field_validation_v001`) contains images, YOLO labels of all visible balls, a manifest with role and active box, a completeness report and a `dataset.yaml` with `train: null`. Excluded images are not exported as negatives. No reference export is created before the review is complete.

## Contamination protection

`configs/reserved_sessions.json` reserves the group `personal_field_session_20261014`, its source paths and SHA-256 hashes. The dataset builder and the Hugging Face launcher check that this group is absent from `train`, including when submitting an already-prepared training. Adding other clips of this match requires the same `match_id`.

The annotation configuration is `configs/field_validation_20261021.json`. The historical field `role: test_primary` is only used by the existing annotation transport; the review project and the export correctly say `val` and `development_validation`. This workflow launches no training.

## Comparison after human validation

The plan `configs/field_evaluation_20261021.json` fixes the three checkpoints to compare: Nano 960, Nano 1280 and Small 960. The first comparison uses a common threshold of 0.25 and one-to-one matching at IoU >= 0.50, separating:

- detection of all visible balls: TP/FP/FN, precision and recall, with results by ball size in source pixels;
- active-ball choice: correct target chosen, secondary ball chosen, wrong target, and no target while the active ball is visible.

Active-ball measurements must be limited to references with `active_identity_reviewed: true`. A box validation by Submit does not justify claiming validation of the active identity.

A box on a spare ball is a good detection but may be a bad choice for the camera; do not mix the two verdicts. Compare models on the same images, without tuning their thresholds separately on this set. Keep the old benchmark separate. Uncertainties must account for temporal correlation; with a single match, nothing can be concluded about variability between stadiums. Latency and power consumption of the three models must be measured separately on the iPhone.
