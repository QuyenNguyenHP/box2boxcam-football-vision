# First field model (Nano): annotated iPhone match

The first field model (Nano size) retrains from COCO on the previous dataset (`mobile_v003`) plus 148 human-annotated frames of an iPhone match. On the unchanged 168-image validation it gains mAP50-95 (43.6% to 45.8%) and 5 more detected balls, but F1 is flat (75.3% to 75.4%): no net gain at this operating point. Model naming used in these docs: see the [vocabulary](README.md#vocabulary).

## Dataset

`mobile_v004` (the dataset folder of this model) reuses the 2,826 frozen entries of `mobile_v003` and adds 148 images of the iPhone match validated in Label Studio: 141 positive, 7 negative, 148 boxes in total, 6 images containing several balls. The 1,095 match images not yet submitted or still in draft are not used. Corrections submitted in older projects were compared to the exports: they match the labels already in the reference.

| Source | Train | Validation |
|---|---:|---:|
| Base compilation (frames proposed by a multimodal model, human-reviewed) | 739 | 168 |
| Personal `club_match` footage | 808 | 0 |
| Already-reviewed shorts | 603 | 0 |
| Close-up balls | 445 | 0 |
| False positives: videos with no ball | 63 | 0 |
| iPhone match: human annotations | 148 | 0 |
| **Total** | **2,806** | **168** |

Train holds 2,415 positive and 391 negative images. Infactory TV data is excluded. The 168 validation images and labels keep their previous hashes exactly; images reserved for test are absent from the GPU batch.

## Starting point and recipe

The experiment is a fresh start from **COCO-pretrained YOLO26n** on all the data together, not a random initialization. The previous Nano (trained on `mobile_v003`) is the comparison reference on validation, not initial weights. The `sports ball` head keeps the COCO classifier features and its output row through the transfer verified in earlier experiments (see `mobile_sports_ball_transfer_20261010.md`).

Restarting from the previous Nano weights and retraining on the same batch would be another protocol (different starting point, continuation recipe); training on the 148 new images alone would be a third. Neither is used here.

Resolution 960 px, batch 8, AdamW, initial lr 0.001, seed 42, maximum 30 epochs, patience 8, A10G GPU, 90-minute total limit. Checkpoints are published privately every two epochs. Best-checkpoint selection uses validation only, never the test set.

Epoch count, starting point and added images change together: the comparison with the previous Nano measures this whole new recipe, not the isolated effect of the 148 images. The existing validation covers few close-ups and does not contain this new match session, so it cannot measure generalization to another night match. The diagnostic on the 63 false-positive images is run on training images and is described as such.

The new session annotates all visible balls, including stationary or touchline ones. Older batches used a "ball in play" policy; their possibly omitted secondary balls remain a point to harmonize in future reviews.

## Commands

- Configuration: `configs/mobile_nano_field_all_balls_960.json`.
- Freezing of the new review is implemented in `src/box2boxcam/data/build_reviewed_field.py` and checked by tests for human selection, explicit negatives, multiple boxes and excluded drafts.

```sh
python scripts/mobile_training_hf.py status --config configs/mobile_nano_field_all_balls_960.json
python scripts/mobile_training_hf.py collect --config configs/mobile_nano_field_all_balls_960.json
```

The Hugging Face dataset and model repositories are private. Launching refuses a job already registered; earlier trainings are never overwritten. `collect` refuses to present a progress checkpoint as a finished result.

## Results

All 30 epochs ran in 25.5 minutes of training; the best validation mAP50-95 on the curve is at epoch 28.

| Same 168-image validation | Previous Nano | First field model (Nano) |
|---|---:|---:|
| mAP50 | 75.1% | 76.2% |
| mAP50-95 | 43.6% | 45.8% |
| Detected balls (TP) | 90 | 95 |
| Missed balls (FN) | 44 | 39 |
| Predictions not matched to annotations (FP) | 15 | 23 |
| Precision | 85.7% | 80.5% |
| Recall | 67.2% | 70.9% |
| F1 | 75.3% | 75.4% |

TP/FP/FN and precision/recall/F1 use the same threshold 0.25 and matching IoU 0.50. The first field model recovers 5 more balls and produces 8 more predictions not matched to an annotation. F1 changes by only 0.08 point: no clear global gain at this operating point. The mAP50-95 gain suggests better detection/localization on this validation, but does not prove improvement on another match or on spare balls. The diagnostic on the 63 negative training images stays at zero false boxes for both models; it is not an independent test.

Historical validation annotations target the ball in play, while the new session annotates all visible balls. A measured FP is therefore a prediction unmatched to these annotations, not automatically proof that the visible object is not a real ball. Extra boxes need inspection.

Final model (Nano) SHA-256: `cd3ad62956b77987d830414231fcaa9246350cb8caaaace8c3923b6213d38a71`. The previous Nano is kept for comparison. The reserved test was not used for selection, and the 148 images of the new match were in train, not in this validation.
