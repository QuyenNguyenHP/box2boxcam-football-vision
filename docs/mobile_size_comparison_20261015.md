# Nano 1280 vs Small 960 (15 October 2026)

Two trainings compared against the first field model (Nano, 960 px) on the frozen `mobile_v004` export. Small 960 gives the best overall balance (101 TP / 21 FP / 33 FN, mAP50-95 49.4%); raising Nano resolution to 1280 did not help.

Each job used an A10G GPU at about 1 USD/h with a 90-minute timeout, so total GPU cost was capped at about 3 USD. Model repositories are private.

## Common protocol

- Frozen `mobile_v004` export, 2,806 train / 168 val images, no TV data; same dataset revision and same archive SHA-256 (`a668ba60c58a64afe24c68e0227e9cfa11e4d411ef82019f85dd2318378f293d`).
- COCO initialization of each architecture, full transfer of the `sports ball` classifier (row 32 to 0), without restarting from the fine-tuned first-field-model weights.
- 30 epochs maximum, patience 8, batch 8, seed 42, AdamW, lr0 0.001, unchanged augmentations. Private checkpoints every 2 epochs.
- Reference: the first field model (Nano) at 960, re-evaluated at each trial's own resolution (1280 or 960) on the same validation. No reserved test is used to select the checkpoint.
- The iPhone clips provided on 15 October stay outside this frozen export; compare on them afterwards without feeding them back into this training.

## Pipeline adaptation and checks

The launcher and worker now take the configured `base_model` checkpoint instead of imposing Nano. Small keeps its 128 classifier channels, Nano its 80, in both branches. The 25 related tests pass: transferred-weight identity, raw and decoded box predictions, finite loss and gradients, save/reload, trainer, continuation, and protection against duplicate paid submissions. Full local checks before upload pass for both architectures; hashes of the frozen images/labels are verified. The worker repeats the checks on GPU.

The app keeps the first field model (Nano). Core ML conversion and real iPhone performance remain to be verified; an A10G GPU time does not measure iPhone time.

Configurations: `configs/mobile_nano_field_all_balls_1280.json` (yolo26n.pt at 1280 px) and `configs/mobile_small_field_all_balls_960.json` (yolo26s.pt at 960 px), both run with `scripts/mobile_training_hf.py` (`status`, `collect`).

## Results (collected 16 October 2026)

Same validation: 168 images, 134 annotated balls, confidence threshold 25%, matching IoU 50%.

| Model | TP | FP | FN | Precision | Recall | F1 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26n.pt 960 | 95 | 23 | 39 | 80.5% | 70.9% | 75.4% | 45.8% |
| yolo26n.pt 1280 | 92 | 31 | 42 | 74.8% | 68.7% | 71.6% | 45.9% |
| yolo26s.pt 960 | 101 | 21 | 33 | 82.8% | 75.4% | 78.9% | 49.4% |

Small 960 has the best overall result: six more TP, six fewer FN and two fewer FP than the current Nano. On validation images without a ball, however, its FP go from 4 to 8. Nano 1280 does worse at the fixed threshold; raising the resolution was not enough in this trial. FP are defined by the existing annotations, whose coverage of secondary balls remains a limit. The validation does not specifically measure the small balls of the two new night clips.

Both jobs completed. Nano 1280: 15 epochs, best mAP50-95 at epoch 7, 19.1 minutes of training. Small 960: 26 epochs, best mAP50-95 at epoch 18, 25.1 minutes of training. Both hit the early-stop criterion after eight epochs without improvement. Best-weight SHA-256 hashes are in `models/reports/mobile_size_comparison_20261015.json`.

The first field model (Nano) re-evaluated directly at 1280 finds 103 balls but produces 62 FP: this control confirms that upscaling can also bring out more false proposals. The Nano retrained at 1280 brings FP down to 31 but does not beat the existing Nano 960.

Reported job times are A10G measurements, not iPhone measurements. No Core ML export, model replacement or iPhone installation was done.
