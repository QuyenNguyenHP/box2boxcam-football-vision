# Nano 960: do the old TV images help?

Adding the v002 TV train set (3,048 images) to the recent iPhone-style data improves mAP50-95 (42.5% to 46.1%) and cuts false detections (34 to 11), at the cost of recall (92 to 87 detected balls). The gain is not uniform.

This second experiment adds the **v002 train set**, without its old validations or tests. Already-excluded images stay excluded. Exported post-correction annotations are used, not the old YOLO proposals. Quality is heterogeneous: this trial assumes neither that all TV images are bad nor that adding them will help.

| Data | Recent only (finished) | Recent + TV (new trial) |
|---|---:|---:|
| Recent train | 1,547 | 1,547 |
| TV train v002 | 0 | 3,048 |
| Total train | 1,547 | 4,595 |
| Train with a ball | 1,390 | 3,503 |
| Train without a ball | 157 | 1,092 |
| Validation, strictly identical | 168 | 168 |
| Test images used for training | 0 | 0 |

The comparison uses **YOLO26n at 960 px**, same COCO initialization, batch 8, AdamW, seed 42, same augmentations, 60 epochs maximum and patience 12. The recent model is **not** the starting point of the new training: its `best.pt` only serves as reference on the common validation, and the new trial has its own output folder.

Interpretation caveat: quantity, visual domain, annotation quality and negative proportion (10.1% to 23.8%) change together, and an epoch also contains more updates. This is therefore not an isolated measure of the effect of "average quality". The metrics mainly say whether **this specific addition** helps our model.

The checkpoint is selected on validation, never on the 292 personal images. That personal benchmark has already been looked at; a new comparison on it would be exploratory, not a new blind test. The short test clip and the long train clip share the 30 September match; another fully reserved match would be needed for an independent generalization measure.

Configuration: `configs/mobile_nano_tv_960.json`. The derived export `mobile_tv_v001` uses hard-linked JPEGs, with no second permanent copy. The private job was limited to two hours on an A10G (cap about 2 USD at 1 USD/h, not a billing forecast), doubled from the first trial because train is three times larger.

## Final results on the common validation

The job completed: 59 epochs, 4,574.8 seconds of training (about 76 minutes). Fixed-threshold metrics use confidence >= 0.25 and IoU >= 0.50.

| Metric | Recent only | Recent + TV |
|---|---:|---:|
| mAP50 | 75.6% | 75.0% |
| mAP50-95 | 42.5% | 46.1% |
| Precision at fixed threshold | 73.0% | 88.8% |
| Recall at fixed threshold | 68.7% | 64.9% |
| F1 at fixed threshold | 70.8% | 75.0% |
| Good detections | 92 | 87 |
| False detections | 34 | 11 |
| Missed balls | 42 | 47 |
| False detections on negative images | 13 | 3 |

The addition improves overall localization and clearly reduces false positives, with an unfavorable recall trade-off at the chosen threshold: 23 fewer false detections but 5 more missed balls. It is not a uniform improvement. The personal benchmark and close-ups have not yet been re-evaluated with this checkpoint; the results above do not prove improvement in those situations.

```bash
python scripts/mobile_training_hf.py status --config configs/mobile_nano_tv_960.json
python scripts/mobile_training_hf.py collect --config configs/mobile_nano_tv_960.json
```

These commands launch no new paid job; `collect` refuses to present an incomplete training as a final result. Section 10 of `notebooks/09_mobile_training.ipynb` shows the comparison, confusion matrices and learning curves.
