# High-resolution 960 px checkpoint

`best.pt` is the primary PitchLens annotation teacher. It continues from the 640 px v002 checkpoint and was fine-tuned for 12 epochs at 960 px on an NVIDIA A10G.

## Locked validation results at 960 px

- mAP50: **75.2%**
- mAP50–95: **36.8%**
- precision at confidence 0.25: **82.5%**
- recall at confidence 0.25: **70.9%**
- F1 at confidence 0.25: **76.3%**
- false positives on negative images: **16**

The test split was not used. Complete training curves, batches, confusion matrices and machine-readable metrics are stored alongside the checkpoint.

This is an offline dataset-construction model, not the mobile deployment model. Every pseudo-label should remain subject to human review.
