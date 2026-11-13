# Truly reusing the COCO `sports ball` class

A single-class YOLO26n trained from COCO normally loses the pretrained `sports ball` classifier. This document explains the transfer that keeps it (COCO class 32 becomes class 0) and the checks done before training, with the training setup of the first experiment (`configs/mobile_nano_sports_ball_960.json`).

## What was verified

In the COCO model, `sports ball` is class **32**. In our single-class model it becomes **0**, with the same name. Index 0 therefore does not mean "person": that numbering belonged to the COCO list.

With Ultralytics **8.4.153**, a simple rename is not enough for YOLO26n: when nc goes from 80 to 1, the constructor shrinks the classifier's inner channels from 80 to 64. Name-based remapping then keeps the six biases but not the six output weight rows, whose dimensions differ. This is confirmed by the code and by the [remapping documentation](https://docs.ultralytics.com/reference/nn/tasks/).

To keep the learned classifier, `sports_ball_transfer.py`:

1. Copies the pretrained COCO graph without modifying the original.
2. Keeps the visual features and localization.
3. In the three scales of **each of the two detection branches**, keeps only the weights and biases of output 32.
4. Renumbers that output to 0, named `sports ball`; nc becomes 1.
5. Keeps the 80 inner channels instead of rebuilding a 64-channel head.

A trainer only adapts model construction, following the [Ultralytics customization](https://docs.ultralytics.com/usage/engine/) mechanism. Loss, optimizer, EMA, validation and saves stay Ultralytics'. The checkpoint contains a standard `DetectionModel`, not a new model class depending on our helper for inference.

## Evidence before training

- The 12 output tensors match COCO row 32 exactly.
- The other 696 tensors are kept and the source stays unchanged.
- Both branches are tested: ball logits identical up to float rounding, raw and decoded boxes identical.
- Save/reload preserves the graph and predictions.
- Detection loss and gradients of both branches are finite.
- The trainer really keeps width 80, with no hidden rebuild.

These checks are in `tests/test_sports_ball_transfer.py`. The launcher repeats the check before submission, and the GPU worker before optimization. The score check applies to predictions **before top-k**: removing the other classes can change that selection even though raw scores are preserved.

## Data and comparison

**No Infactory/TV image of v002 added.** The same private revision of dataset `mobile_v001` is reused: 1,547 train (1,390 with a ball, 157 negatives); 168 val (134 with a ball, 34 negatives). Annotations stay numbered 0. Images and labels are verified by SHA-256. Only the worker YAML takes the name `sports ball`. No test is used to train or select weights.

Other settings stay those of the first recent trial: YOLO26n, 960 px, batch 8, AdamW, lr 0.001, seed 42, 60 epochs maximum, patience 12, same augmentations. Initialization restarts from the original COCO, **not** from the recent best.pt, which only serves as reference on the same 168 images.

The model has 2,533,886 parameters, versus 2,504,190 in the old single-class model (+1.19%), because the pretrained inner width is kept. We are therefore testing full classifier transfer, **not merely a rename on a strictly identical architecture**. Weights stay free to evolve during training, which does not guarantee that all close-up capabilities are preserved; dedicated images would be needed to measure that.

The private job uses an A10G with a one-hour limit; at the observed 1 USD/h the compute cap is about 1 USD. Old weights are preserved. Before submission the 77 project tests passed. A real CPU mini-training was also verified: 4 train and 2 validation positive images, 1 epoch at 128 px, finite loss and a checkpoint reloadable with width 80. That temporary smoke test is not a performance measurement.

```bash
python scripts/mobile_training_hf.py status --config configs/mobile_nano_sports_ball_960.json
python scripts/mobile_training_hf.py collect --config configs/mobile_nano_sports_ball_960.json
```

## Caution for resuming

`YOLO(best.pt).train()` with the standard trainer can rebuild the head at width 64. Use `SportsBallTransferTrainer` again to preserve the learned channels. Ordinary inference with `YOLO(best.pt)` has no such limitation.
