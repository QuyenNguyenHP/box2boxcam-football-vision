# Training artifacts

This directory is local and is not uploaded to Git. It contains generated
artifacts, not source data.

```text
checkpoints/
├── yolo26n.pt                 downloaded pretrained starting point
├── teachers/
│   └── yolo26x.pt             large offline annotation teacher
└── baseline_runs/
    └── yolo26n_v002_local_<YYYYMMDD_HHMMSS>/
```

Each timestamped baseline directory is one independent experiment. It references
`data/processed/v002` directly; images and labels are not copied into the run.

Important files inside one run:

- `dataset.yaml`: points to the dataset version used by the run.
- `experiment.json`: exact training and evaluation configuration.
- `training.log`: detailed Ultralytics log.
- `train/weights/best.pt`: best weights according to validation.
- `train/weights/last.pt`: weights after the final epoch.
- `train/results.csv`: one row of losses and metrics per epoch.
- `comparison.json` and `comparison.md`: before/after metrics.
- `comparison.png`: readable before/after chart.
- `confusion_matrix_fixed.png`: TP, FP and FN at confidence 0.25 / IoU 0.50.
- `learning_curves.png`: train/validation losses and validation mAP.
- `decision_curves.png`: precision, recall and F1 as the confidence threshold changes.
- `examples_corrected.jpg`: examples where fine-tuning fixed an error.
- `examples_missed_after.jpg`: annotated balls still missed after fine-tuning.
- `examples_false_alarms_after.jpg`: false detections still present after fine-tuning.

The `before/`, `after/`, and `train/` image files are automatic Ultralytics
diagnostics. They can be regenerated. The `.pt`, JSON, CSV and YAML files are
the important artifacts to retain for a useful experiment.

Smoke tests were temporary pipeline checks. They were removed once the complete
training pipeline had been validated.

`teachers/yolo26x.pt` is the generic COCO starting point and is not a deployment
candidate. At the same 640 px resolution it scores below the fine-tuned nano.
`teachers/yolo26x_v002_hf/best.pt` is the useful annotation teacher fine-tuned
on the private Hugging Face v002 dataset with an A10G. It reaches 69.1% mAP50,
34.3% mAP50-95 and 69.1% fixed-threshold F1 on validation. It runs on CPU on
this 8 GB Mac because the x architecture does not fit safely in MPS memory.
See `models/reports/teacher_benchmark_v002_val/`.

`teachers/yolo26x_v002_hf/highres_960/best.pt` continues that checkpoint at
960 px for 12 epochs. Evaluated at its training resolution, it reaches 75.2%
mAP50, 36.8% mAP50-95 and 76.3% fixed-threshold F1. Training curves, confusion
matrices, examples and resolution comparisons are deliberately kept outside
the checkpoint directory in `models/reports/teacher_highres_960/`.
