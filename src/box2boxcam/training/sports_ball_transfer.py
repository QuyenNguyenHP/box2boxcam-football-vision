"""Keep the COCO sports-ball classifier, not just its name or output bias.

YOLO26n normally shrinks classifier features from 80 to 64 channels for nc=1.
Copying the original graph and pruning ONLY final class-output rows avoids
that information loss. The returned graph uses standard Ultralytics classes.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel


def retain_sports_ball(source: DetectionModel):
    """Return an independent one-class model with the original learned head."""
    if not isinstance(source, DetectionModel) or type(source.model[-1]) is not Detect:
        raise ValueError("Expected a standard Ultralytics detection model")
    candidates = [int(k) for k, name in source.names.items()
                  if str(name).strip().lower() == "sports ball"]
    if len(candidates) != 1:
        raise ValueError("Source must contain exactly one named sports ball class")
    class_id = candidates[0]
    source_head = source.model[-1]
    if not 0 <= class_id < source_head.nc:
        raise ValueError("Source class ID is outside the classifier")
    if source_head.cv3 is None or getattr(source_head, "one2one_cv3", None) is None:
        raise ValueError("Both unfused YOLO26 classification branches are required")
    if source_head.nc != len(source.names):
        raise ValueError("Source metadata and classifier disagree")
    model = deepcopy(source).float()
    head = model.model[-1]
    head_prefix = next(name for name, module in source.named_modules() if module is source_head)
    mapped_keys, widths = [], []
    for branch in ("cv3", "one2one_cv3"):
        sequences = getattr(head, branch)
        if len(sequences) != source_head.nl:
            raise ValueError("Incomplete multi-scale classifier")
        for scale, sequence in enumerate(sequences):
            output = sequence[-1]
            if (not isinstance(output, nn.Conv2d) or output.out_channels != source_head.nc
                    or output.bias is None or output.kernel_size != (1, 1) or output.groups != 1):
                raise ValueError("Unexpected class-output convolution")
            widths.append(output.in_channels)
            # Keep the learned features intact; only remove unrelated output rows.
            output.weight = nn.Parameter(output.weight[class_id:class_id + 1].detach().clone())
            output.bias = nn.Parameter(output.bias[class_id:class_id + 1].detach().clone())
            output.out_channels = 1
            prefix = f"{head_prefix}.{branch}.{scale}.{len(sequence) - 1}"
            mapped_keys.extend((prefix + ".weight", prefix + ".bias"))
    head.nc = model.nc = 1
    head.no = 4 * head.reg_max + 1
    head.shape = None
    model.names = {0: "sports ball"}
    model.yaml["nc"] = 1
    model.criterion = None
    if hasattr(model, "class_weights"):
        model.class_weights = None
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    before, after = source.state_dict(), model.state_dict()
    if before.keys() != after.keys():
        raise ValueError("Transfer changed unrelated module structure")
    for key, value in after.items():
        expected = before[key][class_id:class_id + 1] if key in mapped_keys else before[key]
        if not torch.equal(value, expected.to(value.dtype)):
            raise ValueError(f"Pretrained weight was not preserved: {key}")
    report = {
        "method": "retain_full_classifier_features_and_named_output_row",
        "source_class_id": class_id, "source_class_name": "sports ball",
        "target_class_id": 0, "target_class_name": "sports ball",
        "source_classes": source_head.nc, "target_classes": 1,
        "classification_hidden_widths": widths,
        "mapped_output_tensors": mapped_keys,
        "other_tensors_preserved": len(after) - len(mapped_keys),
        "all_weight_checks_passed": True,
        "source_parameters": sum(p.numel() for p in source.parameters()),
        "target_parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, report


def verify_dense_predictions(source, target, sample):
    """Compare both raw branches BEFORE class-dependent top-k selection.

Dropping other classes changes top-k competition, not raw sports-ball logits
or box predictions. Float convolution reduction may introduce tiny rounding.
"""
    class_id = next(int(k) for k, name in source.names.items()
                    if str(name).strip().lower() == "sports ball")
    source_training, target_training = source.training, target.training
    errors = {}
    try:
        source.eval()
        target.eval()
        with torch.no_grad():
            source_raw = source(sample)[1]
            target_raw = target(sample)[1]
            for branch in ("one2many", "one2one"):
                before, after = source_raw[branch], target_raw[branch]
                source_scores = before["scores"][:, class_id:class_id + 1]
                torch.testing.assert_close(after["scores"], source_scores, atol=1e-4, rtol=1e-5)
                torch.testing.assert_close(after["boxes"], before["boxes"], atol=0, rtol=0)
                source_decoded = source.model[-1]._get_decode_boxes(before)
                target_decoded = target.model[-1]._get_decode_boxes(after)
                torch.testing.assert_close(target_decoded, source_decoded, atol=0, rtol=0)
                errors[branch] = {
                    "class_logit_max_error": float((after["scores"] - source_scores).abs().max()),
                    "raw_box_max_error": float((after["boxes"] - before["boxes"]).abs().max()),
                    "decoded_box_max_error": float((target_decoded - source_decoded).abs().max()),
                }
    finally:
        source.train(source_training)
        target.train(target_training)
    return {"passed": True, "sample_shape": list(sample.shape), "branches": errors,
            "scope": "dense predictions, before top-k; not a guarantee after fine-tuning"}


class SportsBallTransferTrainer(DetectionTrainer):
    """Use the retained graph instead of rebuilding a narrower classifier."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None or self.data["nc"] != 1 or self.data["names"] != {0: "sports ball"}:
            raise ValueError("Sports-ball transfer requires pretrained weights and one named class")
        model, report = retain_sports_ball(weights)
        # A local generator leaves the training RNG/seed untouched.
        generator = torch.Generator().manual_seed(42)
        sample = torch.rand(1, 3, 320, 320, generator=generator)
        sample = sample.to(device=next(weights.parameters()).device,
                           dtype=next(weights.parameters()).dtype)
        report["dense_prediction_check"] = verify_dense_predictions(weights, model, sample)
        self.sports_ball_transfer_report = report
        return model
