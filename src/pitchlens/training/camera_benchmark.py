"""Frame-level camera targeting: confidence gate, then exactly one target.

This measures the targeting decision, not box quality at COCO IoU thresholds
or temporal tracking. With multiple annotations, any annotated ball can match;
the dataset cannot identify which ball is currently in play.
"""

from collections import Counter

import numpy as np

from pitchlens.training.diagnostics import box_iou


def score_target(ground_truth, predictions, confidences, confidence=.25, matching_iou=.2):
    """Select before consulting truth; return one mutually exclusive outcome."""
    if not 0 <= confidence <= 1 or not 0 < matching_iou <= 1:
        raise ValueError("Confidence must be in [0,1], IoU in (0,1].")
    truth = np.asarray(ground_truth, dtype=float).reshape(-1, 4)
    boxes = np.asarray(predictions, dtype=float).reshape(-1, 4)
    scores = np.asarray(confidences, dtype=float).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("One confidence is required per prediction.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction confidences must be finite values in [0,1].")
    # Stable tie policy: first maximum. No ground-truth-assisted selection.
    index = int(np.argmax(scores)) if len(scores) else None
    if index is None or scores[index] < confidence:
        return {
            "outcome": "missed" if len(truth) else "correct_silence",
            "selected_index": None, "selected_confidence": None, "iou": 0.,
        }
    overlap = float(box_iou(truth, boxes[index:index + 1]).max()) if len(truth) else 0.
    outcome = (
        "correct_target" if overlap >= matching_iou else
        "wrong_target" if len(truth) else "false_alarm"
    )
    return {
        "outcome": outcome, "selected_index": index,
        "selected_confidence": float(scores[index]), "iou": overlap,
    }


def benchmark_targets(rows, model_key, confidence=.25, matching_iou=.2, cached_min_confidence=.25):
    """Summarize a cache of normalized boxes, one target decision per frame.

    Lowering confidence below the cache's generation threshold needs new
    predictions; missing boxes must never silently bias the benchmark.
    """
    if confidence < cached_min_confidence:
        raise ValueError("The cache is filtered at 0.25: re-run inference for a lower threshold.")
    counts = Counter()
    decisions = []
    for row in rows:
        decision = score_target(
            row["ground_truth"], row[model_key], row[model_key + "_confidence"],
            confidence=confidence, matching_iou=matching_iou,
        )
        counts.update(images=1, **{decision["outcome"]: 1})
        counts["positive_images" if row["ground_truth"] else "negative_images"] += 1
        decisions.append({"item_id": row["item_id"], "match_id": row["match_id"], **decision})
    tp = counts["correct_target"]
    fp = counts["wrong_target"] + counts["false_alarm"]
    fn = counts["wrong_target"] + counts["missed"]
    tn = counts["correct_silence"]
    metrics = dict(counts)
    metrics.update(
        confidence=confidence, matching_iou=matching_iou,
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=tp / (tp + fp) if tp + fp else 0.,
        recall=tp / (tp + fn) if tp + fn else 0.,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        correct_decision_rate=(tp + tn) / counts["images"] if counts["images"] else 0.,
        false_alarm_rate=counts["false_alarm"] / counts["negative_images"] if counts["negative_images"] else None,
    )
    return {"metrics": metrics, "decisions": decisions}
