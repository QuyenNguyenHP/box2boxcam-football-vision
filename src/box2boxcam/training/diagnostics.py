"""Small, explicit diagnostics used by the training notebook."""

from contextlib import contextmanager

import numpy as np


@contextmanager
def quiet_ultralytics_progress():
    """Hide per-batch progress bars while preserving summaries and log files.

    Ultralytics' lightweight progress bar writes carriage returns that some
    notebook frontends store as hundreds of separate output lines. The global
    flag is restored even when training or validation raises an exception.
    """

    import ultralytics.utils as ultralytics_utils

    previous = ultralytics_utils.VERBOSE
    ultralytics_utils.VERBOSE = False
    try:
        yield
    finally:
        ultralytics_utils.VERBOSE = previous


def box_iou(first, second):
    """Return the pairwise IoU matrix for two arrays of xyxy boxes."""

    first = np.asarray(first, dtype=float).reshape(-1, 4)
    second = np.asarray(second, dtype=float).reshape(-1, 4)
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=float)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    first_area = np.clip(first[:, 2:] - first[:, :2], 0, None).prod(axis=1)
    second_area = np.clip(second[:, 2:] - second[:, :2], 0, None).prod(axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def detection_counts(ground_truth, predictions, iou_threshold=0.5):
    """Greedily match boxes and return TP, FP and FN at one operating point."""

    ground_truth = np.asarray(ground_truth, dtype=float).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 4)
    overlaps = box_iou(ground_truth, predictions)
    candidates = []
    for truth_index, prediction_index in np.argwhere(overlaps >= iou_threshold):
        overlap = overlaps[truth_index, prediction_index]
        candidates.append((float(overlap), int(truth_index), int(prediction_index)))
    matched_truth, matched_predictions = set(), set()
    for _, truth_index, prediction_index in sorted(candidates, reverse=True):
        if truth_index not in matched_truth and prediction_index not in matched_predictions:
            matched_truth.add(truth_index)
            matched_predictions.add(prediction_index)
    true_positives = len(matched_truth)
    return {
        "tp": true_positives,
        "fp": len(predictions) - true_positives,
        "fn": len(ground_truth) - true_positives,
    }


def change_category(before, after):
    """Describe how one image changed at the fixed confidence/IoU thresholds."""

    before_errors = before["fp"] + before["fn"]
    after_errors = after["fp"] + after["fn"]
    if before_errors and not after_errors:
        return "corrected"
    if after_errors < before_errors:
        return "improved"
    if not before_errors and after_errors:
        return "regressed"
    if after_errors:
        return "still_wrong"
    return "correct_both"
