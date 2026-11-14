"""Frame-level matching and counting metrics.

A detection is a true positive when it overlaps a not-yet-matched ground-truth box
with IoU >= ``iou_threshold`` (default 0.5); detections are consumed by decreasing
confidence, so the matching does not depend on the confidence threshold. Everything
else is a false positive. A ground-truth ball counts as *found* at threshold ``t``
when one of its matched detections has confidence >= ``t``.

``matches[i]`` is the list ``[(confidence, gt_index or -1), ...]`` of frame ``i``.
"""

import numpy as np

IOU_THRESHOLD = 0.5
SMALL_MAX_SIDE = 16.0


def iou(a, b):
    """IoU of two ``xyxy`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def match_frame(gts, boxes, confidences, iou_threshold=IOU_THRESHOLD):
    """Greedy matching of one frame by decreasing confidence."""
    used = set()
    out = []
    for k in np.argsort(-np.asarray(confidences, dtype=float)):
        best, best_j = iou_threshold, -1
        for j, gt in enumerate(gts):
            if j in used:
                continue
            value = iou(boxes[k], gt)
            if value >= best:
                best, best_j = value, j
        if best_j >= 0:
            used.add(best_j)
        out.append((float(confidences[k]), best_j))
    return out


def match_all(gts, detections, iou_threshold=IOU_THRESHOLD):
    """``gts``: list of ``(n, 4)`` arrays; ``detections``: list of ``(boxes, confidences)``."""
    if len(gts) != len(detections):
        raise ValueError("one detection entry is required per frame")
    return [match_frame(g, b, c, iou_threshold) for g, (b, c) in zip(gts, detections)]


def is_small(box, max_side=SMALL_MAX_SIDE):
    """Small ball: the longest side is at most ``max_side`` pixels (full-resolution image)."""
    return max(box[2] - box[0], box[3] - box[1]) <= max_side


def small_balls(gts, max_side=SMALL_MAX_SIDE):
    """Set of ``(frame, gt_index)`` for small ground-truth balls."""
    return {
        (i, j)
        for i, g in enumerate(gts)
        for j, box in enumerate(g)
        if is_small(box, max_side)
    }


def all_balls(gts):
    return {(i, j) for i, g in enumerate(gts) for j in range(len(g))}


def count_tp_fp(matches, threshold, frames=None):
    """``(tp, fp, found)`` at a confidence threshold.

    ``frames`` may repeat indices (bootstrap resampling); ``found`` is the set of
    ``(frame, gt_index)`` pairs found at least once.
    """
    tp = fp = 0
    found = set()
    for i in frames if frames is not None else range(len(matches)):
        for confidence, j in matches[i]:
            if confidence < threshold:
                continue
            if j >= 0:
                tp += 1
                found.add((i, j))
            else:
                fp += 1
    return tp, fp, found


def count_small_tp(matches, small, threshold, frames=None):
    """True positives on small balls (counted with multiplicity over ``frames``)."""
    return sum(
        1
        for i in (frames if frames is not None else range(len(matches)))
        for c, j in matches[i]
        if c >= threshold and j >= 0 and (i, j) in small
    )


def max_tp_under_fp_budget(matches, budget, frames=None):
    """Most true positives reachable when sweeping the threshold down while
    the number of false positives stays <= ``budget``."""
    events = sorted(
        (
            (confidence, j >= 0)
            for i in (frames if frames is not None else range(len(matches)))
            for confidence, j in matches[i]
        ),
        key=lambda event: -event[0],
    )
    tp = fp = best = 0
    for _, is_tp in events:
        if is_tp:
            tp += 1
        else:
            fp += 1
        if fp <= budget:
            best = max(best, tp)
    return best


def recovery(matches_a, matches_b, gts, threshold, small):
    """Balls missed by B at ``threshold`` that A finds, and B's hits that A loses."""
    n = len(gts)
    everything = all_balls(gts)
    frames = range(n)
    found_a = count_tp_fp(matches_a, threshold, frames)[2]
    found_b = count_tp_fp(matches_b, threshold, frames)[2]
    missed_b = everything - found_b
    return {
        "b_misses": len(missed_b),
        "a_recovers": len(missed_b & found_a),
        "b_misses_small": len(missed_b & small),
        "a_recovers_small": len(missed_b & found_a & small),
        "a_loses": len(found_b - found_a),
    }
