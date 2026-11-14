"""Player tracks, velocities and the weighted group target."""

from collections import deque

import numpy as np


class PlayerTargets:
    def __init__(self, size, view):
        self.size = np.asarray(size)
        self.view = view
        self.previous = np.empty((0, 5))
        self.histories = []
        self.track_ids = []
        self.next_track_id = 0

    def step(self, boxes, timestamp, center=None):
        if center is not None and len(boxes):
            feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
            inside = (np.abs(feet - center) <= self.view / 2).all(1)
            boxes = boxes[inside]
        feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
        histories = [deque([(timestamp, point)], maxlen=7) for point in feet]
        track_ids = list(range(self.next_track_id, self.next_track_id + len(feet)))
        self.next_track_id += len(feet)
        if len(boxes) and len(self.previous):
            old_feet = np.column_stack(
                ((self.previous[:, 0] + self.previous[:, 2]) / 2, self.previous[:, 3])
            )
            delta = (feet[:, None] - old_feet[None]) * self.size
            heights = np.maximum(8, (boxes[:, 3] - boxes[:, 1]) * self.size[1])
            old_heights = np.maximum(
                8, (self.previous[:, 3] - self.previous[:, 1]) * self.size[1]
            )
            distance = np.linalg.norm(delta, axis=-1) / (
                (heights[:, None] + old_heights[None]) / 2
            )
            cost = distance + 0.4 * np.abs(np.log(heights[:, None] / old_heights[None]))
            used_current, used_previous = set(), set()
            for flat_index in np.argsort(cost, axis=None):
                current, previous = np.unravel_index(flat_index, cost.shape)
                if cost[current, previous] > 1.8:
                    break
                if current in used_current or previous in used_previous:
                    continue
                used_current.add(current)
                used_previous.add(previous)
                histories[current] = self.histories[previous]
                histories[current].append((timestamp, feet[current].copy()))
                track_ids[current] = self.track_ids[previous]
        velocities = np.zeros_like(feet)
        for index, history in enumerate(histories):
            if len(history) >= 3:
                times = np.array([time for time, _ in history])
                positions = np.array([point for _, point in history])
                centered = times - times.mean()
                velocities[index] = (
                    centered[:, None] * (positions - positions.mean(0))
                ).sum(0) / (centered @ centered)
        self.previous, self.histories, self.track_ids = boxes, histories, track_ids
        if not len(feet):
            return None
        heights = np.maximum(8, (boxes[:, 3] - boxes[:, 1]) * self.size[1])
        relative_speed = np.linalg.norm(velocities * self.size, axis=1) / heights
        movement = 1 + 3 * np.clip((relative_speed - 0.35) / 2.0, 0, 1)
        separation = (feet[:, None] - feet[None]) / self.view
        neighbors = np.exp(-0.5 * np.square(separation / 0.25).sum(-1)).sum(1)
        density = np.sqrt(neighbors / max(1, float(np.median(neighbors))))
        weights = movement * density
        weighted = np.average(feet, axis=0, weights=weights)
        lead = np.clip(
            np.average(velocities, axis=0, weights=weights) * 0.35,
            -0.12 * self.view,
            0.12 * self.view,
        )
        return {
            "mean": feet.mean(0),
            "motion": weighted + lead,
            "feet": feet,
            "velocity": velocities,
            "weight": weights,
            "track_ids": track_ids,
        }
