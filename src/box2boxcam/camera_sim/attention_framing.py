"""Frame ahead of a persistent body-direction consensus from visible players."""

import cv2
import numpy as np


def dominant_heading(
    row,
    center,
    view,
    size,
    *,
    minimum_dominance=0.75,
    minimum_bodies=3,
    visible_fraction=0.25,
):
    selected = set(row["unidirectional_owners"])
    visible_owners, votes = set(), {}
    for owner, ray in zip(row["owners"], row["rays"]):
        ray = np.asarray(ray, dtype=float)
        if (
            not np.isfinite(ray).all()
            or (np.abs(ray[:2] / size - center) > view / 2).any()
        ):
            continue
        visible_owners.add(owner)
        if (
            owner in selected
            and ray[4] > 0
            and (owner not in votes or ray[4] > votes[owner][4])
        ):
            votes[owner] = ray
    required = max(minimum_bodies, int(np.ceil(len(visible_owners) * visible_fraction)))
    if len(votes) < required:
        return None
    rays = np.asarray(list(votes.values()))
    directions = rays[:, 2:4]
    norms = np.linalg.norm(directions, axis=1)
    if (norms < 1e-6).any():
        return None
    directions = directions / norms[:, None]
    weights = rays[:, 4]
    agreement = directions @ directions.T >= np.cos(np.deg2rad(35))
    support_weights = agreement @ weights
    support = agreement[int(support_weights.argmax())]
    dominance = float(weights[support].sum() / weights.sum())
    direction = np.average(directions[support], axis=0, weights=weights[support])
    coherence = float(np.linalg.norm(direction))
    if support.sum() < required or dominance < minimum_dominance or coherence < 0.85:
        return None
    return {
        "direction": direction / coherence,
        "support": int(support.sum()),
        "dominance": dominance,
        "coherence": coherence,
    }


class AttentionFraming:
    def __init__(self, size, view, field):
        self.size = np.asarray(size)
        self.view = np.asarray(view)
        self.field = np.asarray(field, dtype=np.float32)
        self.pending_direction = None
        self.pending_count = 0
        self.direction = None
        self.last_support = -np.inf
        self.target = None
        self.last_time = None
        self.lead_widths = 0.0

    def anchor(self, point):
        """Transfer the shared reference while another source owns the camera.

        Body-direction evidence keeps its own freshness; this cannot invent
        support. It only prevents an inactive planner retaining a distant target.
        """
        self.target = self._on_field(np.asarray(point, dtype=float)).copy()

    def _on_field(self, point):
        if cv2.pointPolygonTest(self.field, tuple(point), False) >= 0:
            return point
        a = self.field / self.view
        b = np.roll(a, -1, axis=0)
        edges = b - a
        position = point / self.view
        along = np.clip(
            ((position - a) * edges).sum(1) / np.maximum(np.square(edges).sum(1), 1e-9),
            0,
            1,
        )
        nearest = a + along[:, None] * edges
        chosen = (
            nearest[np.linalg.norm(nearest - position, axis=1).argmin()] * self.view
        )
        return 0.98 * chosen + 0.02 * self.field.mean(0)

    def _approach(self, point, dt, rate, respect_direction=False):
        point = self._on_field(point)
        if self.target is None:
            self.target = point.copy()
            return self.target.copy()
        change = point - self.target
        if respect_direction:
            pixels = change * self.size
            if pixels @ self.direction < 0:
                rate = min(rate, 0.15)
        if np.linalg.norm(change / self.view) >= 0.03:
            change = np.clip(change, -rate * self.view * dt, rate * self.view * dt)
            self.target = self._on_field(self.target + change)
        return self.target.copy()

    def step(self, row, players, center, timestamp):
        dt = (
            0.1
            if self.last_time is None
            else np.clip(timestamp - self.last_time, 0, 0.2)
        )
        self.last_time = timestamp
        if timestamp - self.last_support > 4:
            self.direction = None
        heading = dominant_heading(row, center, self.view, self.size)
        observed_people = 0 if players is None else len(players.get("feet", []))
        if heading is not None and heading["support"] < max(
            3, int(np.ceil(observed_people * 0.35))
        ):
            heading = None
        continuation = False
        if self.direction is not None:
            retained = dominant_heading(
                row,
                center,
                self.view,
                self.size,
                minimum_dominance=0.55,
                minimum_bodies=2,
                visible_fraction=0.25,
            )
            if (
                retained is not None
                and retained["support"] >= max(2, int(np.ceil(observed_people * 0.25)))
                and retained["direction"] @ self.direction >= np.cos(np.deg2rad(35))
            ):
                heading, continuation = retained, True
        base = None if players is None else players["motion"]
        support = 0 if heading is None else heading["support"]
        if heading is None:
            self.pending_count = 0
        else:
            direction = heading["direction"]
            aligned = (
                self.pending_direction is not None
                and self.pending_direction @ direction >= np.cos(np.deg2rad(30))
            )
            self.pending_count = self.pending_count + 1 if aligned else 1
            self.pending_direction = direction
            opposite = self.direction is not None and self.direction @ direction < 0
            required = 7 if opposite else 3
            if (continuation or self.pending_count >= required) and base is not None:
                respect_direction = self.direction is not None and not opposite
                new_intent = self.direction is None or opposite
                if new_intent:
                    self.direction = direction
                else:
                    self.direction = 0.8 * self.direction + 0.2 * direction
                    self.direction /= np.linalg.norm(self.direction)
                agreement = np.clip((heading["dominance"] - 0.75) / 0.25, 0, 1)
                represented = heading["support"] / max(
                    heading["support"], observed_people
                )
                coverage = np.clip((represented - 0.20) / 0.40, 0, 1)
                lead_widths = 0.15 + 0.27 * agreement * coverage
                self.lead_widths = (
                    lead_widths if new_intent else max(self.lead_widths, lead_widths)
                )
                displacement = (
                    self.direction
                    * (self.lead_widths * self.view[0] * self.size[0])
                    / self.size
                )
                displacement[1] = np.clip(
                    displacement[1], -0.25 * self.view[1], 0.25 * self.view[1]
                )
                target = self._approach(
                    base + displacement, dt, 0.45, respect_direction
                )
                self.last_support = timestamp
                return target, "shared orientation", support
        # Keep the last world target briefly, without adding unseen people to a vote.
        age = timestamp - self.last_support
        if self.target is not None and age <= 1.2:
            return self.target.copy(), "remembered direction", support
        enough_players = players is not None and len(players.get("feet", [])) >= 3
        if base is not None and (self.direction is None or enough_players or age > 3):
            return self._approach(base, dt, 0.15), "gradual recentring", support
        if self.target is not None and age <= 3:
            return self.target.copy(), "observing the area", support
        self.direction, self.target = None, None
        self.lead_widths = 0.0
        return None, "search", support
