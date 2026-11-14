"""Causal ball continuity and a shared target for ball/player camera guidance.

Coordinates are in one motion-compensated image reference. In replay that is
the fixed source image; live use must supply an equivalent camera transform.
Detector scores are an eligibility floor, never the continuity weight.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class BallTrack:
    point: np.ndarray | None
    velocity: np.ndarray
    reliability: float
    coverage: float
    age: float
    measured: bool
    confirmed: bool
    exit_direction: np.ndarray | None


class ContinuousBallTrack:
    WINDOW = 0.8
    EXPIRE = 0.65

    def __init__(self, view, field=None):
        self.view = np.asarray(view, dtype=float)
        self.field = None if field is None else np.asarray(field, dtype=float)
        self.point = None
        self.velocity = np.zeros(2)
        self.last_seen = self.started = None
        self.hits = 0
        self.history = deque()
        self.error = 0.0
        self.previous_time = self.previous_center = None
        self.exit_direction = None
        self.last_reliability = 0.0
        self.last_local = None
        self.last_relative_velocity = np.zeros(2)

    def field_mask(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        allowed = np.isfinite(points).all(1)
        if self.field is None or not len(points):
            return allowed
        # Image-plane test only: an airborne ball may be ABOVE the projected
        # pitch. Reject below the near touchline, not above the far touchline.
        a, b = self.field, np.roll(self.field, -1, axis=0)
        for index, (x, y) in enumerate(points):
            if not allowed[index]:
                continue
            crossing = ((a[:, 0] <= x) & (b[:, 0] >= x)) | (
                (b[:, 0] <= x) & (a[:, 0] >= x)
            )
            crossing &= np.abs(b[:, 0] - a[:, 0]) > 1e-8
            if not crossing.any():
                allowed[index] = False
                continue
            along = (x - a[crossing, 0]) / (b[crossing, 0] - a[crossing, 0])
            near_edge = np.max(
                a[crossing, 1] + along * (b[crossing, 1] - a[crossing, 1])
            )
            allowed[index] = y <= near_edge + 0.06 * self.view[1]
        return allowed

    def step(self, proposals, center, timestamp, hint=None):
        if not np.isfinite(timestamp) or (
            self.previous_time is not None and timestamp <= self.previous_time
        ):
            raise ValueError(
                "Observations must have finite, strictly increasing capture timestamps"
            )
        center = np.asarray(center, dtype=float)
        dt = 0.1 if self.previous_time is None else timestamp - self.previous_time
        camera_velocity = (
            np.zeros(2)
            if self.previous_center is None
            else (center - self.previous_center) / dt
        )
        self.previous_time, self.previous_center = timestamp, center.copy()
        proposals = np.asarray(proposals, dtype=float).reshape(-1, 5)
        valid = (
            np.isfinite(proposals).all(1)
            & (proposals[:, 4] >= 0.10)
            & (proposals[:, 4] <= 1)
            & (proposals[:, 2:4] > 0).all(1)
            & (proposals[:, 2:4] <= self.view).all(1)
            & (np.abs(proposals[:, :2] - center) <= self.view / 2 + 1e-8).all(1)
        )
        # Apply the same calibrated geometry before EVERY association. A nearby
        # outside false positive must not inherit an already reliable track.
        proposals = proposals[valid & self.field_mask(proposals[:, :2])]
        age = np.inf if self.last_seen is None else timestamp - self.last_seen
        if age > self.EXPIRE:
            self.point = None
            self.velocity = np.zeros(2)
            self.history.clear()
            self.hits = 0
            self.started = None
            self.error = 0.0

        chosen = None
        if self.point is not None and len(proposals):
            predicted = self.point + self.velocity * age
            distance = np.linalg.norm(
                (proposals[:, :2] - predicted) / self.view, axis=1
            )
            gate = min(0.23, 0.055 + 0.65 * age)
            nearest = int(distance.argmin())
            if distance[nearest] <= gate:
                chosen = proposals[nearest]
        elif len(proposals):
            # On acquisition use action context as a weak spatial prior. Once a
            # track exists, temporal association alone chooses between candidates.
            prior = center if hint is None else np.asarray(hint)
            distance = np.linalg.norm((proposals[:, :2] - prior) / self.view, axis=1)
            chosen = proposals[np.argmin(distance - 0.08 * proposals[:, 4])]

        measured = chosen is not None
        if measured:
            if self.point is None:
                self.point = chosen[:2].copy()
                self.started = timestamp
                self.exit_direction = None
            else:
                predicted = self.point + self.velocity * age
                residual = chosen[:2] - predicted
                self.error = 0.7 * self.error + 0.3 * np.linalg.norm(
                    residual / self.view
                )
                self.point = predicted + 0.78 * residual
                beta = 1 - np.exp(-age / 0.22)
                self.velocity += beta * residual / age
                self.velocity = np.clip(self.velocity / self.view, -2, 2) * self.view
            self.last_seen = timestamp
            self.hits += 1
            age = 0.0
            self.last_local = (self.point - center) / self.view
            self.last_relative_velocity = (self.velocity - camera_velocity) / self.view
        self.history.append((timestamp, measured))
        while self.history and self.history[0][0] < timestamp - self.WINDOW + 1e-8:
            self.history.popleft()
        coverage = sum(hit for _, hit in self.history) / max(1, len(self.history))
        span = 0 if self.started is None else self.last_seen - self.started
        confirmed = self.hits >= 3 and span >= 0.18
        maturity = min(1.0, span / 0.5) if confirmed else 0.0
        precision = np.clip(1 - self.error / 0.16, 0.35, 1)
        freshness = max(0, 1 - age / 0.5)
        reliability = float(coverage * maturity * precision * freshness)
        if measured:
            self.last_reliability = reliability
        elif (
            self.last_reliability >= 0.55
            and self.last_local is not None
            and self.exit_direction is None
            and age <= 0.35
        ):
            outward = (np.abs(self.last_local) >= 0.37) & (
                self.last_local * self.last_relative_velocity > 0.03
            )
            if outward.any():
                direction = np.where(outward, np.sign(self.last_local), 0.0)
                self.exit_direction = direction / np.linalg.norm(direction)
        target = None
        if self.point is not None and confirmed and age <= 0.5:
            # Bound extrapolation even when a small ball moves exceptionally fast.
            displacement = self.velocity * min(age + 0.10, 0.25)
            length = np.linalg.norm(displacement / self.view)
            if length > 0.15:
                displacement *= 0.15 / length
            target = self.point + displacement
            if not self.field_mask([target])[0]:
                target = None
        return BallTrack(
            target,
            self.velocity.copy(),
            reliability,
            float(coverage),
            float(age),
            measured,
            confirmed,
            None if self.exit_direction is None else self.exit_direction.copy(),
        )


class HybridFraming:
    def __init__(
        self, view, *, acquire=0.70, retain=0.25, transition_rate=0.85, exit_hold=1.2
    ):
        self.view = np.asarray(view, dtype=float)
        self.acquire, self.retain = acquire, retain
        self.transition_rate, self.exit_hold = transition_rate, exit_hold
        self.ball_active = False
        self.ball_weight = 0.0
        self.target = None
        self.previous_time = None
        self.exit_direction = None
        self.exit_until = -np.inf
        self.exit_target = None

    def step(self, ball, player_target, player_mode, center, timestamp, heading=None):
        dt = (
            0.1
            if self.previous_time is None
            else min(0.2, timestamp - self.previous_time)
        )
        if dt <= 0:
            raise ValueError("Capture timestamps must increase")
        self.previous_time = timestamp
        center = np.asarray(center)
        if (
            ball.point is not None
            and ball.measured
            and ball.reliability >= self.acquire
        ):
            self.ball_active = True
            self.exit_until = -np.inf
            self.exit_direction = None
        elif ball.reliability < self.retain or ball.point is None:
            self.ball_active = False

        if (
            ball.exit_direction is not None
            and self.exit_direction is None
            and ball.age <= 0.35
        ):
            self.exit_direction = ball.exit_direction.copy()
            self.exit_until = timestamp + self.exit_hold
            origin = self.target if self.target is not None else center
            self.exit_target = origin + self.exit_direction * self.view * 0.38

        # Hysteresis selects the authority. Do not turn every small reliability
        # fluctuation into a different point between the ball and the players.
        desired_weight = 1.0 if self.ball_active else 0.0
        if ball.point is None:
            self.ball_weight = 0.0
        else:
            rate = 2.2 if desired_weight > self.ball_weight else 3.0
            self.ball_weight += np.clip(
                desired_weight - self.ball_weight, -rate * dt, rate * dt
            )

        fallback = player_target
        mode = "players: " + player_mode
        if self.exit_direction is not None and timestamp < self.exit_until:
            # A persistent opposite body consensus may contradict the exit
            # hypothesis. An unconfirmed group centroid may not erase it.
            contradiction = (
                heading is not None
                and player_mode == "shared orientation"
                and heading @ self.exit_direction < -0.5
            )
            if contradiction:
                self.exit_until = timestamp
            else:
                fallback = self.exit_target.copy()
                if player_target is not None:
                    projection = (
                        (player_target - fallback) / self.view
                    ) @ self.exit_direction
                    if projection > 0:
                        fallback = player_target
                mode = "search exit side"

        desired = fallback
        if ball.point is not None and self.ball_weight > 0:
            base = center if fallback is None else fallback
            desired = (1 - self.ball_weight) * base + self.ball_weight * ball.point
            mode = (
                "continuous ball"
                if self.ball_weight >= 0.85
                else "ball/player transition"
            )
        if desired is None:
            return None, "search without cue"
        desired = np.clip(desired, self.view / 2, 1 - self.view / 2)
        if self.target is None:
            self.target = center.copy()
        # One shared, rate-limited reference; the motor controller keeps its
        # existing velocity and acceleration through every source transition.
        delta = desired - self.target
        self.target += np.clip(
            delta,
            -self.transition_rate * dt * self.view,
            self.transition_rate * dt * self.view,
        )
        return self.target.copy(), mode
