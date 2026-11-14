"""Causal digital zoom from current-crop evidence, with slow tightening and fast widening."""

import numpy as np


class ZoomPolicy:
    def __init__(self, base_view, maximum=1.0, mode="fixed"):
        if (
            mode not in ("fixed", "dynamic")
            or not np.isfinite(maximum)
            or not 1 <= maximum <= 3
        ):
            raise ValueError("Zoom must be fixed/dynamic and between 1 and 3")
        self.base_view = np.asarray(base_view, dtype=float)
        self.maximum, self.mode = maximum, mode
        self.zoom = maximum if mode == "fixed" else 1.0
        self.previous_time = None
        self.stable_since = None
        self.last_ball = -np.inf
        self.reason = "fixed zoom" if mode == "fixed" else "wide search"

    @property
    def view(self):
        return self.base_view / self.zoom

    def step(self, timestamp, center, ball, players):
        dt = 0.1 if self.previous_time is None else timestamp - self.previous_time
        if not np.isfinite(timestamp) or not 0 < dt <= 1:
            raise ValueError("Zoom timestamps must increase, without large gaps")
        self.previous_time = timestamp
        if self.mode == "fixed":
            return self.zoom
        target, reason = 1.0, "wide search"
        visible_ball = (
            ball.measured
            and ball.confirmed
            and ball.reliability >= 0.7
            and ball.point is not None
        )
        if visible_ball:
            self.last_ball = timestamp
            extent = 2 * np.abs(ball.point - center) + self.view * 0.45
            target = min(self.maximum, float(np.min(self.base_view / extent)))
            reason = "continuous ball: tighten"
        elif timestamp - self.last_ball <= 0.35:
            target, reason = self.zoom, "brief loss: hold zoom"
        elif timestamp - self.last_ball < 1.8:
            reason = "ball lost: widen"
        elif players is not None and len(players["feet"]) >= 3:
            # Only people observed in this branch's current viewport can vote.
            feet = np.asarray(players["feet"])
            feet = feet[(np.abs(feet - center) <= self.view / 2 + 1e-8).all(1)]
            if len(feet) >= 3:
                focus = np.asarray(players["motion"])
                order = np.argsort(np.linalg.norm((feet - focus) / self.view, axis=1))
                group = feet[order[: max(3, int(np.ceil(len(feet) * 0.6)))]]
                # Fit a majority of the visible action, plus headroom and lead.
                # Fit around the observed action, where the pan/tilt planner is
                # heading. Fitting around the old wide-frame center deadlocks
                # zoom and tilt when the players occupy its upper portion.
                extent = 2 * np.max(np.abs(group - focus), axis=0) + self.view * [
                    0.20,
                    0.35,
                ]
                target = min(
                    self.maximum,
                    float(np.min(self.base_view / np.maximum(extent, 0.001))),
                )
                reason = "group visible: cautious framing"
        target = float(np.clip(target, 1, self.maximum))
        if target > self.zoom * 1.08:
            if self.stable_since is None:
                self.stable_since = timestamp
            if timestamp - self.stable_since < 0.8:
                target = self.zoom
                reason = "wait for stable framing"
        else:
            self.stable_since = None
        rate = 0.12 if target > self.zoom else 0.4
        log_step = float(np.clip(np.log(target / self.zoom), -rate * dt, rate * dt))
        self.zoom = float(np.clip(self.zoom * np.exp(log_step), 1, self.maximum))
        self.reason = reason
        return self.zoom
