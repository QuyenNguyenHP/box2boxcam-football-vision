"""Causal camera control for replay, in viewport widths and heights per second."""

import numpy as np


class SmoothController:
    DT = 0.1
    ACQUIRE = 0.62
    RETAIN = 0.53
    MAX_SPEED = np.array([0.55, 0.44])
    MAX_ACCELERATION = np.array([1.25, 1.05])
    MAX_JERK = np.array([4.8, 4.0])
    DEADBAND = np.array([0.04, 0.05])
    LEAD = 0.28

    def __init__(self, view, start, search=True):
        self.view = np.asarray(view, dtype=float)
        self.lower, self.upper = self.view / 2, 1 - self.view / 2
        if np.ndim(start) == 0:
            start = [start, 0.5]
        self.center = np.clip(start, self.lower, self.upper)
        self.search = search
        self.state = "holding"
        self._frame = -1
        self._last_seen = None
        self._point = self._raw_point = None
        self._track_velocity = np.zeros(2)
        self._pending = None
        self._pending_count = 0
        self._speed = np.zeros(2)
        self._acceleration = np.zeros(2)
        self._search_bounds = None
        self._search_side = 0

    def resize(self, view):
        """Change FOV without resetting physical positions or controller memory."""
        view = np.asarray(view, dtype=float)
        if (
            view.shape != (2,)
            or not np.isfinite(view).all()
            or (view <= 0).any()
            or (view > 1).any()
        ):
            raise ValueError("Invalid viewport")
        ratio = self.view / view
        for name in ("_point", "_raw_point", "_pending"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value * ratio)
        self._track_velocity *= ratio
        self._speed = np.clip(self._speed * ratio, -self.MAX_SPEED, self.MAX_SPEED)
        self._acceleration = np.clip(
            self._acceleration * ratio, -self.MAX_ACCELERATION, self.MAX_ACCELERATION
        )
        if self._search_bounds is not None:
            self._search_bounds *= ratio[0]
        self.view = view.copy()
        self.lower, self.upper = view / 2, 1 - view / 2
        self.center = np.clip(self.center, self.lower, self.upper)

    def _accept(self, point, probability, age):
        if point is not None and probability >= self.RETAIN and age <= 0.4:
            expected = self._point + age * self._track_velocity
            residual = point - expected
            if (np.abs(residual) <= [0.24, 0.21]).all():
                observed_speed = np.clip((point - self._raw_point) / age, -0.9, 0.9)
                self._track_velocity = (
                    0.72 * self._track_velocity + 0.28 * observed_speed
                )
                # Larger coherent movements get more weight than local detector jitter.
                gain = np.clip(0.32 + 2.4 * np.abs(residual), 0.32, 0.66)
                self._point = expected + gain * residual
                return True

        if point is None or probability < self.ACQUIRE:
            self._pending_count = 0
            return False
        consistent = (
            self._pending_count
            and (np.abs(point - self._pending) <= [0.13, 0.13]).all()
        )
        self._pending_count = self._pending_count + 1 if consistent else 1
        previous, self._pending = self._pending, point
        if self._pending_count < (3 if age <= 0.4 else 2):
            return False

        self._point = 0.7 * point + 0.3 * previous
        displacement = point - previous
        motion = np.sign(displacement) * np.maximum(
            np.abs(displacement) - self.DEADBAND, 0
        )
        self._track_velocity = 0.5 * np.clip(motion / self.DT, -0.9, 0.9)
        return True

    def _sweep(self, center, age):
        limits = np.array([0.5, 1 / self.view[0] - 0.5])
        if self._search_bounds is None:
            origin = self._point[0] if self._point is not None else center[0]
            self._search_bounds = np.clip(origin + np.array([-1.45, 1.45]), *limits)
            direction = self._track_velocity[0]
            if abs(direction) < 0.08:
                direction = 0.5 - self.center[0]
            self._search_side = int(direction >= 0)
        if age >= 4.0:
            self._search_bounds = limits
        goal = self._search_bounds[self._search_side]
        if abs(goal - center[0]) < 0.08 and abs(self._speed[0]) < 0.16:
            self._search_side = 1 - self._search_side
            goal = self._search_bounds[self._search_side]
        return np.array([goal, center[1]])

    def step(self, seen, probability):
        """Update from one frame. `seen` and the returned center use world coordinates."""
        self._frame += 1
        age = (
            np.inf
            if self._last_seen is None
            else (self._frame - self._last_seen) * self.DT
        )
        center = self.center / self.view
        point = None
        if seen is not None and np.isfinite(seen).all() and np.isfinite(probability):
            point = np.asarray(seen) / self.view

        target = center
        speed_limit = self.MAX_SPEED
        deadband = self.DEADBAND
        if self._accept(point, probability, age):
            self._raw_point, self._last_seen = point, self._frame
            self._pending_count = 0
            self._search_bounds = None
            self.state = "tracking"
            target = self._point + self.LEAD * self._track_velocity
        elif age <= 0.4:
            self.state = "coasting"
            target = self._point + min(age + self.LEAD, 0.45) * self._track_velocity
        elif self.search and self._frame >= 5:
            self.state = "searching"
            target = self._sweep(center, age)
            speed_limit = np.array([0.42, 0.30])
            deadband = self.DEADBAND * 0.4
        else:
            self.state = "holding"

        error = target - center
        error = np.sign(error) * np.maximum(np.abs(error) - deadband, 0)
        # Apply the deadband around the action before projecting into the available view.
        feasible_error = (
            np.clip(center + error, self.lower / self.view, self.upper / self.view)
            - center
        )
        requested = np.clip(2.8 * feasible_error, -speed_limit, speed_limit)
        desired_acceleration = np.clip(
            (requested - self._speed) / 0.24,
            -self.MAX_ACCELERATION,
            self.MAX_ACCELERATION,
        )
        jerk_limit = self.MAX_JERK * self.DT
        acceleration = self._acceleration + np.clip(
            desired_acceleration - self._acceleration, -jerk_limit, jerk_limit
        )
        speed = np.clip(
            self._speed + acceleration * self.DT, -self.MAX_SPEED, self.MAX_SPEED
        )
        next_center = np.clip(
            self.center + speed * self.DT * self.view, self.lower, self.upper
        )
        actual_speed = (next_center - self.center) / (self.DT * self.view)
        self._acceleration = (actual_speed - self._speed) / self.DT
        # A viewport boundary is a hard stop. Discard outward momentum there.
        stopped = ~np.isclose(speed, actual_speed)
        self._speed = np.where(stopped, 0, actual_speed)
        self._acceleration[stopped] = 0
        self.center = next_center
        return self.center
