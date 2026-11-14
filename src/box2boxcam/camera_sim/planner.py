"""Player-guidance planner of the camera simulation.

``Planner()`` with default parameters reproduces ``AttentionFraming`` exactly;
``guard=True`` with the default guard parameters adds the group guard (stay with
the densest group of players when the visible count collapses). Every knob is a
plain number so a variant can be ported to another language as constants.
"""

from collections import deque

import numpy as np

from box2boxcam.camera_sim.attention_framing import AttentionFraming, dominant_heading

DEFAULTS = {
    # orientation consensus
    "min_support": 3,
    "support_frac": 0.35,
    "min_dominance": 0.75,
    "pending_required": 3,
    "opposite_required": 7,
    "lead_min": 0.15,
    "lead_gain": 0.27,
    "lead_scale": 1.0,
    "approach_rate": 0.45,
    "recenter_rate": 0.15,
    "memory_hold": 1.2,
    "observe_hold": 3.0,
    "direction_reset": 4.0,
    # orientation input filters (off by default)
    "min_torso_px": 0.0,
    "motion_mode": "none",
    "motion_min": 0.3,
    "use_orientation": True,
    # group guard (off by default)
    "guard": False,
    "g_window": 4.0,
    "g_full": 0.70,
    "g_zero": 0.40,
    "g_enter": 0.40,
    "g_exit": 0.60,
    "g_min_peak": 5,
    "g_return_rate": 0.30,
    "g_timeout": 6.0,
    "g_peak_strict": 6,
    # players-only baselines
    "base": "motion",  # 'motion' (PlayerTargets weighted centroid + 0.35 s velocity lead) or 'mean'
    "vel_lead": 0.0,  # extra lead along the visible group's mean velocity, in seconds
    "vel_lead_cap": 0.25,  # cap of that lead, in view widths
    "density_bw": 0.12,
    "speed_gain": 2.0,
    "few_hold": False,
    "dead_zone": 0.0,
    "veto_speed": 0.0,  # view widths/s; 0 = off
}


def filter_row(row, p):
    """Optional orientation-input fixes (all off by default):
    min_torso_px  : bodies whose shoulder-hip height is below this many pixels do not vote;
    motion_mode   : 'agree'   -> a body votes only if it has a travel cue and its ray points along travel (dot > 0);
                    'resolve' -> a body with a travel cue of strength >= motion_min keeps the torso axis but takes the
                                 front/back sense from travel (selected rays flipped, ambiguous pairs resolved)."""
    mode = p["motion_mode"]
    if p["min_torso_px"] <= 0 and mode == "none":
        return row
    poses = row["poses"]
    evid = {e["owner"]: e for e in row.get("evidence", [])}
    by_owner = {}
    for ray, owner in zip(row["rays"], row["owners"]):
        by_owner.setdefault(owner, []).append(np.asarray(ray, float))
    sel_in = set(row["unidirectional_owners"])
    rays, owners, sel = [], [], []
    for owner, rs in by_owner.items():
        pts = np.asarray(poses[owner])
        if p["min_torso_px"] > 0:
            torso = pts[[5, 6, 11, 12]]
            if abs(torso[2:, 1].mean() - torso[:2, 1].mean()) < p["min_torso_px"]:
                continue
        mo = evid.get(owner, {}).get("motion")
        travel = None if mo is None else np.asarray(mo["direction"], float)
        selected = owner in sel_in
        if mode == "agree":
            if travel is None:
                continue
            rs = [r for r in rs if r[2:4] @ travel > 0]
            if not rs:
                continue
            selected = selected or len(rs) == 1
        elif (
            mode == "resolve"
            and travel is not None
            and mo["strength"] >= p["motion_min"]
        ):
            r = max(rs, key=lambda r: abs(r[2:4] @ travel)).copy()
            if r[2:4] @ travel < 0:
                r[2:4] = -r[2:4]
            rs = [r]
            selected = True
        for r in rs:
            rays.append(r.tolist())
            owners.append(owner)
        if selected:
            sel.append(owner)
    return {**row, "rays": rays, "owners": owners, "unidirectional_owners": sel}


class Planner(AttentionFraming):
    def __init__(self, size, view, field, **params):
        super().__init__(size, view, field)
        self.p = {**DEFAULTS, **params}
        self.hist = deque()
        self.n = 0
        self.peak = 0
        self.peak_center = None
        self.returning = False
        self.ret_peak = 0
        self.ret_center = None
        self.ret_since = 0.0

    # ---------------- group guard bookkeeping
    def observe(self, players, center, t):
        p = self.p
        self.n = 0 if players is None else len(players.get("feet", []))
        self.hist.append((t, self.n, np.asarray(center, float).copy()))
        while self.hist[0][0] < t - p["g_window"]:
            self.hist.popleft()
        best = max(self.hist, key=lambda h: (h[1], h[0]))
        self.peak, self.peak_center = best[1], best[2]
        if self.returning and (
            self.n >= p["g_exit"] * self.ret_peak or t - self.ret_since > p["g_timeout"]
        ):
            self.returning = False

    def anchor(self, point):
        self.returning = False
        super().anchor(point)

    def _base(self, players):
        """Players-only target. base: 'motion' (PlayerTargets: density x movement weighted centroid + 0.35 s lead),
        'mean', 'median', 'density' (Gaussian peak, bandwidth density_bw view widths, mean of players within it),
        'speedw' (weights 1 + speed_gain * relative speed), 'converge' (least-squares meeting point of velocity rays,
        fallback 'motion'). Then optional extra lead along the weighted group velocity (vel_lead s, capped)."""
        if players is None:
            return None
        p = self.p
        feet = np.asarray(players["feet"], float)
        vel = np.asarray(players["velocity"], float)
        if p["base"] == "mean":
            b = np.asarray(players["mean"], float).copy()
        elif p["base"] == "median":
            b = np.median(feet, axis=0)
        elif p["base"] == "density":
            d = (feet[:, None] - feet[None]) / self.view
            dens = np.exp(-0.5 * (d**2).sum(-1) / p["density_bw"] ** 2).sum(1)
            c = feet[dens.argmax()]
            near = (
                np.linalg.norm((feet - c) / self.view, axis=1) <= 1.5 * p["density_bw"]
            )
            b = feet[near].mean(0)
        elif p["base"] == "speedw":
            rel = np.linalg.norm(vel / self.view, axis=1)  # view widths per s
            w = 1 + p["speed_gain"] * np.clip(rel / 0.5, 0, 1)
            b = np.average(feet, axis=0, weights=w)
        elif p["base"] == "converge":
            b = np.asarray(players["motion"], float).copy()
            sp = np.linalg.norm(vel / self.view, axis=1)
            mv = sp > 0.1
            if mv.sum() >= 3:
                dirs = vel[mv] / np.linalg.norm(vel[mv], axis=1, keepdims=True)
                A = np.zeros((2, 2))
                r = np.zeros(2)
                for f, u in zip(feet[mv] / self.view, dirs):
                    Pm = np.eye(2) - np.outer(u, u)
                    A += Pm
                    r += Pm @ f
                ev = np.linalg.eigvalsh(A)
                if ev[0] / ev[-1] > 0.1:
                    q = np.linalg.solve(A, r) * self.view
                    if (
                        np.linalg.norm((q - b) / self.view) < 0.6
                    ):  # only a nearby meeting point
                        b = q
        else:
            b = np.asarray(players["motion"], float).copy()
        if p["vel_lead"] > 0 and len(feet):
            v = np.average(vel, axis=0, weights=players["weight"])
            cap = p["vel_lead_cap"] * self.view
            b = b + np.clip(v * p["vel_lead"], -cap, cap)
        return b

    def step(self, row, players, center, timestamp):
        p = self.p
        guard = p["guard"]
        ratio = self.n / self.peak if self.peak else 1.0
        if guard and (
            not self.returning
            and self.peak >= p["g_min_peak"]
            and ratio < p["g_enter"]
            and self.peak_center is not None
        ):
            self.returning = True
            self.ret_peak, self.ret_center, self.ret_since = (
                self.peak,
                self.peak_center.copy(),
                timestamp,
            )
            self.direction = None
            self.pending_count = 0
        if guard and self.returning:
            dt = (
                0.1
                if self.last_time is None
                else np.clip(timestamp - self.last_time, 0, 0.2)
            )
            self.last_time = timestamp
            return (
                self._approach(self.ret_center, dt, p["g_return_rate"]),
                "return to group",
                0,
            )
        scale = (
            (
                float(
                    np.clip((ratio - p["g_zero"]) / (p["g_full"] - p["g_zero"]), 0, 1)
                )
                if self.peak
                else 1.0
            )
            if guard
            else 1.0
        )
        dt = (
            0.1
            if self.last_time is None
            else np.clip(timestamp - self.last_time, 0, 0.2)
        )
        self.last_time = timestamp
        if timestamp - self.last_support > p["direction_reset"]:
            self.direction = None
        observed_people = 0 if players is None else len(players.get("feet", []))
        heading = None
        continuation = False
        if not p["use_orientation"]:
            base = self._base(players)
            if base is None:
                return (
                    (None if self.target is None else self.target.copy()),
                    "centroid (memory)",
                    0,
                )
            rate = p["approach_rate"]
            if p["few_hold"] and len(players["feet"]) < 3:
                rate = 0.15  # few players visible: drift slowly
            if (
                self.target is not None
                and p["dead_zone"] > 0
                and np.all(np.abs(base - self.target) <= p["dead_zone"] * self.view)
            ):
                return self.target.copy(), "centroid (dead zone)", 0
            return self._approach(base, dt, rate), "centroid", 0
        if p["use_orientation"]:
            row = filter_row(row, p)
            heading = dominant_heading(
                row,
                center,
                self.view,
                self.size,
                minimum_dominance=p["min_dominance"],
                minimum_bodies=p["min_support"],
            )
            if heading is not None and heading["support"] < max(
                p["min_support"], int(np.ceil(observed_people * p["support_frac"]))
            ):
                heading = None
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
                    and retained["support"]
                    >= max(2, int(np.ceil(observed_people * 0.25)))
                    and not (
                        guard
                        and self.peak >= p["g_peak_strict"]
                        and retained["support"] < 3
                    )
                    and retained["direction"] @ self.direction >= np.cos(np.deg2rad(35))
                ):
                    heading, continuation = retained, True
        base = self._base(players)
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
            required = p["opposite_required"] if opposite else p["pending_required"]
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
                lead_widths = (
                    p["lead_min"] + p["lead_gain"] * agreement * coverage
                ) * p["lead_scale"]
                self.lead_widths = (
                    lead_widths if new_intent else max(self.lead_widths, lead_widths)
                )
                displacement = (
                    self.direction
                    * (self.lead_widths * scale * self.view[0] * self.size[0])
                    / self.size
                )
                displacement[1] = np.clip(
                    displacement[1], -0.25 * self.view[1], 0.25 * self.view[1]
                )
                if (
                    p["veto_speed"] > 0
                    and players is not None
                    and len(players["feet"]) >= 3
                ):
                    # tracked-velocity veto: the visible group clearly moves the other way in x -> no body lead
                    gv = (
                        np.average(
                            players["velocity"], axis=0, weights=players["weight"]
                        )
                        / self.view
                    )
                    if abs(gv[0]) >= p["veto_speed"] and np.sign(gv[0]) != np.sign(
                        self.direction[0]
                    ):
                        displacement = np.zeros(2)
                target = self._approach(
                    base + displacement, dt, p["approach_rate"], respect_direction
                )
                self.last_support = timestamp
                return target, "shared orientation", support
        age = timestamp - self.last_support
        if self.target is not None and age <= p["memory_hold"]:
            return self.target.copy(), "remembered direction", support
        enough_players = players is not None and len(players.get("feet", [])) >= 3
        if base is not None and (
            self.direction is None or enough_players or age > p["observe_hold"]
        ):
            return (
                self._approach(base, dt, p["recenter_rate"]),
                "gradual recentring",
                support,
            )
        if self.target is not None and age <= p["observe_hold"]:
            return self.target.copy(), "observing the area", support
        self.direction, self.target = None, None
        self.lead_widths = 0.0
        return None, "search", support


def make_attention(cfg, size, view, field):
    params = {k: v for k, v in cfg.items() if k in DEFAULTS}
    return Planner(size, view, field, **params)
