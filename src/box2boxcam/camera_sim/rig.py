"""Closed-loop camera rig: one virtual camera driven by detections of its own crop.

Each camera only "sees" the people, poses and ball proposals that fall inside its
current crop, then runs the player/ball/body-orientation logic: ``PlayerTargets``,
``BodyEvidence``, ``direction_row``, the planner (``AttentionFraming`` or a
variant), ``ContinuousBallTrack``, ``HybridFraming``, ``SmoothController`` and a
fixed x2 ``ZoomPolicy``, with the pitch filter and a centre clip.
"""

import cv2
import numpy as np

from box2boxcam.camera_sim.body_evidence import BodyEvidence
from box2boxcam.camera_sim.hybrid_framing import ContinuousBallTrack, HybridFraming
from box2boxcam.camera_sim.planner import make_attention
from box2boxcam.camera_sim.players import PlayerTargets
from box2boxcam.camera_sim.pose_directions import direction_row
from box2boxcam.camera_sim.smooth_control import SmoothController
from box2boxcam.camera_sim.zoom_policy import ZoomPolicy

FPS = 10


class Rig:
    def __init__(self, cfg, size, field, start_center=None):
        self.cfg = cfg
        self.size = np.asarray(size)
        self.field = np.asarray(field)
        self.poly_px = (
            np.rint(self.field * self.size).astype(np.float32).reshape(-1, 1, 2)
        )
        self.poly_n = self.field.astype(np.float32)
        self.lo, self.hi = self.field.min(0), self.field.max(0)
        base_view = np.array([0.45, 0.45 * size[0] / size[1] * 9 / 16])
        self.zoom = ZoomPolicy(base_view, 2.0, "fixed")
        view = self.zoom.view
        y0 = float(self.field[:, 1].min() + 0.35 * np.ptp(self.field[:, 1]))
        start = [0.5, y0] if start_center is None else start_center
        self.control = SmoothController(view, start, search=False)
        ms = cfg.get("max_speed", 0.55)
        self.control.MAX_SPEED = np.array([ms, ms * 0.8])
        self.oracle = None
        self.track = ContinuousBallTrack(view, field)
        self.framing = HybridFraming(
            view, acquire=0.70, retain=0.25, transition_rate=0.85, exit_hold=1.2
        )
        self.players = PlayerTargets(size, view)
        self.evidence = BodyEvidence(size, view)
        self.attention = make_attention(cfg, size, view, field)
        self.row = None
        self.rows = []
        self.discarded = 0

    def step(self, k, det, t):
        size = self.size
        view = self.zoom.view.copy()
        self.control.resize(view)
        self.track.view = view
        self.framing.view = view
        self.players.view = self.evidence.view = self.evidence.tracker.view = (
            self.attention.view
        ) = view
        center = self.control.center.copy()
        origin = np.rint((center - view / 2) * size).astype(int)
        end = np.rint((center + view / 2) * size).astype(int)
        pic_wh = end - origin
        # ---- people seen by this crop
        boxes = det["people"]
        poses_all = det["pose"]
        if len(boxes):  # PeopleDetector keeps feet inside the pitch polygon
            feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
            onpitch = np.array(
                [
                    cv2.pointPolygonTest(self.poly_n, (float(x), float(y)), False) >= 0
                    for x, y in feet
                ]
            )
            boxes, poses_all = boxes[onpitch], poses_all[onpitch]
        if len(boxes):
            feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
            inside = (np.abs(feet - center) <= view / 2).all(1)
            boxes, poses_all = boxes[inside], poses_all[inside]
        context = self.players.step(boxes, t, center=center)
        n_view = 0 if context is None else len(context["feet"])
        if hasattr(self.attention, "observe"):
            self.attention.observe(context, center, t)
        if k % (FPS // 5) == 0:
            poses = []
            for b, p in zip(boxes, poses_all):
                if not np.isfinite(p).all():
                    continue
                # the closed loop crops the pose input to the camera picture: joints outside are lost
                p = p.copy()
                out = ((p[:, :2] < origin) | (p[:, :2] >= end)).any(1)
                p[out, 2] = 0.0
                poses.append(p)
            row = direction_row(
                poses,
                origin,
                tuple(pic_wh),
                center,
                size,
                view,
                self.evidence,
                t,
                horizontal_fov=60,
            )
            ptarget, pmode, _support = self.attention.step(row, context, center, t)
            self.row = {
                "poses": [np.asarray(p, np.float32) for p in row["poses"]],
                "rays": np.asarray(row["rays"], float).reshape(-1, 5),
                "owners": list(row["owners"]),
                "selected": list(row["unidirectional_owners"]),
                "t": t,
                "evidence": row["evidence"],
            }
        else:
            ptarget = (
                None if self.attention.target is None else self.attention.target.copy()
            )
            pmode = (
                "return to group"
                if getattr(self.attention, "returning", False)
                else "remembered direction"
            )
        heading = self.attention.direction
        # ---- ball proposals seen by this crop
        cand = det["ball"]
        if len(cand):
            cand = cand[(np.abs(cand[:, :2] - center) <= view / 2).all(1)]
            cand = cand[np.argsort(-cand[:, 4])][:10]
        if len(cand):
            keep = np.array(
                [
                    cv2.pointPolygonTest(
                        self.poly_px, (float(x * size[0]), float(y * size[1])), True
                    )
                    >= -float(w * size[0])
                    for x, y, w, h, _ in cand
                ]
            )
            self.discarded += int((~keep).sum())
            cand = cand[keep]
        cand = np.asarray(cand, float).reshape(-1, 5)
        if self.cfg.get("no_ball"):
            cand = np.empty((0, 5))
        ball = self.track.step(
            cand, center, t, hint=None if context is None else context["motion"]
        )
        target, source = self.framing.step(ball, ptarget, pmode, center, t, heading)
        if self.cfg.get("static"):
            target, source = None, "fixed"
        if self.cfg.get("oracle") and self.oracle is not None:
            o = self.oracle[min(k, len(self.oracle) - 1)]
            target, source = (
                (None, "oracle")
                if not np.isfinite(o).all()
                else (np.asarray(o, float), "oracle")
            )
        if self.framing.ball_active and target is not None:
            self.attention.anchor(target)
        self.rows.append(
            {
                "center": center,
                "view": view,
                "measured": bool(ball.measured),
                "ball": None if ball.point is None else ball.point,
                "cand": cand,
                "mode": source,
                "target": None if target is None else np.asarray(target),
                "n": n_view,
                "heading": None if heading is None else np.asarray(heading).copy(),
                "gvel": None
                if context is None
                else np.average(context["velocity"], axis=0, weights=context["weight"]),
                "row": self.row if k % (FPS // 5) == 0 else None,
            }
        )
        self.control.step(target, 0 if target is None else 1)
        self.control.center = np.clip(self.control.center, self.lo, self.hi)
        self.zoom.step(t, center, ball, context)
