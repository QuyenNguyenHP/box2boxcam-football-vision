"""Ground truth per decision (10 Hz) from the official SoccerTrack v2 files.

GT people   : mot/<match>.txt (MOT boxes in clip pixels; the released file is 0-indexed = video frame, conf -1).
              Feet = bottom-centre. 'On pitch' = foot inside the calibrated pitch polygon grown by 1.5 m
              (players + goalkeepers + referee; MOT has no role column).
Action centre (AC): pitch coordinates; Gaussian kernel density (sigma 8 m) of on-pitch GT people evaluated at each
              person; AC = mean position of the people within 15 m of the densest person. Smoothed over 1 s.
              Projected to panorama pixels with the TPS calibration.
Events      : BAS 12-class events (frame of the half video == clip frame for the first 240 s).
Event location proxy: reconstructed ball track (ball/<m>_1st_ball.npz) AT the event frame only (= annotated anchor).
"""

import json
from pathlib import Path

import cv2
import numpy as np

from box2boxcam.benchmark.calib import Calib

FPS_SRC, FPS = 25, 10


def load_mot(root, match):
    a = np.loadtxt(Path(root) / f"mot/{match}.txt", delimiter=",", usecols=range(9))
    # released file: 0-indexed frames = video frame index, conf column is -1 (all rows valid)
    return a  # frame,id,x,y,w,h,conf,class,vis


class GT:
    def __init__(self, match, root, calib=None):
        self.match = match
        raw = Path(root)
        self.cal = calib or Calib(match, root)
        self.size = self.cal.size
        mot = load_mot(root, match)
        self.frames = {}
        order = np.argsort(mot[:, 0], kind="stable")
        mot = mot[order]
        f = mot[:, 0].astype(int)
        cuts = np.flatnonzero(np.diff(f)) + 1
        for chunk in np.split(mot, cuts):
            self.frames[int(chunk[0, 0])] = chunk
        self.max_frame = max(self.frames)
        ev = json.loads(
            (raw / f"bas/{match}/{match}_12_class_events.json").read_text()
        )["actions"]
        self.events = [
            e
            for e in ev
            if e["gameTime"].startswith("1 ") and e["frame"] < self.max_frame
        ]
        try:
            b = np.load(raw / f"ball/{match}_1st_ball.npz")
            self.ball = (b["frame"], b["x"], b["y"], bool(b["y_flipped"]))
        except (OSError, KeyError, ValueError):
            self.ball = None
        # pitch polygon grown by 1.5 m, in pixels
        xs = np.linspace(-1.5, 106.5, 40)
        ys = np.linspace(-1.5, 69.5, 24)
        edge = np.r_[
            np.c_[xs, np.full_like(xs, -1.5)],
            np.c_[np.full_like(ys, 106.5), ys],
            np.c_[xs[::-1], np.full_like(xs, 69.5)],
            np.c_[np.full_like(ys, -1.5), ys[::-1]],
        ]
        self._poly = self.cal.to_img(edge).astype(np.float32)
        self._cv2 = cv2

    def boxes_at(self, t_abs):
        fr = round(t_abs * FPS_SRC)
        return self.frames.get(fr, np.empty((0, 9)))

    def people_at(self, t_abs):
        """ids, feet px (N,2), heights px, pitch xy (N,2) for on-pitch GT people."""
        b = self.boxes_at(t_abs)
        if not len(b):
            return np.empty(0, int), np.empty((0, 2)), np.empty(0), np.empty((0, 2))
        feet = np.c_[b[:, 2] + b[:, 4] / 2, b[:, 3] + b[:, 5]]
        on = np.array(
            [
                self._cv2.pointPolygonTest(self._poly, (float(x), float(y)), False) >= 0
                for x, y in feet
            ]
        )
        b, feet = b[on], feet[on]
        return b[:, 1].astype(int), feet, b[:, 5], self.cal.to_pitch(feet)

    def series(self, t0, t1):
        """Per-decision GT arrays for [t0, t1) at 10 Hz (absolute clip seconds)."""
        n = round((t1 - t0) * FPS)
        out = {
            "t": t0 + np.arange(n) / FPS,
            "feet": [],
            "pitch": [],
            "ids": [],
            "h": [],
        }
        ac = np.full((n, 2), np.nan)
        for k in range(n):
            ids, feet, h, pitch = self.people_at(t0 + k / FPS)
            out["feet"].append(feet / self.size)
            out["pitch"].append(pitch)
            out["ids"].append(ids)
            out["h"].append(h)
            if len(pitch) >= 3:
                d2 = ((pitch[:, None] - pitch[None]) ** 2).sum(-1)
                dens = np.exp(-d2 / (2 * 8**2)).sum(1)
                c = pitch[dens.argmax()]
                near = np.sqrt(((pitch - c) ** 2).sum(1)) <= 15
                ac[k] = pitch[near].mean(0)
        # 1 s centred moving average (causality irrelevant: evaluation only)
        sm = ac.copy()
        for k in range(n):
            w = ac[max(0, k - 5) : k + 6]
            w = w[np.isfinite(w).all(1)]
            if len(w):
                sm[k] = w.mean(0)
        out["ac_pitch"] = sm
        out["ac"] = np.where(
            np.isfinite(sm).all(1)[:, None],
            self.cal.to_img(np.nan_to_num(sm)) / self.size,
            np.nan,
        )
        evs = [e for e in self.events if t0 * FPS_SRC <= e["frame"] < t1 * FPS_SRC]
        out["events"] = []
        for e in evs:
            loc = None
            loc_m = None
            if self.ball is not None:
                fr, bx, by, flipped = self.ball
                # npz frame k <-> event frame k (largest second difference sits there); only the anchor is used
                i = np.searchsorted(fr, e["frame"])
                if i < len(fr) and fr[i] == e["frame"] and np.isfinite(bx[i]):
                    clamped = (
                        abs(abs(bx[i]) - 52.5) < 0.05 or abs(abs(by[i]) - 34) < 0.05
                    )
                    if (
                        not clamped
                    ):  # positions clamped to the pitch rectangle are skipped
                        p = np.array(
                            [bx[i] + 52.5, (-by[i] if flipped else by[i]) + 34]
                        )
                        loc = (self.cal.to_img(p)[0] / self.size).tolist()
                        loc_m = p.tolist()
            out["events"].append(
                {
                    "t": e["frame"] / FPS_SRC - t0,
                    "label": e["label"],
                    "loc": loc,
                    "loc_m": loc_m,
                }
            )
        out["to_pitch"] = lambda xy: self.cal.to_pitch(
            np.asarray(xy, float) * self.size
        )
        return out
