"""Pitch (metres, corner origin 0..105 x 0..68, y=68 near touchline) <-> panorama pixels, from the
official 65 hand-annotated keypoints, using a smoothed thin-plate spline (handles the fisheye stitch)."""

import json
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

SIZES = {"117092": (3840, 1906), "117093": (4096, 1080), "132831": (3840, 1504)}


def keypoints_path(root, match):
    """Official calibration keypoints inside a SoccerTrack v2 snapshot (release v1.2)."""
    return Path(root) / "raw" / match / f"{match}_keypoints.json"


def keypoints(match, root):
    d = json.loads(keypoints_path(root, match).read_text())
    pitch = np.array([[float(v) for v in re.findall(r"[-\d.]+", k)] for k in d])
    img = np.array(list(d.values()), float)
    return pitch, img


class Calib:
    def __init__(self, match, root, smoothing=1.0):
        self.match = match
        self.size = np.array(SIZES[match])
        p, i = keypoints(match, root)
        self.kp_pitch, self.kp_img = p, i
        self.fwd = RBFInterpolator(
            p, i, kernel="thin_plate_spline", smoothing=smoothing
        )
        self.inv = RBFInterpolator(
            i, p, kernel="thin_plate_spline", smoothing=smoothing
        )

    def to_img(self, pitch):  # pitch metres corner-origin -> pixels
        pitch = np.asarray(pitch, float).reshape(-1, 2)
        return self.fwd(pitch)

    def to_pitch(self, px):
        px = np.asarray(px, float).reshape(-1, 2)
        return self.inv(px)

    def polygon(self, n=24):
        """Pitch boundary as normalised image polygon (for field filters)."""
        xs = np.linspace(0, 105, n)
        ys = np.linspace(0, 68, max(6, n // 2))
        edge = np.r_[
            np.c_[xs, np.zeros_like(xs)],
            np.c_[np.full_like(ys, 105), ys][1:],
            np.c_[xs[::-1], np.full_like(xs, 68)][1:],
            np.c_[np.zeros_like(ys), ys[::-1]][1:-1],
        ]
        return self.to_img(edge) / self.size

    def loo(self, smoothing=1.0):
        errs = []
        for k in range(len(self.kp_pitch)):
            m = np.ones(len(self.kp_pitch), bool)
            m[k] = False
            f = RBFInterpolator(
                self.kp_pitch[m],
                self.kp_img[m],
                kernel="thin_plate_spline",
                smoothing=smoothing,
            )
            errs.append(np.linalg.norm(f(self.kp_pitch[k : k + 1])[0] - self.kp_img[k]))
        return np.array(errs)


def main(argv=None):
    """Leave-one-out calibration error: ``python -m box2boxcam.benchmark.calib ROOT MATCH...``."""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    for m in argv[1:]:
        c = Calib(m, argv[0])
        for s in (0.0, 1.0, 10.0):
            e = c.loo(s)
            print(
                f"{m} smoothing {s} LOO px median {np.median(e):.1f} "
                f"p90 {np.percentile(e, 90):.1f} max {e.max():.1f}"
            )


if __name__ == "__main__":
    main()
