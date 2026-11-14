"""True closed-loop run of camera variants on SoccerTrack v2 segments.

Every variant owns a virtual camera and runs the detectors on its own crop only
(people tiles, poses at 5 Hz, ball detector at 960 px), then the same ``Rig`` /
planner logic and ground-truth scoring for all variants. Decisions are taken at
10 Hz from frames decoded with ffmpeg.
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from box2boxcam.benchmark.aggregate import KEYS, aggregate, segments
from box2boxcam.benchmark.calib import Calib
from box2boxcam.benchmark.gt import GT
from box2boxcam.benchmark.metrics import score
from box2boxcam.benchmark.variants import VARIANTS
from box2boxcam.camera_sim.people_detector import PeopleDetector
from box2boxcam.camera_sim.rig import Rig

DEFAULT_VIDEO_TEMPLATE = "{root}/mot/clips/{match}.mp4"
EMPTY_DETECTIONS = {
    "people": np.empty((0, 5)),
    "pose": np.empty((0, 17, 3)),
    "ball": np.empty((0, 5)),
}


def default_device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_models(people_weights, pose_weights, ball_weights):
    """Load the three YOLO models once; they are shared by every segment."""
    from ultralytics import YOLO

    return {
        "people": YOLO(str(people_weights)),
        "pose": YOLO(str(pose_weights)),
        "ball": YOLO(str(ball_weights)),
    }


def frames(path, size, start, dur):
    args = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        str(start),
        "-i",
        path,
        "-t",
        str(dur),
        "-an",
        "-filter_threads",
        "2",
        "-vf",
        f"fps=10,scale={size[0]}:{size[1]}",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    with subprocess.Popen(args, stdout=subprocess.PIPE) as p:
        n = size[0] * size[1] * 3
        while len(data := p.stdout.read(n)) == n:
            yield np.frombuffer(data, np.uint8).reshape(size[1], size[0], 3)
        p.stdout.close()
        p.wait()


class Live:
    """Detectors run on the crop of one camera: people (tiled), poses (5 Hz), ball proposals.

    ``people``, ``pose`` and ``ball`` are weights paths or already loaded YOLO models.
    """

    def __init__(self, field, size, people, pose, ball, device):
        from ultralytics import YOLO

        self.device = device
        self.size = np.asarray(size)
        self.people = PeopleDetector(people, field, size, device)
        self.pose = pose if hasattr(pose, "predict") else YOLO(str(pose))
        self.ball = ball if hasattr(ball, "predict") else YOLO(str(ball))

    def detect(self, frame, rig, k):
        size = self.size
        view = rig.zoom.view.copy()
        center = rig.control.center.copy()
        origin = np.rint((center - view / 2) * size).astype(int)
        end = np.rint((center + view / 2) * size).astype(int)
        pic = frame[origin[1] : end[1], origin[0] : end[0]]
        boxes = self.people(pic, origin=origin)
        kps = np.full((len(boxes), 17, 3), np.nan, np.float32)
        if k % 2 == 0 and len(boxes):
            feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
            inview = (np.abs(feet - center) <= view / 2).all(1)
            jobs = []
            h, w = pic.shape[:2]
            for j, bx in enumerate(boxes):
                if not inview[j]:
                    continue
                tgt = bx[:4] * np.tile(size, 2) - np.tile(origin, 2)
                pad = (tgt[3] - tgt[1]) * 0.30
                a = np.floor(np.maximum([0, 0], tgt[:2] - pad)).astype(int)
                b = np.ceil(np.minimum([w, h], tgt[2:] + pad)).astype(int)
                if (b <= a).any():
                    continue
                jobs.append((j, tgt, a, pic[a[1] : b[1], a[0] : b[0]]))
            if jobs:
                res = self.pose.predict(
                    [q[3] for q in jobs],
                    imgsz=384,
                    device=self.device,
                    conf=0.15,
                    verbose=False,
                )
                for (j, tgt, a, _), r in zip(jobs, res):
                    if not len(r.boxes):
                        continue
                    cand = r.boxes.xyxy.cpu().numpy() + np.tile(a, 2)
                    cen = (cand[:, :2] + cand[:, 2:]) / 2
                    mid = (tgt[:2] + tgt[2:]) / 2
                    ci = np.linalg.norm(cen - mid, axis=1).argmin()
                    if np.linalg.norm(cen[ci] - mid) > (tgt[3] - tgt[1]) * 0.6:
                        continue
                    pts = r.keypoints.data[ci].cpu().numpy().copy()
                    pts[:, :2] += a + origin
                    kps[j] = pts
        (r,) = self.ball.predict(
            pic,
            imgsz=960,
            device=self.device,
            classes=[0],
            conf=0.10,
            max_det=10,
            verbose=False,
        )
        if len(r.boxes):
            loc = r.boxes.xywhn.cpu().numpy()
            world = loc.copy()
            world[:, :2] = center + (loc[:, :2] - 0.5) * view
            world[:, 2:4] *= view
            cand = np.column_stack((world, r.boxes.conf.cpu().numpy()))
        else:
            cand = np.empty((0, 5))
        return {
            "people": np.asarray(boxes, float).reshape(-1, 5),
            "pose": kps,
            "ball": cand,
        }


def clean_rows(rows):
    """Rows without the heavy per-decision pose/candidate payload (what ``score`` needs)."""
    return [{k: v for k, v in x.items() if k not in ("row", "cand")} for x in rows]


def simulate_segment(root, video_template, seg, names, models, device, keep_rows=False):
    """Closed-loop simulation of ``names`` on one segment.

    Returns ``(ground_truth_series, {variant: rows})``.
    """
    m = seg["match"]
    g = GT(m, root).series(seg["t0"], seg["t1"])
    cal = Calib(m, root)
    size = cal.size
    field = cal.polygon().tolist()
    live = Live(field, size, models["people"], models["pose"], models["ball"], device)
    ac0 = g["ac"][0] if np.isfinite(g["ac"][0]).all() else None
    rigs = {n: Rig(VARIANTS[n], size, field, ac0) for n in names}
    for rg in rigs.values():
        rg.oracle = g["ac"]
    n_dec = round((seg["t1"] - seg["t0"]) * 10)
    video = video_template.format(root=root, match=m)
    for k, frame in enumerate(frames(video, size, seg["t0"], seg["t1"] - seg["t0"])):
        if k >= n_dec:
            break
        for n, rg in rigs.items():
            if VARIANTS[n].get("static") or VARIANTS[n].get("oracle"):
                # bounds need no detections
                rg.step(k, EMPTY_DETECTIONS, k / 10)
            else:
                rg.step(k, live.detect(frame, rg, k), k / 10)
    return g, {n: rg.rows for n, rg in rigs.items()}


def run_benchmark(
    root,
    segments_file,
    which,
    names,
    out_file,
    models,
    device,
    video_template=DEFAULT_VIDEO_TEMPLATE,
    only_ids=None,
):
    """Score ``names`` on the segments of split ``which`` and write ``out_file`` (JSON)."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    chosen = segments(segments_file, which)
    if only_ids:
        chosen = [s for s in chosen if s["id"] in only_ids]
    per_seg = []
    t_start = time.time()
    for seg in chosen:
        g, rows = simulate_segment(root, video_template, seg, names, models, device)
        res = {"id": seg["id"]}
        for n in names:
            res[n] = score(clean_rows(rows[n]), g)
        per_seg.append(res)
        print(
            seg["id"],
            f"{time.time() - t_start:.0f}s",
            " | ".join(
                f"{n}: lt3 {res[n]['lt3']:.2f} acdx {res[n]['ac_dx']:.2f}"
                for n in names
            ),
            flush=True,
        )
        out_file.write_text(
            json.dumps({"set": which, "partial": True, "segments": per_seg}, indent=1)
        )
    agg = aggregate(per_seg, names)
    result = {
        "set": which,
        "mode": "closed loop",
        "variants": {n: VARIANTS[n] for n in names},
        "aggregate": agg,
        "segments": per_seg,
    }
    out_file.write_text(json.dumps(result, indent=1))
    for n in names:
        a = agg[n]
        print(
            f"{n:24s} "
            + " ".join(f"{k}={a[k][0]:.3f}" for k in KEYS if a[k] is not None)
            + f" lag_pooled={a['lag_pooled']} n_trans={a['n_trans']}"
        )
    if which in ("TEST", "FINAL"):
        with open(out_file.parent / "looks.log", "a") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M')} {which} look (closed loop): "
                f"{out_file.stem} variants={names}\n"
            )
    return result
