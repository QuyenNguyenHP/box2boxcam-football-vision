"""Orientation audit against ground-truth motion.

For every pose decision (5 Hz) of a simulated camera, each body with rays is matched to a GT
box (hip midpoint inside the GT box, nearest centre). GT motion = image velocity of the GT foot
point over +-0.5 s (and pitch speed via calibration). Only bodies moving >= ``SPEED`` m/s along a
straight path are audited. ``audit_segment`` takes closed-loop rows that keep the pose payload
(``keep_rows``), as returned by ``closed_loop.simulate_segment``.
"""

import numpy as np

SPEED = 2.5  # m/s


def ang(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def audit_segment(gt, seg, rows, bodies, heads):
    """Append audited bodies and common-heading records of one segment to ``bodies`` / ``heads``."""
    size = gt.size

    # GT trajectories by id: frame -> foot px
    def foot(t_abs, gid):
        b = gt.boxes_at(t_abs)
        b = b[b[:, 1] == gid]
        return (
            None if not len(b) else np.array([b[0, 2] + b[0, 4] / 2, b[0, 3] + b[0, 5]])
        )

    for k, r in enumerate(rows):
        t_abs = seg["t0"] + k / 10
        if r["row"] is not None:
            row = r["row"]
            boxes = gt.boxes_at(t_abs)
            if not len(boxes):
                continue
            sel = set(row["selected"])
            by_owner = {}
            for ray, owner in zip(row["rays"], row["owners"]):
                by_owner.setdefault(owner, []).append(ray)
            for owner, rays in by_owner.items():
                pts = row["poses"][owner]
                hip = pts[[11, 12], :2].mean(0)
                sh = pts[[5, 6], :2].mean(0)
                torso = float(abs(hip[1] - sh[1]))
                inside = (
                    (boxes[:, 2] <= hip[0])
                    & (hip[0] <= boxes[:, 2] + boxes[:, 4])
                    & (boxes[:, 3] <= hip[1])
                    & (hip[1] <= boxes[:, 3] + boxes[:, 5])
                )
                if not inside.any():
                    continue
                cand = boxes[inside]
                cen = np.c_[cand[:, 2] + cand[:, 4] / 2, cand[:, 3] + cand[:, 5] / 2]
                b = cand[np.linalg.norm(cen - hip, axis=1).argmin()]
                gid = int(b[1])
                pa, pb, pm = (
                    foot(t_abs - 0.5, gid),
                    foot(t_abs + 0.5, gid),
                    foot(t_abs, gid),
                )
                if pa is None or pb is None or pm is None:
                    continue
                va = gt.cal.to_pitch(np.array([pa, pm, pb]))
                speed = np.linalg.norm(va[2] - va[0]) / 1.0
                straight = np.linalg.norm(va[2] - va[0]) / (
                    np.linalg.norm(va[1] - va[0]) + np.linalg.norm(va[2] - va[1]) + 1e-9
                )
                vimg = pb - pa
                rec = {
                    "seg": seg["id"],
                    "t": t_abs,
                    "speed": float(speed),
                    "straight": float(straight),
                    "torso_px": torso,
                    "box_h": float(b[5]),
                    "pitch_y": float(va[1, 1]),
                    "foot_y_norm": float(pm[1] / size[1]),
                    "selected": owner in sel,
                    "n_hyp": len(rays),
                }
                errs = [ang(np.asarray(ray)[2:4], vimg) for ray in rays]
                rec["err_best"] = min(errs)
                rec["err_first"] = errs[0]
                if owner in sel:
                    rec["err"] = errs[0]
                    ev = next(
                        (e for e in row.get("evidence", []) if e["owner"] == owner),
                        None,
                    )
                    if ev is not None:
                        rec["sources"] = ev["sources"]
                        rec["raw_err"] = (
                            ang(ev["raw_direction"], vimg)
                            if ev.get("raw_direction") is not None
                            else None
                        )
                bodies.append(rec)
        # common heading the camera follows vs GT group motion of people in the crop
        if r["heading"] is not None and k % 2 == 0:
            c, v = np.asarray(r["center"]), np.asarray(r["view"])
            ids, feet, _heights, _pitch = gt.people_at(t_abs)
            inside = (np.abs(feet / size - c) <= v / 2).all(1)
            vel = []
            for gid in ids[inside]:
                pa, pb = foot(t_abs - 0.5, gid), foot(t_abs + 0.5, gid)
                if pa is not None and pb is not None:
                    vel.append(pb - pa)
            if len(vel) >= 3:
                gv = np.mean(vel, 0)
                dv = r.get("gvel")
                dv_px = (
                    None
                    if dv is None or np.linalg.norm(dv) == 0
                    else np.asarray(dv) * size
                )
                heads.append(
                    {
                        "seg": seg["id"],
                        "t": t_abs,
                        "err": ang(r["heading"], gv),
                        "det_vel_err": None if dv_px is None else ang(dv_px, gv),
                        "det_vel_x_agree": None
                        if dv_px is None
                        else bool(np.sign(dv_px[0]) == np.sign(gv[0])),
                        "x_sign_agree": bool(
                            np.sign(r["heading"][0]) == np.sign(gv[0])
                        ),
                        "group_speed_px": float(np.linalg.norm(gv)),
                        "mode": r["mode"],
                        "group_vx_view_per_s": float(gv[0] / size[0] / v[0]),
                    }
                )


def summarize(bodies, heads):
    B = [b for b in bodies if b["speed"] >= SPEED and b["straight"] >= 0.8]
    sel = [b for b in B if b["selected"]]
    amb = [b for b in B if not b["selected"]]
    e = np.array([b["err"] for b in sel])
    print(
        f"moving bodies audited: {len(B)} (selected/unidirectional {len(sel)}, ambiguous {len(amb)})"
    )
    if len(e):
        print(
            f"selected: median err {np.median(e):.0f} deg, p75 {np.percentile(e, 75):.0f}, "
            f"<45deg {100 * (e < 45).mean():.0f}%, "
            f"front/back wrong (>90deg) {100 * (e > 90).mean():.0f}%"
        )
        axis = np.minimum(e, 180 - e)
        print(f"  axis error ignoring sign: median {np.median(axis):.0f}")
        for lo, hi in ((0, 15), (15, 25), (25, 40), (40, 999)):
            m = [b["err"] for b in sel if lo <= b["torso_px"] < hi]
            if m:
                print(
                    f"  torso {lo}-{hi}px: n={len(m)} median {np.median(m):.0f} wrong-side {100 * np.mean(np.array(m) > 90):.0f}%"
                )
        for lo, hi in ((0, 23), (23, 46), (46, 70)):
            m = [b["err"] for b in sel if lo <= b["pitch_y"] < hi]
            if m:
                print(
                    f"  pitch y {lo}-{hi} m (0=far): n={len(m)} median {np.median(m):.0f} wrong-side {100 * np.mean(np.array(m) > 90):.0f}%"
                )
    if amb:
        eb = np.array([b["err_best"] for b in amb])
        print(f"ambiguous: best hypothesis median {np.median(eb):.0f} deg")
    if heads:
        he = np.array([h["err"] for h in heads])
        fast = [h for h in heads if abs(h["group_vx_view_per_s"]) > 0.15]
        dv = [h for h in heads if h.get("det_vel_err") is not None]
        if dv:
            print(
                f"  same instants, detected-players mean velocity (no orientation): median err {np.median([h['det_vel_err'] for h in dv]):.0f} deg, x-sign agree {100 * np.mean([h['det_vel_x_agree'] for h in dv]):.0f}% (n={len(dv)})"
            )
        print(
            f"common heading: n={len(heads)} median err {np.median(he):.0f} deg, x-sign agree {100 * np.mean([h['x_sign_agree'] for h in heads]):.0f}%;"
            f" when group moves >0.15 view/s in x: n={len(fast)} x-sign agree {100 * np.mean([h['x_sign_agree'] for h in fast]) if fast else float('nan'):.0f}%"
        )
