"""Camera-path metrics against GT (no ball accuracy: there is no per-frame ball GT).

Per decision (10 Hz), crop = centre +- view/2 (normalised panorama coordinates):
  n_in / n_tot      GT on-pitch people with feet inside the crop / on the pitch
  lt3               share of decisions with < 3 GT people in the crop
  ac_dx             |crop centre x - action centre x| in view widths (AC: see gt.py); ac_in = AC inside the crop
  ev_in             share of BAS events whose location proxy (ball-track anchor at the event frame) is in the crop
  ev_dx             |crop centre x - anchor x| in view widths at event frames (anchors clamped to the pitch edge skipped)
  ev_ac_in          share of BAS events at which the AC is in the crop (available for every match)
  lag               reaction lag on fast transitions (see transitions())
  speed, reversals  mean |dx|/view per s and number of x-direction reversals (sign changes of per-decision dx,
                    ignoring pan speeds < 0.05 view widths per s) per minute
  ball_used         share of decisions where the ball tracker had a measurement (behaviour stat only)
The first WARM s of each segment are excluded (all variants start from the same pose at the AC).
"""

import numpy as np

FPS = 10
WARM = 3.0


def transitions(ac_x, view_w, min_move=0.6, window=3.0):
    """Fast transitions: the smoothed AC moves >= min_move (0.6) view widths within `window` (3) s.
    Returns list of (k_start, k_end, sign). Non-overlapping, greedy in time."""
    n = len(ac_x)
    w = int(window * FPS)
    out = []
    k = 0
    while k < n - 1:
        if not np.isfinite(ac_x[k]):
            k += 1
            continue
        seg = ac_x[k : k + w + 1]
        dev = (seg - ac_x[k]) / view_w
        ok = np.isfinite(dev)
        hit = np.flatnonzero(ok & (np.abs(dev) >= min_move))
        if len(hit):
            e = k + hit[0]
            # extend to where the move stops (AC keeps moving in the same direction)
            s = np.sign(dev[hit[0]])
            while (
                e + 1 < n
                and np.isfinite(ac_x[e + 1])
                and (ac_x[e + 1] - ac_x[e]) * s > 0
            ):
                e += 1
            out.append((k, e, s))
            k = e + 1
        else:
            k += 1
    return out


def lag_of(cx, ac_x, k0, k1, s, horizon=8.0):
    """Time between the AC and the crop centre each covering half of the AC move (from k0)."""
    half = ac_x[k0] + 0.5 * (ac_x[k1] - ac_x[k0])
    ta = k0 + np.flatnonzero((ac_x[k0 : k1 + 1] - half) * s >= 0)[0]
    lim = min(len(cx), k1 + int(horizon * FPS))
    after = np.flatnonzero((cx[k0:lim] - half) * s >= 0)
    if not len(after):
        return horizon, False
    tc = k0 + after[0]
    return max(0.0, (tc - ta) / FPS), True


def score(rows, g):
    c = np.array([r["center"] for r in rows])
    v = np.array([r["view"] for r in rows])
    n = min(len(c), len(g["t"]))
    c, v = c[:n], v[:n]
    k0 = int(WARM * FPS)
    n_in, n_tot = [], []
    for k in range(n):
        f = g["feet"][k]
        n_tot.append(len(f))
        n_in.append(int((np.abs(f - c[k]) <= v[k] / 2).all(1).sum()) if len(f) else 0)
    n_in, n_tot = np.array(n_in), np.array(n_tot)
    ac = g["ac"][:n]
    okac = np.isfinite(ac).all(1)
    ac_dx = np.abs(c[:, 0] - ac[:, 0]) / v[:, 0]
    ac_in = (np.abs(ac - c) <= v / 2).all(1)
    sl = slice(k0, n)
    m = {
        "n_in": float(n_in[sl].mean()),
        "frac_in": float((n_in[sl] / np.maximum(n_tot[sl], 1)).mean()),
        "lt3": float((n_in[sl] < 3).mean()),
        "ac_dx": float(ac_dx[sl][okac[sl]].mean()),
        "ac_in": float(ac_in[sl][okac[sl]].mean()),
    }
    ev_in, ev_ac, ev_dx, ev_m = [], [], [], []
    for e in g["events"]:
        k = round(e["t"] * FPS)
        if k < k0 or k >= n:
            continue
        if e["loc"] is not None:
            ev_in.append(bool((np.abs(np.array(e["loc"]) - c[k]) <= v[k] / 2).all()))
            ev_dx.append(abs(e["loc"][0] - c[k, 0]) / v[k, 0])
            if (
                g.get("to_pitch") is not None
            ):  # pan error in metres: crop-centre column at the ball's image row vs the ball
                ev_m.append(
                    float(
                        np.linalg.norm(
                            g["to_pitch"]([c[k, 0], e["loc"][1]])[0]
                            - np.array(e["loc_m"])
                        )
                    )
                )
        if okac[k]:
            ev_ac.append(bool(ac_in[k]))
    m["ev_in"] = float(np.mean(ev_in)) if ev_in else None
    m["n_ev_loc"] = len(ev_in)
    m["ev_dx"] = float(np.mean(ev_dx)) if ev_dx else None
    m["ev_m"] = (
        float(np.mean(ev_m)) if ev_m else None
    )  # metres, crop centre (projected on the pitch) to event-time ball anchor
    m["ev_ac_in"] = float(np.mean(ev_ac)) if ev_ac else None
    lags, missed = [], 0
    for a, b, s in transitions(np.where(okac, ac[:, 0], np.nan), float(v[0, 0])):
        if a < k0:
            continue
        lag, ok = lag_of(c[:, 0], ac[:, 0], a, b, s)
        lags.append(lag)
        missed += not ok
    m["lags"] = lags
    m["lag"] = float(np.mean(lags)) if lags else None
    m["n_trans"] = len(lags)
    m["trans_missed"] = missed
    dx = np.diff(c[sl, 0]) / v[sl][1:, 0]
    m["speed"] = float(np.abs(dx).mean() * FPS)
    mv = dx[np.abs(dx) >= 0.005]  # ignore pan speeds < 0.05 view widths per s
    m["reversals_pm"] = float(((mv[1:] * mv[:-1]) < 0).sum() / ((n - k0) / FPS / 60))
    m["ball_used"] = float(np.mean([r["measured"] for r in rows[k0:n]]))
    m["dur"] = (n - k0) / FPS
    return m
