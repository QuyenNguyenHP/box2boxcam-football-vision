"""Causal limb and motion evidence for disambiguating torso orientations."""

import numpy as np

from box2boxcam.camera_sim.players import PlayerTargets

EVIDENCE_WEIGHT = {"knees": 1.8, "arms": 1.1, "motion": 0.65, "temporal": 0.7}


def refine_direction(direction, cues):
    """Small angular correction from coherent travel, never a facing reversal.

    Torso projection remains the facing hypothesis. Travel can refine an aligned
    hypothesis only; contradictory current limb evidence vetoes the correction.
    """
    motion = cues.get("motion")
    if motion is None or motion.get("angular_strength", motion["strength"]) < 0.55:
        return direction.copy(), 0.0
    travel = np.asarray(motion["direction"])
    cosine = float(direction @ travel)
    if cosine < np.cos(np.deg2rad(60)):
        return direction.copy(), 0.0
    for name in ("knees", "arms"):
        cue = cues.get(name)
        if (
            cue is not None
            and cue["strength"] >= 0.55
            and cue["direction"] @ travel < np.cos(np.deg2rad(60))
        ):
            return direction.copy(), 0.0
    angle = np.arctan2(direction[0] * travel[1] - direction[1] * travel[0], cosine)
    correction = float(
        np.clip(
            angle * 0.45 * motion.get("angular_strength", motion["strength"]),
            -np.pi / 9,
            np.pi / 9,
        )
    )
    c, s = np.cos(correction), np.sin(correction)
    return np.array(
        [c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]]
    ), float(np.rad2deg(correction))


def motion_cue(history, size, height):
    """Reject jitter, recent turns and bad fits before using travel as evidence."""
    if len(history) < 4:
        return None
    times = np.array([time for time, _ in history])
    if times[-1] - times[0] < 0.5:
        return None
    points = np.array([point for _, point in history]) * size
    centered = times - times.mean()
    velocity = (centered[:, None] * (points - points.mean(0))).sum(0) / (
        centered @ centered
    )
    speed = np.linalg.norm(velocity)
    if speed < 1e-6:
        return None
    residual = np.sqrt(
        np.square(points - points.mean(0) - centered[:, None] * velocity).sum(1).mean()
    )
    segments = np.diff(points, axis=0)
    traveled = np.linalg.norm(segments, axis=1).sum()
    coherence = np.linalg.norm(points[-1] - points[0]) / max(traveled, 1e-6)
    recent = points[-1] - points[-3]
    alignment = max(0, recent @ velocity / max(np.linalg.norm(recent) * speed, 1e-6))
    if coherence < 0.55 or alignment < 0.5:
        return None
    fit = np.clip(
        1 - 2 * residual / max(speed * (times[-1] - times[0]), 0.2 * height), 0, 1
    )
    reliability = float(coherence * alignment**2 * fit)
    speed_strength = float(np.clip((speed / height - 0.2) / 0.8, 0, 1))
    strength = speed_strength * (0.5 + 0.5 * reliability)
    return (
        {
            "direction": velocity / speed,
            "strength": strength,
            "angular_strength": speed_strength * reliability,
        }
        if strength >= 0.15
        else None
    )


def limb_cues(points):
    """Estimate a projected forward tendency from visible limb curvature."""
    torso = points[[5, 6, 11, 12]]
    if (torso[:, 2] < 0.55).any():
        return {}
    vertical = torso[2:, :2].mean(0) - torso[:2, :2].mean(0)
    length = np.linalg.norm(vertical)
    if length < 8:
        return {}
    vertical /= length
    groups = {"knees": ((11, 13, 15), (12, 14, 16)), "arms": ((5, 7, 9), (6, 8, 10))}
    cues = {}
    for name, triples in groups.items():
        vectors, strengths = [], []
        for indices in triples:
            joints = points[list(indices)]
            quality = float(joints[:, 2].min())
            if quality < 0.55:
                continue
            start, joint, end = joints[:, :2]
            chord = end - start
            chord_length = np.linalg.norm(chord)
            bone_length = (
                np.linalg.norm(joint - start) + np.linalg.norm(end - joint)
            ) / 2
            if chord_length < 5 or bone_length < 5 or chord_length < 0.6 * bone_length:
                continue
            projection = (
                start + np.clip((joint - start) @ chord / chord_length**2, 0, 1) * chord
            )
            bow = joint - projection
            bend = np.linalg.norm(bow)
            if bend < 2:
                continue
            # The knee bows forward. Elbow flexion gives the opposite tendency.
            direction = bow / bend * (1 if name == "knees" else -1)
            alignment = abs(chord @ vertical / chord_length)
            strength = (
                (quality - 0.55)
                / 0.45
                * np.clip((bend / bone_length - 0.12) / 0.30, 0, 1)
                * alignment**2
            )
            vectors.append(direction)
            strengths.append(strength)
        if strengths and sum(strengths) > 0:
            total = np.sum(np.asarray(vectors) * np.asarray(strengths)[:, None], axis=0)
            agreement = np.linalg.norm(total) / sum(strengths)
            if agreement > 0.25:
                cues[name] = {
                    "direction": total / np.linalg.norm(total),
                    "strength": min(1.0, sum(strengths)) * agreement,
                }
    return cues


def resolve_rays(proposals, cues):
    """Choose one direction only when geometry or corroborating cues separate it."""
    rays = np.asarray(proposals, dtype=float).reshape(-1, 5).copy()
    if not len(rays):
        return {"rays": [], "selected": False, "probability": 0.0, "sources": []}
    sources = [name for name, cue in cues.items() if cue["strength"] >= 0.15]
    if len(rays) == 1:
        return {
            "rays": rays.tolist(),
            "selected": True,
            "probability": 1.0,
            "sources": ["geometry"],
        }
    similarity = rays[:, 2:4] @ rays[:, 2:4].T
    if similarity.min() >= np.cos(np.deg2rad(20)):
        ray = np.average(rays, axis=0, weights=rays[:, 4])
        ray[2:4] /= np.linalg.norm(ray[2:4])
        return {
            "rays": [ray.tolist()],
            "selected": True,
            "probability": 1.0,
            "sources": ["geometry"],
        }
    scores = np.log(np.maximum(rays[:, 4], 1e-6))
    for name, cue in cues.items():
        scores += (
            EVIDENCE_WEIGHT[name] * cue["strength"] * (rays[:, 2:4] @ cue["direction"])
        )
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    winner = int(probabilities.argmax())
    real_sources = [name for name in sources if name != "temporal"]
    strong_knees = cues.get("knees", {}).get("strength", 0) >= 0.8
    corroborated = (
        len(real_sources) >= 2
        or strong_knees
        or (real_sources and "temporal" in sources)
    )
    selected = bool(probabilities[winner] >= 0.78 and corroborated)
    if selected:
        ray = rays[winner]
        raw_direction = ray[2:4].copy()
        ray[2:4], adjustment = refine_direction(raw_direction, cues)
        ray[4] *= probabilities[winner]
        output = [ray.tolist()]
    else:
        rays[:, 4] *= 0.5 + 0.5 * probabilities
        output = rays.tolist()
    return {
        "rays": output,
        "selected": selected,
        "probability": float(probabilities[winner]),
        "sources": sources,
        "angle_adjustment_degrees": adjustment if selected else 0.0,
        "raw_direction": raw_direction.tolist() if selected else None,
    }


class BodyEvidence:
    def __init__(self, size, view):
        self.size = np.asarray(size)
        self.view = np.asarray(view)
        self.tracker = PlayerTargets(size, view)
        self.previous = {}

    def observe(self, poses, center, timestamp):
        boxes, owners = [], []
        cues = [limb_cues(points) for points in poses]
        for owner, points in enumerate(poses):
            torso = points[[5, 6, 11, 12]]
            if (torso[:, 2] < 0.55).any():
                continue
            shoulder, hip = torso[:2, :2].mean(0), torso[2:, :2].mean(0)
            height = hip[1] - shoulder[1]
            if height < 8 or (np.abs(hip / self.size - center) > self.view / 2).any():
                continue
            box = np.array(
                [hip[0] - height / 2, shoulder[1], hip[0] + height / 2, hip[1]]
            )
            boxes.append([*(box / np.tile(self.size, 2)), float(torso[:, 2].mean())])
            owners.append(owner)
        rows = self.tracker.step(
            np.asarray(boxes).reshape(-1, 5), timestamp, center=center
        )
        track_ids = {}
        self.previous = {
            key: value
            for key, value in self.previous.items()
            if timestamp - value[1] <= 0.4
        }
        if rows is not None:
            for index, (owner, track_id) in enumerate(zip(owners, rows["track_ids"])):
                track_ids[owner] = track_id
                height = (boxes[index][3] - boxes[index][1]) * self.size[1]
                movement = motion_cue(self.tracker.histories[index], self.size, height)
                if movement is not None:
                    cues[owner]["motion"] = movement
                if track_id in self.previous:
                    direction, time, confidence = self.previous[track_id]
                    cues[owner]["temporal"] = {
                        "direction": direction,
                        "strength": confidence * np.exp(-(timestamp - time) / 0.3),
                    }
        return cues, track_ids

    def remember(self, result, track_id, timestamp):
        if track_id is not None and result["selected"]:
            direction = np.asarray(result["rays"][0][2:4])
            self.previous[track_id] = (direction, timestamp, result["probability"])
