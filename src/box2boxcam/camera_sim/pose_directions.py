"""Pose-derived direction rays (torso normals, head pose, convergence) for the simulated viewport."""

import cv2
import numpy as np

from box2boxcam.camera_sim.body_evidence import (
    resolve_rays,
)

# Approximate facial geometry in arbitrary units, ordered as COCO nose, eyes, ears.
# This template is a geometric hypothesis, not a trained gaze estimator.
FACE = np.array(
    [[0, 0, 0], [22, -18, 17], [-22, -18, 17], [65, -8, 55], [-65, -8, 55]],
    dtype=np.float64,
)
TORSO = np.array(
    [[0.20, 0, 0], [-0.20, 0, 0], [0.14, 0.50, 0], [-0.14, 0.50, 0]], dtype=float
)
SKELETON = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def head_ray(points, image_size, horizontal_fov=60):
    face = points[:5]
    valid = face[:, 2] >= 0.55
    if not valid[0] or valid.sum() < 4:
        return None
    observed = face[valid, :2].astype(np.float64)
    span = float(np.linalg.norm(np.ptp(observed, axis=0)))
    # Enlarging the pose input does not create new face detail in the source.
    if span < 8:
        return None
    width, height = image_size
    focal = width / (2 * np.tan(np.deg2rad(horizontal_fov) / 2))
    camera = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=float
    )
    try:
        ok, rotation, translation = cv2.solvePnP(
            FACE[valid], observed, camera, None, flags=cv2.SOLVEPNP_SQPNP
        )
    except cv2.error:
        return None
    if not ok or translation[2, 0] <= 0:
        return None
    reconstructed, _ = cv2.projectPoints(
        FACE[valid], rotation, translation, camera, None
    )
    error = float(np.sqrt(np.square(reconstructed[:, 0] - observed).sum(1).mean()))
    if error > max(1.0, 0.12 * span):
        return None
    axis, _ = cv2.projectPoints(
        np.array([[0, 0, 0], [0, 0, -translation[2, 0] * 0.2]]),
        rotation,
        translation,
        camera,
        None,
    )
    direction = axis[1, 0] - axis[0, 0]
    length = np.linalg.norm(direction)
    if length < 0.03 * width:
        return None
    confidence = float(face[valid, 2].mean() * np.exp(-error / span))
    return np.r_[axis[0, 0], direction / length, confidence]


def body_rays(points, image_size, horizontal_fov=60):
    """Project plausible torso normals, retaining the two planar pose hypotheses."""
    joints = points[[5, 6, 11, 12]]
    if (
        (joints[:, 2] < 0.55).any()
        or (joints[:, :2] < 0).any()
        or (joints[:, :2] > image_size).any()
    ):
        return []
    observed = joints[:, :2].astype(float)
    span = np.linalg.norm(np.ptp(observed, axis=0))
    if span < 12 or np.linalg.norm(observed[0] - observed[1]) < 2:
        return []
    width, height = image_size
    focal = width / (2 * np.tan(np.deg2rad(horizontal_fov) / 2))
    camera = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=float
    )
    try:
        ok, rotations, translations, _ = cv2.solvePnPGeneric(
            TORSO, observed, camera, None, flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        return []
    if not ok:
        return []
    proposals = []
    for rotation, translation in zip(rotations, translations):
        if translation[2, 0] <= 0:
            continue
        projected, _ = cv2.projectPoints(TORSO, rotation, translation, camera, None)
        error = float(np.sqrt(np.square(projected[:, 0] - observed).sum(1).mean()))
        if error > max(1, span * 0.12):
            continue
        axis, _ = cv2.projectPoints(
            np.array([[0, 0.25, 0], [0, 0.25, -translation[2, 0] * 0.2]]),
            rotation,
            translation,
            camera,
            None,
        )
        direction = axis[1, 0] - axis[0, 0]
        length = np.linalg.norm(direction)
        if length >= 0.03 * width:
            confidence = float(joints[:, 2].mean() * np.exp(-error / span))
            proposals.append(np.r_[axis[0, 0], direction / length, confidence])
    return proposals


def convergence(rays, lower, upper, owners=None):
    """Find a forward intersection supported by at least three nonparallel rays."""
    rays = np.asarray(rays, dtype=float).reshape(-1, 5)
    if len(rays) < 3:
        return None
    owners = np.arange(len(rays)) if owners is None else np.asarray(owners)
    origins, directions, weights = rays[:, :2], rays[:, 2:4], rays[:, 4]
    best_support, best_score = None, 0.0
    for i in range(len(rays)):
        for j in range(i):
            if owners[i] == owners[j]:
                continue
            matrix = np.column_stack((directions[i], -directions[j]))
            if abs(np.linalg.det(matrix)) < np.sin(np.deg2rad(15)):
                continue
            distances = np.linalg.solve(matrix, origins[j] - origins[i])
            point = origins[i] + distances[0] * directions[i]
            if min(distances) < 0 or (point < lower).any() or (point > upper).any():
                continue
            difference = point - origins
            distance = np.linalg.norm(difference, axis=1)
            cosine = (difference * directions).sum(1) / np.maximum(distance, 1e-6)
            eligible = (cosine >= np.cos(np.deg2rad(18))) & (distance >= 8)
            support = np.zeros(len(rays), dtype=bool)
            for owner in np.unique(owners[eligible]):
                indices = np.flatnonzero(eligible & (owners == owner))
                support[indices[np.argmax(weights[indices] * cosine[indices])]] = True
            support_score = float(weights[support].sum())
            if support.sum() >= 3 and support_score > best_score:
                best_support, best_score = support, support_score
    if best_support is None:
        return None
    normals = np.eye(2)[None] - directions[:, :, None] * directions[:, None, :]
    weighted = normals[best_support] * weights[best_support, None, None]
    matrix = weighted.sum(0)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] / eigenvalues[-1] < 0.035:
        return None
    point = np.linalg.solve(matrix, (weighted @ origins[best_support, :, None]).sum(0))[
        :, 0
    ]
    forward = ((point - origins[best_support]) * directions[best_support]).sum(1)
    if (forward < 0).any() or (point < lower).any() or (point > upper).any():
        return None
    return point


def direction_row(
    poses,
    origin,
    image_size,
    center,
    size,
    view,
    evidence=None,
    timestamp=0,
    horizontal_fov=60,
):
    rays, owners, head_directions, selected, cue_rows = [], [], [], [], []
    cues, track_ids = (
        evidence.observe(poses, center, timestamp) if evidence is not None else ({}, {})
    )
    for owner, points in enumerate(poses):
        local = points.copy()
        local[:, :2] -= origin
        proposals = body_rays(local, image_size, horizontal_fov=horizontal_fov)
        if evidence is not None:
            result = resolve_rays(proposals, cues[owner])
            evidence.remember(result, track_ids.get(owner), timestamp)
            proposals = result["rays"]
            cue_rows.append(
                {
                    "owner": owner,
                    "selected": result["selected"],
                    "probability": result["probability"],
                    "sources": result["sources"],
                    "angle_adjustment_degrees": result.get(
                        "angle_adjustment_degrees", 0
                    ),
                    "raw_direction": result.get("raw_direction"),
                    "motion": None
                    if "motion" not in cues[owner]
                    else {
                        "direction": cues[owner]["motion"]["direction"].tolist(),
                        "strength": cues[owner]["motion"]["strength"],
                    },
                }
            )
        if len(proposals) == 1:
            selected.append(owner)
        for ray in proposals:
            ray = np.asarray(ray, dtype=float)
            ray[:2] += origin
            rays.append(ray)
            owners.append(owner)
        head = head_ray(local, image_size, horizontal_fov=horizontal_fov)
        if head is not None:
            head[:2] += origin
            head_directions.append(head)
    intersection = convergence(
        rays,
        np.maximum(0, (center - view) * size),
        np.minimum(size, (center + view) * size),
        owners,
    )
    return {
        "center": center.tolist(),
        "poses": [points.tolist() for points in poses],
        "rays": [ray.tolist() for ray in rays],
        "owners": owners,
        "unidirectional_owners": selected,
        "evidence": cue_rows,
        "head_rays": [ray.tolist() for ray in head_directions],
        "intersection": None if intersection is None else intersection.tolist(),
    }
