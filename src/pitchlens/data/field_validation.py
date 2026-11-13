"""Strict human-review parsing for an all-ball field development benchmark."""

from __future__ import annotations

from pitchlens.data.label_studio_review import _box_from_result

# The choice values below (French) are persisted in existing Label Studio
# exports and must match them exactly: do not translate. In English:
# Actif = active, Secondaire = secondary, visible/masqué/hors champ/indéterminé
# = visible/occluded/out of frame/undetermined, Vérifié = verified,
# Image inexploitable = unusable image, Identité vérifiée = identity verified.
ACTIVE_STATES = {
    "Actif visible",
    "Actif masqué",
    "Actif hors champ",
    "Actif indéterminé",
}
ROLES = {"Actif", "Secondaire", "Rôle indéterminé"}
SUBMIT_PROTOCOL = "submit_boxes_v2"


def proposed_role(ball: dict) -> str:
    prefix = ball.get("temporal_evidence", "").split(":", 1)[0].strip().upper()
    return {"ACTIF": "Actif", "SECONDAIRE": "Secondaire"}.get(
        prefix, "Rôle indéterminé"
    )


def choice(results: list[dict], name: str) -> str | None:
    values = [r["value"]["choices"] for r in results if r.get("from_name") == name]
    if not values:
        return None
    if len(values) != 1 or len(values[0]) != 1:
        raise ValueError(f"Expected exactly one choice for {name}")
    return values[0][0]


def parse_reviewed_task(task: dict) -> dict | None:
    """Never treat predictions, unsubmitted drafts or skipped tasks as labels."""
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not annotations:
        return None
    if len(annotations) != 1:
        raise ValueError("Multiple submitted reviews require explicit adjudication")
    data = task["data"]
    results = annotations[0].get("result", [])
    single_submit = data.get("review_protocol") == SUBMIT_PROTOCOL
    legacy_review = not single_submit or annotations[0]["id"] in data.get(
        "legacy_active_annotation_ids", []
    )
    status = choice(results, "review_status")
    if status is None and single_submit and not legacy_review:
        # A submitted annotation is the user's explicit confirmation. Predictions
        # and autosaved drafts still return None above, even with this protocol.
        status = "Vérifié"
    if status not in {"Vérifié", "Image inexploitable"}:
        raise ValueError("Human completeness decision missing")
    record = {
        key: data[key]
        for key in (
            "item_id",
            "match_id",
            "video_id",
            "source_video",
            "source_video_sha256",
            "source_frame",
            "timestamp_seconds",
            "width",
            "height",
            "image_sha256",
        )
    }
    record.update(
        annotation_id=annotations[0]["id"],
        label_source="submitted_human",
        annotation_scope="all_visible",
        split="val",
        boxes_pixels=[],
        roles=[],
        active_box_pixels=None,
        review_confirmation="review_status" if legacy_review else "submit",
        active_identity_reviewed=False,
    )
    if status == "Image inexploitable":
        return record | {"status": "excluded", "active_state": "Actif indéterminé"}
    state = choice(results, "active_state")
    active_reviewed = legacy_review or (
        choice(results, "active_review_status") == "Identité vérifiée"
    )
    if active_reviewed and state not in ACTIVE_STATES:
        raise ValueError("Active-ball visibility must be explicit")
    rectangles = [
        r
        for r in results
        if r.get("from_name") == "bbox" and r.get("type") == "rectangle"
    ]
    ids = [r["id"] for r in rectangles]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate box region IDs")
    if active_reviewed and any(
        r.get("id") not in ids for r in results if r.get("from_name") == "ball_role"
    ):
        raise ValueError("Ball role without a matching box region")
    active = []
    for rectangle in rectangles:
        region_results = [r for r in results if r.get("id") == rectangle["id"]]
        role = choice(region_results, "ball_role")
        if active_reviewed and role not in ROLES:
            raise ValueError("Each real ball needs a role, including uncertain roles")
        box = _box_from_result(rectangle, int(data["width"]), int(data["height"]))
        record["boxes_pixels"].append(box)
        record["roles"].append(role if active_reviewed else "Rôle indéterminé")
        if active_reviewed and role == "Actif":
            active.append(box)
    if active_reviewed and (
        (state == "Actif visible" and len(active) != 1)
        or (state != "Actif visible" and active)
    ):
        raise ValueError("Active visibility and the number of active boxes disagree")
    return record | {
        "status": "positive" if rectangles else "negative",
        "active_state": state if active_reviewed else "Actif indéterminé",
        "active_box_pixels": active[0] if active else None,
        "active_identity_reviewed": active_reviewed,
    }
