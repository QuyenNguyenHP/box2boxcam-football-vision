"""Named camera variants (parameters of policies.Planner; empty dict = current app rule). 'no_ball' = player branch only."""

VARIANTS = {
    "current": {},
    "guard": {"guard": True},
    "centroid": {"use_orientation": False, "base": "mean"},
    "centroid_motion": {"use_orientation": False, "base": "motion"},
    # orientation fixes (audit: axis ok, front/back wrong 16 % of fast movers, 59 % of movers left ambiguous)
    "orient_resolve": {"motion_mode": "resolve"},
    "orient_agree": {"motion_mode": "agree"},
    "short_memory": {"memory_hold": 0.6, "observe_hold": 1.5},
    "half_lead": {"lead_scale": 0.5},
    # predictive group motion instead of bodies
    "cm_vlead05": {"use_orientation": False, "base": "motion", "vel_lead": 0.5},
    "cm_vlead10": {"use_orientation": False, "base": "motion", "vel_lead": 1.0},
    "cm_guard": {"use_orientation": False, "base": "motion", "guard": True},
}
for k in list(VARIANTS):
    VARIANTS[k + "_nb"] = {**VARIANTS[k], "no_ball": True}
for vl in (0.3, 0.5, 0.7):
    for ar in (0.3, 0.45, 0.6):
        VARIANTS[f"cmv{int(vl * 10)}_ar{int(ar * 100)}"] = {
            "use_orientation": False,
            "base": "motion",
            "vel_lead": vl,
            "approach_rate": ar,
        }
VARIANTS["cmv5_mean"] = {"use_orientation": False, "base": "mean", "vel_lead": 0.5}
VARIANTS["cmv5_guard"] = {
    "use_orientation": False,
    "base": "motion",
    "vel_lead": 0.5,
    "guard": True,
}
VARIANTS["halflead_v5"] = {"lead_scale": 0.5, "vel_lead": 0.5}
# coordinator's survey candidates (cache TUNE)
NB = {"use_orientation": False}
VARIANTS.update(
    {
        "median_v5": {**NB, "base": "median", "vel_lead": 0.5},
        "density_v5": {**NB, "base": "density", "vel_lead": 0.5},
        "density8_v5": {**NB, "base": "density", "density_bw": 0.08, "vel_lead": 0.5},
        "speedw_v5": {**NB, "base": "speedw", "vel_lead": 0.5},
        "converge_v5": {**NB, "base": "converge", "vel_lead": 0.5},
        "cmv5_dz10": {**NB, "base": "motion", "vel_lead": 0.5, "dead_zone": 0.10},
        "cmv5_dz05": {**NB, "base": "motion", "vel_lead": 0.5, "dead_zone": 0.05},
        "cmv5_spd40": {**NB, "base": "motion", "vel_lead": 0.5, "max_speed": 0.40},
        "cmv5_few": {**NB, "base": "motion", "vel_lead": 0.5, "few_hold": True},
        "static": {"static": True},
        "oracle_ac": {"oracle": True},
    }
)
VARIANTS.update(
    {
        "guard_v5": {"guard": True, "vel_lead": 0.5},
        "guard_half_v5": {"guard": True, "lead_scale": 0.5, "vel_lead": 0.5},
    }
)
VARIANTS.update(
    {
        "guard_e50": {"guard": True, "g_enter": 0.5},
        "guard_e50_half": {"guard": True, "g_enter": 0.5, "lead_scale": 0.5},
    }
)
VARIANTS.update(
    {
        "guard_veto10": {"guard": True, "veto_speed": 0.10},
        "guard_veto05": {"guard": True, "veto_speed": 0.05},
        "veto10": {"veto_speed": 0.10},
    }
)
