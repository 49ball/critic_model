"""Interpretable weak labels and conservative PoC vetoes, NOT safety certification."""

from copy import deepcopy
import numpy as np
from .schema import validate_scene
from .geometry import trajectory_features, min_swept_clearance

RULE_VERSION = "cv-disc-corridor-v1"


def score_rules(scene):
    validate_scene(scene)
    poses = np.asarray(scene["candidates"], dtype=np.float64)
    ego, dt = scene["ego"], scene["dt"]
    f = trajectory_features(poses, dt, ego["speed"])
    k, t = poses.shape[:2]
    clearance = min_swept_clearance(poses, dt, scene["agents"], ego)
    lane = scene.get("lane_half_width")
    # Oriented footprint lateral extent at samples; no curved-map semantics.
    extent = (
        np.abs(np.sin(poses[..., 2])) * ego["length"] + np.abs(np.cos(poses[..., 2])) * ego["width"]
    ) * 0.5
    road_margin = (
        np.full(k, 1000.0) if lane is None else lane - (np.abs(poses[..., 1]) + extent).max(axis=1)
    )
    yaw = np.unwrap(np.concatenate([np.zeros((k, 1)), poses[..., 2]], axis=1), axis=1)
    yaw_rate = np.diff(yaw, axis=1) / dt
    lat_acc = f[..., 4] * yaw_rate
    # Broad prototype bounds; not comfort thresholds or validated vehicle limits.
    dynamics = (
        (np.abs(f[..., 5]).max(1) > 10.0)
        | (np.abs(lat_acc).max(1) > 8.0)
        | (f[..., 4].max(1) > 55.0)
    )
    risk = np.stack([clearance < 0.5, road_margin < 0.0, dynamics], -1).astype("float32")
    risk_mask = np.ones_like(risk)
    risk_mask[:, 0] = float(scene["observation_complete"])
    risk_mask[:, 1] = float(lane is not None)
    comfort = np.exp(
        -np.mean(np.abs(f[..., 5]) / 3.0 + np.abs(f[..., 6]) / 10.0 + np.abs(lat_acc) / 3.0, axis=1)
    )
    # Straight-forward displacement ONLY, not route progress or social quality.
    progress = np.clip(poses[:, -1, 0] / max(ego["desired_speed"] * t * dt, 1.0), 0, 1)
    quality = np.stack([comfort, progress], -1).astype("float32")
    utility = 0.4 * comfort + 0.6 * progress
    return {
        "risk": risk,
        "risk_mask": risk_mask,
        "quality": quality,
        "utility": utility.astype("float32"),
        "clearance": clearance,
        "road_margin": road_margin,
        "feasible": ((risk * risk_mask).max(1) == 0) & risk_mask.all(1),
    }


def label_scene(scene):
    result = deepcopy(scene)
    result.pop("labels", None)  # Explicit relabeling discards stale candidate-aligned targets.
    rules = score_rules(result)
    pairs = []
    for a in range(len(rules["utility"])):
        for b in range(a + 1, len(rules["utility"])):
            if rules["feasible"][a] and rules["feasible"][b]:
                delta = rules["utility"][a] - rules["utility"][b]
                if abs(delta) >= 0.03:
                    pairs.append([a, b, float(delta > 0)])
    result["labels"] = {
        "source": "rules_cv",
        "version": RULE_VERSION,
        "risk": rules["risk"].tolist(),
        "risk_mask": rules["risk_mask"].tolist(),
        "quality": rules["quality"].tolist(),
        "quality_mask": np.ones_like(rules["quality"]).tolist(),
        "utility": rules["utility"].tolist(),
        "utility_mask": rules["feasible"].astype(float).tolist(),
        "pairs": pairs,
    }
    return result


def select_candidate(
    utility,
    probabilities,
    rules,
    observation_complete,
    threshold,
    uncertainty=None,
    max_uncertainty=0.15,
    beta=1.0,
):
    """Return no selected index if required checks are missing or all candidates fail."""
    utility = np.asarray(utility, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    uncertainty = np.zeros_like(utility) if uncertainty is None else np.asarray(uncertainty)
    if not (0 < threshold < 1) or max_uncertainty < 0:
        raise ValueError("Invalid selection thresholds")
    if probabilities.shape != (len(utility), 3) or uncertainty.shape != utility.shape:
        raise ValueError("Selection shape mismatch")
    finite = np.isfinite(utility) & np.isfinite(probabilities).all(1)
    finite &= np.isfinite(uncertainty)
    eligible = (
        rules["feasible"]
        & finite
        & observation_complete
        & (probabilities.max(1) < threshold)
        & (uncertainty <= max_uncertainty)
    )
    adjusted = np.where(eligible, utility - beta * uncertainty, -np.inf)
    order = np.flatnonzero(eligible)
    order = order[np.argsort(-adjusted[order], kind="stable")]
    selected = int(order[0]) if len(order) else None
    return {
        "selected_index": selected,
        "eligible_indices": order.tolist(),
        "abstained": selected is None,
        "requires_external_planner": True,
        "status": "advisory_only" if selected is not None else "abstain",
        "reason": (
            "proxy_checks_passed"
            if selected is not None
            else "missing_observation_or_check_or_no_candidate_passes"
        ),
    }
