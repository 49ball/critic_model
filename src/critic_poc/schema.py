"""Versioned data contract. Only t0 observations enter the network."""

import json
from pathlib import Path
import numpy as np

RISK_NAMES = ("collision_cv", "offroad_corridor", "dynamics")
QUALITY_NAMES = ("comfort", "progress_proxy")


def finite_number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid {name}: {value}")
    return float(value)


def validate_scene(scene):
    if scene.get("schema_version") != 1 or scene.get("frame") != "ego_t0":
        raise ValueError("Expected schema_version=1, frame=ego_t0")
    for key in ("scene_id", "group_id", "domain"):
        if not isinstance(scene.get(key), str) or not scene[key]:
            raise ValueError(f"{key} must be a nonempty string")
    finite_number(scene.get("dt"), "dt", 0.001)
    if scene["dt"] > 1:
        raise ValueError("dt must be <= 1 second")
    finite_number(scene.get("t0_us"), "t0_us", 0)
    if type(scene.get("observation_complete")) is not bool:
        raise ValueError("observation_complete must be explicit boolean")
    ego = scene.get("ego", {})
    for key in ("speed", "length", "width", "desired_speed"):
        finite_number(ego.get(key), f"ego.{key}", 0 if key == "speed" else 0.01)
    lane = scene.get("lane_half_width")
    if lane is not None:
        finite_number(lane, "lane_half_width", 0.1)
    agents = scene.get("agents")
    if not isinstance(agents, list) or len(agents) > 128:
        raise ValueError("agents must be a list with at most 128 entries")
    for agent in agents:
        for key in ("x", "y", "yaw", "vx", "vy", "length", "width"):
            finite_number(
                agent.get(key), f"agent.{key}", 0.01 if key in ("length", "width") else None
            )
    try:
        poses = np.asarray(scene["candidates"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidates must have shape [K,T,3]") from exc
    if (
        poses.ndim != 3
        or poses.shape[-1] != 3
        or not 1 <= poses.shape[0] <= 128
        or not 2 <= poses.shape[1] <= 256
    ):
        raise ValueError("Expected candidates [K<=128, 2<=T<=256, 3]")
    if not np.isfinite(poses).all():
        raise ValueError("candidates contain NaN/Inf")
    labels = scene.get("labels")
    if labels is not None:
        if not labels.get("source"):
            raise ValueError("labels.source required (rules_cv / simulation / human)")
        k = len(poses)
        for key, shape in (
            ("risk", (k, 3)),
            ("risk_mask", (k, 3)),
            ("quality", (k, 2)),
            ("quality_mask", (k, 2)),
            ("utility", (k,)),
            ("utility_mask", (k,)),
        ):
            value = np.asarray(labels.get(key), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"labels.{key} must be finite, shape {shape}")
            if np.any((value < 0) | (value > 1)):
                raise ValueError(f"labels.{key} must be in [0,1]")
            if key.endswith("mask") and not np.isin(value, [0, 1]).all():
                raise ValueError(f"labels.{key} must be binary")
        for pair in labels.get("pairs", []):
            if len(pair) != 3:
                raise ValueError("pair = [candidate_a, candidate_b, preference_a]")
            a, b, pref = pair
            if (
                type(a) is not int
                or type(b) is not int
                or a == b
                or min(a, b) < 0
                or max(a, b) >= k
            ):
                raise ValueError("Invalid preference candidate indices")
            finite_number(pref, "preference", 0)
            if pref > 1:
                raise ValueError("preference must be in [0,1]")
    return scene


def read_scenes(path):
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result.append(validate_scene(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not result:
        raise ValueError(f"Empty dataset: {path}")
    ids = [s["scene_id"] for s in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scene_id in dataset")
    return result


def write_scenes(path, scenes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for scene in scenes:
            validate_scene(scene)
            handle.write(json.dumps(scene, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
