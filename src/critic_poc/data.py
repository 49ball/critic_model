"""Batching with explicit agent, candidate, temporal and label masks."""

import numpy as np
import torch
from .schema import validate_scene
from .geometry import trajectory_features


def check_disjoint(left, right):
    overlap = {s["group_id"] for s in left} & {s["group_id"] for s in right}
    if overlap:
        raise ValueError(f"group_id leakage across splits: {sorted(overlap)[:5]}")


def collate_scenes(scenes):
    for scene in scenes:
        validate_scene(scene)
    b = len(scenes)
    k = max(len(s["candidates"]) for s in scenes)
    t = max(len(s["candidates"][0]) for s in scenes)
    n = max(1, max(len(s["agents"]) for s in scenes))
    batch = {
        "agents": torch.zeros(b, n, 8),
        "agent_mask": torch.zeros(b, n, dtype=torch.bool),
        "ego": torch.zeros(b, 7),
        "trajectory": torch.zeros(b, k, t, 8),
        "time_mask": torch.zeros(b, k, t, dtype=torch.bool),
        "candidate_mask": torch.zeros(b, k, dtype=torch.bool),
        "risk": torch.zeros(b, k, 3),
        "risk_mask": torch.zeros(b, k, 3),
        "quality": torch.zeros(b, k, 2),
        "quality_mask": torch.zeros(b, k, 2),
        "utility": torch.zeros(b, k),
        "utility_mask": torch.zeros(b, k),
    }
    pairs = []
    for i, scene in enumerate(scenes):
        ego, lane = scene["ego"], scene.get("lane_half_width")
        batch["ego"][i] = torch.tensor(
            [
                ego["speed"],
                ego["length"],
                ego["width"],
                ego["desired_speed"],
                lane or 0.0,
                float(lane is not None),
                float(scene["observation_complete"]),
            ]
        )
        for j, a in enumerate(scene["agents"]):
            batch["agents"][i, j] = torch.tensor(
                [
                    a["x"],
                    a["y"],
                    np.sin(a["yaw"]),
                    np.cos(a["yaw"]),
                    a["vx"],
                    a["vy"],
                    a["length"],
                    a["width"],
                ]
            )
            batch["agent_mask"][i, j] = True
        f = trajectory_features(scene["candidates"], scene["dt"], ego["speed"])
        ki, ti = f.shape[:2]
        batch["trajectory"][i, :ki, :ti] = torch.from_numpy(f)
        batch["time_mask"][i, :ki, :ti] = True
        batch["candidate_mask"][i, :ki] = True
        labels = scene.get("labels")
        if labels:
            for name in ("risk", "risk_mask", "quality", "quality_mask", "utility", "utility_mask"):
                batch[name][i, :ki] = torch.tensor(labels[name])
            for a, c, pref in labels.get("pairs", []):
                pairs.append([i, a, c, pref])
    batch["pairs"] = torch.tensor(pairs, dtype=torch.float32).reshape(-1, 4)
    return batch


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
