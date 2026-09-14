"""Toy straight-corridor scenes. Validates plumbing, NOT real driving performance."""

from pathlib import Path
import numpy as np
from .rules import label_scene
from .schema import write_scenes, write_json


def generate_dataset(out, scenes=400, candidates=8, steps=32, seed=7):
    if scenes < 20 or not 2 <= candidates <= 128 or not 2 <= steps <= 256:
        raise ValueError("scenes>=20, 2<=candidates<=128, 2<=steps<=256 required")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    names = ("train", "val", "calibration", "test")
    paths = {name: root / f"{name}.jsonl" for name in names}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("Output dataset exists; choose a new output directory.")
    rng = np.random.default_rng(seed)
    samples = []
    dt = 0.1
    time = np.arange(1, steps + 1) * dt
    for i in range(scenes):
        v0 = float(rng.uniform(4, 16))
        lane = float(rng.uniform(3.2, 6))
        actors = []
        for _ in range(int(rng.integers(0, 4))):
            actors.append(
                dict(
                    x=float(rng.uniform(12, 60)),
                    y=float(rng.choice([0.0, 3.0, -3.0])),
                    yaw=0.0,
                    vx=float(rng.uniform(0, v0)),
                    vy=0.0,
                    length=4.5,
                    width=1.9,
                )
            )
        trajectories = []
        for j in range(candidates):
            acc = 0.0 if j == 0 else float(rng.choice([-4.0, -2.0, 0.0, 1.0, 3.0, 14.0]))
            speed = np.maximum(0.0, v0 + acc * time)
            x = np.cumsum(speed) * dt
            offset = 0.0 if j == 0 else float(rng.choice([0.0, 0.0, 1.8, -1.8, 7.0]))
            # Quintic lateral offset has zero initial/final velocity and acceleration.
            u = time / time[-1]
            y = offset * (10 * u**3 - 15 * u**4 + 6 * u**5)
            dx = np.diff(x, prepend=0.0)
            dy = np.diff(y, prepend=0.0)
            yaw = np.arctan2(dy, np.maximum(dx, 1e-6))
            trajectories.append(np.stack([x, y, yaw], -1).tolist())
        rng.shuffle(trajectories)
        sample = dict(
            schema_version=1,
            scene_id=f"toy-{seed}-{i}",
            group_id=f"toy-{seed}-{i}",
            domain="synthetic",
            frame="ego_t0",
            dt=dt,
            t0_us=1000000,
            ego=dict(speed=v0, length=4.8, width=2.0, desired_speed=v0),
            observation_complete=True,
            lane_half_width=lane,
            agents=actors,
            candidates=trajectories,
            provenance={"generator": "synthetic-v1", "seed": seed},
        )
        samples.append(label_scene(sample))
    rng.shuffle(samples)
    edges = [0, int(scenes * 0.65), int(scenes * 0.8), int(scenes * 0.9), scenes]
    for j, name in enumerate(names):
        write_scenes(paths[name], samples[edges[j] : edges[j + 1]])
    write_json(
        root / "manifest.json",
        {
            "seed": seed,
            "scenes": scenes,
            "candidates": candidates,
            "steps": steps,
            "counts": {n: edges[i + 1] - edges[i] for i, n in enumerate(names)},
            "notice": "Synthetic plumbing benchmark only; labels derived from rule scorer.",
        },
    )
    return paths
