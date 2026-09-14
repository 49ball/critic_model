"""All geometry is in metres, seconds, radians, in the ego frame at t0."""

import numpy as np


def trajectory_features(poses, dt, initial_speed):
    poses = np.asarray(poses, dtype=np.float64)
    origin = np.zeros((poses.shape[0], 1, 2))
    velocity = np.diff(np.concatenate([origin, poses[..., :2]], axis=1), axis=1) / dt
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = (
        np.diff(np.concatenate([np.full((len(poses), 1), initial_speed), speed], axis=1), axis=1)
        / dt
    )
    # No previous acceleration observation in v1: initial jerk is unknown, set to zero.
    jerk = np.diff(acceleration, axis=1, prepend=acceleration[:, :1]) / dt
    time = np.broadcast_to(np.arange(1, poses.shape[1] + 1) * dt, speed.shape)
    return np.stack(
        [
            poses[..., 0],
            poses[..., 1],
            np.sin(poses[..., 2]),
            np.cos(poses[..., 2]),
            speed,
            acceleration,
            jerk,
            time,
        ],
        -1,
    ).astype("float32")


def min_swept_clearance(poses, dt, agents, ego):
    """Conservative circumscribed discs; linear sweep between samples.
    Constant-velocity other agents. This is NOT a realistic collision oracle.
    """
    if not agents:
        return np.full(len(poses), 1000.0, dtype=np.float64)
    path = np.concatenate([np.zeros((len(poses), 1, 2)), poses[..., :2]], axis=1)
    times = np.arange(path.shape[1]) * dt
    agent_xy = np.array([[a["x"], a["y"]] for a in agents])
    velocity = np.array([[a["vx"], a["vy"]] for a in agents])
    future = agent_xy[None, :, :] + times[:, None, None] * velocity[None, :, :]
    relative = path[:, :, None, :] - future[None, :, :, :]
    start, delta = relative[:, :-1], np.diff(relative, axis=1)
    fraction = np.clip(-(start * delta).sum(-1) / np.maximum((delta * delta).sum(-1), 1e-12), 0, 1)
    distance = np.linalg.norm(start + fraction[..., None] * delta, axis=-1)
    radius = 0.5 * np.hypot(ego["length"], ego["width"])
    others = np.array([0.5 * np.hypot(a["length"], a["width"]) for a in agents])
    return (distance - radius - others[None, None, :]).min(axis=(1, 2))
