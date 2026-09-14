import copy
import numpy as np
import pytest

from critic_poc.schema import validate_scene
from critic_poc.geometry import trajectory_features
from critic_poc.rules import score_rules, select_candidate
from critic_poc.alpamayo import normalize_predictions


def scene():
    return {
        "schema_version": 1,
        "scene_id": "a",
        "group_id": "log-a",
        "domain": "synthetic",
        "frame": "ego_t0",
        "dt": 0.1,
        "t0_us": 1000000,
        "ego": {"speed": 10.0, "length": 4.0, "width": 2.0, "desired_speed": 10.0},
        "observation_complete": True,
        "lane_half_width": 4.0,
        "agents": [],
        "candidates": [[[float(t), 0.0, 0.0] for t in range(1, 11)]],
    }


def test_nonfinite_and_bad_time_rejected():
    s = scene()
    s["dt"] = 0
    with pytest.raises(ValueError):
        validate_scene(s)
    s = scene()
    s["candidates"][0][0][0] = float("nan")
    with pytest.raises(ValueError):
        validate_scene(s)


def test_initial_speed_is_used_without_artificial_acceleration():
    s = scene()
    f = trajectory_features(np.asarray(s["candidates"]), 0.1, 10.0)
    np.testing.assert_allclose(f[..., 4], 10.0, atol=1e-5)
    np.testing.assert_allclose(f[..., 5:7], 0.0, atol=1e-4)


def test_alpamayo_axes_and_yaw_preserved():
    xyz = np.zeros((1, 1, 2, 4, 3))
    xyz[0, 0, 1, :, 0] = [1, 2, 3, 4]
    rot = np.broadcast_to(np.eye(3), (1, 1, 2, 4, 3, 3)).copy()
    rot[0, 0, 1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    result = normalize_predictions(xyz, rot)
    assert result.shape == (2, 4, 3)
    np.testing.assert_allclose(result[1, :, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(result[1, :, 2], np.pi / 2)
    with pytest.raises(ValueError):
        normalize_predictions(np.zeros((2, 1, 2, 4, 3)), np.zeros((2, 1, 2, 4, 3, 3)))


def test_collision_veto_overrides_high_utility():
    s = scene()
    s["agents"] = [
        {"x": 10.0, "y": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "length": 4.0, "width": 2.0}
    ]
    # Second candidate decelerates with physically modest braking, stops at x=5.
    t = np.arange(1, 11) * 0.1
    s["candidates"].append(np.stack([10 * t - 5 * t * t, t * 0, t * 0], -1).tolist())
    r = score_rules(s)
    assert r["risk"][0, 0] == 1
    result = select_candidate([100.0, 0.0], [[0, 0, 0], [0, 0, 0]], r, True, 0.5)
    # All candidates may fail conservative envelopes; never return colliding candidate.
    assert result["selected_index"] != 0


def test_all_rejected_and_missing_observation_abstain():
    s = scene()
    r = score_rules(s)
    assert select_candidate([1.0], [[1, 1, 1]], r, True, 0.5)["selected_index"] is None
    assert select_candidate([1.0], [[0, 0, 0]], r, False, 0.5)["selected_index"] is None
    s["observation_complete"] = False
    assert score_rules(s)["risk_mask"][0, 0] == 0


def test_missing_lane_is_not_labeled_compliant():
    s = scene()
    s["lane_half_width"] = None
    assert score_rules(s)["risk_mask"][0, 1] == 0


def test_future_labels_do_not_change_input_contract():
    s = scene()
    t = copy.deepcopy(s)
    t["future_outcome"] = {"collision": True}
    validate_scene(s)
    validate_scene(t)
