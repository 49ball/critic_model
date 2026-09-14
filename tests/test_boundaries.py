import copy
import numpy as np
import pytest
import torch
from critic_poc.alpamayo import merge_world
from critic_poc.data import collate_scenes
from critic_poc.model import TrajectoryCritic
from critic_poc.rules import score_rules, label_scene
from test_contract import scene


def test_swept_collision_between_waypoints():
    s = scene()
    s["dt"] = 1.0
    s["ego"]["speed"] = 20.0
    s["candidates"] = [[[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]]]
    s["agents"] = [dict(x=10.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, length=1.0, width=1.0)]
    assert score_rules(s)["risk"][0, 0] == 1


def test_ground_truth_future_is_never_model_input():
    a = scene()
    b = copy.deepcopy(a)
    b["future_outcome"] = {"x": [999, -999]}
    ba, bb = collate_scenes([a]), collate_scenes([b])
    for key in ba:
        torch.testing.assert_close(ba[key], bb[key])


def test_permuting_candidates_only_permutes_outputs():
    s = scene()
    s["candidates"].append((np.asarray(s["candidates"][0]) * 0.8).tolist())
    model = TrajectoryCritic(d_model=32, layers=1, dropout=0).eval()
    a = model(collate_scenes([s]))
    s["candidates"].reverse()
    b = model(collate_scenes([s]))
    for key in a:
        torch.testing.assert_close(a[key].flip(1), b[key], atol=1e-5, rtol=1e-5)


def test_world_sidecar_cannot_silently_mix_coordinate_frames():
    with pytest.raises(ValueError, match="frame"):
        merge_world(scene(), dict(scene_id="a", t0_us=1000000, frame="world"))


def test_relabel_does_not_mutate_old_labels():
    s = label_scene(scene())
    old = copy.deepcopy(s)
    _ = label_scene(s)
    assert s == old
