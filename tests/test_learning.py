import copy
import torch
import pytest

from critic_poc.data import collate_scenes, check_disjoint
from critic_poc.model import TrajectoryCritic
from critic_poc.losses import critic_loss
from critic_poc.rules import label_scene
from test_contract import scene


def test_variable_candidates_empty_agents_and_padding_invariance():
    torch.manual_seed(2)
    s = label_scene(scene())
    t = copy.deepcopy(s)
    t["candidates"] = t["candidates"] * 2
    t["agents"] = [{"x": 20, "y": 3, "yaw": 0, "vx": 2, "vy": 0, "length": 4, "width": 2}]
    t = label_scene(t)
    model = TrajectoryCritic(d_model=32, layers=1, dropout=0).eval()
    one = model(collate_scenes([s]))
    batch = model(collate_scenes([s, t]))
    assert batch["risk_logits"].shape == (2, 2, 3)
    assert torch.isfinite(batch["utility"]).all()
    torch.testing.assert_close(one["utility"][0, 0], batch["utility"][0, 0], atol=1e-5, rtol=1e-5)


def test_unlabeled_targets_do_not_create_loss():
    s = label_scene(scene())
    batch = collate_scenes([s])
    model = TrajectoryCritic(d_model=32, layers=1, dropout=0)
    output = model(batch)
    batch["risk_mask"].zero_()
    batch["quality_mask"].zero_()
    batch["utility_mask"].zero_()
    batch["pairs"] = torch.empty((0, 4))
    loss, _ = critic_loss(output, batch)
    assert loss.item() == 0.0
    loss.backward()


def test_optimizer_reduces_loss_on_small_fixture():
    torch.manual_seed(3)
    batch = collate_scenes([label_scene(scene())])
    model = TrajectoryCritic(d_model=32, layers=1, dropout=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    start = critic_loss(model(batch), batch)[0].item()
    for _ in range(20):
        loss, _ = critic_loss(model(batch), batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert critic_loss(model(batch), batch)[0].item() < start * 0.5


def test_group_split_blocks_adjacent_frames():
    a = scene()
    b = copy.deepcopy(a)
    b["scene_id"] = "another-frame"
    with pytest.raises(ValueError, match="group"):
        check_disjoint([a], [b])
