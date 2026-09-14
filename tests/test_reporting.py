import copy
import torch
from critic_poc.model import TrajectoryCritic
from critic_poc.inference import evaluate_checkpoint, rank_scene
from critic_poc.rules import label_scene
from critic_poc.schema import write_scenes
from test_contract import scene


def constant_checkpoint(path):
    model = TrajectoryCritic(d_model=32, layers=1, dropout=0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save(
        {
            "format_version": 1,
            "model_config": model.config,
            "model": model.state_dict(),
            "domains": ["synthetic"],
            "risk_supported": [True, True, True],
            "temperatures": [1.0, 1.0, 1.0],
            "calibration_status": ["not_calibrated"] * 3,
            "groups": {"train": [], "val": [], "calibration": []},
        },
        path,
    )


def test_unknown_selected_outcome_has_null_violation_rate_and_regret(tmp_path):
    checkpoint = tmp_path / "constant.pt"
    constant_checkpoint(checkpoint)
    s = scene()
    s["candidates"] *= 2
    s = label_scene(s)
    s["labels"]["risk_mask"][0] = [0, 0, 0]
    s["labels"]["utility_mask"][0] = 0
    write_scenes(tmp_path / "unknown.jsonl", [s])
    report = evaluate_checkpoint(checkpoint, tmp_path / "unknown.jsonl", threshold=0.9)
    assert report["selection_rate"] == 1.0
    assert report["selected_label_violation_rate"] is None
    assert report["selected_label_coverage"] == 0.0
    assert report["top1_regret_on_selected_labeled_scenes"] is None
    assert report["fixed_first_regret_on_oracle_feasible_scenes"] is None


def test_domain_shift_abstention_clears_eligible_indices(tmp_path):
    checkpoint = tmp_path / "constant.pt"
    constant_checkpoint(checkpoint)
    s = copy.deepcopy(scene())
    s["domain"] = "physical_ai_av"
    report = rank_scene(checkpoint, s, threshold=0.9, allow_domain_shift=True, mc_samples=1)
    assert report["selected_index"] is None
    assert report["eligible_indices"] == []
    assert report["quality_order"] == [0]
