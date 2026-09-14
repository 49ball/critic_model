import copy
import json
import pytest
from critic_poc.synthetic import generate_dataset
from critic_poc.training import train_model
from critic_poc.inference import evaluate_checkpoint, rank_scene
from critic_poc.schema import read_scenes
from critic_poc.metrics import binary_metrics


def test_train_save_reload_evaluate_and_domain_guard(tmp_path):
    paths = generate_dataset(tmp_path / "data", scenes=40, candidates=4, steps=16, seed=4)
    config = dict(
        epochs=2, batch_size=8, d_model=32, layers=1, dropout=0.1, lr=0.002, seed=3, threads=1
    )
    result = train_model(
        paths["train"],
        paths["val"],
        tmp_path / "run",
        config,
        calibration_path=paths["calibration"],
    )
    checkpoint = tmp_path / "run" / "best.pt"
    assert checkpoint.exists()
    assert result["epochs_completed"] == 2
    report = evaluate_checkpoint(checkpoint, paths["test"])
    assert report["scenes"] == 4
    assert 0 <= report["risk"]["dynamics"]["brier"] <= 1
    scene = read_scenes(paths["test"])[0]
    ranked = rank_scene(checkpoint, scene, mc_samples=3)
    assert len(ranked["candidates"]) == 4
    assert all(0 <= c["uncertainty"] <= 1 for c in ranked["candidates"])
    assert isinstance(json.dumps(ranked, allow_nan=False), str)
    other = copy.deepcopy(scene)
    other["domain"] = "physical_ai_av"
    with pytest.raises(ValueError, match="domain"):
        rank_scene(checkpoint, other)
    assert rank_scene(checkpoint, other, allow_domain_shift=True)["domain_shift"] is True
    with pytest.raises(ValueError, match="group"):
        evaluate_checkpoint(checkpoint, paths["train"])


def test_metrics_single_class_and_threshold_semantics():
    m = binary_metrics([0, 1], [0.2, 0.8])
    assert m["recall"] == 1.0
    assert m["false_positive_rate"] == 0.0
    assert m["brier"] == pytest.approx(0.04)
    assert binary_metrics([0, 0], [0.1, 0.2])["auroc"] is None
