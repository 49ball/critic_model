"""Evaluation and advisory reranking; no vehicle controller or fallback generator."""

import numpy as np
import torch
from .data import collate_scenes, to_device
from .metrics import binary_metrics
from .rules import score_rules, select_candidate
from .schema import read_scenes, RISK_NAMES, QUALITY_NAMES, validate_scene
from .training import load_checkpoint, device_for


def predict(model, checkpoint, scene, device, mc_samples=1):
    if mc_samples < 1 or mc_samples > 100:
        raise ValueError("mc_samples must be in [1,100]")
    batch = to_device(collate_scenes([scene]), device)
    draws = []
    model.eval()
    if mc_samples > 1:
        model.train()  # Only dropout changes; network has no BatchNorm.
    temperatures = torch.tensor(checkpoint["temperatures"], device=device)
    with torch.inference_mode():
        for _ in range(mc_samples):
            output = model(batch)
            draws.append({**output, "risk": (output["risk_logits"] / temperatures).sigmoid()})
    model.eval()
    risk = torch.stack([d["risk"] for d in draws])[:, 0]
    utility = torch.stack([d["utility"] for d in draws])[:, 0]
    quality = torch.stack([d["quality"] for d in draws])[:, 0]
    uncertainty = torch.maximum(
        risk.std(0, unbiased=False).amax(-1), utility.std(0, unbiased=False)
    )
    return {
        "risk": risk.mean(0).cpu().numpy(),
        "utility": utility.mean(0).cpu().numpy(),
        "quality": quality.mean(0).cpu().numpy(),
        "uncertainty": uncertainty.cpu().numpy(),
    }


def check_domain(checkpoint, scene, allow):
    shift = scene["domain"] not in checkpoint["domains"]
    if shift and not allow:
        raise ValueError(
            f"domain {scene['domain']} not in checkpoint {checkpoint['domains']}; "
            "use --allow-domain-shift for an explicitly experimental report"
        )
    return shift


def rank_scene(
    checkpoint_path,
    scene,
    device="cpu",
    threshold=0.5,
    mc_samples=8,
    allow_domain_shift=False,
    max_uncertainty=0.15,
):
    validate_scene(scene)
    device = device_for(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    shift = check_domain(checkpoint, scene, allow_domain_shift)
    prediction = predict(model, checkpoint, scene, device, mc_samples)
    rules = score_rules(scene)
    result = select_candidate(
        prediction["utility"],
        prediction["risk"],
        rules,
        scene["observation_complete"],
        threshold,
        prediction["uncertainty"],
        max_uncertainty,
    )
    missing = [RISK_NAMES[h] for h in range(3) if not checkpoint["risk_supported"][h]]
    if missing or shift:
        result.update(
            selected_index=None,
            eligible_indices=[],
            abstained=True,
            status="abstain",
            reason="unsupported_risk_heads_or_domain_shift",
        )
    # Report quality order even on abstention; never call it a safe trajectory.
    order = np.argsort(-prediction["utility"], kind="stable").tolist()
    result.update(
        scene_id=scene["scene_id"],
        domain_shift=shift,
        quality_order=order,
        unsupported_heads=missing,
        calibration_status=checkpoint["calibration_status"],
        uncertainty_method="MC dropout heuristic" if mc_samples > 1 else "not_estimated",
        notice="Research advisory only. CV/corridor proxies; no real collision probability.",
        candidates=[
            {
                "index": i,
                "utility": float(prediction["utility"][i]),
                "risk_proxy": dict(zip(RISK_NAMES, map(float, prediction["risk"][i]))),
                "quality": dict(zip(QUALITY_NAMES, map(float, prediction["quality"][i]))),
                "uncertainty": float(prediction["uncertainty"][i]),
                "rule_risk": dict(zip(RISK_NAMES, map(float, rules["risk"][i]))),
                "rule_known": dict(zip(RISK_NAMES, map(bool, rules["risk_mask"][i]))),
                "rule_feasible": bool(rules["feasible"][i]),
                "cv_clearance_m": float(rules["clearance"][i]) if scene["agents"] else None,
            }
            for i in range(len(scene["candidates"]))
        ],
    )
    return result


def evaluate_checkpoint(
    checkpoint_path,
    data_path,
    device="cpu",
    allow_seen_groups=False,
    allow_domain_shift=False,
    threshold=0.5,
):
    device = device_for(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    scenes = read_scenes(data_path)
    seen = set().union(*(set(v) for v in checkpoint["groups"].values()))
    overlap = {s["group_id"] for s in scenes} & seen
    if overlap and not allow_seen_groups:
        raise ValueError(
            "Evaluation group leakage into train/val/calibration; "
            "--allow-seen-groups is diagnostic only"
        )
    ys = [[] for _ in range(3)]
    ps = [[] for _ in range(3)]
    errors = [[] for _ in range(2)]
    regrets, baseline_regrets = [], []
    pairs_correct = pairs_total = selected = 0
    safety_violations = labeled_selected = 0
    for scene in scenes:
        shift = check_domain(checkpoint, scene, allow_domain_shift)
        if "labels" not in scene:
            raise ValueError(
                "Evaluation requires outcome labels; run label or import external labels."
            )
        pred = predict(model, checkpoint, scene, device)
        labels = scene["labels"]
        target = np.asarray(labels["risk"])
        mask = np.asarray(labels["risk_mask"], dtype=bool)
        # Event-count metrics require complete binary outcomes; missing/soft labels
        # remain usable for per-head Brier/NLL but do not mean zero violations.
        observed = mask.all(1) & np.isin(target, [0, 1]).all(1)
        utility_known = np.asarray(labels["utility_mask"], dtype=bool)
        for h in range(3):
            ys[h].extend(target[mask[:, h], h].tolist())
            ps[h].extend(pred["risk"][mask[:, h], h].tolist())
        for h in range(2):
            valid = np.asarray(labels["quality_mask"])[:, h].astype(bool)
            error = np.abs(pred["quality"][:, h] - np.asarray(labels["quality"])[:, h])
            errors[h].extend(error[valid].tolist())
        rules = score_rules(scene)
        decision = select_candidate(
            pred["utility"], pred["risk"], rules, scene["observation_complete"], threshold
        )
        idx = decision["selected_index"]
        if shift or not all(checkpoint["risk_supported"]):
            idx = None
        if idx is not None:
            selected += 1
            if observed[idx]:
                labeled_selected += 1
                safety_violations += int(target[idx].any())
            u = np.asarray(labels["utility"])
            # Oracle constrained by observed evaluation labels, not learned output.
            eligible = (target * mask).max(1) == 0
            eligible &= observed & utility_known
            if eligible.any() and eligible[idx]:
                regrets.append(float(u[eligible].max() - u[idx]))
            elif eligible.any() and observed[idx] and target[idx].any():
                regrets.append(float(u[eligible].max()))
        # Fixed candidate zero baseline, same label feasibility condition.
        u = np.asarray(labels["utility"])
        eligible = (target * mask).max(1) == 0
        eligible &= observed & utility_known
        if eligible.any() and observed[0] and (utility_known[0] or target[0].any()):
            baseline_regrets.append(float(u[eligible].max() - (u[0] if eligible[0] else 0.0)))
        for a, b, pref in labels.get("pairs", []):
            if pref == 0.5:
                continue
            pairs_total += 1
            pairs_correct += int((pred["utility"][a] > pred["utility"][b]) == (pref > 0.5))
    return {
        "scenes": len(scenes),
        "label_sources": sorted({s["labels"]["source"] for s in scenes}),
        "risk": {
            name: binary_metrics(ys[h], ps[h], threshold) for h, name in enumerate(RISK_NAMES)
        },
        "quality_mae": {
            name: float(np.mean(errors[h])) if errors[h] else None
            for h, name in enumerate(QUALITY_NAMES)
        },
        "preference_accuracy": pairs_correct / pairs_total if pairs_total else None,
        "preference_pairs": pairs_total,
        "selection_rate": selected / len(scenes),
        "abstention_rate": 1 - selected / len(scenes),
        "selected_label_violation_count": safety_violations,
        "selected_with_complete_binary_labels": labeled_selected,
        "selected_label_coverage": labeled_selected / selected if selected else None,
        "selected_label_violation_rate": safety_violations / labeled_selected
        if labeled_selected
        else None,
        "regret_scene_count": len(regrets),
        "fixed_first_regret_scene_count": len(baseline_regrets),
        "top1_regret_on_selected_labeled_scenes": float(np.mean(regrets)) if regrets else None,
        "fixed_first_regret_on_oracle_feasible_scenes": float(np.mean(baseline_regrets))
        if baseline_regrets
        else None,
        "seen_groups": bool(overlap),
        "evaluation_uncertainty": "disabled for deterministic evaluation",
        "notice": "Offline proxy benchmark. Selection-conditioned regret needs coverage context. "
        "Rule-labeled metrics measure proxy imitation, not real safety improvement.",
    }
