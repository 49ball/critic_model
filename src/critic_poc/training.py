"""Supervised and pairwise training with held-out temperature calibration."""

import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import check_disjoint, collate_scenes, to_device
from .losses import critic_loss
from .model import TrajectoryCritic
from .schema import read_scenes, write_json

DEFAULT_CONFIG = dict(
    epochs=10,
    batch_size=16,
    d_model=64,
    layers=2,
    dropout=0.1,
    lr=0.001,
    weight_decay=0.0001,
    seed=7,
    threads=2,
)


def device_for(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N, auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    return device


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = TrajectoryCritic(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def fit_temperatures(model, scenes, device):
    if not scenes:
        return [1.0, 1.0, 1.0], ["not_calibrated"] * 3
    logits, targets, masks = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(scenes, batch_size=8, collate_fn=collate_scenes):
            batch = to_device(batch, device)
            logits.append(model(batch)["risk_logits"].reshape(-1, 3).cpu())
            targets.append(batch["risk"].reshape(-1, 3).cpu())
            masks.append(batch["risk_mask"].reshape(-1, 3).cpu())
    logits, targets, masks = [torch.cat(v) for v in (logits, targets, masks)]
    temps, status = [], []
    for h in range(3):
        mask = masks[:, h].bool()
        x, y = logits[mask, h], targets[mask, h]
        if len(y) < 10 or not ((y >= 0.5).any() and (y < 0.5).any()):
            temps.append(1.0)
            status.append("insufficient_two_class_calibration_data")
            continue
        candidates = torch.logspace(-1, 1.3, 80)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            x[:, None] / candidates[None, :],
            y[:, None].expand(-1, len(candidates)),
            reduction="none",
        ).mean(0)
        temps.append(float(candidates[losses.argmin()]))
        status.append("heldout_temperature_scaled_proxy")
    return temps, status


def train_model(train_path, val_path, out, config=None, calibration_path=None, device="cpu"):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(config)
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or cfg["threads"] < 1 or cfg["lr"] <= 0:
        raise ValueError("epochs, batch_size, threads and lr must be positive")
    root = Path(out)
    if (root / "best.pt").exists() or (root / "last.pt").exists():
        raise FileExistsError("Training output contains a checkpoint; use a new output directory.")
    train, val = read_scenes(train_path), read_scenes(val_path)
    check_disjoint(train, val)
    calibration = read_scenes(calibration_path) if calibration_path else []
    check_disjoint(train, calibration)
    check_disjoint(val, calibration)
    train_domains = sorted({s["domain"] for s in train})
    if {s["domain"] for s in val + calibration} - set(train_domains):
        raise ValueError("Validation/calibration domain not present in training")
    for split in (train, val, calibration):
        for s in split:
            if "labels" not in s:
                raise ValueError("Training/validation/calibration scenes require labels")
    observed_targets = sum(
        float(np.sum(s["labels"][key]))
        for s in train
        for key in ("risk_mask", "quality_mask", "utility_mask")
    )
    if observed_targets == 0 and not any(s["labels"].get("pairs") for s in train):
        raise ValueError("No supervised targets or preferences in training data")
    root.mkdir(parents=True, exist_ok=True)
    device = device_for(device)
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    model_config = {key: cfg[key] for key in ("d_model", "layers", "dropout")}
    model = TrajectoryCritic(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    metadata = {
        "format_version": 1,
        "model_config": model_config,
        "training_config": cfg,
        "domains": train_domains,
        "label_sources": sorted({s["labels"]["source"] for s in train}),
        "groups": {
            "train": sorted({s["group_id"] for s in train}),
            "val": sorted({s["group_id"] for s in val}),
            "calibration": sorted({s["group_id"] for s in calibration}),
        },
        "data_sha256": {
            "train": sha(train_path),
            "val": sha(val_path),
            "calibration": sha(calibration_path) if calibration_path else None,
        },
        "risk_supported": [
            any(np.asarray(s["labels"]["risk_mask"])[:, h].any() for s in train) for h in range(3)
        ],
        "temperatures": [1.0, 1.0, 1.0],
        "calibration_status": ["not_calibrated"] * 3,
        "torch_version": str(torch.__version__),
    }
    history, best = [], float("inf")
    for epoch in range(cfg["epochs"]):
        generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        loader = DataLoader(
            train,
            batch_size=cfg["batch_size"],
            shuffle=True,
            generator=generator,
            collate_fn=collate_scenes,
        )
        model.train()
        totals = []
        for batch in loader:
            batch = to_device(batch, device)
            loss, _ = critic_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; check units and labels.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.detach()))
        model.eval()
        validation = []
        with torch.inference_mode():
            for batch in DataLoader(val, batch_size=cfg["batch_size"], collate_fn=collate_scenes):
                batch = to_device(batch, device)
                validation.append(float(critic_loss(model(batch), batch)[0]))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(totals)),
            "val_loss": float(np.mean(validation)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            **metadata,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "epoch": epoch + 1,
        }
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save(checkpoint, root / "best.pt")
        torch.save({**checkpoint, "optimizer": optimizer.state_dict()}, root / "last.pt")
        write_json(root / "history.json", history)
    selected, checkpoint = load_checkpoint(root / "best.pt", device)
    temps, status = fit_temperatures(selected, calibration, device)
    checkpoint["temperatures"], checkpoint["calibration_status"] = temps, status
    torch.save(checkpoint, root / "best.pt")
    summary = {
        "epochs_completed": cfg["epochs"],
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "calibration_status": status,
        "temperatures": temps,
        "device": str(device),
        "domains": train_domains,
        "label_sources": metadata["label_sources"],
        "notice": "Proxy labels do not demonstrate real-world safety.",
    }
    write_json(root / "training-summary.json", summary)
    write_json(root / "config.json", cfg)
    return summary
