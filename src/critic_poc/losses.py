"""Masked outcome / attribute / preference learning; no TD or RL claim."""

from torch.nn import functional as F


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def critic_loss(output, batch):
    risk = masked_mean(
        F.binary_cross_entropy_with_logits(output["risk_logits"], batch["risk"], reduction="none"),
        batch["risk_mask"],
    )
    quality = masked_mean(
        F.smooth_l1_loss(output["quality"], batch["quality"], reduction="none"),
        batch["quality_mask"],
    )
    utility = masked_mean(
        F.smooth_l1_loss(output["utility"], batch["utility"], reduction="none"),
        batch["utility_mask"],
    )
    pairs = batch["pairs"]
    preference = output["utility"].sum() * 0
    if len(pairs):
        bi, ai, ci = (pairs[:, i].long() for i in range(3))
        # Small temperature expands bounded utility difference for preference learning.
        difference = (output["utility"][bi, ai] - output["utility"][bi, ci]) / 0.2
        preference = F.binary_cross_entropy_with_logits(difference, pairs[:, 3])
    loss = risk + quality + utility + 0.2 * preference
    return loss, {
        name: float(value.detach())
        for name, value in (
            ("risk", risk),
            ("quality", quality),
            ("utility", utility),
            ("preference", preference),
            ("total", loss),
        )
    }
