"""Small dependency-free metrics, explicit undefined cases and tie handling."""

import numpy as np


def binary_metrics(target, probability, threshold=0.5):
    y, p = np.asarray(target, dtype=float), np.asarray(probability, dtype=float)
    if not len(y):
        return {
            "count": 0,
            "brier": None,
            "nll": None,
            "ece": None,
            "auroc": None,
            "auprc": None,
            "recall": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
        }
    positive = y >= 0.5
    prediction = p >= threshold
    tp = int((prediction & positive).sum())
    fp = int((prediction & ~positive).sum())
    pos, neg = int(positive.sum()), int((~positive).sum())
    ece = 0.0
    bins = np.minimum((p * 10).astype(int), 9)
    for i in range(10):
        mask = bins == i
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    auroc = ap = None
    if pos and neg:
        order = np.argsort(-p, kind="stable")
        sorted_p, sorted_y = p[order], positive[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(p) - 1]
        tps = np.r_[0, np.cumsum(sorted_y)[ends]]
        fps = np.r_[0, np.cumsum(~sorted_y)[ends]]
        recall = tps / pos
        auroc = float(np.trapezoid(recall, fps / neg))
        ap = float((np.diff(recall) * (tps[1:] / (tps[1:] + fps[1:]))).sum())
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "count": len(y),
        "positives": pos,
        "negatives": neg,
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "ece": float(ece),
        "auroc": auroc,
        "auprc": ap,
        "true_positives": tp,
        "false_positives": fp,
        "recall": tp / pos if pos else None,
        "false_negative_rate": (pos - tp) / pos if pos else None,
        "false_positive_rate": fp / neg if neg else None,
    }
