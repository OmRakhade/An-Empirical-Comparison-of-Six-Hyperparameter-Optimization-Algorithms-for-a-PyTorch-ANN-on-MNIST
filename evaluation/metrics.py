"""Classification metrics computed from integer class labels."""

from __future__ import annotations

import torch


def classification_metrics(
    targets: torch.Tensor, predictions: torch.Tensor, num_classes: int = 10
) -> dict[str, float]:
    if targets.numel() == 0:
        raise ValueError("Cannot calculate metrics for an empty set.")
    targets = targets.reshape(-1).to(dtype=torch.long)
    predictions = predictions.reshape(-1).to(dtype=torch.long)
    confusion = torch.bincount(
        targets * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).to(dtype=torch.float64)
    true_positives = confusion.diag()
    precision = true_positives / confusion.sum(dim=0).clamp(min=1)
    recall = true_positives / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "accuracy": float(true_positives.sum() / confusion.sum()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }
