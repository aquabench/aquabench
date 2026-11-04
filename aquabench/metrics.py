"""Confusion-matrix mIoU over all classes (background included)."""

from __future__ import annotations

from typing import List, Tuple

import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                     ignore_index: int) -> torch.Tensor:
    """``[K, K]`` counts, rows = target, cols = prediction. Ignored pixels are dropped."""
    pred, target = pred.flatten(), target.flatten()
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    idx = target[valid] * num_classes + pred[valid]
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def iou_from_confusion(cm: torch.Tensor) -> Tuple[float, List[float]]:
    """(mIoU, per-class IoU). A class with an empty union (absent in GT and prediction)
    gets NaN and is left out of the mean."""
    cm = cm.double()
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    iou = torch.where(union > 0, tp / union.clamp(min=1), torch.full_like(tp, float("nan")))
    present = ~torch.isnan(iou)
    miou = iou[present].mean().item() if present.any() else float("nan")
    return miou, iou.tolist()

