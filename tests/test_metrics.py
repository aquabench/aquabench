"""mIoU from the confusion matrix."""

import math

import pytest
import torch

from aquabench.metrics import confusion_matrix, iou_from_confusion


def test_miou_known_values():
    """Hand-made example: per-class IoU 1/3, 2/3, 1/2; ignored pixels and absent classes don't count."""
    target = torch.tensor([0, 0, 1, 1, 2, 2, 255])
    pred = torch.tensor([0, 1, 1, 1, 2, 0, 3])
    miou, per_class = iou_from_confusion(confusion_matrix(pred, target, num_classes=4, ignore_index=255))
    assert per_class[:3] == pytest.approx([1 / 3, 2 / 3, 1 / 2])
    assert math.isnan(per_class[3])  # class 3 absent in target and (non-ignored) prediction
    assert miou == pytest.approx(0.5)


def test_miou_includes_background():
    """Background counts as a class: inclusive mean (0.65), not foreground-only (0.5)."""
    target = torch.tensor([0, 0, 0, 0, 1, 1])
    pred = torch.tensor([0, 0, 0, 0, 1, 0])
    miou, per_class = iou_from_confusion(confusion_matrix(pred, target, num_classes=2, ignore_index=255))
    assert per_class == pytest.approx([0.8, 0.5])
    assert miou == pytest.approx(0.65)
