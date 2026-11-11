"""mIoU from the confusion matrix."""

import math

import pytest
import torch

from aquabench.metrics import confusion_matrix, iou_from_confusion


def test_miou_known_values():
    """Hand-made example: per-class IoU 1/3, 2/3, 1/2; ignored pixels don't count, and the
    mean is over all 4 classes of the dataset."""
    target = torch.tensor([0, 0, 1, 1, 2, 2, 255])
    pred = torch.tensor([0, 1, 1, 1, 2, 0, 3])
    miou, per_class = iou_from_confusion(confusion_matrix(pred, target, num_classes=4, ignore_index=255))
    assert per_class[:3] == pytest.approx([1 / 3, 2 / 3, 1 / 2])
    assert math.isnan(per_class[3])  # class 3 absent in target and (non-ignored) prediction
    assert miou == pytest.approx(1.5 / 4)


def test_miou_divides_by_the_number_of_classes():
    """Predicting a class that is absent from the ground truth doesn't change the denominator."""
    target = torch.tensor([0, 0, 1, 1])
    never = confusion_matrix(torch.tensor([0, 0, 1, 1]), target, num_classes=3, ignore_index=255)
    once = confusion_matrix(torch.tensor([0, 2, 1, 1]), target, num_classes=3, ignore_index=255)
    assert iou_from_confusion(never)[0] == pytest.approx(2 / 3)
    assert iou_from_confusion(once)[0] == pytest.approx(1.5 / 3)


def test_miou_includes_background():
    """Background counts as a class: inclusive mean (0.65), not foreground-only (0.5)."""
    target = torch.tensor([0, 0, 0, 0, 1, 1])
    pred = torch.tensor([0, 0, 0, 0, 1, 0])
    miou, per_class = iou_from_confusion(confusion_matrix(pred, target, num_classes=2, ignore_index=255))
    assert per_class == pytest.approx([0.8, 0.5])
    assert miou == pytest.approx(0.65)
