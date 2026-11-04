"""Plain PyTorch training / evaluation loop with gradient accumulation."""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aquabench.metrics import confusion_matrix, iou_from_confusion


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(device: torch.device, precision: str):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16")


def upsample_to(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def train_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer, *, grad_accum_steps: int,
                ignore_index: int, device: torch.device, precision: str = "fp32",
                on_step: Optional[Callable[[float], None]] = None) -> float:
    """One epoch. An optimizer step happens every ``grad_accum_steps`` batches, so
    ``batch_size x grad_accum_steps`` is the effective batch. Returns the mean batch loss.
    ``on_step(loss)`` is called after every optimizer step with that step's mean loss."""
    model.train()
    params = [p for g in optimizer.param_groups for p in g["params"]]
    optimizer.zero_grad(set_to_none=True)
    total, n_batches, group_loss, in_group = 0.0, 0, 0.0, 0
    last = len(loader) - 1
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model(x)
        loss = F.cross_entropy(upsample_to(logits, y), y, ignore_index=ignore_index)
        (loss / grad_accum_steps).backward()
        value = loss.item()
        total, n_batches = total + value, n_batches + 1
        group_loss, in_group = group_loss + value, in_group + 1
        if in_group == grad_accum_steps or i == last:
            if in_group < grad_accum_steps:  # short last group: keep it a mean, not a partial sum
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(grad_accum_steps / in_group)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if on_step is not None:
                on_step(group_loss / in_group)
            group_loss, in_group = 0.0, 0
    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, *, num_classes: int, ignore_index: int,
                    device: torch.device, precision: str = "fp32") -> Tuple[float, List[float]]:
    """(all-class mIoU, per-class IoU) over a whole split."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model(x)
        cm += confusion_matrix(upsample_to(logits, y).argmax(1), y, num_classes, ignore_index)
    return iou_from_confusion(cm.cpu())
