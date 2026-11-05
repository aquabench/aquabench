"""Segmentation heads and the encoder output contract.

Heads return **logits** at their native resolution; the training loop upsamples them
to the mask size before loss and metric.

Encoder contract:
    linear          forward(x) -> Tensor [B, C, h, w]
    upernet_*       forward(x) -> list of 4 Tensors [B, C_i, h_i, w_i] (high-res first)
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

RUN_TYPES = ("linear", "upernet_frozen", "upernet_finetune")
UPERNET_LEVELS = 4
#: Applied when all 4 maps share one spatial size (plain ViT features), as in v1:
#: the maps are resized x4, x2, x1, x0.5 to form a pyramid. Pyramidal inputs pass through.
VIT_PYRAMID_SCALES = (4.0, 2.0, 1.0, 0.5)


class EncoderContractError(ValueError):
    """The encoder output does not match what the run type expects."""


def _check_token_grid(feat: torch.Tensor, grid: Sequence[int], run_type: str) -> None:
    """The encoder's last map must have exactly one token per grid cell."""
    if tuple(feat.shape[-2:]) != tuple(grid):
        raise EncoderContractError(
            f"run_type {run_type!r}: the encoder's last feature map is {list(feat.shape[-2:])}, but the "
            f"dataset's token grid is {list(grid)}. Set patch_size to the encoder's total stride "
            f"(e.g. 14 for a ViT/14, 32 for a ResNet)."
        )


def check_encoder_output(out, run_type: str, grid: Sequence[int]) -> List[int]:
    """Validate the encoder output for ``run_type`` and the dataset's token ``grid``;
    return the channel count per level."""
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}, got {run_type!r}")

    def describe(o):
        if isinstance(o, torch.Tensor):
            return f"a tensor of shape {list(o.shape)}"
        if isinstance(o, (list, tuple)):
            return f"a {type(o).__name__} of {len(o)} items"
        return f"an object of type {type(o).__name__}"

    if run_type == "linear":
        if not isinstance(out, torch.Tensor) or out.ndim != 4:
            raise EncoderContractError(
                f"run_type 'linear' expects encoder.forward(x) to return one tensor [B, C, h, w], "
                f"but it returned {describe(out)}. Return only the last feature map for linear "
                f"probing, or use run_type 'upernet_frozen' / 'upernet_finetune' for 4 maps."
            )
        _check_token_grid(out, grid, run_type)
        return [out.shape[1]]

    ok = (isinstance(out, (list, tuple)) and len(out) == UPERNET_LEVELS
          and all(isinstance(o, torch.Tensor) and o.ndim == 4 for o in out))
    if not ok:
        raise EncoderContractError(
            f"run_type {run_type!r} expects encoder.forward(x) to return a list of "
            f"{UPERNET_LEVELS} tensors [B, C_i, h_i, w_i], but it returned {describe(out)}. "
            f"For a single feature map use run_type 'linear'."
        )
    if len({o.shape[0] for o in out}) != 1:
        raise EncoderContractError(f"feature maps disagree on batch size: {[list(o.shape) for o in out]}")
    _check_token_grid(out[-1], grid, run_type)
    return [o.shape[1] for o in out]


class LinearHead(nn.Module):
    """BatchNorm + 1x1 conv on a single feature map."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.bn(feat))


def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def to_pyramid(feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Turn 4 same-size maps into a x4/x2/x1/x0.5 pyramid; leave pyramids unchanged."""
    if len({tuple(f.shape[-2:]) for f in feats}) > 1:
        return list(feats)
    return [F.interpolate(f, scale_factor=s, mode="bilinear", align_corners=False)
            for f, s in zip(feats, VIT_PYRAMID_SCALES)]


class UPerNetHead(nn.Module):
    """UPerNet decoder: Pyramid Pooling Module on the deepest level + FPN + fusion."""

    def __init__(self, in_channels: Sequence[int], num_classes: int, channels: int = 512,
                 pool_scales: Sequence[int] = (1, 2, 3, 6)):
        super().__init__()
        *shallow, deepest = in_channels
        self.ppm = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(deepest, channels, 1, bias=False),
                          nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for s in pool_scales
        )
        self.psp_bottleneck = _conv_bn_relu(deepest + len(pool_scales) * channels, channels)
        self.lateral_convs = nn.ModuleList(_conv_bn_relu(c, channels, k=1) for c in shallow)
        self.fpn_convs = nn.ModuleList(_conv_bn_relu(channels, channels) for _ in shallow)
        self.fpn_bottleneck = _conv_bn_relu(len(in_channels) * channels, channels)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        feats = to_pyramid(feats)
        top = feats[-1]
        pooled = [_resize(m(top), top.shape[-2:]) for m in self.ppm]
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]
        laterals.append(self.psp_bottleneck(torch.cat([top, *pooled], dim=1)))
        for i in range(len(laterals) - 1, 0, -1):  # top-down
            laterals[i - 1] = laterals[i - 1] + _resize(laterals[i], laterals[i - 1].shape[-2:])
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)] + [laterals[-1]]
        outs = [_resize(o, outs[0].shape[-2:]) for o in outs]
        return self.classifier(self.fpn_bottleneck(torch.cat(outs, dim=1)))


def build_head(run_type: str, in_channels: Sequence[int], num_classes: int) -> nn.Module:
    if run_type == "linear":
        return LinearHead(in_channels[0], num_classes)
    return UPerNetHead(in_channels, num_classes)


class SegModel(nn.Module):
    """Encoder + head. A frozen encoder runs in eval mode under ``no_grad``."""

    def __init__(self, encoder: nn.Module, head: nn.Module, freeze_encoder: bool):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.freeze_encoder = freeze_encoder
        for p in encoder.parameters():
            p.requires_grad_(not freeze_encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        return self.head(feats)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self
