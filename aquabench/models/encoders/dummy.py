"""A tiny weightless encoder for offline plumbing tests.

Mirrors the DenseBackbone interface: produces `len(scales)` multi-scale feature
maps and exposes `out_channels` / `patch_size` / `feature_extractor` (for freeze),
so any head/task path can be validated without downloading real weights.
"""

import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BackboneOutput


class DummyEncoder(nn.Module):
    def __init__(self, scales=(1.0,), out_channels: int = 64, patch_size: int = 16):
        super().__init__()
        self.scales = list(scales)
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stem = nn.Conv2d(3, out_channels, 3, padding=1)

    @property
    def feature_extractor(self) -> nn.Module:
        return self.stem

    def forward(self, pixel_values, **_) -> BackboneOutput:
        b, _, h, w = pixel_values.shape
        gh, gw = max(1, h // self.patch_size), max(1, w // self.patch_size)
        base = self.stem(F.interpolate(pixel_values, size=(gh, gw), mode="bilinear", align_corners=False))
        maps = tuple(
            F.interpolate(base, scale_factor=s, mode="bilinear", align_corners=False) for s in self.scales
        )
        return BackboneOutput(feature_maps=maps)
