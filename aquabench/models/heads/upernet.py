"""Self-contained UPerNet decoder (PSP + FPN) — no HuggingFace UperNet dependency.

Consumes the encoder's multi-scale feature maps (ordered high-res -> low-res, the
last one deepest) and produces per-pixel **logits only** — the Task computes the
CrossEntropy loss. The head owns the encoder and exposes `freeze_backbone()`, so
encoders only need to return feature maps.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from aquabench.data.tokens import pad_to_patch, resize_to_tokens


def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, pad: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _PPM(nn.ModuleList):
    """Pyramid Pooling Module on the deepest feature."""

    def __init__(self, pool_scales, in_ch: int, channels: int):
        super().__init__()
        for s in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_ch, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        return [F.interpolate(m(x), size=x.shape[-2:], mode="bilinear", align_corners=False) for m in self]


class UPerNetDecoder(nn.Module):
    """UPerHead: PSP on the deepest level + an FPN over all levels + fusion."""

    def __init__(self, in_channels: int, n_levels: int, channels: int = 512,
                 num_classes: int = 2, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = _PPM(pool_scales, in_channels, channels)
        self.psp_bottleneck = _conv_bn_relu(in_channels + len(pool_scales) * channels, channels)
        self.lateral_convs = nn.ModuleList([_conv_bn_relu(in_channels, channels, k=1, pad=0) for _ in range(n_levels - 1)])
        self.fpn_convs = nn.ModuleList([_conv_bn_relu(channels, channels) for _ in range(n_levels - 1)])
        self.fpn_bottleneck = _conv_bn_relu(n_levels * channels, channels)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        psp = self.psp_bottleneck(torch.cat([inputs[-1], *self.ppm(inputs[-1])], dim=1))
        laterals = [lat(inputs[i]) for i, lat in enumerate(self.lateral_convs)]
        laterals.append(psp)
        # top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False)
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(self.fpn_convs))] + [laterals[-1]]
        target = outs[0].shape[-2:]
        outs = [F.interpolate(o, size=target, mode="bilinear", align_corners=False) for o in outs]
        return self.classifier(self.fpn_bottleneck(torch.cat(outs, dim=1)))


class UPerNetSegHead(nn.Module):
    def __init__(self, encoder, num_classes: int, in_channels: int, n_levels: int,
                 channels: int = 512, auto_pad: bool = True, token_grid=None):
        super().__init__()
        self.encoder = encoder
        self.patch_size = getattr(encoder, "patch_size", 16)
        self.auto_pad = auto_pad
        self.token_grid = token_grid
        self.decoder = UPerNetDecoder(in_channels=in_channels, n_levels=n_levels,
                                      channels=channels, num_classes=num_classes)

    def forward(self, pixel_values):
        if self.token_grid is not None:
            pixel_values = resize_to_tokens(pixel_values, self.patch_size, self.token_grid)
        elif self.auto_pad:
            pixel_values, _ = pad_to_patch(pixel_values, self.patch_size)
        feats = list(self.encoder(pixel_values).feature_maps)
        # logits stay at the decoder resolution; the Task upsamples to label res
        return self.decoder(feats)

    def _encoder_params(self):
        fe = getattr(self.encoder, "feature_extractor", self.encoder)
        return fe.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self._encoder_params()):
            self.encoder.eval()  # encoder is frozen — keep BN in eval
        return self

    def freeze_backbone(self):
        for p in self._encoder_params():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self._encoder_params():
            p.requires_grad = True
