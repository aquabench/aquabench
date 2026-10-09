"""Linear probing head: frozen backbone + a single 1x1 conv on the last feature map.

Returns **logits at the token grid** (DINOv3-style linear probe) — the Task
upsamples them to the label resolution and computes the CrossEntropy loss. The
backbone's feature extractor is frozen at init. When a `token_grid` is given the
input is resized so the encoder emits that fixed token count (matched across
encoders with different patch sizes, no padding); otherwise it falls back to
padding the input up to the patch size.
"""

import torch
import torch.nn as nn

from aquabench.data.tokens import pad_to_patch, resize_to_tokens


class LinearProbingSegmentationModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        hidden_size: int = 512,
        auto_pad: bool = True,
        token_grid=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.auto_pad = auto_pad
        self.token_grid = token_grid
        self.patch_size = getattr(backbone, "patch_size", 14)

        self.backbone = backbone
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Conv2d(hidden_size, num_classes, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.token_grid is not None:
            pixel_values = resize_to_tokens(pixel_values, self.patch_size, self.token_grid)
        elif self.auto_pad:
            pixel_values, _ = pad_to_patch(pixel_values, self.patch_size)

        with torch.no_grad():
            features = self.backbone(pixel_values).feature_maps[-1]

        # logits stay at the token grid; the Task upsamples to the label resolution
        return self.classifier(features)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # backbone is always frozen — keep BN in eval
        return self

    def freeze_backbone(self):
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = True
