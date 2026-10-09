"""
ResNet Backbone Adapter for UperNet.

This adapter wraps a timm ResNet (or any CNN with features_only support)
for use with UperNet. Unlike the ViT-specific DenseBackbone,
this handles CNN backbones that already produce multi-scale spatial features.

Usage:
    backbone = ResNetBackboneAdapter.from_timm("resnet50", out_channels=512)
    output = backbone(pixel_values)  # BackboneOutput with 4 feature maps
"""

import inspect
from typing import List, Optional

import timm
import torch
import torch.nn as nn
from transformers.modeling_outputs import BackboneOutput


class ResNetBackboneAdapter(nn.Module):
    """
    Backbone adapter for CNN models (ResNet, etc.) with UperNet.

    Uses timm's features_only mode to extract multi-scale feature maps
    from CNN stages, then projects them to uniform channel width.

    Unlike the ViT adapter, no sequence-to-spatial reshaping or artificial
    multi-scaling is needed — CNNs naturally produce multi-scale spatial features.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: List[int],
        out_channels: int = 512,
    ):
        super().__init__()

        self.backbone = backbone
        self.out_channels = out_channels

        # For UperNetSegmentationModel auto-padding (ResNet needs input divisible by 32)
        self.patch_size = 32

        # Lateral 1x1 convs to project each stage's channels to out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in feature_channels
        ])

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> BackboneOutput:
        # Extract multi-scale features from CNN backbone
        stage_features = self.backbone(pixel_values)

        # Project each stage to uniform channel width
        feature_maps = []
        for i, x in enumerate(stage_features):
            x = self.lateral_convs[i](x)
            feature_maps.append(x)

        return BackboneOutput(
            feature_maps=tuple(feature_maps),
            hidden_states=None,
            attentions=None,
        )

    # TODO: remove once ResNet jobs are re-run; alias so freeze_backbone() works without changes
    @property
    def feature_extractor(self) -> nn.Module:
        """Alias for self.backbone to match the DenseBackbone interface."""
        return self.backbone

    def forward_with_filtered_kwargs(self, *args, **kwargs) -> BackboneOutput:
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}
        return self(*args, **filtered_kwargs)

    @classmethod
    def from_timm(
        cls,
        model_name: str,
        out_channels: int = 512,
        pretrained: bool = True,
    ) -> "ResNetBackboneAdapter":
        """
        Create adapter with a timm CNN backbone.

        Args:
            model_name: timm model name (e.g., "resnet50", "resnet101")
            out_channels: Number of output channels after projection
            pretrained: Whether to load ImageNet-pretrained weights

        Returns:
            ResNetBackboneAdapter instance
        """
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )

        feature_channels = backbone.feature_info.channels()

        return cls(
            backbone=backbone,
            feature_channels=feature_channels,
            out_channels=out_channels,
        )
