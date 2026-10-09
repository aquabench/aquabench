"""
Dense Backbone Adapter.

This adapter composes a FeatureExtractor with the common logic for:
- Reshaping sequence features to spatial format
- Applying lateral convolutions for channel projection
- Scaling features to different spatial resolutions
- Smoothing with 3x3 convolutions

The adapter provides a clean separation between model-specific feature
extraction and task-specific feature processing.
"""

import inspect
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BackboneOutput

from aquabench.models.encoders.extractors.base import FeatureExtractor


class DenseBackbone(nn.Module):
    """
    Backbone adapter that composes a FeatureExtractor for multi-scale spatial features.

    This class handles the common logic for converting transformer features to
    multi-scale feature maps suitable for dense prediction tasks (depth, segmentation):
    1. Extract features at specified layer indices via FeatureExtractor
    2. Remove special tokens (CLS, register tokens)
    3. Reshape from sequence [B, N, C] to spatial [B, C, H, W]
    4. Apply lateral convolutions to project to out_channels
    5. Scale features to different spatial resolutions
    6. Apply smoothing convolutions

    Example:
        >>> from aquabench.models.encoders.extractors import DinoFeatureExtractor
        >>> extractor = DinoFeatureExtractor.from_pretrained("facebook/dinov2-base")
        >>> backbone = DenseBackbone(
        ...     feature_extractor=extractor,
        ...     backbone_indices=[2, 5, 8, 11],
        ...     scales=[4.0, 2.0, 1.0, 0.5],
        ...     out_channels=512,
        ... )
        >>> output = backbone(pixel_values)
        >>> print([f.shape for f in output.feature_maps])
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone_indices: List[int],
        scales: List[float] = [4.0, 2.0, 1.0, 0.5],
        out_channels: int = 512,
        img_size: int = 224,
        use_lateral_convs: bool = True,
    ):
        """
        Initialize the ViT backbone adapter.

        Args:
            feature_extractor: FeatureExtractor instance for the backbone model
            backbone_indices: Layer indices to extract features from
            scales: Spatial scales for each feature level
            out_channels: Number of output channels after projection (ignored if use_lateral_convs=False)
            img_size: Input image size (for grid size calculation)
            use_lateral_convs: Whether to use lateral convolutions for channel projection.
                If False, uses ViT output channels directly (for linear probing).
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.backbone_indices = backbone_indices
        self.scales = scales
        self.use_lateral_convs = use_lateral_convs

        # Get config from feature extractor
        config = feature_extractor.config
        self.in_channels = config.hidden_size
        self.patch_size = config.patch_size
        self.num_special_tokens = config.num_special_tokens

        # When not using lateral convs, output channels = input channels (ViT dim)
        self.out_channels = out_channels if use_lateral_convs else self.in_channels

        # Calculate grid size
        if isinstance(img_size, (list, tuple)):
            self.grid_size = [img_size[0] // self.patch_size, img_size[1] // self.patch_size]
        else:
            self.grid_size = [img_size // self.patch_size for _ in range(2)]

        assert len(scales) == len(backbone_indices), \
            f"scales ({len(scales)}) and backbone_indices ({len(backbone_indices)}) must match"

        # Projection layers (only if use_lateral_convs=True)
        if self.use_lateral_convs:
            self.lateral_convs = nn.ModuleList([
                nn.Conv2d(self.in_channels, out_channels, 1)
                for _ in range(len(backbone_indices))
            ])
        

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Forward pass through the backbone.

        Args:
            pixel_values: Input images [B, C, H, W]
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights

        Returns:
            BackboneOutput with feature_maps, hidden_states, and attentions
        """
        # Calculate actual grid size from input
        _, _, input_h, input_w = pixel_values.shape
        grid_h = input_h // self.patch_size
        grid_w = input_w // self.patch_size

        # Extract features via the feature extractor
        extracted = self.feature_extractor.extract_features(
            pixel_values,
            indices=self.backbone_indices,
            output_hidden_states=output_hidden_states or False,
            output_attentions=output_attentions or False,
        )

        # Process features: reshape, project, scale
        feature_maps = []
        for i, x in enumerate(extracted.features):
            # Remove special tokens: [B, N, C] -> [B, H*W, C]
            if self.num_special_tokens > 0:
                x = x[:, self.num_special_tokens:, :]

            # Reshape to spatial: [B, H*W, C] -> [B, C, H, W]
            B, N, C = x.shape
            x = x.reshape(B, grid_h, grid_w, C).permute(0, 3, 1, 2).contiguous()

            # Apply lateral conv if enabled: [B, C, H, W] -> [B, out_channels, H, W]
            if self.use_lateral_convs:
                x = self.lateral_convs[i](x)

            # Scale features
            x = F.interpolate(
                x.contiguous(),
                scale_factor=self.scales[i],
                mode="bilinear",
                align_corners=False
            )


            feature_maps.append(x)

        return BackboneOutput(
            feature_maps=tuple(feature_maps),
            hidden_states=extracted.hidden_states,
            attentions=extracted.attentions,
        )

    def forward_with_filtered_kwargs(self, *args, **kwargs) -> BackboneOutput:
        """
        Forward pass that filters kwargs based on what the model actually supports.

        This method handles cases where extra kwargs are passed that aren't
        part of the forward signature.
        """
        # Filter kwargs to only include those in our forward signature
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}

        return self(*args, **filtered_kwargs)

    @classmethod
    def from_dino(
        cls,
        model_name: str,
        backbone_indices: List[int],
        scales: List[float] = [4.0, 2.0, 1.0, 0.5],
        out_channels: int = 512,
        num_register_tokens: int = 0,
        img_size: int = 224,
        use_lateral_convs: bool = True,
        **kwargs,
    ) -> "DenseBackbone":
        """
        Create adapter with a DINO feature extractor.

        Args:
            model_name: HuggingFace model name (e.g., "facebook/dinov2-base")
            backbone_indices: Layer indices to extract features from
            scales: Spatial scales for each feature level
            out_channels: Number of output channels (ignored if use_lateral_convs=False)
            num_register_tokens: Number of register tokens
            img_size: Input image size
            use_lateral_convs: Whether to use lateral convolutions for projection
            **kwargs: Additional arguments for DinoFeatureExtractor

        Returns:
            DenseBackbone instance
        """
        from aquabench.models.encoders.extractors import DinoFeatureExtractor

        extractor = DinoFeatureExtractor.from_pretrained(
            model_name,
            num_register_tokens=num_register_tokens,
            **kwargs,
        )

        return cls(
            feature_extractor=extractor,
            backbone_indices=backbone_indices,
            scales=scales,
            out_channels=out_channels,
            img_size=img_size,
            use_lateral_convs=use_lateral_convs,
        )

    @classmethod
    def from_franca(
        cls,
        model_name: str,
        backbone_indices: List[int],
        scales: List[float] = [4.0, 2.0, 1.0, 0.5],
        out_channels: int = 512,
        weights: str = "IN21K",
        use_rasa_head: bool = True,
        use_lateral_convs: bool = True,
        **kwargs,
    ) -> "DenseBackbone":
        """
        Create adapter with a Franca feature extractor.

        Args:
            model_name: Model name (e.g., "franca_vitl14")
            backbone_indices: Layer indices to extract features from
            scales: Spatial scales for each feature level
            out_channels: Number of output channels (ignored if use_lateral_convs=False)
            weights: Weight variant ("IN21K" or "LAION")
            use_rasa_head: Whether to use RASA head
            use_lateral_convs: Whether to use lateral convolutions for projection
            **kwargs: Additional arguments

        Returns:
            DenseBackbone instance
        """
        from aquabench.models.encoders.extractors import FrancaFeatureExtractor

        extractor = FrancaFeatureExtractor.from_pretrained(
            model_name,
            weights=weights,
            use_rasa_head=use_rasa_head,
            **kwargs,
        )

        # Franca uses img_size=518 for pretrained weights
        return cls(
            feature_extractor=extractor,
            backbone_indices=backbone_indices,
            scales=scales,
            out_channels=out_channels,
            img_size=518,
            use_lateral_convs=use_lateral_convs,
        )

    @classmethod
    def from_radio(
        cls,
        model_name: str,
        backbone_indices: Optional[List[int]] = None,
        scales: List[float] = [4.0, 2.0, 1.0, 0.5],
        out_channels: int = 512,
        img_size: int = 224,
        use_lateral_convs: bool = True,
        **kwargs,
    ) -> "DenseBackbone":
        """
        Create adapter with a RADIO feature extractor.

        Args:
            model_name: Model name (e.g., "c-radio_v4-h")
            backbone_indices: Layer indices to extract features from.
                If None, evenly-spaced indices across model depth are used.
            scales: Spatial scales for each feature level
            out_channels: Number of output channels (ignored if use_lateral_convs=False)
            img_size: Input image size
            use_lateral_convs: Whether to use lateral convolutions for projection
            **kwargs: Additional arguments

        Returns:
            DenseBackbone instance
        """
        from aquabench.models.encoders.extractors import RadioFeatureExtractor
        from aquabench.models.encoders.extractors.radio import RADIO_CONFIGS

        extractor = RadioFeatureExtractor.from_pretrained(
            model_name,
            **kwargs,
        )

        if backbone_indices is None:
            # Sensible defaults: evenly-spaced indices across model depth
            num_layers = RADIO_CONFIGS[model_name]["num_layers"]
            num_features = len(scales)
            step = num_layers // num_features
            backbone_indices = [step * (i + 1) - 1 for i in range(num_features)]

        return cls(
            feature_extractor=extractor,
            backbone_indices=backbone_indices,
            scales=scales,
            out_channels=out_channels,
            img_size=img_size,
            use_lateral_convs=use_lateral_convs,
        )

    @classmethod
    def from_eva(
        cls,
        model_name: str,
        backbone_indices: List[int],
        scales: List[float] = [4.0, 2.0, 1.0, 0.5],
        out_channels: int = 512,
        img_size: int = 224,
        use_lateral_convs: bool = True,
        **kwargs,
    ) -> "DenseBackbone":
        """
        Create adapter with an EVA-02 feature extractor (timm).

        Args:
            model_name: Catalog key ("eva02_base", "eva02_large")
            backbone_indices: Block indices to extract features from
            scales: Spatial scales for each feature level
            out_channels: Number of output channels (ignored if use_lateral_convs=False)
            img_size: Input image size
            use_lateral_convs: Whether to use lateral convolutions for projection
            **kwargs: Additional arguments for EvaFeatureExtractor

        Returns:
            DenseBackbone instance
        """
        from aquabench.models.encoders.extractors import EvaFeatureExtractor

        extractor = EvaFeatureExtractor.from_pretrained(model_name, **kwargs)

        return cls(
            feature_extractor=extractor,
            backbone_indices=backbone_indices,
            scales=scales,
            out_channels=out_channels,
            img_size=img_size,
            use_lateral_convs=use_lateral_convs,
        )

    @classmethod
    def from_vjepa(
        cls,
        model_name: str,
        backbone_indices: List[int],
        scales: List[float] = [4.0, 2.0, 1.0, 0.5],
        out_channels: int = 512,
        img_size: int = 224,
        use_lateral_convs: bool = True,
        **kwargs,
    ) -> "DenseBackbone":
        """
        Create adapter with a V-JEPA 2.1 feature extractor (torch.hub, image pathway).

        Args:
            model_name: Catalog key / hub entrypoint ("vjepa2_1_vit_base_384",
                "vjepa2_1_vit_gigantic_384")
            backbone_indices: Layer indices to extract features from (V-JEPA 2.1
                uses the final layer only; the same map is returned per index)
            scales: Spatial scales for each feature level
            out_channels: Number of output channels (ignored if use_lateral_convs=False)
            img_size: Input image size (grid is recomputed from the actual input)
            use_lateral_convs: Whether to use lateral convolutions for projection
            **kwargs: Additional arguments for VJepaFeatureExtractor

        Returns:
            DenseBackbone instance
        """
        from aquabench.models.encoders.extractors import VJepaFeatureExtractor

        extractor = VJepaFeatureExtractor.from_pretrained(model_name, **kwargs)

        return cls(
            feature_extractor=extractor,
            backbone_indices=backbone_indices,
            scales=scales,
            out_channels=out_channels,
            img_size=img_size,
            use_lateral_convs=use_lateral_convs,
        )
