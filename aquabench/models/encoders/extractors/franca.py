"""
Franca feature extractor.

Handles loading Franca models via torch.hub and extracting raw features.
Franca uses RASA (Relative Absolute Spatial Attention) for position-debiased features.

Reference: https://github.com/valeoai/Franca
"""

from typing import List

import torch
import torch.nn as nn

from aquabench.models.encoders.extractors.base import (
    FeatureExtractor,
    FeatureExtractorConfig,
    ExtractedFeatures,
)


# Model configurations for Franca variants
FRANCA_CONFIGS = {
    "franca_vitb14": {
        "arch_name": "vit_base",
        "hidden_size": 768,
        "patch_size": 14,
        "num_layers": 12,
        "hub_name": "franca_vitb14",
        "checkpoint_img_size": 518,
    },
    "franca_vitl14": {
        "arch_name": "vit_large",
        "hidden_size": 1024,
        "patch_size": 14,
        "num_layers": 24,
        "hub_name": "franca_vitl14",
        "checkpoint_img_size": 518,
    },
    "franca_vitg14": {
        "arch_name": "vit_giant",
        "hidden_size": 1536,
        "patch_size": 14,
        "num_layers": 40,
        "hub_name": "franca_vitg14",
        "checkpoint_img_size": 224,  # Giant checkpoint has 224-sized pos_embed
    },
}


class FrancaFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for Franca models.

    Franca models are loaded via torch.hub and provide:
        - x_norm_clstoken: Classification token
        - x_norm_patchtokens: Standard patch features
        - patch_token_rasa: Position-debiased RASA features (when enabled)
        - intermediate_features: Features from intermediate layers (when requested)

    Note: Franca's intermediate feature extraction may not be available in all
    versions. When not available, final features are duplicated for all indices.

    Example:
        >>> extractor = FrancaFeatureExtractor.from_pretrained("franca_vitl14")
        >>> features = extractor.extract_features(pixel_values, indices=[5, 11, 17, 23])
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        patch_size: int,
        num_layers: int,
        use_rasa_head: bool = True,
    ):
        """
        Initialize the Franca feature extractor.

        Args:
            model: Franca model loaded via torch.hub
            hidden_size: Hidden dimension of the model
            patch_size: Patch size (typically 14)
            num_layers: Number of transformer layers
            use_rasa_head: Whether to use RASA head for position-debiased features
        """
        super().__init__()
        self._model = model
        self._hidden_size = hidden_size
        self._patch_size = patch_size
        self._num_layers = num_layers
        self.use_rasa_head = use_rasa_head

        # Check if model supports intermediate features
        self._supports_intermediate = self._check_intermediate_support()

    @property
    def config(self) -> FeatureExtractorConfig:
        return FeatureExtractorConfig(
            hidden_size=self._hidden_size,
            patch_size=self._patch_size,
            num_layers=self._num_layers,
            num_special_tokens=1,  # CLS token only
            supports_intermediate=self._supports_intermediate,
        )

    def _check_intermediate_support(self) -> bool:
        """Check if the model supports intermediate feature extraction."""
        return hasattr(self._model, "get_intermediate_layers")

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        indices: List[int],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ExtractedFeatures:
        """
        Extract features from the Franca model.

        Args:
            pixel_values: Input images [B, C, H, W]
            indices: Layer indices to extract features from
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights (not supported)

        Returns:
            ExtractedFeatures with features at specified indices
        """
        # Franca's forward_features has no intermediate-output option; use the repo's
        # get_intermediate_layers: 0-based block indices (no embedding entry), with
        # norm=True applying the final LayerNorm to each selected block.
        with torch.set_grad_enabled(self.training):
            outs = self._model.get_intermediate_layers(
                pixel_values, n=list(indices), norm=True, return_class_token=True,
            )

        # Rebuild [B, 1+N, C] (CLS + patches) to match the DINO format DenseBackbone expects.
        features = [torch.cat([cls.unsqueeze(1), patches], dim=1) for patches, cls in outs]
        hidden_states = None

        return ExtractedFeatures(
            features=features,
            hidden_states=hidden_states,
            attentions=None,  # Franca doesn't expose attention weights
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        weights: str = "IN21K",
        use_rasa_head: bool = True,
        **kwargs,
    ) -> "FrancaFeatureExtractor":
        """
        Create a FrancaFeatureExtractor from a pretrained model via torch.hub.

        Args:
            model_name: Model name (e.g., "franca_vitl14", "franca_vitb14", "franca_vitg14")
            weights: Weight variant ("IN21K" or "LAION")
            use_rasa_head: Whether to use RASA head
            **kwargs: Additional arguments

        Returns:
            FrancaFeatureExtractor instance
        """
        if model_name not in FRANCA_CONFIGS:
            available = ", ".join(FRANCA_CONFIGS.keys())
            raise ValueError(f"Unknown Franca model: {model_name}. Available: {available}")

        config = FRANCA_CONFIGS[model_name]

        # Load model via torch.hub
        # img_size must match the checkpoint's positional embeddings to avoid
        # pos_embed shape mismatch. Base/large use 518, giant uses 224.
        # DINOv2 interpolates pos_embed during forward, so any input size works.
        model = torch.hub.load(
            'valeoai/Franca',
            config["hub_name"],
            weights=weights,
            use_rasa_head=use_rasa_head,
            pretrained=True,
            img_size=config["checkpoint_img_size"],
            trust_repo=True,
        )
        model.eval()

        return cls(
            model=model,
            hidden_size=config["hidden_size"],
            patch_size=config["patch_size"],
            num_layers=config["num_layers"],
            use_rasa_head=use_rasa_head,
        )
