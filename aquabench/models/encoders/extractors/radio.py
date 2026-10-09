"""
NVIDIA RADIO feature extractor.

Handles loading RADIO models via torch.hub and extracting raw features.
RADIO combines knowledge from CLIP, DINOv2, and SAM teachers.

Reference: https://github.com/NVlabs/RADIO
"""

from typing import List

import torch
import torch.nn as nn

from aquabench.models.encoders.extractors.base import (
    FeatureExtractor,
    FeatureExtractorConfig,
    ExtractedFeatures,
)


# Model configurations for RADIO variants
RADIO_CONFIGS = {
    # C-RADIOv4 (commercially permissive)
    "c-radio_v4-h": {
        "version": "c-radio_v4-h",
        "hidden_size": 1280,
        "patch_size": 16,
        "num_layers": 32,
    },
    "c-radio_v4-so400m": {
        "version": "c-radio_v4-so400m",
        "hidden_size": 1152,
        "patch_size": 16,
        "num_layers": 27,
    },
    # C-RADIOv3 (commercially viable)
    "c-radio_v3-b": {
        "version": "c-radio_v3-b",
        "hidden_size": 768,
        "patch_size": 16,
        "num_layers": 12,
    },
    "c-radio_v3-l": {
        "version": "c-radio_v3-l",
        "hidden_size": 1024,
        "patch_size": 16,
        "num_layers": 24,
    },
    "c-radio_v3-h": {
        "version": "c-radio_v3-h",
        "hidden_size": 1280,
        "patch_size": 16,
        "num_layers": 32,
    },
    "c-radio_v3-g": {
        "version": "c-radio_v3-g",
        "hidden_size": 1536,
        "patch_size": 14,
        "num_layers": 40,
    },
    # Legacy RADIO models (non-commercial)
    "radio_v2.5-l": {
        "version": "radio_v2.5-l",
        "hidden_size": 1024,
        "patch_size": 16,
        "num_layers": 24,
    },
    "radio_v2.5-h": {
        "version": "radio_v2.5-h",
        "hidden_size": 1280,
        "patch_size": 16,
        "num_layers": 32,
    },
}


class   RadioFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for NVIDIA RADIO models.

    RADIO models are loaded via torch.hub and return features at specified
    transformer layer indices using RADIOModel.forward_intermediates().

    Example:
        >>> extractor = RadioFeatureExtractor.from_pretrained("c-radio_v4-h")
        >>> features = extractor.extract_features(pixel_values, indices=[7, 15, 23, 31])
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        patch_size: int,
        num_layers: int,
    ):
        """
        Initialize the RADIO feature extractor.

        Args:
            model: RADIO model loaded via torch.hub
            hidden_size: Hidden dimension of the model
            patch_size: Patch size
            num_layers: Number of transformer layers (for reference)
        """
        super().__init__()
        self._model = model
        self._hidden_size = hidden_size
        self._patch_size = patch_size
        self._num_layers = num_layers

        # Update patch_size from model if available
        if hasattr(model, 'patch_size'):
            self._patch_size = model.patch_size

    @property
    def config(self) -> FeatureExtractorConfig:
        return FeatureExtractorConfig(
            hidden_size=self._hidden_size,
            patch_size=self._patch_size,
            num_layers=self._num_layers,
            num_special_tokens=0,  # RADIO returns spatial features without CLS
            supports_intermediate=True,
        )

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        indices: List[int],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_prefix_tokens: bool = False,
    ) -> ExtractedFeatures:
        """
        Extract features from the RADIO model at specified layer indices.

        Uses RADIOModel.forward_intermediates() to get actual per-layer features.
        Features are returned in [B, N, C] format.

        Args:
            pixel_values: Input images [B, C, H, W]
            indices: Layer indices to extract features from (0-based block indices)
            output_hidden_states: Not used (kept for interface compatibility)
            output_attentions: Not used (kept for interface compatibility)
            return_prefix_tokens: If True, include prefix tokens (CLS) in output.
                When False (default), only spatial patch tokens are returned.

        Returns:
            ExtractedFeatures with features at specified layer indices
        """
        with torch.set_grad_enabled(self.training):
            intermediates = self._model.forward_intermediates(
                pixel_values,
                indices=indices,
                return_prefix_tokens=return_prefix_tokens,
                norm=True,
                stop_early=True,
                output_fmt='NLC',
                intermediates_only=True,
                aggregation='sparse',
            )

        # Handle duplicate indices (e.g. [31,31,31,31] for last-layer-only):
        # forward_intermediates deduplicates via set, so expand back to match
        # the caller's request
        if len(intermediates) != len(indices):
            unique_indices = sorted(set(indices))
            index_map = {idx: feat for idx, feat in zip(unique_indices, intermediates)}
            intermediates = [index_map[idx] for idx in indices]

        return ExtractedFeatures(
            features=intermediates,
            hidden_states=None,
            attentions=None,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **kwargs,
    ) -> "RadioFeatureExtractor":
        """
        Create a RadioFeatureExtractor from a pretrained model via torch.hub.

        Args:
            model_name: Model name (e.g., "c-radio_v4-h", "c-radio_v3-l")
            **kwargs: Additional arguments

        Returns:
            RadioFeatureExtractor instance
        """
        if model_name not in RADIO_CONFIGS:
            available = ", ".join(RADIO_CONFIGS.keys())
            raise ValueError(f"Unknown RADIO model: {model_name}. Available: {available}")

        config = RADIO_CONFIGS[model_name]

        # Load model via torch.hub
        model = torch.hub.load(
            'NVlabs/RADIO',
            'radio_model',
            version=config["version"],
            progress=True,
            trust_repo=True,
        )
        model.eval()

        return cls(
            model=model,
            hidden_size=config["hidden_size"],
            patch_size=config["patch_size"],
            num_layers=config["num_layers"],
        )
