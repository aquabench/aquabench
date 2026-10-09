"""
EVA-02 feature extractor.

Handles loading EVA-02 models via timm and extracting raw features at specified
transformer block indices. EVA-02 are masked-image-modeling (MIM) pretrained ViTs.

Features are pulled through timm's `forward_intermediates` with `output_fmt='NLC'`,
which already returns *patch tokens only* (CLS/prefix stripped) — so the composing
DenseBackbone sees `num_special_tokens=0`.

Reference: https://github.com/baaivision/EVA / timm `eva02_*` models.
"""

from typing import List

import torch
import torch.nn as nn

from aquabench.models.encoders.extractors.base import (
    FeatureExtractor,
    FeatureExtractorConfig,
    ExtractedFeatures,
)


# Model configurations for EVA-02 variants (MIM self-supervised weights).
EVA_CONFIGS = {
    "eva02_base": {
        "timm_name": "eva02_base_patch14_224.mim_in22k",
        "hidden_size": 768,
        "patch_size": 14,
        "num_layers": 12,
    },
    "eva02_large": {
        "timm_name": "eva02_large_patch14_224.mim_m38m",
        "hidden_size": 1024,
        "patch_size": 14,
        "num_layers": 24,
    },
}


class EvaFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for EVA-02 models loaded via timm.

    Uses `forward_intermediates(..., output_fmt='NLC', intermediates_only=True)` to
    pull patch-token features at the requested block indices. The returned tensors
    are already `[B, N, C]` with prefix tokens removed.

    Example:
        >>> extractor = EvaFeatureExtractor.from_pretrained("eva02_base")
        >>> features = extractor.extract_features(pixel_values, indices=[2, 5, 8, 11])
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        patch_size: int,
        num_layers: int,
    ):
        super().__init__()
        self._model = model
        self._hidden_size = hidden_size
        self._patch_size = patch_size
        self._num_layers = num_layers

    @property
    def config(self) -> FeatureExtractorConfig:
        return FeatureExtractorConfig(
            hidden_size=self._hidden_size,
            patch_size=self._patch_size,
            num_layers=self._num_layers,
            # forward_intermediates(output_fmt='NLC') already drops the prefix tokens.
            num_special_tokens=0,
            supports_intermediate=True,
        )

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        indices: List[int],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ExtractedFeatures:
        """
        Extract features from the EVA-02 model at specified block indices.

        Args:
            pixel_values: Input images [B, C, H, W]
            indices: Block indices to extract features from
            output_hidden_states: Whether to return all block features
            output_attentions: Unused (timm path does not expose attentions here)

        Returns:
            ExtractedFeatures with patch-token features [B, N, C] at each index.
        """
        with torch.set_grad_enabled(self.training):
            features = self._model.forward_intermediates(
                pixel_values,
                indices=list(indices),
                norm=True,
                output_fmt="NLC",
                intermediates_only=True,
            )

        hidden_states = None
        if output_hidden_states:
            hidden_states = self._model.forward_intermediates(
                pixel_values,
                norm=True,
                output_fmt="NLC",
                intermediates_only=True,
            )
            hidden_states = tuple(hidden_states)

        return ExtractedFeatures(
            features=list(features),
            hidden_states=hidden_states,
            attentions=None,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **kwargs,
    ) -> "EvaFeatureExtractor":
        """
        Create an EvaFeatureExtractor from a pretrained timm model.

        Args:
            model_name: Catalog key ("eva02_base", "eva02_large")
            **kwargs: Additional arguments forwarded to timm.create_model

        Returns:
            EvaFeatureExtractor instance
        """
        import timm

        if model_name not in EVA_CONFIGS:
            available = ", ".join(EVA_CONFIGS.keys())
            raise ValueError(f"Unknown EVA-02 model: {model_name}. Available: {available}")

        config = EVA_CONFIGS[model_name]

        # dynamic_img_size lets the ViT accept the (resized-to-token-grid) inputs the
        # probe/heads feed in without a pos-embed shape mismatch.
        model = timm.create_model(
            config["timm_name"],
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            **kwargs,
        )
        model.eval()

        return cls(
            model=model,
            hidden_size=config["hidden_size"],
            patch_size=config["patch_size"],
            num_layers=config["num_layers"],
        )
