"""
Abstract base class for feature extractors.

Feature extractors handle model-specific loading and raw feature extraction.
They are composed with DenseBackbone which handles the common
reshaping, projection, and scaling logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn


@dataclass
class FeatureExtractorConfig:
    """Configuration for a feature extractor."""
    hidden_size: int
    patch_size: int
    num_layers: int
    num_special_tokens: int = 1  # CLS token + register tokens
    supports_intermediate: bool = True  # Whether model supports intermediate layer extraction


@dataclass
class ExtractedFeatures:
    """Container for extracted features from a backbone model."""
    features: List[torch.Tensor]  # Features at specified indices, each [B, N, C]
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None  # All hidden states if requested
    attentions: Optional[Tuple[torch.Tensor, ...]] = None  # Attention weights if requested


class FeatureExtractor(ABC, nn.Module):
    """
    Abstract base class for feature extractors.

    Feature extractors are responsible for:
    1. Loading the underlying model (HuggingFace, torch.hub, etc.)
    2. Extracting raw features at specified layer indices
    3. Returning features in a consistent format [B, N, C]

    The common logic for reshaping, projecting, and scaling features
    is handled by DenseBackbone which composes a FeatureExtractor.

    Subclasses must implement:
        - config: Property returning FeatureExtractorConfig
        - extract_features(): Extract raw features from the model

    Example:
        >>> extractor = DinoFeatureExtractor.from_pretrained("facebook/dinov2-base")
        >>> features = extractor.extract_features(pixel_values, indices=[2, 5, 8, 11])
        >>> print([f.shape for f in features.features])  # [B, N, C] for each index
    """

    def __init__(self):
        super().__init__()
        self._model: Optional[nn.Module] = None

    @property
    @abstractmethod
    def config(self) -> FeatureExtractorConfig:
        """Return the configuration for this feature extractor."""
        pass

    @property
    def model(self) -> nn.Module:
        """Return the underlying model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call from_pretrained() or set model manually.")
        return self._model

    @model.setter
    def model(self, value: nn.Module):
        """Set the underlying model."""
        self._model = value

    @abstractmethod
    def extract_features(
        self,
        pixel_values: torch.Tensor,
        indices: List[int],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ExtractedFeatures:
        """
        Extract features from the model at specified layer indices.

        Args:
            pixel_values: Input images [B, C, H, W]
            indices: Layer indices to extract features from
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights

        Returns:
            ExtractedFeatures containing:
                - features: List of tensors [B, N, C] at specified indices
                - hidden_states: All hidden states if requested
                - attentions: Attention weights if requested
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "FeatureExtractor":
        """
        Load a pretrained feature extractor.

        Args:
            model_name: Model identifier (HuggingFace name, torch.hub name, etc.)
            **kwargs: Additional model-specific arguments

        Returns:
            FeatureExtractor instance
        """
        pass

    def forward(
        self,
        pixel_values: torch.Tensor,
        indices: List[int],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ExtractedFeatures:
        """Forward pass - alias for extract_features."""
        return self.extract_features(
            pixel_values,
            indices=indices,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

    def get_info(self) -> Dict[str, Any]:
        """Get information about the feature extractor."""
        cfg = self.config
        return {
            "hidden_size": cfg.hidden_size,
            "patch_size": cfg.patch_size,
            "num_layers": cfg.num_layers,
            "num_special_tokens": cfg.num_special_tokens,
            "supports_intermediate": cfg.supports_intermediate,
        }
