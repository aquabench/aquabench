"""
Feature extractors for different vision foundation models.

Feature extractors handle model-specific loading and raw feature extraction.
They are composed with DenseBackbone which handles the common
reshaping, projection, and scaling logic.

Available extractors:
- DinoFeatureExtractor: DINOv2/v3 models (HuggingFace)
- FrancaFeatureExtractor: Franca models (torch.hub)
- RadioFeatureExtractor: NVIDIA RADIO models (torch.hub)
- EvaFeatureExtractor: EVA-02 models (timm)
- VJepaFeatureExtractor: V-JEPA 2.1 models (torch.hub, image pathway)

Usage:
    from aquabench.models.encoders.extractors import DinoFeatureExtractor

    extractor = DinoFeatureExtractor.from_pretrained("facebook/dinov2-base")
    features = extractor.extract_features(pixel_values, indices=[2, 5, 8, 11])
"""

from aquabench.models.encoders.extractors.base import (
    FeatureExtractor,
    FeatureExtractorConfig,
    ExtractedFeatures,
)
from aquabench.models.encoders.extractors.dino import DinoFeatureExtractor
from aquabench.models.encoders.extractors.franca import FrancaFeatureExtractor, FRANCA_CONFIGS
from aquabench.models.encoders.extractors.radio import RadioFeatureExtractor, RADIO_CONFIGS
from aquabench.models.encoders.extractors.eva import EvaFeatureExtractor, EVA_CONFIGS
from aquabench.models.encoders.extractors.vjepa import VJepaFeatureExtractor, VJEPA_CONFIGS

__all__ = [
    # Base classes
    "FeatureExtractor",
    "FeatureExtractorConfig",
    "ExtractedFeatures",
    # Extractors
    "DinoFeatureExtractor",
    "FrancaFeatureExtractor",
    "RadioFeatureExtractor",
    "EvaFeatureExtractor",
    "VJepaFeatureExtractor",
    # Configs
    "FRANCA_CONFIGS",
    "RADIO_CONFIGS",
    "EVA_CONFIGS",
    "VJEPA_CONFIGS",
]
