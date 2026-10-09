"""
DINO feature extractor for DINOv2 and DINOv3 models.

Handles loading HuggingFace DINOv2/v3 models and extracting
raw features at specified transformer layer indices.
"""

import inspect
from typing import Optional, List, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from aquabench.models.encoders.extractors.base import (
    FeatureExtractor,
    FeatureExtractorConfig,
    ExtractedFeatures,
)


class DinoFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for DINOv2 and DINOv3 models from HuggingFace.

    Handles:
        - Loading DINOv2/v3 models from HuggingFace
        - Extracting features from intermediate transformer layers
        - Positional embedding interpolation for variable input sizes
        - Support for register tokens

    Example:
        >>> extractor = DinoFeatureExtractor.from_pretrained("facebook/dinov2-base")
        >>> features = extractor.extract_features(pixel_values, indices=[2, 5, 8, 11])
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        patch_size: int,
        num_layers: int,
        num_register_tokens: int = 0,
    ):
        """
        Initialize the DINO feature extractor.

        Args:
            model: HuggingFace ViT model (DINOv2 or DINOv3)
            hidden_size: Hidden dimension of the model
            patch_size: Patch size of the ViT
            num_layers: Number of transformer layers
            num_register_tokens: Number of register tokens (4 for DINOv2-reg, DINOv3)
        """
        super().__init__()
        self._model = model
        self._hidden_size = hidden_size
        self._patch_size = patch_size
        self._num_layers = num_layers
        self._num_register_tokens = num_register_tokens

        # Check model capabilities
        self._has_attentions = self._check_has_attentions()
        self._supports_native_interpolation = self._check_native_interpolation()

        # For positional embedding interpolation
        self._original_img_size = self._get_original_img_size()
        self._pos_embed_cache = None

    @property
    def config(self) -> FeatureExtractorConfig:
        return FeatureExtractorConfig(
            hidden_size=self._hidden_size,
            patch_size=self._patch_size,
            num_layers=self._num_layers,
            num_special_tokens=1 + self._num_register_tokens,  # CLS + registers
            supports_intermediate=True,
        )

    def _check_has_attentions(self) -> bool:
        """Check if the underlying model supports output_attentions."""
        try:
            sig = inspect.signature(self._model.forward)
            return "output_attentions" in sig.parameters
        except (ValueError, TypeError):
            return False

    def _check_native_interpolation(self) -> bool:
        """Check if the model supports interpolate_pos_encoding parameter."""
        try:
            sig = inspect.signature(self._model.forward)
            return "interpolate_pos_encoding" in sig.parameters
        except (ValueError, TypeError):
            return False

    def _get_original_img_size(self) -> Tuple[int, int]:
        """Get the original image size the model was trained on."""
        try:
            if hasattr(self._model, 'config'):
                img_size = self._model.config.image_size
                if isinstance(img_size, (list, tuple)):
                    return tuple(img_size)
                return (img_size, img_size)
        except:
            pass
        return (224, 224)

    def _get_pos_embed_module(self):
        """Find the positional embedding module in the model."""
        candidates = [
            'embeddings.position_embeddings',
            'vit.embeddings.position_embeddings',
            'pos_embed',
            'encoder.pos_embed',
        ]

        for path in candidates:
            try:
                module = self._model
                for attr in path.split('.'):
                    module = getattr(module, attr)
                if isinstance(module, (nn.Parameter, torch.Tensor)):
                    return module, path
            except AttributeError:
                continue

        return None, None

    def _interpolate_pos_encoding(self, pixel_values: torch.Tensor):
        """Manually interpolate positional embeddings for variable input sizes."""
        _, _, H, W = pixel_values.shape
        current_size = (H, W)

        if current_size == self._original_img_size:
            return

        cache_key = (H, W)
        if self._pos_embed_cache is not None and self._pos_embed_cache[0] == cache_key:
            return

        pos_embed, pos_embed_path = self._get_pos_embed_module()

        if pos_embed is None:
            warnings.warn(
                "Could not find positional embeddings to interpolate. "
                "Model may not work correctly with non-standard image sizes."
            )
            return

        orig_h, orig_w = self._original_img_size
        orig_grid_h = orig_h // self._patch_size
        orig_grid_w = orig_w // self._patch_size

        new_grid_h = H // self._patch_size
        new_grid_w = W // self._patch_size

        num_prefix_tokens = 1 + self._num_register_tokens

        if pos_embed.shape[1] <= num_prefix_tokens:
            return

        prefix_tokens = pos_embed[:, :num_prefix_tokens, :]
        pos_tokens = pos_embed[:, num_prefix_tokens:, :]

        embedding_dim = pos_tokens.shape[-1]
        pos_tokens = pos_tokens.reshape(1, orig_grid_h, orig_grid_w, embedding_dim)
        pos_tokens = pos_tokens.permute(0, 3, 1, 2)

        pos_tokens = F.interpolate(
            pos_tokens,
            size=(new_grid_h, new_grid_w),
            mode='bicubic',
            align_corners=False
        )

        pos_tokens = pos_tokens.permute(0, 2, 3, 1)
        pos_tokens = pos_tokens.reshape(1, new_grid_h * new_grid_w, embedding_dim)

        new_pos_embed = torch.cat([prefix_tokens, pos_tokens], dim=1)

        module = self._model
        attrs = pos_embed_path.split('.')
        for attr in attrs[:-1]:
            module = getattr(module, attr)

        if isinstance(getattr(module, attrs[-1]), nn.Parameter):
            setattr(module, attrs[-1], nn.Parameter(new_pos_embed.to(pos_embed.device)))
        else:
            setattr(module, attrs[-1], new_pos_embed.to(pos_embed.device))

        self._pos_embed_cache = (cache_key, new_pos_embed)

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        indices: List[int],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ExtractedFeatures:
        """
        Extract features from the DINO model at specified layer indices.

        Args:
            pixel_values: Input images [B, C, H, W]
            indices: Layer indices to extract features from
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights

        Returns:
            ExtractedFeatures with features at specified indices
        """
        # Handle positional encoding interpolation if needed
        if not self._supports_native_interpolation:
            self._interpolate_pos_encoding(pixel_values)

        # Build kwargs for the underlying model
        model_kwargs = {"output_hidden_states": True}

        if output_attentions and self._has_attentions:
            model_kwargs["output_attentions"] = output_attentions

        if self._supports_native_interpolation:
            model_kwargs["interpolate_pos_encoding"] = True

        # Forward pass
        outputs = self._model(pixel_values, **model_kwargs)

        # `indices` are 0-based block indices (as in the official DINOv3 repo's
        # get_intermediate_layers). HF hidden_states[0] is the embedding output, so
        # block i lives at hidden_states[i + 1]. Like the official norm=True, apply
        # the final LayerNorm to every selected block output.
        norm = getattr(self._model, "norm", None) or getattr(self._model, "layernorm", None)
        features = []
        for idx in indices:
            feat = outputs.hidden_states[idx + 1]
            features.append(norm(feat) if norm is not None else feat)

        # Prepare outputs
        hidden_states = outputs.hidden_states if output_hidden_states else None
        attentions = None
        if output_attentions and hasattr(outputs, "attentions") and outputs.attentions is not None:
            attentions = outputs.attentions

        return ExtractedFeatures(
            features=features,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_register_tokens: int = 0,
        **kwargs,
    ) -> "DinoFeatureExtractor":
        """
        Create a DinoFeatureExtractor from a pretrained HuggingFace model.

        Args:
            model_name: HuggingFace model name (e.g., "facebook/dinov2-base")
            num_register_tokens: Number of register tokens (auto-detected for some models)
            **kwargs: Additional arguments

        Returns:
            DinoFeatureExtractor instance
        """
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Auto-detect config
        hidden_size = model.config.hidden_size
        patch_size = model.config.patch_size
        num_layers = model.config.num_hidden_layers

        # Auto-detect register tokens for known models
        if "reg" in model_name.lower() or "dinov3" in model_name.lower():
            num_register_tokens = num_register_tokens or 4

        return cls(
            model=model,
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_register_tokens=num_register_tokens,
        )
