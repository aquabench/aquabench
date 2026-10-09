"""
V-JEPA 2.1 feature extractor (image pathway).

Loads the official V-JEPA 2.1 encoders via torch.hub (Meta does not host these on
HuggingFace) and returns the **frozen final-layer patch tokens** for dense
prediction. By design V-JEPA 2.1's last layer suffices — we do NOT concatenate
intermediate layers (this differs from V-JEPA 2).

Image pathway details (verified against facebookresearch/vjepa2):
    - A single image frame is fed as a 5D tensor [B, C, T=1, H, W]. T == 1 matches
      the encoder's ``img_temporal_dim_size`` and routes the input through the 2D
      conv patchifier (``patch_embed_img``: Conv3d with a (1, 16, 16) kernel), i.e.
      patch size 16, plus a learnable image modality embedding.
    - The modality embedding is *added* to the tokens (not concatenated), so there
      are no special/CLS tokens in the output: N = (H/16) * (W/16) patch tokens in
      row-major order, matching DenseBackbone's spatial reshape.
    - The encoder uses 3D RoPE with ``interpolate_rope=True``, so positional
      frequencies are interpolated to the input resolution inside each block.

Reference: https://github.com/facebookresearch/vjepa2
"""

import os
import pathlib
from typing import List

import torch
import torch.nn as nn

from aquabench.models.encoders.extractors.base import (
    FeatureExtractor,
    FeatureExtractorConfig,
    ExtractedFeatures,
)


# Catalog of supported V-JEPA 2.1 variants. Keys are the torch.hub entrypoints in
# facebookresearch/vjepa2; dims/depths are read off the encoder arch defs.
VJEPA_CONFIGS = {
    # ViT-B (80M), distilled from ViT-G -> the lightweight model.
    "vjepa2_1_vit_base_384": {
        "hub_name": "vjepa2_1_vit_base_384",
        "hidden_size": 768,
        "patch_size": 16,
        "num_layers": 12,
        # Top-level key holding the encoder weights in the checkpoint dict.
        "checkpoint_key": "ema_encoder",
    },
    # ViT-G (2B) -> the large / best-quality model.
    "vjepa2_1_vit_gigantic_384": {
        "hub_name": "vjepa2_1_vit_gigantic_384",
        "hidden_size": 1664,
        "patch_size": 16,
        "num_layers": 48,
        "checkpoint_key": "target_encoder",
    },
}

VJEPA_HUB_REPO = "facebookresearch/vjepa2"

# Direct CDN URLs — used instead of the hub's VJEPA_BASE_URL which currently
# points to http://localhost:8300 (see facebookresearch/vjepa2 #137).
VJEPA_CHECKPOINT_URLS: dict[str, str] = {
    "vjepa2_1_vit_base_384":     "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
    "vjepa2_1_vit_gigantic_384": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
}


def _get_or_download_checkpoint(model_name: str) -> pathlib.Path:
    # VJEPA_CACHE_DIR overrides everything; default is checkpoints/vjepa2/ at
    # the repo root (resolved from __file__ so it works regardless of DATA_ROOT).
    cache_dir = pathlib.Path(
        os.environ.get(
            "VJEPA_CACHE_DIR",
            str(pathlib.Path(__file__).parents[4] / "checkpoints" / "vjepa2"),
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = VJEPA_CHECKPOINT_URLS[model_name]
    filename = url.rsplit("/", 1)[-1]
    local_path = cache_dir / filename

    if not local_path.exists():
        print(f"Downloading V-JEPA 2.1 checkpoint → {local_path}")
        torch.hub.download_url_to_file(url, str(local_path), progress=True)

    return local_path


class VJepaFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for V-JEPA 2.1 image-pathway encoders (torch.hub).

    Returns the final-layer patch tokens [B, N, C] (N = H/16 * W/16). Intermediate
    layers are not used; when several indices are requested (e.g. a 4-level UPerNet
    head) the same final-layer map is returned for each, mirroring Franca's fallback.

    Example:
        >>> extractor = VJepaFeatureExtractor.from_pretrained("vjepa2_1_vit_base_384")
        >>> feats = extractor.extract_features(pixel_values, indices=[11])
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
            num_special_tokens=0,  # image modality embed is added, not concatenated; no CLS
            supports_intermediate=False,  # V-JEPA 2.1: the final layer suffices
        )

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        indices: List[int],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ExtractedFeatures:
        """Extract final-layer patch tokens via the V-JEPA 2.1 image pathway.

        Args:
            pixel_values: Input images [B, C, H, W].
            indices: Requested layer indices. V-JEPA 2.1 uses the final layer only,
                so the same final-layer map is returned for each requested index.

        Returns:
            ExtractedFeatures with `features` = [final_tokens] * len(indices),
            each [B, N, C] with N = (H/16) * (W/16).
        """
        # Image pathway: T=1 routes through the 2D-conv patchifier (patch 16).
        x = pixel_values.unsqueeze(2)  # [B, C, H, W] -> [B, C, T=1, H, W]

        with torch.set_grad_enabled(self.training):
            tokens = self._model(x)  # [B, N, C] final-layer patch tokens (RoPE interpolated)

        features = [tokens for _ in indices]
        return ExtractedFeatures(features=features, hidden_states=None, attentions=None)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **kwargs,
    ) -> "VJepaFeatureExtractor":
        """Load a V-JEPA 2.1 encoder with pretrained weights.

        The hub architecture is loaded with ``pretrained=False`` to bypass the
        broken ``VJEPA_BASE_URL`` (localhost) in facebookresearch/vjepa2 #137.
        Weights are fetched from ``VJEPA_CHECKPOINT_URLS`` and cached under
        ``$VJEPA_CACHE_DIR`` (default ``<repo>/checkpoints/vjepa2/``). The encoder
        sub-dict is selected by ``checkpoint_key`` and its ``module.``/``backbone.``
        prefixes stripped before a strict ``load_state_dict`` (mirrors the repo's
        ``_clean_backbone_key``).

        Args:
            model_name: Catalog key in VJEPA_CONFIGS, e.g. "vjepa2_1_vit_base_384".
        """
        if model_name not in VJEPA_CONFIGS:
            available = ", ".join(VJEPA_CONFIGS.keys())
            raise ValueError(f"Unknown V-JEPA model: {model_name}. Available: {available}")

        config = VJEPA_CONFIGS[model_name]

        # Build architecture only — avoids the hub's broken localhost download
        # (VJEPA_BASE_URL = http://localhost:8300 in facebookresearch/vjepa2 #137).
        encoder, _predictor = torch.hub.load(
            VJEPA_HUB_REPO,
            config["hub_name"],
            trust_repo=True,
            pretrained=False,
        )

        # Load weights from local cache or download from the real CDN.
        ckpt_path = _get_or_download_checkpoint(model_name)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # The checkpoint is {checkpoint_key: encoder_weights, "predictor": ...}; the
        # encoder weights are prefixed "module.backbone." (DDP + wrapper). Mirror the
        # repo's _clean_backbone_key: select the encoder sub-dict and strip prefixes.
        encoder_sd = state_dict[config["checkpoint_key"]]
        encoder_sd = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in encoder_sd.items()
        }
        encoder.load_state_dict(encoder_sd, strict=True)
        encoder.eval()

        return cls(
            model=encoder,
            hidden_size=config["hidden_size"],
            patch_size=config["patch_size"],
            num_layers=config["num_layers"],
        )
