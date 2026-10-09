"""Encoder axis: a flat catalog + a generic builder.

`build_encoder(name, out_indices, scales, use_lateral_convs, out_channels, img_size)`
returns a module producing `feature_maps` (DenseBackbone for ViTs via the existing
`from_{dino,franca,radio}` factories; ResNetBackboneAdapter for CNNs; DummyEncoder
for tests).

`out_indices` is generic: None/"auto" -> evenly-spaced over depth; "last" ->
last layer repeated; an int N -> N evenly-spaced; or an explicit list. For a linear
head (1 level) the default already resolves to the last layer.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

OutIndices = Union[None, str, int, List[int]]


@dataclass
class EncoderSpec:
    family: str  # dino | franca | radio | resnet | dummy
    model_id: str
    num_layers: int = 0
    num_register_tokens: int = 0
    weights: Optional[str] = None
    use_rasa_head: bool = False


# Representative catalog (extend freely — these reuse the same builders).
ENCODERS = {
    "dinov3_small":   EncoderSpec("dino", "facebook/dinov3-vits16-pretrain-lvd1689m", 12, 4),
    "dinov3_base":    EncoderSpec("dino", "facebook/dinov3-vitb16-pretrain-lvd1689m", 12, 4),
    "dinov3_large":   EncoderSpec("dino", "facebook/dinov3-vitl16-pretrain-lvd1689m", 24, 4),
    "dinov3_7b":      EncoderSpec("dino", "facebook/dinov3-vit7b16-pretrain-lvd1689m", 40, 4),
    "franca_base":    EncoderSpec("franca", "franca_vitb14", 12, 0, "IN21K", False),
    "franca_large":   EncoderSpec("franca", "franca_vitl14", 24, 0, "LAION", False),
    "franca_giant":   EncoderSpec("franca", "franca_vitg14", 40, 0, "LAION", False),
    "radio_v3_base":  EncoderSpec("radio", "c-radio_v3-b", 12),
    "radio_v3_large": EncoderSpec("radio", "c-radio_v3-l", 24),
    "radio_v4_h":     EncoderSpec("radio", "c-radio_v4-h", 32),
    "radio_v4_so400m":EncoderSpec("radio", "c-radio_v4-so400m", 27),
    "eva02_base":     EncoderSpec("eva", "eva02_base", 12),
    "eva02_large":    EncoderSpec("eva", "eva02_large", 24),
    "vjepa21_vitb":   EncoderSpec("vjepa", "vjepa2_1_vit_base_384", 12, 0),
    "vjepa21_vitg":   EncoderSpec("vjepa", "vjepa2_1_vit_gigantic_384", 48, 0),
    "resnet50":       EncoderSpec("resnet", "resnet50"),
    "resnet101":      EncoderSpec("resnet", "resnet101"),
    "dummy":          EncoderSpec("dummy", "dummy", 12),
}


def _evenly_spaced(num_layers: int, n: int) -> List[int]:
    step = max(1, num_layers // n)
    return [step * (i + 1) - 1 for i in range(n)]


def resolve_indices(out_indices: OutIndices, num_layers: int, n_levels: int) -> List[int]:
    if out_indices in (None, "auto"):
        return _evenly_spaced(num_layers, n_levels)
    if out_indices == "last":
        return [num_layers - 1] * n_levels
    if isinstance(out_indices, int):
        return _evenly_spaced(num_layers, out_indices)
    return list(out_indices)


def build_encoder(name, out_indices, scales, use_lateral_convs, out_channels, img_size):
    spec = ENCODERS[name]
    n_levels = len(scales)

    if spec.family == "dummy":
        from aquabench.models.encoders.dummy import DummyEncoder
        return DummyEncoder(scales=scales, out_channels=out_channels or 64)

    if spec.family == "resnet":
        from aquabench.models.encoders.resnet import ResNetBackboneAdapter
        return ResNetBackboneAdapter.from_timm(spec.model_id, out_channels=out_channels or 512)

    from aquabench.models.encoders.dense_backbone import DenseBackbone

    idx = resolve_indices(out_indices, spec.num_layers, n_levels)
    oc = out_channels or 512
    if spec.family == "dino":
        return DenseBackbone.from_dino(
            spec.model_id, backbone_indices=idx, scales=scales, out_channels=oc,
            num_register_tokens=spec.num_register_tokens, img_size=img_size[0],
            use_lateral_convs=use_lateral_convs,
        )
    if spec.family == "franca":
        return DenseBackbone.from_franca(
            spec.model_id, backbone_indices=idx, scales=scales, out_channels=oc,
            weights=spec.weights, use_rasa_head=spec.use_rasa_head, use_lateral_convs=use_lateral_convs,
        )
    if spec.family == "radio":
        return DenseBackbone.from_radio(
            spec.model_id, backbone_indices=idx, scales=scales, out_channels=oc,
            img_size=img_size[0], use_lateral_convs=use_lateral_convs,
        )
    if spec.family == "eva":
        return DenseBackbone.from_eva(
            spec.model_id, backbone_indices=idx, scales=scales, out_channels=oc,
            img_size=img_size[0], use_lateral_convs=use_lateral_convs,
        )
    if spec.family == "vjepa":
        return DenseBackbone.from_vjepa(
            spec.model_id, backbone_indices=idx, scales=scales, out_channels=oc,
            img_size=img_size[0], use_lateral_convs=use_lateral_convs,
        )
    raise ValueError(f"Unknown encoder family {spec.family!r}")
