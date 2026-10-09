"""Shared encoder-family and parameter-count metadata, used by both compute_rankings.py
and make_ranking_figures.py so the two never drift apart.
"""

from __future__ import annotations

# encoder families, in a fixed order, mapping vitb-tier name -> largest-tier name
FAMILIES: dict[str, tuple[str, str]] = {
    "dinov3": ("dinov3_base", "dinov3_7b"),
    "franca": ("franca_base", "franca_giant"),
    "radio":  ("radio_v3_base", "radio_v4_h"),
    "eva":    ("eva02_base", "eva02_large"),
    "vjepa":  ("vjepa21_vitb", "vjepa21_vitg"),
}
VITB_ENCODERS = [v[0] for v in FAMILIES.values()]
LARGEST_ENCODERS = [v[1] for v in FAMILIES.values()]
RESNET_ENCODER = "resnet50"

# encoder -> family, inverse of FAMILIES (both tiers map to the same family)
ENCODER_TO_FAMILY: dict[str, str] = {
    encoder: family for family, (vitb, largest) in FAMILIES.items() for encoder in (vitb, largest)
}

# Approximate parameter counts (millions), from published model cards -- used only
# for bubble/size annotations, not for any quantitative claim.
PARAMS_M: dict[str, int] = {
    "dinov3_base": 86, "dinov3_7b": 6600,
    "franca_base": 86, "franca_giant": 1100,
    "radio_v3_base": 86, "radio_v4_h": 630,
    "eva02_base": 86, "eva02_large": 304,
    "vjepa21_vitb": 86, "vjepa21_vitg": 1011,
    "resnet50": 25,
}
