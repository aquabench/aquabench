"""UIQM — Underwater Image Quality Measure (Panetta et al., 2015).

Reference:
  K. Panetta, C. Gao, S. Agaian, "Human-Visual-System-Inspired Underwater
  Image Quality Measures", IEEE Journal of Oceanic Engineering, 2015.

Standard combination coefficients:
  UIQM = 0.0282·UICM + 0.2953·UISM + 3.5753·UIConM

This follows the reference implementation from the FUnIE-GAN paper's evaluation
code (https://github.com/xahidbuffon/FUnIE-GAN/blob/master/Evaluation/uqim_utils.py),
which is the de facto standard used across the underwater image enhancement
literature for reporting UIQM. Constants (e.g. lambda_b=0.144 in UISM rather than
the "standard" luma weight 0.114, window_size=10) are kept exactly as in that
reference so scores are comparable to numbers reported elsewhere.
"""

from __future__ import annotations

import math

import numpy as np


def _mu_a(x: np.ndarray, alpha_l: float = 0.1, alpha_r: float = 0.1) -> float:
    """Asymmetric alpha-trimmed mean, matching the reference off-by-one exactly
    (weight divides by K-T_a_L-T_a_R but the sum starts one index later)."""
    sorted_x = np.sort(x)
    k = len(sorted_x)
    t_a_l = math.ceil(alpha_l * k)
    t_a_r = math.floor(alpha_r * k)
    s, e = t_a_l + 1, k - t_a_r
    if e <= s:
        return float(sorted_x.mean()) if k > 0 else 0.0
    weight = 1.0 / (k - t_a_l - t_a_r)
    return float(weight * sorted_x[s:e].sum())


def _s_a(x: np.ndarray, mu: float) -> float:
    return float(np.mean((x - mu) ** 2))


def _uicm(img: np.ndarray) -> float:
    """Chroma-based color measurement.  img: (H,W,3) float32 in [0,1]."""
    r, g, b = img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()
    rg = r - g
    yb = 0.5 * (r + g) - b

    mu_rg, mu_yb = _mu_a(rg), _mu_a(yb)
    s_rg, s_yb = _s_a(rg, mu_rg), _s_a(yb, mu_yb)

    l = math.sqrt(mu_rg ** 2 + mu_yb ** 2)
    r_ = math.sqrt(s_rg + s_yb)
    return (-0.0268 * l) + (0.1586 * r_)


def _sobel_mag(x: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    dx = ndimage.sobel(x, axis=1)
    dy = ndimage.sobel(x, axis=0)
    mag = np.hypot(dx, dy)
    max_val = mag.max()
    if max_val > 0:
        mag = mag * (255.0 / max_val)
    return mag


def _eme(x: np.ndarray, window_size: int = 10) -> float:
    """Enhancement measure estimation: mean log(block_max/block_min) over a
    non-overlapping window_size x window_size grid."""
    h, w = x.shape
    k2, k1 = h // window_size, w // window_size
    if k1 == 0 or k2 == 0:
        return 0.0
    cropped = x[: k2 * window_size, : k1 * window_size]
    blocks = (
        cropped.reshape(k2, window_size, k1, window_size)
        .transpose(0, 2, 1, 3)
        .reshape(k2 * k1, -1)
    )
    block_max = blocks.max(axis=1)
    block_min = blocks.min(axis=1)
    valid = (block_min > 0) & (block_max > 0)
    val = np.log(block_max[valid] / block_min[valid]).sum()
    return float((2.0 / (k1 * k2)) * val)


def _uism(img: np.ndarray) -> float:
    """Sharpness measurement: per-channel Sobel-edge-map x intensity, then EME.
    img: (H,W,3) float32 in [0,1]."""
    lambdas = (0.299, 0.587, 0.144)  # matches the reference exactly (not 0.114)
    total = 0.0
    for c, lam in enumerate(lambdas):
        ch = img[:, :, c]
        edge_map = _sobel_mag(ch) * ch
        total += lam * _eme(edge_map, window_size=10)
    return total


def _uiconm(img: np.ndarray, window_size: int = 10) -> float:
    """Log-AMEE contrast measure, jointly over R/G/B within each block.
    img: (H,W,3) float32 in [0,1]."""
    h, w, _ = img.shape
    k2, k1 = h // window_size, w // window_size
    if k1 == 0 or k2 == 0:
        return 0.0
    cropped = img[: k2 * window_size, : k1 * window_size, :]
    blocks = (
        cropped.reshape(k2, window_size, k1, window_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(k2 * k1, -1)
    )
    block_max = blocks.max(axis=1)
    block_min = blocks.min(axis=1)
    top = block_max - block_min
    bot = block_max + block_min
    valid = (top > 0) & (bot > 0)
    ratio = top[valid] / bot[valid]
    val = (ratio * np.log(ratio)).sum()
    return float((-1.0 / (k1 * k2)) * val)


def uiqm(img: np.ndarray) -> float:
    """Compute UIQM for a single image.

    Args:
        img: uint8 RGB array of shape (H, W, 3).

    Returns:
        UIQM score as a float.
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * _uicm(img) + c2 * _uism(img) + c3 * _uiconm(img)
