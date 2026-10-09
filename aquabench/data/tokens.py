"""Match the encoder token grid across patch sizes via minimal scaling.

To compare encoders with *different* patch sizes on an equal footing, every
encoder is made to emit the same token grid per task. `target_token_grid` fixes
that grid `(gh, gw)` from the task's default image size at a reference patch of
16; `resize_to_tokens` then resizes the input directly to `(gh*p, gw*p)` so a
`patch_size` encoder lands on exactly that grid. Because the grid is derived from
the task's own aspect ratio, this is a sub-percent stretch ("minimal scaling")
and needs **no padding** — no reflect/black border tokens for any encoder.

`pad_to_patch` remains only as a fallback for heads built without a token grid
(it reflect-pads H,W up to a multiple of the patch size so a ViT accepts them).
Loss/metric run against the un-padded labels: heads return logits at their native
(token/decoder) resolution and the task upsamples to the label resolution.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

#: Patch size that defines the shared token budget (DINOv3/RADIO are 16).
REFERENCE_PATCH_SIZE = 16


def pad_to_patch(image: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Reflect-pad `image` to a multiple of `patch_size`; return (padded, original_hw)."""
    _, _, h, w = image.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image, (h, w)


def target_token_grid(img_size: Tuple[int, int], reference_patch: int = REFERENCE_PATCH_SIZE) -> Tuple[int, int]:
    """The per-task (gh, gw) token grid every encoder should match.

    Fixed by a reference patch (16) on the task's default image size, so an
    encoder with a smaller patch shrinks its input to produce this same count.
    """
    h, w = img_size
    return max(1, round(h / reference_patch)), max(1, round(w / reference_patch))


def resize_to_tokens(image: torch.Tensor, patch_size: int, grid: Tuple[int, int]) -> torch.Tensor:
    """Resize `image` to exactly `grid=(gh, gw)` * `patch_size` pixels so the
    encoder emits exactly that token grid — no padding.

    The grid is derived from the task aspect ratio at patch 16, so the aspect
    change here is a sub-percent stretch (minimal scaling) rather than a crop or
    a padded border.
    """
    gh, gw = grid
    th, tw = gh * patch_size, gw * patch_size
    if tuple(image.shape[-2:]) != (th, tw):
        image = F.interpolate(image, size=(th, tw), mode="bilinear", align_corners=False)
    return image
