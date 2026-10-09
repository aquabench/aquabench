"""Head axis: pluggable decoders that consume encoder feature maps.

Each head declares how the encoder should be configured (number of feature
levels, scales, lateral convs, channel width) and whether the encoder is frozen
by default. The build fn passes the encoder in as the `backbone` and returns a
module whose `forward(pixel_values)` gives **logits only** — the Task computes the
loss.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class HeadSpec:
    n_levels: int
    scales: List[float]
    use_lateral_convs: bool
    out_channels: Optional[int]   # None -> use the encoder's native channel width
    freeze_encoder: bool          # default freeze policy (overridable on CLI)
    build: Callable


def _build_linear(encoder, num_classes, token_grid=None, **_):
    from aquabench.models.heads.linear import LinearProbingSegmentationModel
    return LinearProbingSegmentationModel(
        backbone=encoder, num_classes=num_classes, hidden_size=encoder.out_channels,
        token_grid=token_grid,
    )


def _build_upernet(encoder, num_classes, n_levels, token_grid=None, **_):
    from aquabench.models.heads.upernet import UPerNetSegHead
    return UPerNetSegHead(
        encoder=encoder, num_classes=num_classes, in_channels=encoder.out_channels,
        n_levels=n_levels, token_grid=token_grid,
    )


HEADS = {
    # frozen encoder + 1x1 conv on the last (native-width) feature map
    "linear": HeadSpec(1, [1.0], False, None, True, _build_linear),
    # trainable encoder + our own UPerNet decoder (full fine-tuning / research)
    "upernet": HeadSpec(4, [4.0, 2.0, 1.0, 0.5], False, None, False, _build_upernet),
    # frozen encoder + our own UPerNet decoder (benchmark secondary evaluation)
    "upernet_frozen": HeadSpec(4, [4.0, 2.0, 1.0, 0.5], False, None, True, _build_upernet),
}
