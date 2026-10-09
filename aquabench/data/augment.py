"""Kornia GPU augmentation for semantic segmentation.

One `SegAugmentation` class driven by a small `PRESETS` dict — the same geometric
transforms hit image + mask, color transforms hit the image only, and the mask's
integer dtype is preserved. Applied train-only (the DataModule gates it on
`trainer.training`); images already arrive at the dataset's default size.
"""

import torch
import torch.nn as nn
import kornia.augmentation as K

# preset -> params; (brightness, contrast, saturation, hue) for color jitter.
PRESETS = {
    "none":   None,
    "light":  dict(p=0.5, hflip=True, vflip=False, rotate=0.0,  scale=None,       color=(0.1, 0.1, 0.1, 0.05)),
    "medium": dict(p=0.5, hflip=True, vflip=False, rotate=10.0, scale=(0.9, 1.1), color=(0.2, 0.2, 0.2, 0.10)),
    "heavy":  dict(p=0.7, hflip=True, vflip=True,  rotate=20.0, scale=(0.8, 1.2), color=(0.3, 0.3, 0.3, 0.15)),
}


class SegAugmentation(nn.Module):
    def __init__(self, img_size=(512, 512), preset: str = "medium"):
        super().__init__()
        self.img_size = img_size
        cfg = PRESETS.get(preset)
        self.transform = self._build(cfg) if cfg else None

    @staticmethod
    def _build(cfg):
        aug = []
        if cfg["hflip"]:
            aug.append(K.RandomHorizontalFlip(p=cfg["p"], same_on_batch=False))
        if cfg["vflip"]:
            aug.append(K.RandomVerticalFlip(p=cfg["p"], same_on_batch=False))
        if cfg["rotate"]:
            aug.append(K.RandomRotation(degrees=cfg["rotate"], p=cfg["p"], same_on_batch=False,
                                        resample="bilinear", align_corners=False))
        if cfg["scale"]:
            aug.append(K.RandomAffine(degrees=0, scale=cfg["scale"], p=cfg["p"], same_on_batch=False,
                                      resample="bilinear", align_corners=False))
        b, c, s, h = cfg["color"]
        aug.append(K.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h, p=cfg["p"], same_on_batch=False))
        return K.AugmentationSequential(*aug, data_keys=["input", "mask"], same_on_batch=False)

    def forward(self, images: torch.Tensor, masks: torch.Tensor):
        """images [B,C,H,W] in [0,1]; masks [B,H,W]. Same geometry on both; mask dtype preserved."""
        if self.transform is None:
            return images, masks
        is_int = masks.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)
        masks_4d = masks.unsqueeze(1).float() if masks.ndim == 3 else masks.float()
        aug_images, aug_masks = self.transform(images, masks_4d)
        aug_masks = aug_masks.squeeze(1)
        return aug_images, (aug_masks.long() if is_int else aug_masks)
