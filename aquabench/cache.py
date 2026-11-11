"""Frozen features computed once and reused every epoch.

Used for ``linear`` without augmentation: the encoder is frozen and the images never change, so
its features are the same in every epoch. Each split is encoded once into two ``.npy`` files
(features and masks) that are read back through a memory map, so they don't have to fit in RAM.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from aquabench.train import autocast


def feature_dtype(precision: str):
    """bf16 features are stored as float16, which holds every bf16 value in its range exactly."""
    return np.float16 if precision == "bf16" else np.float32


def cache_bytes(samples: int, feature_shape: Sequence[int], mask_size: Sequence[int], precision: str) -> int:
    """Disk space one split needs: features ``[N, C, h, w]`` plus uint8 masks ``[N, H, W]``."""
    features = int(np.prod(feature_shape)) * np.dtype(feature_dtype(precision)).itemsize
    return samples * (features + int(np.prod(mask_size)))


def _create(path: Path, dtype, shape) -> np.memmap:
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))
    if hasattr(os, "posix_fallocate"):  # reserve the space now: a full disk fails here, not mid-write
        with open(path, "r+b") as f:
            os.posix_fallocate(f.fileno(), 0, os.path.getsize(path))
    return arr


class CachedFeatures(Dataset):
    """One encoded split. Returns ``(features [C, h, w] float32, mask [H, W] long)``."""

    def __init__(self, features_path: Union[str, Path], masks_path: Union[str, Path]):
        self.features = np.load(features_path, mmap_mode="r")
        self.masks = np.load(masks_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int):
        return (torch.from_numpy(np.array(self.features[i], dtype=np.float32)),
                torch.from_numpy(np.array(self.masks[i], dtype=np.int64)))


@torch.no_grad()
def cache_features(encoder: nn.Module, loader, folder: Union[str, Path], split: str, *,
                   feature_shape: Sequence[int], device: torch.device, precision: str) -> CachedFeatures:
    """Encode every sample of ``loader`` (in order) into ``folder`` and return the cached split.

    Raises ``OSError`` if ``folder`` has too little space, and ``ValueError`` if a feature
    does not fit the storage type.
    """
    folder, n = Path(folder), len(loader.dataset)
    dtype = feature_dtype(precision)
    features_path, masks_path = folder / f"{split}_features.npy", folder / f"{split}_masks.npy"
    features = _create(features_path, dtype, (n, *feature_shape))
    masks = _create(masks_path, np.uint8, (n, *loader.dataset.size))
    encoder.eval()
    i = 0
    for x, y in loader:
        with autocast(device, precision):
            out = encoder(x.to(device, non_blocking=True))
        chunk = out.float().cpu().numpy().astype(dtype)
        if not np.isfinite(chunk).all():
            raise ValueError(f"{split}: encoder features are not finite as {np.dtype(dtype).name}; "
                             f"set \"cache_features\": false in the config")
        features[i:i + len(chunk)] = chunk
        masks[i:i + len(chunk)] = y.numpy().astype(np.uint8)
        i += len(chunk)
    features.flush()
    masks.flush()
    del features, masks
    return CachedFeatures(features_path, masks_path)
