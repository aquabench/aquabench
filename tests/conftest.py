"""Shared fixtures: a tiny synthetic dataset and tiny encoders that follow the contract."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from aquabench.download import write_dataset

NUM_CLASSES = 3
SIZE = (64, 64)  # divisible by 16 -> token grid 4x4
COLORS = np.array([[20, 40, 120], [200, 180, 40], [40, 200, 90]], dtype=np.uint8)

#: Small, fast training settings for tests (CPU, fp32, no workers).
FAST = {"epochs": 2, "batch_size": 2, "grad_accum_steps": 1, "num_workers": 0,
        "precision": "fp32", "log_every": 1}


def make_samples(n, seed, size=SIZE, ignore_border=True):
    """Images whose colour encodes the class, plus noise; masks with an ignore strip."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h, w = size
        mask = np.zeros((h, w), dtype=np.uint8)
        for c in range(1, NUM_CLASSES):
            y, x = rng.integers(0, h - 32), rng.integers(0, w - 32)
            mask[y:y + 32, x:x + 32] = c
        img = COLORS[mask] + rng.integers(0, 30, (h, w, 3), dtype=np.uint8)
        if ignore_border:
            mask[:, :2] = 255
        out.append((Image.fromarray(img), mask))
    return out


def write_toy(root, name="Toy", val_seed=2):
    meta = {"name": name, "num_classes": NUM_CLASSES, "ignore_index": 255, "size": list(SIZE),
            "class_names": ["background", "fish", "plant"]}
    write_dataset(Path(root) / name, meta, {
        "train": make_samples(8, seed=1),
        "val": make_samples(4, seed=val_seed),
        "test": make_samples(4, seed=3),
    })
    return Path(root)


@pytest.fixture
def data_root(tmp_path):
    return write_toy(tmp_path / "data")


@pytest.fixture(params=["synthetic", "real"])
def any_data_root(request, tmp_path):
    """Synthetic data always; the downloaded datasets too when $AQUABENCH_DATA is set."""
    if request.param == "real":
        real = os.environ.get("AQUABENCH_DATA")
        if not real:
            pytest.skip("set $AQUABENCH_DATA to also check the downloaded benchmark")
        names = sorted(d.name for d in Path(real).iterdir() if (d / "meta.json").exists())
        if not names:
            pytest.skip(f"no datasets in {real}")
        return Path(real), names
    return write_toy(tmp_path / "data"), ["Toy"]


class TinyEncoder(nn.Module):
    """Patchify conv + 4 conv blocks. ``pyramid=True`` returns all 4 maps (UPerNet),
    otherwise the last one (linear). Has BatchNorm so buffer changes are visible."""

    def __init__(self, patch_size=16, dim=8, pyramid=False):
        super().__init__()
        self.name = "tiny"
        self.patch = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(dim)
        self.blocks = nn.ModuleList(nn.Conv2d(dim, dim, 3, padding=1) for _ in range(4))
        self.pyramid = pyramid

    def forward(self, x):
        h = self.norm(self.patch(x))
        outs = []
        for b in self.blocks:
            h = torch.relu(b(h))
            outs.append(h)
        return outs if self.pyramid else outs[-1]


def tiny_encoder(run_type, seed=0, **kw):
    torch.manual_seed(seed)
    return TinyEncoder(pyramid=run_type != "linear", **kw)


class ListLogger:
    def __init__(self):
        self.records = []

    def log(self, metrics, step):
        self.records.append((step, dict(metrics)))

    def keys(self):
        return {k for _, m in self.records for k in m}
