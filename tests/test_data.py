"""Data layout, sizes and augmentation."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from aquabench.config import default_config, load_config
from aquabench.data import (SegDataset, augment_pair, input_size, load_meta, resolve_datasets,
                            token_grid, verify_data_root)


def test_download_creates_expected_layout(any_data_root):
    """Every dataset has meta.json, train/val/test splits and matching image/mask pairs."""
    root, names = any_data_root
    assert verify_data_root(root, names, check_masks=False) == []


def test_masks_in_label_range(any_data_root):
    """Mask values are only in [0, num_classes) or the ignore index."""
    root, names = any_data_root
    problems = verify_data_root(root, names, check_masks=True)
    assert not [p for p in problems if "values" in p], problems


def test_dataset_size_divisible_by_16(any_data_root, tmp_path):
    """Every dataset's defined size is a multiple of 16; a bad size is rejected at load."""
    root, names = any_data_root
    for d in resolve_datasets(root, names):
        h, w = load_meta(d)["size"]
        assert h % 16 == 0 and w % 16 == 0
    (tmp_path / "Bad").mkdir()
    (tmp_path / "Bad" / "meta.json").write_text(json.dumps({"num_classes": 2, "ignore_index": 255, "size": [60, 64]}))
    with pytest.raises(ValueError, match="divisible by 16"):
        load_meta(tmp_path / "Bad")


def test_input_size_matches_token_grid():
    """Input size = token grid (size / 16) x patch size, aspect ratio kept."""
    assert token_grid((448, 768)) == (28, 48)
    assert input_size((448, 768), 16) == (448, 768)
    assert input_size((448, 768), 14) == (392, 672)


def test_patch14_keeps_token_count(data_root):
    """Patch 14 shrinks the image but keeps the token grid; the mask stays at dataset size."""
    img16, mask16 = SegDataset(data_root / "Toy", "test", patch_size=16)[0]
    img14, mask14 = SegDataset(data_root / "Toy", "test", patch_size=14)[0]
    assert img16.shape[-2:] == (64, 64) and img14.shape[-2:] == (56, 56)
    assert (img16.shape[-2] // 16, img16.shape[-1] // 16) == (img14.shape[-2] // 14, img14.shape[-1] // 14)
    assert torch.equal(mask16, mask14) and mask14.shape == (64, 64)


def test_augment_defaults_per_run_type():
    """Augmentation is off / off / on for linear / upernet_frozen / upernet_finetune; override works."""
    assert [default_config(r)["augment"] for r in ("linear", "upernet_frozen", "upernet_finetune")] == [False, False, True]
    assert load_config("linear", augment=True)["augment"] is True
    assert load_config("upernet_finetune", augment=False)["augment"] is False


def test_no_augment_is_deterministic(data_root):
    """Without augmentation the same sample gives the identical tensor twice."""
    ds = SegDataset(data_root / "Toy", "train", augment=False)
    a, b = ds[0], ds[0]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_augment_never_pads():
    """Augmented image / mask always have the target size and no artificial ignore border."""
    torch.manual_seed(0)
    img = torch.rand(3, 48, 80)
    mask = torch.randint(0, 3, (48, 80))
    for _ in range(50):
        ai, am = augment_pair(img, mask)
        assert ai.shape == img.shape and am.shape == mask.shape
        assert am.max() < 3 and am.min() >= 0
