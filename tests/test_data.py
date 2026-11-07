"""Data layout, sizes and augmentation."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from aquabench.config import default_config, load_config
from aquabench.data import (DATASET_SPECS, DATASETS, SegDataset, augment_pair, check_dataset, input_size,
                            load_meta, resolve_datasets, token_grid, verify_data_root)

from conftest import write_toy


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


def _write_meta(root, folder, size, num_classes=8):
    (root / folder).mkdir()
    (root / folder / "meta.json").write_text(json.dumps({"num_classes": num_classes, "ignore_index": 255, "size": size}))
    return root / folder


@pytest.mark.parametrize("folder", ["SUIM", "suim"])
def test_benchmark_dataset_with_wrong_size_is_rejected(folder, tmp_path):
    """A benchmark dataset must have its expected size; the folder name is matched like find_dataset."""
    with pytest.raises(ValueError, match=r"SUIM must have size \[480, 640\], got \[272, 480\]"):
        load_meta(_write_meta(tmp_path, folder, [272, 480]))


def test_benchmark_dataset_with_wrong_class_count_is_rejected(tmp_path):
    """Same-size datasets differ in class count: UIIS10K data in a SUIM folder is caught."""
    with pytest.raises(ValueError, match="SUIM must have 8 classes, got 11"):
        load_meta(_write_meta(tmp_path, "SUIM", [480, 640], num_classes=11))


def test_expected_specs_pass_and_other_datasets_are_not_checked(tmp_path):
    """Every benchmark dataset loads with its spec; custom folder names keep any valid meta."""
    assert DATASET_SPECS.keys() == DATASETS.keys()
    for name, spec in DATASET_SPECS.items():
        meta = load_meta(_write_meta(tmp_path, name, list(spec["size"]), spec["num_classes"]))
        assert meta["size"] == list(spec["size"])
    assert load_meta(_write_meta(tmp_path, "MyData", [272, 480]))["size"] == [272, 480]


def test_verify_reports_wrong_benchmark_size(tmp_path):
    """verify_data_root lists a wrong size as a problem instead of raising."""
    _write_meta(tmp_path, "SUIM", [272, 480])
    assert any("must have size" in p for p in verify_data_root(tmp_path, ["SUIM"], check_masks=False))


@pytest.fixture
def toy_suim(tmp_path, monkeypatch):
    """A toy dataset named SUIM whose spec matches it, so only a changed count can fail."""
    monkeypatch.setitem(DATASET_SPECS, "SUIM", {"size": (64, 64), "num_classes": 3,
                                                "samples": {"train": 8, "val": 4, "test": 4}})
    return write_toy(tmp_path, name="SUIM")


def test_matching_sample_counts_pass(toy_suim):
    """A benchmark dataset with the expected samples per split passes both checks."""
    assert check_dataset(toy_suim / "SUIM")["num_classes"] == 3
    assert verify_data_root(toy_suim, ["SUIM"], check_masks=False) == []


def test_wrong_sample_count_is_rejected(toy_suim):
    """A missing sample is caught by check_dataset and listed by verify_data_root."""
    (toy_suim / "SUIM" / "test" / "images" / "000000.png").unlink()
    (toy_suim / "SUIM" / "test" / "masks" / "000000.png").unlink()
    with pytest.raises(ValueError, match="expected 4 samples, found 3"):
        check_dataset(toy_suim / "SUIM")
    assert any("expected 4 samples, found 3" in p for p in verify_data_root(toy_suim, ["SUIM"], check_masks=False))


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
    """Augmentation is off / on / on for linear / upernet_frozen / upernet_finetune; override works."""
    assert [default_config(r)["augment"] for r in ("linear", "upernet_frozen", "upernet_finetune")] == [False, True, True]
    assert load_config("linear", augment=True)["augment"] is True
    assert load_config("upernet_finetune", augment=False)["augment"] is False


def test_batch_arguments_override_config():
    """batch_size and grad_accum_steps arguments win over the config; invalid values are rejected."""
    cfg = load_config("linear", {"batch_size": 8, "grad_accum_steps": 4}, batch_size=4, grad_accum_steps=8)
    assert (cfg["batch_size"], cfg["grad_accum_steps"]) == (4, 8)
    assert load_config("linear", {"batch_size": 8})["batch_size"] == 8
    with pytest.raises(ValueError, match="batch_size must be"):
        load_config("linear", batch_size=0)
    with pytest.raises(ValueError, match="grad_accum_steps must be"):
        load_config("linear", grad_accum_steps=0)


def test_normalize_options(data_root):
    """imagenet, none and a callable all work on the [0, 1] image; anything else is rejected."""
    raw, _ = SegDataset(data_root / "Toy", "test", normalize="none")[0]
    assert raw.min() >= 0 and raw.max() <= 1
    img, _ = SegDataset(data_root / "Toy", "test", normalize="imagenet")[0]
    assert img.min() < 0
    doubled, _ = SegDataset(data_root / "Toy", "test", normalize=lambda x: x * 2)[0]
    assert torch.equal(doubled, raw * 2)
    with pytest.raises(ValueError, match="a callable, got 'clip'"):
        SegDataset(data_root / "Toy", "test", normalize="clip")
    with pytest.raises(ValueError, match="a callable, got 'clip'"):
        load_config("linear", {"normalize": "clip"})


def test_normalize_callable_must_keep_the_shape(data_root):
    """A callable that changes the shape or returns no tensor fails with a clear message."""
    with pytest.raises(ValueError, match="must return a tensor of shape"):
        SegDataset(data_root / "Toy", "test", normalize=lambda x: x[0])[0]
    with pytest.raises(ValueError, match="got float"):
        SegDataset(data_root / "Toy", "test", normalize=lambda x: 1.0)[0]


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
