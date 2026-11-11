"""Run types, encoder contract, freezing, logging, seeds, grad accumulation."""

import copy
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import aquabench
from aquabench.config import default_config
from aquabench.train import train_epoch

from conftest import FAST, ListLogger, tiny_encoder, write_toy


def _run(enc, run_type, data_root, tmp_path, **kw):
    kw.setdefault("config", FAST)
    kw.setdefault("logger", ListLogger())
    return aquabench.run(enc, run_type=run_type, data_root=data_root, datasets="Toy",
                         output_dir=tmp_path / "runs", device="cpu", **kw)


@pytest.mark.parametrize("run_type, pyramid", [("linear", True), ("upernet_frozen", False), ("upernet_finetune", False)])
def test_linear_rejects_wrong_encoder_output(run_type, pyramid, data_root, tmp_path):
    """Clear error if the encoder returns a list for linear or a single map for UPerNet."""
    enc = tiny_encoder("linear")
    enc.pyramid = pyramid
    with pytest.raises(aquabench.EncoderContractError, match="expects encoder.forward"):
        _run(enc, run_type, data_root, tmp_path)


@pytest.mark.parametrize("run_type", ["linear", "upernet_frozen", "upernet_finetune"])
def test_wrong_patch_size_is_rejected(run_type, data_root, tmp_path):
    """A patch 8 encoder run with the default patch_size=16 misses the token grid."""
    with pytest.raises(aquabench.EncoderContractError, match="token grid is \\[4, 4\\]"):
        _run(tiny_encoder(run_type, patch_size=8), run_type, data_root, tmp_path)


@pytest.mark.parametrize("run_type", ["linear", "upernet_frozen"])
def test_matching_patch_size_runs(run_type, data_root, tmp_path):
    """The same patch 14 encoder runs once patch_size says so."""
    res = _run(tiny_encoder(run_type, patch_size=14), run_type, data_root, tmp_path, patch_size=14)
    assert 0.0 <= res["Toy"].test_miou <= 1.0


class CnnPyramid(nn.Module):
    """Stages at strides 4, 8, 16, 32, as a CNN returns them."""

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(nn.Conv2d(3, 8, s, stride=s) for s in (4, 8, 16, 32))

    def forward(self, x):
        return [stage(x) for stage in self.stages]


def test_cnn_pyramid_uses_its_total_stride(data_root, tmp_path):
    """A CNN pyramid passes with patch_size=32, its total stride, and fails with the default."""
    res = _run(CnnPyramid(), "upernet_frozen", data_root, tmp_path, patch_size=32)
    assert 0.0 <= res["Toy"].test_miou <= 1.0
    with pytest.raises(aquabench.EncoderContractError, match="total stride"):
        _run(CnnPyramid(), "upernet_frozen", data_root, tmp_path / "default")


def test_bad_second_dataset_stops_the_run_before_training(data_root, tmp_path):
    """All datasets are checked first, so a bad later one does not waste the earlier runs."""
    (data_root / "SUIM").mkdir()
    (data_root / "SUIM" / "meta.json").write_text('{"num_classes": 8, "ignore_index": 255, "size": [272, 480]}')
    with pytest.raises(ValueError, match="SUIM must have size"):
        aquabench.run(tiny_encoder("linear"), run_type="linear", data_root=data_root, datasets=["Toy", "SUIM"],
                      config=FAST, logger=ListLogger(), output_dir=tmp_path / "runs", device="cpu")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("run_type", ["linear", "upernet_frozen"])
def test_frozen_encoder_stays_unchanged(run_type, data_root, tmp_path):
    """Encoder weights and buffers (BN stats) are identical before and after a frozen run."""
    enc = tiny_encoder(run_type)
    before = copy.deepcopy(enc.state_dict())
    _run(enc, run_type, data_root, tmp_path)
    after = enc.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.requires_grad for p in enc.parameters())  # caller's flags restored


def test_finetune_updates_encoder(data_root, tmp_path):
    """upernet_finetune changes the encoder weights."""
    enc = tiny_encoder("upernet_finetune")
    before = copy.deepcopy(enc.state_dict())
    _run(enc, "upernet_finetune", data_root, tmp_path)
    assert any(not torch.equal(before[k], v) for k, v in enc.state_dict().items())


def test_logger_receives_metrics(data_root, tmp_path):
    """The user logger gets train loss, val mIoU and the final test mIoU."""
    logger = ListLogger()
    res = _run(tiny_encoder("linear"), "linear", data_root, tmp_path, logger=logger)
    keys = logger.keys()
    assert {"Toy/train/loss", "Toy/val/miou", "Toy/test/miou", "Toy/test/iou/fish"} <= keys
    assert logger.records[-1][1]["Toy/test/miou"] == res.test_miou
    steps = [s for s, _ in logger.records]
    assert steps == sorted(steps)


def test_seed_reproducible(data_root, tmp_path):
    """Same seed (with augmentation on) gives identical mIoU."""
    a = _run(tiny_encoder("upernet_finetune"), "upernet_finetune", data_root, tmp_path / "a", seed=3)
    b = _run(tiny_encoder("upernet_finetune"), "upernet_finetune", data_root, tmp_path / "b", seed=3)
    assert a.test_miou == b.test_miou
    assert a.per_class_iou == b.per_class_iou


def test_reported_metric_is_last_epoch(tmp_path):
    """Val never changes the reported number: a different val split gives the same test result."""
    root_a = write_toy(tmp_path / "a", val_seed=2)
    root_b = write_toy(tmp_path / "b", val_seed=99)
    cfg = {**FAST, "lr": 1e-2, "augment": False}  # learns enough for the two val splits to score differently
    a = _run(tiny_encoder("upernet_finetune"), "upernet_finetune", root_a, tmp_path / "ra", config=cfg)
    b = _run(tiny_encoder("upernet_finetune"), "upernet_finetune", root_b, tmp_path / "rb", config=cfg)
    assert a.val_miou_last != b.val_miou_last
    assert a.test_miou == b.test_miou and a.per_class_iou == b.per_class_iou


def test_datasets_are_independent(tmp_path):
    """In a multi-dataset finetune call, each dataset starts from the original encoder:
    B's result equals running B alone."""
    root = write_toy(tmp_path / "data", name="A")
    write_toy(root, name="B", val_seed=5)
    both = aquabench.run(tiny_encoder("upernet_finetune"), run_type="upernet_finetune", data_root=root,
                         datasets=["A", "B"], config=FAST, output_dir=tmp_path / "r1", device="cpu",
                         logger=ListLogger())
    alone = aquabench.run(tiny_encoder("upernet_finetune"), run_type="upernet_finetune", data_root=root,
                          datasets="B", config=FAST, output_dir=tmp_path / "r2", device="cpu",
                          logger=ListLogger())
    assert list(both) == ["A", "B"]
    assert both["B"].test_miou == alone["B"].test_miou


def test_grad_accum_equivalent():
    """16 x 2 gives (numerically close to) the same update as 32 x 1 on a tiny model."""
    torch.manual_seed(0)
    x = torch.randn(32, 3, 8, 8)
    y = torch.randint(0, 4, (32, 8, 8))
    model = nn.Conv2d(3, 4, 3, padding=1)
    params = {}
    for bs, accum in [(32, 1), (16, 2)]:
        m = copy.deepcopy(model)
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        loader = DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=False)
        train_epoch(m, loader, opt, grad_accum_steps=accum, ignore_index=255, device=torch.device("cpu"))
        params[accum] = [p.detach() for p in m.parameters()]
    for p1, p2 in zip(params[1], params[2]):
        assert torch.allclose(p1, p2, atol=1e-6)
    assert not torch.allclose(params[1][0], model.weight)  # an update happened


def test_all_run_types_50_epochs():
    """Shipped configs agree on 50 epochs and an effective batch of 32."""
    for rt in aquabench.RUN_TYPES:
        cfg = default_config(rt)
        assert cfg["epochs"] == 50
        assert cfg["batch_size"] * cfg["grad_accum_steps"] == 32


def test_lr_factors():
    """Head: plain cosine. Encoder: 0 while frozen, linear ramp, then the same cosine."""
    from aquabench.train import lr_factors

    head, enc = lr_factors(10, "cosine", freeze_epochs=1, warmup_epochs=2)
    assert head(0) == 1.0 and head(5) == pytest.approx(0.5)
    assert enc(0) == 0.0
    assert enc(1) == pytest.approx(0.5 * head(1)) and enc(2) == pytest.approx(head(2)) and enc(7) == head(7)
    head, enc = lr_factors(10, "none")
    assert head(3) == enc(3) == 1.0


def test_finetune_warmup_defaults_and_validation():
    """Only upernet_finetune has warm-up settings; impossible values are rejected."""
    from aquabench.config import load_config

    cfg = default_config("upernet_finetune")
    assert (cfg["encoder_lr_scale"], cfg["freeze_encoder_epochs"], cfg["encoder_warmup_epochs"]) == (0.1, 1, 2)
    for rt in ("linear", "upernet_frozen"):
        assert "freeze_encoder_epochs" not in default_config(rt)
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(rt, {"freeze_encoder_epochs": 1})
    with pytest.raises(ValueError, match="never trains"):
        load_config("upernet_finetune", {"epochs": 2, "freeze_encoder_epochs": 2})
    with pytest.raises(ValueError, match="encoder_lr_scale"):
        load_config("upernet_finetune", {"encoder_lr_scale": 0})


class SnapshotLogger(ListLogger):
    """Keeps a copy of the encoder weights and buffers at the end of each epoch."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder, self.snapshots = encoder, {}

    def log(self, metrics, step):
        super().log(metrics, step)
        if "Toy/epoch" in metrics:
            self.snapshots[metrics["Toy/epoch"]] = copy.deepcopy(self.encoder.state_dict())


def test_finetune_freezes_then_trains_the_encoder(data_root, tmp_path):
    """Epoch 1 leaves the encoder (weights and BatchNorm statistics) untouched; later epochs train it."""
    enc = tiny_encoder("upernet_finetune")
    before = copy.deepcopy(enc.state_dict())
    logger = SnapshotLogger(enc)
    _run(enc, "upernet_finetune", data_root, tmp_path, logger=logger, config={**FAST, "epochs": 3})
    assert all(torch.equal(before[k], logger.snapshots[1][k]) for k in before)
    assert any(not torch.equal(before[k], logger.snapshots[3][k]) for k in before)
    enc_lr = [m["Toy/train/encoder_lr"] for _, m in logger.records if "Toy/train/encoder_lr" in m]
    # epoch 2 (index 1): half the ramp times the cosine factor; epoch 3 is lower again
    assert enc_lr[0] == 0.0 and max(enc_lr) == pytest.approx(1e-5 * 0.5 * 0.5 * (1 + math.cos(math.pi / 3)))


def test_frozen_run_types_have_one_learning_rate(data_root, tmp_path):
    logger = ListLogger()
    _run(tiny_encoder("upernet_frozen"), "upernet_frozen", data_root, tmp_path, logger=logger)
    assert "Toy/train/lr" in logger.keys() and "Toy/train/encoder_lr" not in logger.keys()


class NoGradEncoder(nn.Module):
    """Follows the output contract, but cuts the gradient."""

    def __init__(self):
        super().__init__()
        self.inner = tiny_encoder("upernet_finetune")

    def forward(self, x):
        with torch.no_grad():
            return self.inner(x)


class NoParameters(nn.Module):
    def forward(self, x):
        return [torch.nn.functional.avg_pool2d(x, 16)] * 4


def test_finetune_needs_an_encoder_that_can_be_trained(data_root, tmp_path):
    """Full fine-tuning: an encoder that detaches its outputs or has no weights is rejected at
    the start; the same encoder is fine for upernet_frozen."""
    with pytest.raises(aquabench.EncoderContractError, match="carry no gradient"):
        _run(NoGradEncoder(), "upernet_finetune", data_root, tmp_path)
    with pytest.raises(aquabench.EncoderContractError, match="has no parameters"):
        _run(NoParameters(), "upernet_finetune", data_root, tmp_path)
    assert 0.0 <= _run(NoGradEncoder(), "upernet_frozen", data_root, tmp_path)["Toy"].test_miou <= 1.0


def test_finetune_trains_parameters_the_caller_froze(data_root, tmp_path):
    """Every parameter is trained whatever its requires_grad flag, and the flag is restored."""
    enc = tiny_encoder("upernet_finetune")
    enc.patch.weight.requires_grad_(False)
    before = enc.patch.weight.detach().clone()
    res = _run(enc, "upernet_finetune", data_root, tmp_path)
    assert not torch.equal(before, enc.patch.weight)
    assert enc.patch.weight.requires_grad is False
    info = res["Toy"].run_info["encoder"]
    assert info["trained_parameters"] == info["parameters"] == sum(p.numel() for p in enc.parameters())


def test_linear_caches_features_and_gives_the_same_result(data_root, tmp_path):
    """Features computed once give the result of running the encoder every epoch, and the
    cache folder is gone afterwards."""
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "a", cache_dir=cache)
    plain = _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "b", config={**FAST, "cache_features": False})
    assert cached.run_info["features_cached"] is True and plain.run_info["features_cached"] is False
    assert cached.test_miou == pytest.approx(plain.test_miou, abs=1e-6)
    assert cached.val_miou_last == pytest.approx(plain.val_miou_last, abs=1e-6)
    assert list(cache.iterdir()) == []


def test_cache_runs_the_encoder_once_per_sample(data_root, tmp_path):
    """8 train + 4 val + 4 test images and the probe: 17 images in total, whatever the epochs."""
    enc, seen = tiny_encoder("linear"), []
    enc.register_forward_hook(lambda m, inputs, out: seen.append(len(inputs[0])))
    _run(enc, "linear", data_root, tmp_path, config={**FAST, "epochs": 3})
    assert sum(seen) == 17


def test_augmentation_turns_the_cache_off(data_root, tmp_path):
    assert _run(tiny_encoder("linear"), "linear", data_root, tmp_path, augment=True).run_info["features_cached"] is False


def test_cache_without_space_falls_back(data_root, tmp_path, monkeypatch):
    """A full cache folder gives a warning and a normal run, not a crash."""
    import aquabench.cache

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(aquabench.cache, "_create", full)
    with pytest.warns(UserWarning, match="features not cached"):
        res = _run(tiny_encoder("linear"), "linear", data_root, tmp_path)
    assert res.run_info["features_cached"] is False and 0.0 <= res.test_miou <= 1.0
