"""Run types, encoder contract, freezing, logging, seeds, grad accumulation."""

import copy

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
