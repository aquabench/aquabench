"""Run folder, result files, checkpoints and evaluate()."""

import json

import pytest

import aquabench

from conftest import FAST, ListLogger, tiny_encoder


def _run(enc, run_type, data_root, output_dir, **kw):
    return aquabench.run(enc, run_type=run_type, data_root=data_root, datasets="Toy", config=FAST,
                         output_dir=output_dir, device="cpu", logger=ListLogger(), **kw)


def test_run_folder_contents(data_root, tmp_path):
    """Default run creates config / history / result / checkpoint in the expected path."""
    res = _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "runs", seed=0)
    run_dir = tmp_path / "runs" / "tiny" / "linear" / "Toy_seed0"
    assert {p.name for p in run_dir.iterdir()} == {"config.json", "history.json", "result.json", "checkpoint.pt"}
    result = json.loads((run_dir / "result.json").read_text())
    assert result["test_miou"] == pytest.approx(res.test_miou)
    assert result["dataset"] == "Toy" and result["run_type"] == "linear" and result["seed"] == 0
    assert result["aquabench_version"] == aquabench.__version__
    assert len(result["per_class_iou"]) == 3
    history = json.loads((run_dir / "history.json").read_text())
    assert [h["epoch"] for h in history] == [1, 2]
    assert history[-1]["val_miou"] == pytest.approx(result["val_miou_last"])


def test_config_copied_to_run_folder(data_root, tmp_path):
    """The effective config (defaults + user overrides + run params) is written to config.json."""
    res = _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "runs", patch_size=16, augment=True)
    cfg = json.loads((tmp_path / "runs" / "tiny" / "linear" / "Toy_seed0" / "config.json").read_text())
    assert cfg == res.config
    assert cfg["augment"] is True and cfg["epochs"] == FAST["epochs"] and cfg["lr"] == 1e-3
    assert cfg["patch_size"] == 16 and cfg["run_type"] == "linear"
    _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "batch", batch_size=4, grad_accum_steps=3)
    cfg = json.loads((tmp_path / "batch" / "tiny" / "linear" / "Toy_seed0" / "config.json").read_text())
    assert (cfg["batch_size"], cfg["grad_accum_steps"]) == (4, 3)


def _times_two(img):
    return img * 2


class Scale:
    def __call__(self, img):
        return img * 2

    def __repr__(self):
        return "Scale(2)"


@pytest.mark.parametrize("fn, label", [(_times_two, "callable:_times_two"), (Scale(), "callable:Scale(2)")])
def test_normalize_callable_is_recorded_by_name(fn, label, data_root, tmp_path):
    """The run works, config.json and the checkpoint stay plain, and evaluate needs the callable again."""
    enc = tiny_encoder("linear")
    res = _run(enc, "linear", data_root, tmp_path / "runs", normalize=fn)
    run_dir = tmp_path / "runs" / "tiny" / "linear" / "Toy_seed0"
    assert json.loads((run_dir / "config.json").read_text())["normalize"] == label
    assert json.loads((run_dir / "result.json").read_text())["config"]["normalize"] == label
    with pytest.raises(ValueError, match="pass it again"):
        aquabench.evaluate(enc, run_dir, data_root=data_root, device="cpu", logger=ListLogger())
    again = aquabench.evaluate(enc, run_dir, data_root=data_root, device="cpu", logger=ListLogger(), normalize=fn)
    assert again.test_miou == pytest.approx(res.test_miou)
    assert again.config["normalize"] == label


def test_normalize_callable_gives_the_same_result_as_the_equivalent_string(data_root, tmp_path):
    """A callable with the ImageNet stats matches normalize='imagenet' exactly."""
    from aquabench.data import IMAGENET_MEAN, IMAGENET_STD
    import torch

    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    a = _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "a", normalize="imagenet")
    b = _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "b", normalize=lambda x: (x - mean) / std)
    assert a.test_miou == pytest.approx(b.test_miou)


def test_save_checkpoint_false(data_root, tmp_path):
    """No checkpoint file, results still written."""
    _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "runs", save_checkpoint=False)
    run_dir = tmp_path / "runs" / "tiny" / "linear" / "Toy_seed0"
    assert not (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "result.json").exists() and (run_dir / "history.json").exists()


def test_checkpoint_dir_override(data_root, tmp_path):
    """Checkpoint lands in checkpoint_dir, results stay in output_dir."""
    _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "runs", checkpoint_dir=tmp_path / "scratch")
    rel = "tiny/linear/Toy_seed0"
    assert (tmp_path / "scratch" / rel / "checkpoint.pt").exists()
    assert not (tmp_path / "runs" / rel / "checkpoint.pt").exists()
    assert (tmp_path / "runs" / rel / "result.json").exists()


@pytest.mark.parametrize("run_type", ["linear", "upernet_finetune"])
def test_evaluate_matches_run(run_type, data_root, tmp_path):
    """evaluate() on a saved checkpoint reproduces the test mIoU of the original run."""
    enc = tiny_encoder(run_type)
    res = _run(enc, run_type, data_root, tmp_path / "runs")
    run_dir = tmp_path / "runs" / "tiny" / run_type / "Toy_seed0"
    # finetune: a freshly initialised encoder, the checkpoint must restore its weights
    eval_enc = tiny_encoder(run_type, seed=123) if run_type == "upernet_finetune" else enc
    ev = aquabench.evaluate(eval_enc, checkpoint_dir=run_dir, dataset="Toy", data_root=data_root,
                            device="cpu", logger=ListLogger())
    assert ev.test_miou == pytest.approx(res.test_miou, abs=1e-9)
    assert (run_dir / "result_eval.json").exists()


def test_evaluate_warns_on_different_frozen_encoder(data_root, tmp_path):
    """Frozen checkpoints hold only the head; evaluating with other encoder weights warns."""
    _run(tiny_encoder("linear"), "linear", data_root, tmp_path / "runs")
    with pytest.warns(UserWarning, match="weights differ"):
        aquabench.evaluate(tiny_encoder("linear", seed=123), tmp_path / "runs" / "tiny" / "linear" / "Toy_seed0",
                           data_root=data_root, device="cpu", logger=ListLogger())
