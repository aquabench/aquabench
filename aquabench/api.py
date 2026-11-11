"""``aquabench.run`` and ``aquabench.evaluate``.

One ``run`` call = one encoder x one run type x one seed, over one or more datasets.
Each dataset is trained, tested and saved independently into its own run folder::

    <output_dir>/<encoder_name>/<run_type>/<dataset>_seed<seed>/
        config.json       effective training config
        history.json      per-epoch train loss and val mIoU
        result.json       test mIoU (all classes), per-class IoU, run_info, ...
        checkpoint.pt     head (+ encoder for upernet_finetune); unless save_checkpoint=False

The reported number is always the test mIoU of the **last** epoch; val is only logged.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aquabench import __version__
from aquabench.cache import cache_bytes, cache_features
from aquabench.config import ConfigLike, load_config
from aquabench.data import (Normalize, SegDataset, check_dataset, find_dataset, load_meta, maybe_dataset,
                            normalize_label, resolve_datasets, token_grid)
from aquabench.heads import EncoderContractError, SegModel, build_head, check_encoder_output
from aquabench.loggers import Logger, PrintLogger
from aquabench.train import evaluate_loader, lr_factors, seed_everything, train_epoch

DATA_ROOT_ENV = "AQUABENCH_DATA"


@dataclass
class Result:
    encoder_name: str
    run_type: str
    dataset: str
    seed: int
    config: dict
    test_miou: float
    per_class_iou: List[Optional[float]]
    class_names: List[str]
    val_miou_last: Optional[float]
    seconds: float
    aquabench_version: str
    #: What ran: the encoder (class, parameters, feature maps), the dataset (classes, sizes,
    #: samples per split) and the environment (torch, device, start time).
    run_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _json_safe(asdict(self))


class RunResults(dict):
    """Dataset name -> :class:`Result`. With a single dataset, the fields of that
    result are also available directly, e.g. ``results.test_miou``."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if len(self) == 1:
            return getattr(next(iter(self.values())), name)
        raise AttributeError(
            f"{name!r}: these results hold {len(self)} datasets; index by name, e.g. results[{next(iter(self), '')!r}]"
        )


def run(
    encoder: nn.Module,
    *,
    run_type: str,
    logger: Optional[Logger] = None,
    seed: int = 0,
    data_root: Union[None, str, Path] = None,
    datasets: Union[str, Sequence[str]] = "all",
    patch_size: int = 16,
    augment: Optional[bool] = None,
    batch_size: Optional[int] = None,
    grad_accum_steps: Optional[int] = None,
    normalize: Optional[Normalize] = None,
    config: ConfigLike = None,
    output_dir: Union[str, Path] = "runs",
    checkpoint_dir: Union[None, str, Path] = None,
    save_checkpoint: bool = True,
    encoder_name: Optional[str] = None,
    device: Union[None, str, torch.device] = None,
    cache_dir: Union[None, str, Path] = None,
    overwrite: bool = False,
) -> RunResults:
    """Train and test ``encoder`` with ``run_type`` on each dataset; return per-dataset results.

    Args:
        encoder: prepared ``nn.Module``. ``linear`` needs ``forward(x) -> [B, C, h, w]``, one
            tensor. Both ``upernet_*`` run types need ``forward(x) -> [f1, f2, f3, f4]``: a Python
            list (or tuple) of exactly 4 separate tensors ``[B, C_i, h_i, w_i]``, shallow to deep,
            not one stacked or concatenated tensor.
            ``upernet_finetune`` is full fine-tuning: every parameter of the encoder is trained,
            whatever its ``requires_grad`` flag, and the weights are updated in place. So the
            encoder must have parameters and its forward must not detach its outputs or run under
            ``torch.no_grad()``; this is checked before training.
            The encoder is moved to ``device``.
        run_type: ``"linear"``, ``"upernet_frozen"`` or ``"upernet_finetune"``.
        logger: anything with ``log(metrics: dict, step: int)``; default prints.
        data_root: folder from ``aquabench download``; default ``$AQUABENCH_DATA``.
        datasets: ``"all"``, one name (``"SUIM"``) or a list of names.
        patch_size: encoder patch size or total stride (32 for a ResNet); the input is resized
            so every encoder gets the dataset's token grid (defined at patch 16). The encoder's
            last feature map must have that grid, else ``EncoderContractError`` is raised.
        augment: overrides the config's ``augment`` (default off for linear, on for both UPerNet types).
        batch_size: overrides the config's ``batch_size`` (default 16).
        grad_accum_steps: overrides the config's ``grad_accum_steps`` (default 2). The effective
            batch is ``batch_size x grad_accum_steps``.
        normalize: overrides the config's ``normalize`` (default ``"imagenet"``). ``"imagenet"``,
            ``"none"`` for ``[0, 1]`` images, or a callable taking a ``[3, H, W]`` image in
            ``[0, 1]``. Use what the encoder was trained with; this is not checked.
        config: JSON path or dict overriding the shipped default config for ``run_type``.
        output_dir: root of the run folders.
        checkpoint_dir: put checkpoints under this root instead of next to the results.
        save_checkpoint: ``False`` writes only config / history / result.
        encoder_name: folder name; default ``encoder.name`` or the class name.
        cache_dir: where ``linear`` keeps its cached features during a run (see the config's
            ``cache_features``); default the system temp folder (``$TMPDIR``). They can take tens
            of GB, so point this at fast local scratch. The files are deleted when the run ends.
        overwrite: ``False`` raises ``FileExistsError`` if a run folder already holds a finished
            run (a ``result.json``); ``True`` replaces it.
    """
    cfg = load_config(run_type, config, augment, batch_size, grad_accum_steps, normalize)
    normalize = normalize if callable(normalize) else cfg["normalize"]
    logger = logger or PrintLogger()
    ds_dirs = resolve_datasets(_data_root(data_root), datasets)
    for ds_dir in ds_dirs:  # a bad dataset fails now, not after the ones before it have trained
        check_dataset(ds_dir)
    device = _device(device)
    encoder_name = _safe_name(encoder_name or getattr(encoder, "name", None) or type(encoder).__name__)
    rels = [Path(encoder_name) / run_type / f"{load_meta(d)['name']}_seed{seed}" for d in ds_dirs]
    done = [str(Path(output_dir) / rel) for rel in rels if (Path(output_dir) / rel / "result.json").exists()]
    if done and not overwrite:
        raise FileExistsError(f"Finished runs already exist: {done}. Pass overwrite=True to replace them.")
    encoder.to(device)

    # Fine-tuning changes the encoder; every dataset must start from the same weights.
    initial = None
    if run_type == "upernet_finetune" and len(ds_dirs) > 1:
        initial = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

    results, step = RunResults(), [0]
    for i, (ds_dir, rel) in enumerate(zip(ds_dirs, rels)):
        if initial is not None and i > 0:
            encoder.load_state_dict(initial)
        (Path(output_dir) / rel / "result.json").unlink(missing_ok=True)  # an unfinished rerun is not a result
        res = _run_dataset(
            encoder, ds_dir, run_type=run_type, cfg=cfg, seed=seed, patch_size=patch_size, normalize=normalize,
            logger=logger, step=step, device=device, encoder_name=encoder_name,
            run_dir=Path(output_dir) / rel,
            ckpt_dir=(Path(checkpoint_dir) / rel) if checkpoint_dir else Path(output_dir) / rel,
            save_checkpoint=save_checkpoint, cache_dir=cache_dir,
        )
        results[res.dataset] = res
    return results


def evaluate(
    encoder: nn.Module,
    checkpoint_dir: Union[str, Path],
    dataset: Optional[str] = None,
    *,
    data_root: Union[None, str, Path] = None,
    logger: Optional[Logger] = None,
    output_dir: Union[None, str, Path] = None,
    device: Union[None, str, torch.device] = None,
    normalize: Optional[Normalize] = None,
) -> Result:
    """Test a saved checkpoint without training.

    ``checkpoint_dir`` is a run folder (or a ``checkpoint.pt`` path). The encoder is
    supplied by the caller; the checkpoint provides the head, plus encoder weights for
    ``upernet_finetune``. Writes ``result_eval.json`` next to the checkpoint, or into the
    usual run-folder layout under ``output_dir`` if given. ``normalize`` defaults to the
    checkpoint's; a run trained with a callable needs it passed again, as only its name is stored.
    """
    path = Path(checkpoint_dir)
    path = path / "checkpoint.pt" if path.is_dir() else path
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg, run_type = ckpt["config"], ckpt["run_type"]
    if normalize is None and cfg["normalize"].startswith("callable:"):
        raise ValueError(f"This checkpoint was trained with {cfg['normalize']}; pass it again as normalize=...")
    normalize = cfg["normalize"] if normalize is None else normalize
    cfg = {**cfg, "normalize": normalize_label(normalize)}
    ds_dir = find_dataset(_data_root(data_root), dataset or ckpt["dataset"])
    meta = check_dataset(ds_dir)
    if meta["num_classes"] != ckpt["num_classes"]:
        raise ValueError(f"checkpoint has {ckpt['num_classes']} classes, {meta['name']} has {meta['num_classes']}")
    logger = logger or PrintLogger()
    device = _device(device)
    if ckpt["encoder"] is not None:
        encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device)

    t0 = time.time()
    test_set = SegDataset(ds_dir, "test", patch_size=cfg["patch_size"], normalize=normalize)
    with _keep_requires_grad(encoder):
        channels, shapes = _probe_channels(encoder, run_type, test_set, device, training=False)
        if channels != ckpt["in_channels"]:
            raise ValueError(f"encoder outputs channels {channels}, checkpoint expects {ckpt['in_channels']}")
        if not _same_fingerprint(_fingerprint(encoder), ckpt.get("encoder_fingerprint")):
            warnings.warn("The encoder's weights differ from the encoder this checkpoint was trained "
                          "with; the test mIoU will not match the original run.", stacklevel=2)
        head = build_head(run_type, channels, meta["num_classes"])
        head.load_state_dict(ckpt["head"])
        model = SegModel(encoder, head.to(device), freeze_encoder=True)
        test_miou, per_class = evaluate_loader(
            model, _loader(test_set, cfg, device), num_classes=meta["num_classes"],
            ignore_index=meta["ignore_index"], device=device, precision=cfg["precision"])

    result = Result(
        encoder_name=ckpt["encoder_name"], run_type=run_type, dataset=meta["name"], seed=ckpt["seed"],
        config=cfg, test_miou=test_miou, per_class_iou=per_class, class_names=_class_names(meta),
        val_miou_last=None, seconds=round(time.time() - t0, 1), aquabench_version=__version__,
        run_info=_run_info(encoder, run_type, shapes, meta, test_set, {"test": len(test_set)}, device, t0),
    )
    _log_test(logger, meta["name"], result, step=0)
    out = (Path(output_dir) / result.encoder_name / run_type / f"{meta['name']}_seed{result.seed}"
           if output_dir else path.parent)
    _write_json(out / "result_eval.json", result.to_dict())
    return result


# --------------------------------------------------------------------------- one dataset

def _run_dataset(encoder, ds_dir, *, run_type, cfg, seed, patch_size, normalize, logger, step, device,
                 encoder_name, run_dir, ckpt_dir, save_checkpoint, cache_dir=None) -> Result:
    t0 = time.time()
    meta = load_meta(ds_dir)
    name, num_classes, ignore = meta["name"], meta["num_classes"], meta["ignore_index"]
    config = {"run_type": run_type, "patch_size": patch_size, **cfg}
    _write_json(run_dir / "config.json", config)

    seed_everything(seed)
    ds_kw = dict(patch_size=patch_size, normalize=normalize)
    train_set = SegDataset(ds_dir, "train", augment=cfg["augment"], **ds_kw)
    val_set = maybe_dataset(ds_dir, "val", **ds_kw)
    test_set = SegDataset(ds_dir, "test", **ds_kw)
    if len(train_set) < cfg["batch_size"]:
        raise ValueError(f"{name}: {len(train_set)} training samples < batch_size {cfg['batch_size']}")

    samples = {"train": len(train_set), "val": len(val_set) if val_set is not None else 0, "test": len(test_set)}
    cache_folder = None
    with contextlib.ExitStack() as stack:
        stack.enter_context(_keep_requires_grad(encoder))
        channels, shapes = _probe_channels(encoder, run_type, train_set, device)
        run_info = _run_info(encoder, run_type, shapes, meta, train_set, samples, device, t0)
        head = build_head(run_type, channels, num_classes).to(device)
        model = SegModel(encoder, head, freeze_encoder=run_type != "upernet_finetune")

        # linear without augmentation: the frozen features are the same every epoch, so they are
        # computed once and the head trains on the stored copy.
        if run_type == "linear" and cfg.get("cache_features") and not cfg["augment"]:
            cache_folder = Path(tempfile.mkdtemp(prefix="aquabench_features_", dir=cache_dir))
            stack.callback(shutil.rmtree, cache_folder, ignore_errors=True)
            need = sum(cache_bytes(n, shapes[0], train_set.size, cfg["precision"]) for n in samples.values())
            logger.log({f"{name}/cache/gb": round(need / 1e9, 2)}, step[0])
            try:
                cached = [cache_features(encoder, _loader(ds, cfg, device), cache_folder, split,
                                         feature_shape=shapes[0], device=device, precision=cfg["precision"])
                          if ds is not None else None
                          for split, ds in (("train", train_set), ("val", val_set), ("test", test_set))]
            except OSError as e:
                warnings.warn(f"{name}: features not cached ({need / 1e9:.1f} GB in {cache_folder.parent}: {e}); "
                              f"the encoder runs every epoch instead. Pass cache_dir=... with enough space.",
                              stacklevel=2)
                cache_folder = None
            else:
                train_set, val_set, test_set = cached
                model = head  # the stored features go straight to the head
        run_info["features_cached"] = cache_folder is not None
        workers = 0 if cache_folder is not None else None  # reading the memory map needs no workers

        # upernet_finetune: the encoder has its own (lower) learning rate, is frozen for the first
        # freeze_encoder_epochs, then ramps up over encoder_warmup_epochs. Other run types: head only.
        finetune = run_type == "upernet_finetune"
        groups = [{"params": list(head.parameters()), "lr": cfg["lr"]}]
        if finetune:
            groups.append({"params": list(encoder.parameters()), "lr": cfg["lr"] * cfg["encoder_lr_scale"]})
        opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        head_factor, encoder_factor = lr_factors(cfg["epochs"], cfg["scheduler"],
                                                 cfg.get("freeze_encoder_epochs", 0), cfg.get("encoder_warmup_epochs", 0))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, [head_factor, encoder_factor][:len(groups)])
        gen = torch.Generator().manual_seed(seed)
        train_loader = _loader(train_set, cfg, device, shuffle=True, generator=gen, num_workers=workers)
        val_loader = _loader(val_set, cfg, device, num_workers=workers) if val_set is not None else None
        test_loader = _loader(test_set, cfg, device, num_workers=workers)
        eval_kw = dict(num_classes=num_classes, ignore_index=ignore, device=device, precision=cfg["precision"])

        def on_step(loss: float):
            step[0] += 1
            if step[0] % cfg["log_every"] == 0:
                lrs = {f"{name}/train/lr": opt.param_groups[0]["lr"]}
                if finetune:
                    lrs[f"{name}/train/encoder_lr"] = opt.param_groups[1]["lr"]
                logger.log({f"{name}/train/loss": loss, **lrs}, step[0])

        history, val_miou = [], None
        for epoch in range(1, cfg["epochs"] + 1):
            if finetune:
                model.set_encoder_frozen(epoch <= cfg["freeze_encoder_epochs"])
            train_loss = train_epoch(model, train_loader, opt, grad_accum_steps=cfg["grad_accum_steps"],
                                     ignore_index=ignore, device=device, precision=cfg["precision"],
                                     on_step=on_step)
            sched.step()
            if val_loader is not None:
                val_miou = evaluate_loader(model, val_loader, **eval_kw)[0]
            history.append({"epoch": epoch, "train_loss": train_loss, "val_miou": val_miou})
            _write_json(run_dir / "history.json", history)
            logger.log({f"{name}/epoch": epoch, f"{name}/train/epoch_loss": train_loss,
                        f"{name}/val/miou": val_miou}, step[0])

        test_miou, per_class = evaluate_loader(model, test_loader, **eval_kw)

        if save_checkpoint:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "aquabench_version": __version__, "encoder_name": encoder_name, "run_type": run_type,
                "dataset": name, "seed": seed, "num_classes": num_classes, "in_channels": channels,
                "config": config, "head": head.state_dict(),
                "encoder": encoder.state_dict() if run_type == "upernet_finetune" else None,
                "encoder_fingerprint": _fingerprint(encoder),
            }, ckpt_dir / "checkpoint.pt")

    result = Result(
        encoder_name=encoder_name, run_type=run_type, dataset=name, seed=seed, config=config,
        test_miou=test_miou, per_class_iou=per_class, class_names=_class_names(meta),
        val_miou_last=val_miou, seconds=round(time.time() - t0, 1), aquabench_version=__version__,
        run_info=run_info,
    )
    _log_test(logger, name, result, step[0])
    _write_json(run_dir / "result.json", result.to_dict())
    return result


# --------------------------------------------------------------------------- helpers

def _probe_channels(encoder, run_type, dataset, device, training=True):
    """Run one dummy image through the encoder and check its output against the contract.
    Returns the channel count and the shape ``[C, h, w]`` of every feature map.

    Before training, ``upernet_finetune`` also checks that gradients can reach the encoder:
    it has parameters, and no map is detached.
    """
    finetune = training and run_type == "upernet_finetune"
    was_training = encoder.training
    encoder.eval()
    if finetune:
        if not any(True for _ in encoder.parameters()):
            raise EncoderContractError("run_type 'upernet_finetune' trains the encoder, but it has no parameters.")
        for p in encoder.parameters():  # full fine-tuning; the caller's flags are restored after the run
            p.requires_grad_(True)
    with torch.set_grad_enabled(finetune):
        out = encoder(torch.zeros(1, 3, *dataset.input_size, device=device))
    encoder.train(was_training)
    channels = check_encoder_output(out, run_type, token_grid(dataset.size))
    maps = [out] if isinstance(out, torch.Tensor) else list(out)
    if finetune and not all(m.requires_grad for m in maps):
        raise EncoderContractError(
            "run_type 'upernet_finetune' trains every encoder parameter, but "
            f"{sum(not m.requires_grad for m in maps)} of the 4 feature maps carry no gradient. Remove "
            "torch.no_grad() / .detach() from the encoder's forward, or use run_type 'upernet_frozen'."
        )
    return channels, [list(m.shape[1:]) for m in maps]


def _run_info(encoder, run_type, shapes, meta, dataset, samples, device, started) -> dict:
    """What ran, for result.json: the encoder as far as its class tells, the dataset, the environment."""
    cls = type(encoder)
    parameters = sum(p.numel() for p in encoder.parameters())
    return {
        "encoder": {
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "parameters": parameters,
            "trained_parameters": parameters if run_type == "upernet_finetune" else 0,
            "feature_maps": shapes,  # [C, h, w] per map, for this dataset
        },
        "dataset": {
            "name": meta["name"], "num_classes": meta["num_classes"], "ignore_index": meta["ignore_index"],
            "size": list(dataset.size), "token_grid": list(token_grid(dataset.size)),
            "input_size": list(dataset.input_size), "samples": samples,
        },
        "environment": {
            "torch": torch.__version__, "python": platform.python_version(), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        },
    }


def _fingerprint(module: nn.Module) -> List[float]:
    """Cheap identity check of a module's weights: the sum of every floating tensor."""
    sums = [t.detach().double().sum() for t in module.state_dict().values() if t.is_floating_point()]
    return torch.stack(sums).cpu().tolist() if sums else []


def _same_fingerprint(a: List[float], b: Optional[List[float]]) -> bool:
    if b is None:
        return True
    return len(a) == len(b) and torch.allclose(torch.tensor(a), torch.tensor(b), rtol=1e-5, atol=1e-6)


@contextlib.contextmanager
def _keep_requires_grad(module: nn.Module):
    """Restore the caller's ``requires_grad`` flags after freezing for a run."""
    flags = [p.requires_grad for p in module.parameters()]
    try:
        yield
    finally:
        for p, f in zip(module.parameters(), flags):
            p.requires_grad_(f)


def _loader(ds, cfg, device, shuffle=False, generator=None, num_workers=None) -> DataLoader:
    nw = cfg["num_workers"] if num_workers is None else num_workers
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, drop_last=shuffle,
                      num_workers=nw, generator=generator, pin_memory=device.type == "cuda",
                      persistent_workers=nw > 0)


def _log_test(logger, name, result: Result, step: int):
    metrics = {f"{name}/test/miou": result.test_miou}
    for cls, v in zip(result.class_names, result.per_class_iou):
        metrics[f"{name}/test/iou/{cls}"] = v
    logger.log(metrics, step)


def _class_names(meta) -> List[str]:
    names = meta.get("class_names") or []
    return list(names) if len(names) == meta["num_classes"] else [str(i) for i in range(meta["num_classes"])]


def _data_root(data_root) -> Path:
    root = data_root or os.environ.get(DATA_ROOT_ENV)
    if not root:
        raise ValueError(f"Pass data_root=... or set ${DATA_ROOT_ENV} (see `aquabench download`).")
    return Path(root)


def _device(device) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def _json_safe(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2) + "\n")
