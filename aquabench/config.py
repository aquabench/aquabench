"""Training config per run type: a shipped JSON default, optionally overridden by the user."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Optional, Union

from aquabench.data import Normalize, normalize_label
from aquabench.heads import RUN_TYPES

ConfigLike = Union[None, str, Path, dict]


def default_config(run_type: str) -> dict:
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}, got {run_type!r}")
    return json.loads((resources.files("aquabench") / "configs" / f"{run_type}.json").read_text())


def load_config(run_type: str, config: ConfigLike = None, augment: Optional[bool] = None,
                batch_size: Optional[int] = None, grad_accum_steps: Optional[int] = None,
                normalize: Optional[Normalize] = None) -> dict:
    """Shipped default for ``run_type``, updated by ``config`` (path or dict), then the arguments.
    A callable ``normalize`` is stored by name, so the config stays JSON-safe."""
    cfg = default_config(run_type)
    if config is not None:
        user = dict(config) if isinstance(config, dict) else json.loads(Path(config).read_text())
        unknown = sorted(set(user) - set(cfg))
        if unknown:
            raise ValueError(f"Unknown config keys {unknown}; valid keys: {sorted(cfg)}")
        cfg.update(user)
    if augment is not None:
        cfg["augment"] = bool(augment)
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if grad_accum_steps is not None:
        cfg["grad_accum_steps"] = grad_accum_steps
    cfg["normalize"] = normalize_label(cfg["normalize"] if normalize is None else normalize)
    if cfg["scheduler"] not in ("cosine", "none"):
        raise ValueError(f"scheduler must be 'cosine' or 'none', got {cfg['scheduler']!r}")
    if cfg["precision"] not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {cfg['precision']!r}")
    if cfg["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    if cfg["grad_accum_steps"] < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if "encoder_lr_scale" in cfg:  # upernet_finetune only
        if cfg["encoder_lr_scale"] <= 0:
            raise ValueError("encoder_lr_scale must be > 0")
        if cfg["freeze_encoder_epochs"] < 0 or cfg["encoder_warmup_epochs"] < 0:
            raise ValueError("freeze_encoder_epochs and encoder_warmup_epochs must be >= 0")
        if cfg["freeze_encoder_epochs"] >= cfg["epochs"]:
            raise ValueError("freeze_encoder_epochs must be smaller than epochs, else the encoder never trains")
    return cfg
