"""Training config per run type: a shipped JSON default, optionally overridden by the user."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Optional, Union

from aquabench.heads import RUN_TYPES

ConfigLike = Union[None, str, Path, dict]


def default_config(run_type: str) -> dict:
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}, got {run_type!r}")
    return json.loads((resources.files("aquabench") / "configs" / f"{run_type}.json").read_text())


def load_config(run_type: str, config: ConfigLike = None, augment: Optional[bool] = None) -> dict:
    """Shipped default for ``run_type``, updated by ``config`` (path or dict), then ``augment``."""
    cfg = default_config(run_type)
    if config is not None:
        user = dict(config) if isinstance(config, dict) else json.loads(Path(config).read_text())
        unknown = sorted(set(user) - set(cfg))
        if unknown:
            raise ValueError(f"Unknown config keys {unknown}; valid keys: {sorted(cfg)}")
        cfg.update(user)
    if augment is not None:
        cfg["augment"] = bool(augment)
    if cfg["scheduler"] not in ("cosine", "none"):
        raise ValueError(f"scheduler must be 'cosine' or 'none', got {cfg['scheduler']!r}")
    if cfg["precision"] not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {cfg['precision']!r}")
    if cfg["grad_accum_steps"] < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    return cfg
