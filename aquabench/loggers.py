"""Logger contract. AQUABENCH depends on no logging library; bring your own adapter.

    class WandbLogger:
        def __init__(self, run): self.run = run
        def log(self, metrics, step): self.run.log(metrics, step=step)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    def log(self, metrics: dict, step: int) -> None: ...


class PrintLogger:
    """Default logger: one line per call on stdout."""

    def log(self, metrics: dict, step: int) -> None:
        items = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"[step {step}] {items}", flush=True)
