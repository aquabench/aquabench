"""Packaging constraints."""

import subprocess
import sys
from pathlib import Path

HEAVY = ["lightning", "pytorch_lightning", "mmseg", "mmcv", "wandb", "datasets", "transformers",
         "timm", "torchmetrics", "kornia", "sklearn"]


def test_no_heavy_dependencies():
    """Importing aquabench pulls in no Lightning / mmseg / wandb / ...; core deps are torch, numpy, pillow."""
    code = "import sys, aquabench; print(' '.join(m for m in %r if m in sys.modules))" % HEAVY
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""

    try:
        import tomllib
    except ImportError:  # Python 3.10
        return
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    deps = {d.split(">")[0].split("=")[0].split("<")[0].strip() for d in pyproject["project"]["dependencies"]}
    assert deps == {"torch", "numpy", "pillow"}
