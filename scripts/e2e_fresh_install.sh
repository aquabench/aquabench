#!/usr/bin/env bash
# End-to-end check as an external user: install the package fresh into a new venv
# (non-editable), then run one tiny synthetic run per run type plus evaluate().
#
#   bash scripts/e2e_fresh_install.sh            # uses uv if available, else python -m venv
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash scripts/e2e_fresh_install.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "work dir: $WORK"

if command -v uv >/dev/null; then
  uv venv -q "$WORK/venv"
  PIP=(uv pip install -q --python "$WORK/venv/bin/python")
else
  python3 -m venv "$WORK/venv"
  PIP=("$WORK/venv/bin/pip" install -q)
fi
if [[ -n "${TORCH_INDEX:-}" ]]; then "${PIP[@]}" torch --index-url "$TORCH_INDEX"; fi
"${PIP[@]}" "$REPO"

cd "$WORK"  # outside the repo, so the installed package is imported
"$WORK/venv/bin/python" - <<'PY'
import json
from pathlib import Path

import numpy as np
import torch.nn as nn
from PIL import Image

import aquabench

# --- a tiny synthetic dataset in the documented on-disk layout
root = Path("data")
ds = root / "Toy"
rng = np.random.default_rng(0)
for split, n in [("train", 4), ("val", 2), ("test", 2)]:
    (ds / split / "images").mkdir(parents=True)
    (ds / split / "masks").mkdir(parents=True)
    for i in range(n):
        mask = rng.integers(0, 2, (32, 48), dtype=np.uint8)
        img = (mask[..., None] * 200 + rng.integers(0, 40, (32, 48, 3))).astype(np.uint8)
        Image.fromarray(img).save(ds / split / "images" / f"{i}.png")
        Image.fromarray(mask).save(ds / split / "masks" / f"{i}.png")
(ds / "meta.json").write_text(json.dumps({"num_classes": 2, "ignore_index": 255, "size": [32, 48]}))


class Tiny(nn.Module):
    name = "tiny"

    def __init__(self, pyramid):
        super().__init__()
        self.patch, self.pyramid = nn.Conv2d(3, 8, 16, 16), pyramid

    def forward(self, x):
        f = self.patch(x)
        return [f, f, f, f] if self.pyramid else f


cfg = {"epochs": 1, "batch_size": 2, "grad_accum_steps": 1, "num_workers": 0, "precision": "fp32"}
for run_type in aquabench.RUN_TYPES:
    enc = Tiny(pyramid=run_type != "linear")
    res = aquabench.run(enc, run_type=run_type, data_root=root, datasets="Toy", config=cfg, device="cpu")
    run_dir = Path("runs") / "tiny" / run_type / "Toy_seed0"
    assert {p.name for p in run_dir.iterdir()} == {"config.json", "history.json", "result.json", "checkpoint.pt"}
    ev = aquabench.evaluate(enc, run_dir, data_root=root, device="cpu")
    assert abs(ev.test_miou - res.test_miou) < 1e-9, (ev.test_miou, res.test_miou)
    print(f"OK {run_type}: test mIoU {res.test_miou:.3f}")
print("aquabench", aquabench.__version__, "end-to-end OK")
PY
"$WORK/venv/bin/aquabench" verify --data-root "$WORK/data" --datasets Toy
