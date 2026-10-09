"""Save 2 sample images + colorized masks per dataset at preprocessing resolution.

One PNG per image, one PNG per mask, named for easy scp:
  results/analysis/figures/samples/{ds}_{idx}_img.png
  results/analysis/figures/samples/{ds}_{idx}_mask.png

Images are written at the dataset's native preprocessing resolution (no extra
rescaling). Masks are colorized with tab20+tab20b (40 colors); background class 0
is near-black, ignore-index 255 is white.

Usage:
  python -m analysis.make_sample_figures
  python -m analysis.make_sample_figures --ds cs cou
  python -m analysis.make_sample_figures --sample-indices 0,500
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from analysis.utils.ds_names import ALL_DATASETS, DS_HF_NAME

# 40-color palette: background = near-black, classes 1..39 from tab20 + tab20b
def _build_palette() -> np.ndarray:
    """Return (40, 3) uint8 RGB array. Index 0 = background (#111111)."""
    import matplotlib.pyplot as plt
    tab20  = (np.array(plt.colormaps["tab20"].colors)  * 255).astype(np.uint8)   # 20×3
    tab20b = (np.array(plt.colormaps["tab20b"].colors) * 255).astype(np.uint8)   # 20×3
    palette = np.vstack([tab20, tab20b])   # 40×3
    palette[0] = [17, 17, 17]             # background → near-black
    return palette


PALETTE = _build_palette()
IGNORE_COLOR = np.array([255, 255, 255], dtype=np.uint8)


def _colorize(mask: np.ndarray) -> np.ndarray:
    """Map (H,W) int64 mask → (H,W,3) uint8 RGB. Ignore index 255 → white."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    ignore = mask == 255
    valid = ~ignore
    indices = mask[valid] % len(PALETTE)   # wrap-around for safety
    rgb[valid] = PALETTE[indices]
    rgb[ignore] = IGNORE_COLOR
    return rgb


def _load_metadata(ds: str, data_dir: str) -> dict:
    meta_path = Path(data_dir) / "_metadata" / f"{ds}.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def process_dataset(
    ds: str,
    data_dir: str,
    out_dir: Path,
    sample_indices: list[int] | None,
    n_samples: int,
) -> None:
    from datasets import load_dataset

    hf_name = DS_HF_NAME.get(ds, ds)
    print(f"  [{ds}] loading test split ({hf_name})...")
    try:
        hf_ds = load_dataset(data_dir, hf_name)
    except Exception as e:
        print(f"  [{ds}] SKIP — could not load: {e}")
        return

    split = hf_ds.get("test") or hf_ds.get("validation")
    if split is None:
        print(f"  [{ds}] SKIP — no test or validation split found")
        return

    n = len(split)
    if sample_indices is not None:
        indices = [i for i in sample_indices if i < n]
    else:
        step = max(n // n_samples, 1)
        indices = [min(i * step, n - 1) for i in range(n_samples)]

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx_pos, idx in enumerate(indices):
        ex = split[idx]
        img = ex["image"].convert("RGB")
        mask = np.array(ex["mask"], dtype=np.int64)

        img_path  = out_dir / f"{ds}_{idx_pos}_img.png"
        mask_path = out_dir / f"{ds}_{idx_pos}_mask.png"

        img.save(img_path)
        Image.fromarray(_colorize(mask)).save(mask_path)
        print(f"  [{ds}] sample {idx_pos} (idx={idx}) → {img_path.name}, {mask_path.name}")


def main() -> None:
    default_data_dir = os.path.join(
        os.environ.get("DATA_ROOT", "/workspace/data"), "unified"
    )
    parser = argparse.ArgumentParser(
        description="Save sample images + colorized masks per dataset."
    )
    parser.add_argument("--analysis-dir", default="results/analysis", type=Path)
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated explicit sample indices, e.g. '0,500'. Overrides --n-samples.",
    )
    parser.add_argument(
        "--ds",
        nargs="+",
        default=None,
        help="Subset of datasets (default: all in ALL_DATASETS).",
    )
    args = parser.parse_args()

    sample_indices: list[int] | None = None
    if args.sample_indices:
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(",")]

    datasets = args.ds if args.ds else list(ALL_DATASETS)
    out_dir = args.analysis_dir / "figures" / "samples"

    print(f"Writing sample figures to {out_dir}")
    for ds in datasets:
        process_dataset(ds, args.data_dir, out_dir, sample_indices, args.n_samples)

    total = list(out_dir.glob("*.png")) if out_dir.exists() else []
    print(f"\nDone — {len(total)} PNG files in {out_dir}")


if __name__ == "__main__":
    main()
