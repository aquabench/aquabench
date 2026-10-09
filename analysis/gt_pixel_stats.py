"""Per-sample ground-truth pixel counts, read directly from mask data.

results/per_sample/*.csv only stores intersection/union (see
aquabench/callbacks.py:PerSampleTestCallback) -- gt pixel counts can't be recovered
from those two numbers alone, so this reads masks straight from the unified HF
dataset instead. Needs the same $DATA_ROOT/unified access as dataset_summary.py.

For the aggregate per-class-per-split pixel distribution, use
analysis/dataset_summary.py (results/analysis/{ds}_class_pixel_dist.csv) --
no need to recompute it here.

Output (per dataset):
  results/analysis/{ds}_gt_pixels_per_sample.csv
    split, sample_idx, class_idx, gt_pixels   (dense: every class, incl. 0-pixel rows)

And prints, per dataset per split, any classes with 0 gt pixels in that split
(e.g. a class present in train but absent from test won't get a test mIoU).

Usage:
  python -m analysis.gt_pixel_stats --ds cou
  python -m analysis.gt_pixel_stats --all [--data-dir /path/to/unified]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.utils.ds_names import ALL_DATASETS, DS_HF_NAME

OUT_DIR = Path("results/analysis")


def _load_hf_dataset(ds: str, data_dir: str):
    from datasets import load_dataset
    hf_name = DS_HF_NAME.get(ds, ds)
    return load_dataset(data_dir, hf_name)


def _load_meta(ds: str, data_dir: str) -> dict:
    # Metadata JSON files are keyed by the HF long name (e.g. coralscapes.json), not
    # the short abbreviation used on the CLI (cs) -- same as _load_hf_dataset.
    meta_path = Path(data_dir) / "_metadata" / f"{DS_HF_NAME.get(ds, ds)}.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def process_dataset(ds: str, data_dir: str) -> None:
    print(f"[{ds}] Loading dataset...")
    hf_ds = _load_hf_dataset(ds, data_dir)
    meta = _load_meta(ds, data_dir)
    id2label = meta.get("id2label", {})
    n_classes = meta.get("num_classes")
    if n_classes is None:
        raise ValueError(f"[{ds}] Missing num_classes in metadata -- can't build dense per-class rows.")

    def class_name(c: int) -> str:
        return id2label.get(str(c), id2label.get(c, f"class_{c}"))

    per_sample_rows = []
    split_totals: dict[str, np.ndarray] = {}

    for split_name, split_data in hf_ds.items():
        print(f"  [{ds}/{split_name}] {len(split_data)} samples, counting pixels...")
        totals = split_totals.setdefault(split_name, np.zeros(n_classes, dtype=np.int64))
        for idx, ex in enumerate(split_data):
            mask = np.asarray(ex["mask"], dtype=np.int64).ravel()
            valid = mask[(mask >= 0) & (mask < n_classes)]
            counts = np.bincount(valid, minlength=n_classes)[:n_classes]
            totals += counts
            for c, count in enumerate(counts):
                per_sample_rows.append({
                    "split": split_name, "sample_idx": idx,
                    "class_idx": c, "gt_pixels": int(count),
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_sample_path = OUT_DIR / f"{ds}_gt_pixels_per_sample.csv"
    with open(per_sample_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "sample_idx", "class_idx", "gt_pixels"])
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"[{ds}] Per-sample gt pixels -> {per_sample_path}")

    any_zero = False
    for split_name, totals in split_totals.items():
        zero_classes = [c for c in range(n_classes) if totals[c] == 0]
        if zero_classes:
            any_zero = True
            names = ", ".join(f"{c} ({class_name(c)})" for c in zero_classes)
            print(f"[{ds}/{split_name}] Classes with 0 gt pixels: {names}")
    if not any_zero:
        print(f"[{ds}] No classes with 0 gt pixels in any split.")


def main():
    default_data_dir = os.path.join(os.environ.get("DATA_ROOT", "/workspace/data"), "unified")
    parser = argparse.ArgumentParser(description="Ground-truth pixel counts per sample/class, from raw masks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ds", type=str, nargs="+", help="One or more dataset names (e.g. cou, or cs liaci sc tc)")
    group.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--data-dir", default=default_data_dir)
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.all else args.ds
    for ds in datasets:
        process_dataset(ds, args.data_dir)


if __name__ == "__main__":
    main()
