"""Compute dataset split counts and class pixel distributions.

Outputs:
  results/analysis/dataset_splits.csv
    ds, n_train, n_val, n_test, n_classes, metric_ignore_index, loss_ignore_index

  results/analysis/{ds}_class_pixel_dist.csv  (one per dataset)
    split, class_idx, class_name, total_pixels, pixel_fraction

Usage:
  python -m analysis.dataset_summary --ds cou
  python -m analysis.dataset_summary --ds cs liaci sc tc
  python -m analysis.dataset_summary --all [--data-dir /path/to/unified]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from analysis.utils.ds_names import ALL_DATASETS, DS_HF_NAME

OUT_DIR = Path("results/analysis")


def _load_hf_dataset(ds: str, data_dir: str):
    from datasets import load_dataset
    hf_name = DS_HF_NAME.get(ds, ds)
    return load_dataset(data_dir, hf_name)


def _meta_path(ds: str, data_dir: str) -> Path:
    # Metadata JSON files are keyed by the HF long name (e.g. coralscapes.json), not
    # the short abbreviation used on the CLI (cs) -- same as _load_hf_dataset.
    return Path(data_dir) / "_metadata" / f"{DS_HF_NAME.get(ds, ds)}.json"


def _load_id2label(ds: str, data_dir: str) -> dict:
    meta_path = _meta_path(ds, data_dir)
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    return meta.get("id2label", {})


def process_dataset(ds: str, data_dir: str, splits_rows: list[dict], num_workers: int = 4) -> None:
    print(f"[{ds}] Loading dataset...")
    hf_ds = _load_hf_dataset(ds, data_dir)
    id2label = _load_id2label(ds, data_dir)

    split_counts = {}
    class_pixel_rows = []

    for split_name, split_data in hf_ds.items():
        n = len(split_data)
        split_counts[split_name] = n
        print(f"  [{ds}/{split_name}] {n} samples, counting pixels...")

        # Count pixels per class directly — no HF map needed (masks are small)
        pixel_counts: dict[int, int] = {}
        total_pixels = 0
        for ex in split_data:
            mask = np.array(ex["mask"], dtype=np.int64)
            total_pixels += mask.size
            for cls in np.unique(mask):
                pixel_counts[int(cls)] = pixel_counts.get(int(cls), 0) + int((mask == cls).sum())

        for cls_idx, count in sorted(pixel_counts.items()):
            class_name = id2label.get(str(cls_idx), id2label.get(cls_idx, f"class_{cls_idx}"))
            class_pixel_rows.append({
                "split": split_name,
                "class_idx": cls_idx,
                "class_name": class_name,
                "total_pixels": count,
                "pixel_fraction": round(count / total_pixels, 6) if total_pixels > 0 else 0.0,
            })

    # Determine n_classes and ignore indices from the metadata
    meta_path = _meta_path(ds, data_dir)
    n_classes = None
    metric_ignore_index = None
    loss_ignore_index = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        n_classes = meta.get("num_classes")
        metric_ignore_index = meta.get("metric_ignore_index", 0)
        loss_ignore_index = meta.get("loss_ignore_index", 255)

    splits_rows.append({
        "ds": ds,
        "n_train": split_counts.get("train", 0),
        "n_val": split_counts.get("validation", split_counts.get("val", 0)),
        "n_test": split_counts.get("test", 0),
        "n_classes": n_classes or "",
        "metric_ignore_index": metric_ignore_index if metric_ignore_index is not None else "",
        "loss_ignore_index": loss_ignore_index if loss_ignore_index is not None else "",
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pix_path = OUT_DIR / f"{ds}_class_pixel_dist.csv"
    with open(pix_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "class_idx", "class_name", "total_pixels", "pixel_fraction"]
        )
        writer.writeheader()
        writer.writerows(class_pixel_rows)
    print(f"[{ds}] Done → {pix_path}")


def main():
    default_data_dir = os.path.join(
        os.environ.get("DATA_ROOT", "/workspace/data"), "unified"
    )
    parser = argparse.ArgumentParser(description="Compute dataset split counts and class pixel distributions.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ds", type=str, nargs="+", help="One or more dataset names (e.g. cou, or cs liaci sc tc)")
    group.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.all else args.ds
    splits_rows: list[dict] = []
    for ds in datasets:
        process_dataset(ds, args.data_dir, splits_rows, num_workers=args.workers)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits_path = OUT_DIR / "dataset_splits.csv"
    # Merge with existing rows if partially run
    existing: dict[str, dict] = {}
    if splits_path.exists():
        with open(splits_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[row["ds"]] = row
    for row in splits_rows:
        existing[row["ds"]] = row

    with open(splits_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ds", "n_train", "n_val", "n_test", "n_classes",
            "metric_ignore_index", "loss_ignore_index",
        ])
        writer.writeheader()
        writer.writerows(existing.values())
    print(f"Splits table → {splits_path}")


if __name__ == "__main__":
    main()
