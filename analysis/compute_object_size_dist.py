"""Per-class connected-component object-size distribution (pixel area per instance).

compute_image_stats.py already runs connected-component labeling per image, but only
keeps small/medium/large COUNTS per image, pooled across all classes -- class identity
and the actual pixel area of each component are discarded. This script keeps both: one
row per connected component, with its class and exact pixel area.

Output (per dataset):
  results/analysis/{ds}_object_size_dist.csv
    split, sample_idx, class_idx, class_name, area_px
    (one row per connected component; background/ignored classes are NOT filtered here
    -- filter by class_name == "background" or metric_ignore_index at plot time, same
    convention as class_size_vs_miou.)

Needs $DATA_ROOT/unified, like compute_image_stats.py / dataset_summary.py -- run on the
compute node, then copy the CSVs into results/analysis/ locally.

Usage:
  python -m analysis.compute_object_size_dist --ds cs liaci sc tc
  python -m analysis.compute_object_size_dist --all [--workers 4] [--data-dir /path/to/unified]
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
FIELDNAMES = ("split", "sample_idx", "class_idx", "class_name", "area_px")


def _compute_object_sizes_batch(batch: dict) -> dict:
    """HuggingFace-compatible batched map function: adds per-image parallel lists of
    connected-component class indices and pixel areas."""
    from scipy.ndimage import label as cc_label

    all_classes, all_areas = [], []
    for mask_img in batch["mask"]:
        mask = np.array(mask_img, dtype=np.int64)
        classes_this: list[int] = []
        areas_this: list[int] = []
        for cls in np.unique(mask):
            if cls >= 255:  # ignore_index
                continue
            binary = (mask == cls).astype(np.uint8)
            labeled, n_comp = cc_label(binary)
            if n_comp == 0:
                continue
            areas = np.bincount(labeled.ravel())[1:]  # drop label-0 (non-class background)
            classes_this.extend([int(cls)] * len(areas))
            areas_this.extend(int(a) for a in areas)
        all_classes.append(classes_this)
        all_areas.append(areas_this)
    return {"cc_class_idx": all_classes, "cc_area_px": all_areas}


def _load_id2label(ds: str, data_dir: str) -> dict:
    meta_path = Path(data_dir) / "_metadata" / f"{DS_HF_NAME.get(ds, ds)}.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text()).get("id2label", {})


def process_dataset(ds: str, data_dir: str, num_workers: int) -> None:
    from datasets import load_dataset

    hf_name = DS_HF_NAME.get(ds, ds)
    print(f"[{ds}] Loading dataset '{hf_name}' from {data_dir} ...")
    hf_ds = load_dataset(data_dir, hf_name)
    id2label = _load_id2label(ds, data_dir)

    def class_name(c: int) -> str:
        return id2label.get(str(c), id2label.get(c, f"class_{c}"))

    print(f"[{ds}] Computing connected-component object sizes (workers={num_workers})...")
    hf_ds = hf_ds.map(
        _compute_object_sizes_batch,
        batched=True,
        batch_size=16,
        num_proc=num_workers,
        desc=f"{ds} object sizes",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{ds}_object_size_dist.csv"
    n_rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for split_name, split_data in hf_ds.items():
            for idx in range(len(split_data)):
                row_data = split_data[idx]
                for cls, area in zip(row_data["cc_class_idx"], row_data["cc_area_px"]):
                    writer.writerow({
                        "split": split_name,
                        "sample_idx": idx,
                        "class_idx": cls,
                        "class_name": class_name(cls),
                        "area_px": area,
                    })
                    n_rows += 1

    print(f"[{ds}] {n_rows} connected components -> {out_path}")


def main():
    default_data_dir = os.path.join(os.environ.get("DATA_ROOT", "/workspace/data"), "unified")
    parser = argparse.ArgumentParser(description="Per-class connected-component object-size distribution.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ds", type=str, nargs="+", help="One or more dataset names (e.g. cou, or cs liaci sc tc)")
    group.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.all else args.ds
    for ds in datasets:
        process_dataset(ds, args.data_dir, args.workers)


if __name__ == "__main__":
    main()
