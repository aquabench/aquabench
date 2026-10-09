"""Compute per-image UIQM scores and object-size statistics for every split of a dataset.

Outputs (per dataset):
  results/analysis/{ds}_image_stats.csv
    split, sample_idx, uiqm_score, uiqm_tier, n_small_obj, n_medium_obj, n_large_obj,
    has_small, has_medium, has_large

  results/analysis/{ds}_uiqm_thresholds.csv
    p33, p66   (global percentiles computed over all splits)

UIQM tiers (three equal-population buckets using global p33/p66):
  low  : uiqm_score < p33
  mid  : p33 <= uiqm_score < p66
  high : uiqm_score >= p66

Object-size thresholds follow COCO definitions (applied to the preprocessed image resolution):
  small  : area < 32^2  = 1024 px
  medium : 1024 <= area < 96^2 = 9216 px
  large  : area >= 9216 px

Usage:
  python -m analysis.compute_image_stats --ds cou [--workers 4]
  python -m analysis.compute_image_stats --all  [--workers 4] [--data-dir /path/to/unified]
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from analysis.utils.ds_names import ALL_DATASETS, DS_HF_NAME

SMALL_AREA = 32 ** 2    # 1024
MEDIUM_AREA = 96 ** 2   # 9216

OUT_DIR = Path(os.environ.get("RESULTS_DIR", "results")) / "analysis"
STATS_FIELDNAMES = (
    "split", "sample_idx", "uiqm_score", "uiqm_tier",
    "n_small_obj", "n_medium_obj", "n_large_obj",
    "has_small", "has_medium", "has_large",
)


def _compute_uiqm_batch(batch: dict) -> dict:
    """HuggingFace-compatible map function: adds 'uiqm_score' column."""
    from analysis.utils.uiqm import uiqm as compute_uiqm
    scores = []
    for img in batch["image"]:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        scores.append(float(compute_uiqm(arr)))
    return {"uiqm_score": scores}


def _compute_size_stats_batch(batch: dict) -> dict:
    """HuggingFace-compatible map function: adds object-size columns."""
    from scipy.ndimage import label as cc_label

    n_small_list, n_medium_list, n_large_list = [], [], []
    for mask_img in batch["mask"]:
        mask = np.array(mask_img, dtype=np.int64)
        n_small = n_medium = n_large = 0
        for cls in np.unique(mask):
            if cls >= 255:  # ignore_index
                continue
            binary = (mask == cls).astype(np.uint8)
            labeled, n_comp = cc_label(binary)
            for comp_id in range(1, n_comp + 1):
                area = int((labeled == comp_id).sum())
                if area < SMALL_AREA:
                    n_small += 1
                elif area < MEDIUM_AREA:
                    n_medium += 1
                else:
                    n_large += 1
        n_small_list.append(n_small)
        n_medium_list.append(n_medium)
        n_large_list.append(n_large)
    return {
        "n_small_obj": n_small_list,
        "n_medium_obj": n_medium_list,
        "n_large_obj": n_large_list,
    }


def process_dataset(ds: str, data_dir: str, num_workers: int) -> None:
    from datasets import load_dataset

    hf_name = DS_HF_NAME.get(ds, ds)
    print(f"\n[{ds}] Loading dataset '{hf_name}' from {data_dir} ...")
    hf_ds = load_dataset(data_dir, hf_name)

    # ---- Pass 1: UIQM scores (parallel via HF map) ----
    print(f"[{ds}] Pass 1/2: Computing UIQM scores (workers={num_workers})...")
    hf_ds = hf_ds.map(
        _compute_uiqm_batch,
        batched=True,
        batch_size=32,
        num_proc=num_workers,
        desc=f"{ds} UIQM",
    )

    # Compute global p33 / p66 across all splits
    all_scores = []
    for split_data in hf_ds.values():
        all_scores.extend(split_data["uiqm_score"])
    all_scores = np.array(all_scores, dtype=np.float32)
    p33 = float(np.percentile(all_scores, 33))
    p66 = float(np.percentile(all_scores, 66))
    print(f"[{ds}] p33={p33:.4f}  p66={p66:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thresh_path = OUT_DIR / f"{ds}_uiqm_thresholds.csv"
    with open(thresh_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p33", "p66"])
        writer.writeheader()
        writer.writerow({"p33": p33, "p66": p66})

    # ---- Pass 2: Object sizes (parallel via HF map) ----
    print(f"[{ds}] Pass 2/2: Computing object sizes (workers={num_workers})...")
    hf_ds = hf_ds.map(
        _compute_size_stats_batch,
        batched=True,
        batch_size=16,
        num_proc=num_workers,
        desc=f"{ds} sizes",
    )

    # ---- Write CSV ----
    def _tier(score: float) -> str:
        if score < p33:
            return "low"
        if score < p66:
            return "mid"
        return "high"

    stats_path = OUT_DIR / f"{ds}_image_stats.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDNAMES)
        writer.writeheader()
        for split_name, split_data in hf_ds.items():
            for idx in range(len(split_data)):
                row_data = split_data[idx]
                score = float(row_data["uiqm_score"])
                ns = int(row_data["n_small_obj"])
                nm = int(row_data["n_medium_obj"])
                nl = int(row_data["n_large_obj"])
                writer.writerow({
                    "split": split_name,
                    "sample_idx": idx,
                    "uiqm_score": round(score, 6),
                    "uiqm_tier": _tier(score),
                    "n_small_obj": ns,
                    "n_medium_obj": nm,
                    "n_large_obj": nl,
                    "has_small": int(ns > 0),
                    "has_medium": int(nm > 0),
                    "has_large": int(nl > 0),
                })

    print(f"[{ds}] Done → {stats_path}")


def main():
    default_data_dir = os.path.join(
        os.environ.get("DATA_ROOT", "/workspace/data"), "unified"
    )
    parser = argparse.ArgumentParser(description="Compute per-image UIQM and object-size stats.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ds", type=str, nargs="+", help="One or more dataset names (e.g. cou, or cs liaci sc tc)")
    group.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for HF map (default: 4)")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.all else args.ds
    for ds in datasets:
        process_dataset(ds, args.data_dir, args.workers)


if __name__ == "__main__":
    main()
