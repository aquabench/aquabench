"""Object-size x quality-tier mIoU grid, per ViT-B encoder family.

Research question: does mIoU degrade more under low image quality specifically for
images dominated by SMALL objects (the failure mode you'd expect: fine detail is what
low quality destroys first), and does that differ by encoder family?

For each image, this assigns two independent tiers:
  - a UIQM quality tier, using the same globally-pooled cutpoints as
    quality_regime_analysis.py (pooled over all 9 datasets, so tiers are comparable
    across datasets -- see that module for details).
  - a dominant object-size tier: whichever of {n_small_obj, n_medium_obj, n_large_obj}
    (COCO area buckets, from compute_image_stats.py) has the most connected components
    in that image. Ties broken toward the larger bucket. Images with zero labeled
    objects are excluded.

CAVEAT: object-size dominance is by component COUNT, not by pixel area covered -- an
image with one huge object and twenty tiny ones is classified "small"-dominant. Treat
this as a first-pass signal, not a precise area-weighted measure.

Then pools per-sample test IoU (intersection/union) into each (quality_tier, size_tier)
cell, restricted to the ViT-B linear regime (5 families x 9 datasets x 5 seeds) -- the
"per ViT-B model family" scope. Reported per (family, ds) and aggregated across datasets
per family.

Prerequisites:
  results/analysis/{ds}_image_stats.csv (compute_image_stats.py)
  results/per_sample/{ds}__{encoder}__linear[__seed{S}]__per_sample_test.csv

Outputs:
  results/analysis/size_quality_grid.csv
    encoder, family, ds, quality_tier, size_tier, mean_miou, std_miou, n_seeds, n_samples
  results/analysis/size_quality_grid_aggregated.csv
    family, quality_tier, size_tier, mean_miou, std_miou, n_ds

Usage:
  python -m analysis.object_size_quality_analysis
  python -m analysis.object_size_quality_analysis --percentiles 25,50,75
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.quality_regime_analysis import (
    _DS_LONG_TO_SHORT,
    _PER_SAMPLE_RE,
    _compute_global_cutpoints,
    _parse_percentiles,
    _tier_names,
)
from analysis.utils.model_meta import ENCODER_TO_FAMILY, VITB_ENCODERS

OUT_DIR = Path("results/analysis")
PER_SAMPLE_DIR = Path("results/per_sample")

SIZE_TIERS = ["small", "medium", "large"]


def _load_size_and_quality_cells(
    ds: str, cutpoints: list[float], quality_tier_names: list[str]
) -> dict[tuple[str, int], tuple[str, str]]:
    """Returns {(split, sample_idx): (quality_tier, size_tier)}. Images with zero
    labeled objects are excluded (no meaningful size tier)."""
    path = OUT_DIR / f"{ds}_image_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run compute_image_stats.py first.")
    out: dict[tuple[str, int], tuple[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ns, nm, nl = int(row["n_small_obj"]), int(row["n_medium_obj"]), int(row["n_large_obj"])
            counts = {"small": ns, "medium": nm, "large": nl}
            if ns + nm + nl == 0:
                continue
            size_tier = max(SIZE_TIERS, key=lambda t: (counts[t], SIZE_TIERS.index(t)))

            score = float(row["uiqm_score"])
            i = 0
            for cp in cutpoints:
                if score < cp:
                    break
                i += 1
            quality_tier = quality_tier_names[i]

            out[(row["split"], int(row["sample_idx"]))] = (quality_tier, size_tier)
    return out


def _find_vitb_linear_csvs() -> dict[tuple[str, str], list[Path]]:
    """(encoder, ds) -> [seed paths], restricted to VITB_ENCODERS x head='linear'."""
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    if not PER_SAMPLE_DIR.exists():
        return grouped
    for path in sorted(PER_SAMPLE_DIR.glob("*__per_sample_test.csv")):
        m = _PER_SAMPLE_RE.match(path.stem)
        if not m:
            continue
        ds_name, encoder, head = m.group("ds"), m.group("encoder"), m.group("head")
        ds_name = _DS_LONG_TO_SHORT.get(ds_name, ds_name)
        if head != "linear" or encoder not in VITB_ENCODERS:
            continue
        grouped[(encoder, ds_name)].append(path)
    return grouped


def _seed_miou_by_cell(
    per_sample_csv: Path, cell_map: dict[tuple[str, int], tuple[str, str]], quality_tiers: list[str]
) -> dict[tuple[str, str], tuple[float | None, int]]:
    inter: dict[tuple[str, str], int] = defaultdict(int)
    union: dict[tuple[str, str], int] = defaultdict(int)
    samples: dict[tuple[str, str], set] = defaultdict(set)

    test_cell = {idx: cell for (split, idx), cell in cell_map.items() if split == "test"}

    with open(per_sample_csv, newline="") as f:
        for row in csv.DictReader(f):
            cell = test_cell.get(int(row["sample_idx"]))
            if cell is None:
                continue
            inter[cell] += int(row["intersection"])
            union[cell] += int(row["union"])
            samples[cell].add(int(row["sample_idx"]))

    out = {}
    for qt in quality_tiers:
        for st in SIZE_TIERS:
            cell = (qt, st)
            u = union[cell]
            out[cell] = ((inter[cell] / u) if u > 0 else None, len(samples[cell]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Object-size x quality-tier mIoU grid, ViT-B linear.")
    parser.add_argument("--percentiles", type=str, default="33,66",
                         help="Comma-separated UIQM percentile cutpoints, same convention as "
                              "quality_regime_analysis.py. Default: '33,66' (3 quality tiers).")
    args = parser.parse_args()

    try:
        percentiles = _parse_percentiles(args.percentiles)
    except ValueError as e:
        parser.error(str(e))
    quality_tiers = _tier_names(percentiles)
    print(f"Quality tiers: {quality_tiers}; size tiers: {SIZE_TIERS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        cutpoints, _ = _compute_global_cutpoints(percentiles)
    except FileNotFoundError as e:
        print(f"  [error] {e}")
        return

    grouped = _find_vitb_linear_csvs()
    if not grouped:
        print("No matching ViT-B linear per-sample CSVs found in results/per_sample/.")
        return
    datasets_needed = sorted({ds for _, ds in grouped})

    cell_maps: dict[str, dict] = {}
    for ds in datasets_needed:
        try:
            cell_maps[ds] = _load_size_and_quality_cells(ds, cutpoints, quality_tiers)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    rows = []
    for (encoder, ds), paths in sorted(grouped.items()):
        if ds not in cell_maps:
            continue
        family = ENCODER_TO_FAMILY[encoder]
        print(f"  Processing {ds}__{encoder} ({len(paths)} seed file(s))...")
        per_seed = [_seed_miou_by_cell(p, cell_maps[ds], quality_tiers) for p in paths]

        for qt in quality_tiers:
            for st in SIZE_TIERS:
                cell = (qt, st)
                vals = [r[cell][0] for r in per_seed if r[cell][0] is not None]
                n_samples = max((r[cell][1] for r in per_seed), default=0)
                rows.append({
                    "encoder": encoder, "family": family, "ds": ds,
                    "quality_tier": qt, "size_tier": st,
                    "mean_miou": round(float(np.mean(vals)), 6) if vals else "",
                    "std_miou": round(float(np.std(vals, ddof=1)), 6) if len(vals) >= 2 else (0.0 if vals else ""),
                    "n_seeds": len(vals), "n_samples": n_samples,
                })

    with open(OUT_DIR / "size_quality_grid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder", "family", "ds", "quality_tier", "size_tier",
                                                "mean_miou", "std_miou", "n_seeds", "n_samples"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {OUT_DIR / 'size_quality_grid.csv'}")

    agg: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["mean_miou"] == "":
            continue
        agg[(r["family"], r["quality_tier"], r["size_tier"])].append(float(r["mean_miou"]))

    families = sorted({r["family"] for r in rows})
    agg_rows = []
    for family in families:
        for qt in quality_tiers:
            for st in SIZE_TIERS:
                vals = agg.get((family, qt, st), [])
                agg_rows.append({
                    "family": family, "quality_tier": qt, "size_tier": st,
                    "mean_miou": round(float(np.mean(vals)), 6) if vals else "",
                    "std_miou": round(float(np.std(vals)), 6) if vals else "",
                    "n_ds": len(vals),
                })

    with open(OUT_DIR / "size_quality_grid_aggregated.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "quality_tier", "size_tier",
                                                "mean_miou", "std_miou", "n_ds"])
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"-> {OUT_DIR / 'size_quality_grid_aggregated.csv'}")


if __name__ == "__main__":
    main()
