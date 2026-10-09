"""Cross-quality-regime generalization analysis.

Research question: When training data is dominated by a given quality tier, do models
generalize across quality regimes in the test split?

This joins per-sample test IoU (from PerSampleTestCallback) with per-image UIQM scores
(from compute_image_stats.py) to measure mIoU per quality tier per model per dataset,
then aggregates by train-data quality dominance.

Tier boundaries are *not* fixed at 33rd/66th percentile, and are computed GLOBALLY across
all datasets (not per-dataset): this script pools the raw `uiqm_score` column from every
{ds}_image_stats.csv it can find, computes cutpoints once from that pooled distribution
using whatever --percentiles you pass (e.g. "25,50,75" for 4 tiers), and applies the same
absolute UIQM thresholds to every dataset. This keeps "low/mid/high quality" comparable
across datasets, rather than relative to each dataset's own distribution. Tier names are
derived from the percentile ranges, e.g. --percentiles "33,66" -> p0-33, p33-66, p66-100.

Per-sample CSVs are seed-suffixed (results/per_sample/{ds}__{encoder}__{head}__seed{S}__
per_sample_test.csv) or unsuffixed for legacy/single-seed runs. All seeds found for a
given (encoder, head, ds) are pooled per-seed (intersection/union summed over samples +
classes within that seed), then reported as mean +/- std mIoU across seeds per tier.

Prerequisites:
  results/per_sample/{ds}__{encoder}__{head}[__seed{S}]__per_sample_test.csv
  results/analysis/{ds}_image_stats.csv   (needs the uiqm_score column, for every dataset
                                            in analysis.utils.ds_names.ALL_DATASETS ideally,
                                            so the global percentiles are well-pooled)

This script runs BOTH tierings every time (no flag): the global one above, and a second
per-dataset-relative tiering where cutpoints are computed from each dataset's own UIQM
distribution alone (not pooled). Under global tiering, since each dataset occupies its own
narrow UIQM band, quality tier ends up almost synonymous with dataset identity -- the
per-dataset tiering isolates the "does quality affect mIoU" question from that confound,
holding dataset identity fixed. Its outputs are the same 4 files with a `_per_dataset` suffix,
plus a Friedman-test significance summary (see below).

Outputs:
  results/analysis/quality_tier_thresholds.csv
    tier, percentile_lower, percentile_upper, uiqm_lower, uiqm_upper, n_datasets_pooled

  results/analysis/train_quality_profile.csv
    ds, tier, n_train, n_val, n_test, frac_train, is_dominant_train_tier
    (dominance is decided from n_train only; n_val/n_test are for reference)

  results/analysis/test_miou_by_quality_tier.csv
    encoder, head, ds, uiqm_tier, mean_miou, std_miou, n_seeds, n_samples

  results/analysis/cross_quality_generalization.csv
    encoder, head, ds, dominant_train_tier, test_tier, mean_miou, std_miou, n_seeds, n_samples

  results/analysis/cross_quality_generalization_aggregated.csv
    dominant_train_tier, test_tier, mean_miou, std_miou, n_model_ds_pairs

  -- per-dataset-relative tiering (same schemas as above, `_per_dataset` suffixed) --
  results/analysis/quality_tier_thresholds_per_dataset.csv
    ds, tier, percentile_lower, percentile_upper, uiqm_lower, uiqm_upper, n_images
  results/analysis/train_quality_profile_per_dataset.csv
  results/analysis/test_miou_by_quality_tier_per_dataset.csv
  results/analysis/cross_quality_generalization_per_dataset.csv
  results/analysis/cross_quality_generalization_aggregated_per_dataset.csv

  results/analysis/quality_tier_effect_per_dataset.csv
    ds, n_tiers, n_models, friedman_stat, friedman_p, lowest_tier, highest_tier,
    mean_miou_diff_extreme_tiers
    (Friedman test per dataset: tier = treatment, model = block, on per-tier mean_miou from
    the per-dataset tiering; only models with a value in every tier are included. This is the
    direct answer to "does quality tier significantly affect mIoU within this dataset?")

By default (no --ds / --models / --percentiles) this processes every model and every
dataset found in results/per_sample/, with the standard 33/66 global split -- i.e. it
gives you everything. Use --ds / --models to narrow the per-sample evaluation down to
specific datasets or models; the global tier cutpoints are unaffected by either (always
pooled over analysis.utils.ds_names.ALL_DATASETS) so tier boundaries stay comparable
across runs.

Usage:
  python -m analysis.quality_regime_analysis
  python -m analysis.quality_regime_analysis --ds cou liaci
  python -m analysis.quality_regime_analysis --models dinov3_base__linear franca_base__upernet
  python -m analysis.quality_regime_analysis --percentiles 25,50,75
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.utils.ds_names import ALL_DATASETS, DS_HF_NAME

# per-sample filenames carry the full HF dataset name (the --ds CLI arg used for the
# sweep); image_stats.csv files are keyed by the short abbreviation. They're identical
# for 5/9 datasets (cou, l4s, liaci, suim, uiis10k) but not for cs/coralscapes,
# df/deepfish, sc/seaclear, tc/trashcan -- convert so ds is the short abbrev everywhere.
_DS_LONG_TO_SHORT = {long: short for short, long in DS_HF_NAME.items()}

OUT_DIR = Path(os.environ.get("RESULTS_DIR", "results")) / "analysis"
PER_SAMPLE_DIR = Path(os.environ.get("RESULTS_DIR", "results")) / "per_sample"

_PER_SAMPLE_RE = re.compile(
    r"^(?P<ds>[^_]+(?:_[^_]+)*?)__(?P<encoder>[^_]+(?:_[^_]+)*?)__(?P<head>linear|upernet_frozen|upernet)"
    r"(?:__seed(?P<seed>-?\d+))?__per_sample_test$"
)


def _parse_percentiles(spec: str) -> list[float]:
    vals = [float(x) for x in spec.split(",") if x.strip() != ""]
    if not vals:
        raise ValueError("--percentiles must contain at least one value")
    if any(v <= 0 or v >= 100 for v in vals):
        raise ValueError("--percentiles values must be strictly between 0 and 100")
    vals = sorted(set(vals))
    return vals


def _tier_names(percentiles: list[float]) -> list[str]:
    bounds = [0.0] + percentiles + [100.0]
    return [f"p{bounds[i]:g}-{bounds[i + 1]:g}" for i in range(len(bounds) - 1)]


def _load_uiqm_scores(ds: str) -> dict[tuple[str, int], float]:
    """Returns {(split, sample_idx): uiqm_score} from the raw per-image stats file."""
    path = OUT_DIR / f"{ds}_image_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run compute_image_stats.py first.")
    out: dict[tuple[str, int], float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "uiqm_score" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no uiqm_score column -- re-run compute_image_stats.py.")
        for row in reader:
            out[(row["split"], int(row["sample_idx"]))] = float(row["uiqm_score"])
    return out


def _compute_global_cutpoints(percentiles: list[float]) -> tuple[list[float], list[str]]:
    """Pools uiqm_score across every available dataset's image_stats.csv (all splits) and
    computes percentile cutpoints once. Returns (cutpoints, datasets_pooled)."""
    all_scores: list[float] = []
    datasets_pooled: list[str] = []
    for ds in ALL_DATASETS:
        path = OUT_DIR / f"{ds}_image_stats.csv"
        if not path.exists():
            print(f"  [skip] no image_stats for '{ds}' -- excluded from global percentile pool")
            continue
        try:
            scores = _load_uiqm_scores(ds)
        except ValueError as e:
            print(f"  [skip] {e}")
            continue
        all_scores.extend(scores.values())
        datasets_pooled.append(ds)

    if not all_scores:
        raise FileNotFoundError(
            "No {ds}_image_stats.csv files found -- run compute_image_stats.py first."
        )
    cutpoints = np.percentile(np.array(all_scores, dtype=np.float64), percentiles).tolist()
    print(f"  Global cutpoints from {len(datasets_pooled)} dataset(s) "
          f"({len(all_scores)} images): {[round(c, 4) for c in cutpoints]}")
    return cutpoints, datasets_pooled


def _assign_tiers(ds: str, cutpoints: list[float], tier_names: list[str]) -> dict:
    """Buckets one dataset's samples into tiers using globally-computed cutpoints."""
    scores = _load_uiqm_scores(ds)

    def assign(score: float) -> str:
        i = 0
        for cp in cutpoints:
            if score < cp:
                break
            i += 1
        return tier_names[i]

    return {key: assign(score) for key, score in scores.items()}


def _write_thresholds(percentiles: list[float], tier_names: list[str], cutpoints: list[float],
                       datasets_pooled: list[str]) -> None:
    bounds = [0.0] + percentiles + [100.0]
    edges = [float("-inf")] + cutpoints + [float("inf")]
    rows = []
    for i, tier in enumerate(tier_names):
        rows.append({
            "tier": tier,
            "percentile_lower": bounds[i], "percentile_upper": bounds[i + 1],
            "uiqm_lower": edges[i] if edges[i] != float("-inf") else "",
            "uiqm_upper": edges[i + 1] if edges[i + 1] != float("inf") else "",
            "n_datasets_pooled": len(datasets_pooled),
        })
    with open(OUT_DIR / "quality_tier_thresholds.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tier", "percentile_lower", "percentile_upper",
                                                "uiqm_lower", "uiqm_upper", "n_datasets_pooled"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Quality-tier thresholds (global, pooled over {datasets_pooled}) "
          f"-> {OUT_DIR / 'quality_tier_thresholds.csv'}")


def _compute_per_dataset_cutpoints(ds: str, percentiles: list[float]) -> tuple[list[float], int]:
    """Percentile cutpoints computed from just this one dataset's own UIQM scores (all
    splits) -- unlike _compute_global_cutpoints, no pooling across datasets. Returns
    (cutpoints, n_images)."""
    scores = _load_uiqm_scores(ds)
    if not scores:
        raise ValueError(f"No uiqm_score rows found for '{ds}'")
    values = np.array(list(scores.values()), dtype=np.float64)
    cutpoints = np.percentile(values, percentiles).tolist()
    return cutpoints, len(values)


def _write_thresholds_per_dataset(percentiles: list[float], tier_names: list[str],
                                   cutpoints_by_ds: dict[str, list[float]],
                                   n_images_by_ds: dict[str, int]) -> None:
    bounds = [0.0] + percentiles + [100.0]
    rows = []
    for ds, cutpoints in sorted(cutpoints_by_ds.items()):
        edges = [float("-inf")] + cutpoints + [float("inf")]
        for i, tier in enumerate(tier_names):
            rows.append({
                "ds": ds, "tier": tier,
                "percentile_lower": bounds[i], "percentile_upper": bounds[i + 1],
                "uiqm_lower": edges[i] if edges[i] != float("-inf") else "",
                "uiqm_upper": edges[i + 1] if edges[i + 1] != float("inf") else "",
                "n_images": n_images_by_ds[ds],
            })
    path = OUT_DIR / "quality_tier_thresholds_per_dataset.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ds", "tier", "percentile_lower", "percentile_upper",
                                                "uiqm_lower", "uiqm_upper", "n_images"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-dataset quality-tier thresholds -> {path}")


def _train_quality_profile(ds: str, tier_map: dict, tier_names: list[str]) -> tuple[list[dict], str]:
    """Returns (rows for this ds, dominant_train_tier). Dominance is decided from the
    train split only (that's the data the model actually learns from); val/test counts
    are reported alongside for reference."""
    n_train = {t: 0 for t in tier_names}
    n_val = {t: 0 for t in tier_names}
    n_test = {t: 0 for t in tier_names}
    for (split, _), tier in tier_map.items():
        if split == "train":
            n_train[tier] += 1
        elif split == "validation":
            n_val[tier] += 1
        elif split == "test":
            n_test[tier] += 1

    total_train = sum(n_train.values())
    if total_train == 0:
        dominant = "unknown"
    else:
        dominant = max(tier_names, key=lambda t: n_train[t])

    rows = []
    for t in tier_names:
        frac_train = round(n_train[t] / total_train, 4) if total_train else 0
        rows.append({
            "ds": ds, "tier": t, "n_train": n_train[t], "n_val": n_val[t], "n_test": n_test[t],
            "frac_train": frac_train, "is_dominant_train_tier": t == dominant,
        })
    return rows, dominant


def _seed_miou_by_tier(per_sample_csv: Path, tier_map: dict, tier_names: list[str]) -> dict[str, tuple[float, int]]:
    """One seed's per-sample CSV -> {tier: (pooled_miou, n_samples)} using the 'test' split tiers."""
    tier_intersection: dict[str, int] = defaultdict(int)
    tier_union: dict[str, int] = defaultdict(int)
    tier_samples: dict[str, set] = defaultdict(set)

    test_tier: dict[int, str] = {idx: tier for (split, idx), tier in tier_map.items() if split == "test"}

    with open(per_sample_csv, newline="") as f:
        for row in csv.DictReader(f):
            sample_idx = int(row["sample_idx"])
            tier = test_tier.get(sample_idx)
            if tier is None:
                continue
            tier_intersection[tier] += int(row["intersection"])
            tier_union[tier] += int(row["union"])
            tier_samples[tier].add(sample_idx)

    out = {}
    for tier in tier_names:
        u = tier_union[tier]
        miou = (tier_intersection[tier] / u) if u > 0 else None
        out[tier] = (miou, len(tier_samples[tier]))
    return out


def _find_per_sample_csvs(datasets: list[str] | None,
                           models: list[str] | None) -> dict[tuple[str, str, str], list[Path]]:
    """Group per-sample CSVs by (encoder, head, ds) -> [path, ...] across all found seeds."""
    grouped: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    if not PER_SAMPLE_DIR.exists():
        return grouped
    datasets_set = set(datasets) if datasets is not None else None
    for path in sorted(PER_SAMPLE_DIR.glob("*__per_sample_test.csv")):
        m = _PER_SAMPLE_RE.match(path.stem)
        if not m:
            print(f"  [skip] unrecognized per-sample filename: {path.name}")
            continue
        ds_name, encoder, head = m.group("ds"), m.group("encoder"), m.group("head")
        ds_name = _DS_LONG_TO_SHORT.get(ds_name, ds_name)
        if datasets_set is not None and ds_name not in datasets_set:
            continue
        if models is not None and f"{encoder}__{head}" not in models:
            continue
        grouped[(encoder, head, ds_name)].append(path)
    return grouped


def _run_tiering_pass(tier_names: list[str], datasets_needed: list[str], grouped: dict,
                       cutpoints_by_ds: dict[str, list[float]], suffix: str) -> list[dict]:
    """Tier assignment + train-quality-profile + per-tier mIoU + cross-regime generalization,
    for a given per-dataset set of cutpoints. Writes train_quality_profile{suffix}.csv,
    test_miou_by_quality_tier{suffix}.csv, cross_quality_generalization{suffix}.csv and
    cross_quality_generalization_aggregated{suffix}.csv. Returns all_tier_rows (used by the
    caller for the Friedman significance test in the per-dataset pass)."""
    tier_maps: dict[str, dict] = {}
    dominant_train_tier: dict[str, str] = {}
    profile_rows: list[dict] = []
    for ds in datasets_needed:
        if ds not in cutpoints_by_ds:
            continue
        try:
            tier_map = _assign_tiers(ds, cutpoints_by_ds[ds], tier_names)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip] {e}")
            continue
        tier_maps[ds] = tier_map
        rows, dominant = _train_quality_profile(ds, tier_map, tier_names)
        profile_rows.extend(rows)
        dominant_train_tier[ds] = dominant

    with open(OUT_DIR / f"train_quality_profile{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ds", "tier", "n_train", "n_val", "n_test", "frac_train",
                                                "is_dominant_train_tier"])
        writer.writeheader()
        writer.writerows(profile_rows)
    print(f"Train quality profiles -> {OUT_DIR / f'train_quality_profile{suffix}.csv'}")

    all_tier_rows: list[dict] = []
    for (encoder, head, ds), paths in sorted(grouped.items()):
        if ds not in tier_maps:
            continue
        print(f"  Processing {ds}__{encoder}__{head} ({len(paths)} seed file(s))...")
        per_seed_results = [_seed_miou_by_tier(p, tier_maps[ds], tier_names) for p in paths]

        for tier in tier_names:
            vals = [r[tier][0] for r in per_seed_results if r[tier][0] is not None]
            n_samples = max((r[tier][1] for r in per_seed_results), default=0)
            all_tier_rows.append({
                "encoder": encoder, "head": head, "ds": ds, "uiqm_tier": tier,
                "mean_miou": round(float(np.mean(vals)), 6) if vals else "",
                "std_miou": round(float(np.std(vals, ddof=1)), 6) if len(vals) >= 2 else 0.0 if vals else "",
                "n_seeds": len(vals),
                "n_samples": n_samples,
            })

    with open(OUT_DIR / f"test_miou_by_quality_tier{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder", "head", "ds", "uiqm_tier", "mean_miou", "std_miou",
                                                "n_seeds", "n_samples"])
        writer.writeheader()
        writer.writerows(all_tier_rows)
    print(f"Per-tier mIoU -> {OUT_DIR / f'test_miou_by_quality_tier{suffix}.csv'}")

    cross_rows: list[dict] = []
    for row in all_tier_rows:
        ds = row["ds"]
        cross_rows.append({
            "encoder": row["encoder"], "head": row["head"], "ds": ds,
            "dominant_train_tier": dominant_train_tier.get(ds, "unknown"),
            "test_tier": row["uiqm_tier"],
            "mean_miou": row["mean_miou"], "std_miou": row["std_miou"],
            "n_seeds": row["n_seeds"], "n_samples": row["n_samples"],
        })

    with open(OUT_DIR / f"cross_quality_generalization{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder", "head", "ds", "dominant_train_tier", "test_tier",
                                                "mean_miou", "std_miou", "n_seeds", "n_samples"])
        writer.writeheader()
        writer.writerows(cross_rows)
    print(f"Cross-quality generalization -> {OUT_DIR / f'cross_quality_generalization{suffix}.csv'}")

    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in cross_rows:
        if row["mean_miou"] == "":
            continue
        key = (row["dominant_train_tier"], row["test_tier"])
        agg[key].append(float(row["mean_miou"]))

    agg_rows = []
    for train_tier in tier_names:
        for test_tier in tier_names:
            vals = agg.get((train_tier, test_tier), [])
            agg_rows.append({
                "dominant_train_tier": train_tier,
                "test_tier": test_tier,
                "mean_miou": round(float(np.mean(vals)), 6) if vals else "",
                "std_miou": round(float(np.std(vals)), 6) if vals else "",
                "n_model_ds_pairs": len(vals),
            })

    with open(OUT_DIR / f"cross_quality_generalization_aggregated{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dominant_train_tier", "test_tier", "mean_miou", "std_miou",
                                                "n_model_ds_pairs"])
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"Aggregated tier grid -> {OUT_DIR / f'cross_quality_generalization_aggregated{suffix}.csv'}")

    return all_tier_rows


def _tier_effect_significance(all_tier_rows: list[dict], tier_names: list[str]) -> list[dict]:
    """Per dataset: Friedman test (tier = treatment, model = block) on per-tier mean_miou,
    restricted to models with a value in every tier. Same test used elsewhere in this repo
    (compute_rankings.py's critical-difference machinery) for 'does this grouping variable
    produce significantly different performance' -- this answers that for quality tier,
    within one dataset at a time."""
    by_ds: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in all_tier_rows:
        if row["mean_miou"] == "":
            continue
        model = f"{row['encoder']}__{row['head']}"
        by_ds[row["ds"]][model][row["uiqm_tier"]] = float(row["mean_miou"])

    out = []
    for ds, model_scores in sorted(by_ds.items()):
        covered = [m for m, tiers in model_scores.items() if all(t in tiers for t in tier_names)]
        n_models, k = len(covered), len(tier_names)
        if n_models < 2 or k < 2:
            print(f"  [skip] {ds}: not enough models with complete tier coverage "
                  f"(n_models={n_models}) for Friedman test")
            out.append({
                "ds": ds, "n_tiers": k, "n_models": n_models,
                "friedman_stat": "", "friedman_p": "",
                "lowest_tier": tier_names[0], "highest_tier": tier_names[-1],
                "mean_miou_diff_extreme_tiers": "",
            })
            continue

        tier_groups = [[model_scores[m][t] for m in covered] for t in tier_names]
        friedman_stat, friedman_p = stats.friedmanchisquare(*tier_groups)
        diffs = [model_scores[m][tier_names[-1]] - model_scores[m][tier_names[0]] for m in covered]

        out.append({
            "ds": ds, "n_tiers": k, "n_models": n_models,
            "friedman_stat": round(float(friedman_stat), 6),
            "friedman_p": round(float(friedman_p), 6),
            "lowest_tier": tier_names[0], "highest_tier": tier_names[-1],
            "mean_miou_diff_extreme_tiers": round(float(np.mean(diffs)), 6),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Cross-quality-regime generalization analysis.")
    parser.add_argument("--ds", nargs="+", default=None,
                         help="Restrict to these datasets (e.g. --ds cou liaci). "
                              "Default: all datasets found in results/per_sample/.")
    parser.add_argument("--models", nargs="+", default=None,
                         help="Restrict to these encoder__head combos (e.g. dinov3_base__linear "
                              "franca_base__upernet). Default: all models found in results/per_sample/.")
    parser.add_argument("--percentiles", type=str, default="33,66",
                         help="Comma-separated percentile cut points in (0,100) defining tier "
                              "boundaries, computed GLOBALLY across all datasets' pooled UIQM "
                              "scores, e.g. '25,50,75' for 4 tiers. Default: '33,66' (3 tiers).")
    args = parser.parse_args()

    try:
        percentiles = _parse_percentiles(args.percentiles)
    except ValueError as e:
        parser.error(str(e))
    tier_names = _tier_names(percentiles)
    print(f"Tiers: {tier_names} (from percentiles {percentiles})")

    grouped = _find_per_sample_csvs(args.ds, args.models)
    if not grouped:
        print("No matching per-sample test CSVs found in results/per_sample/.")
        print("Run: python -m aquabench.cli test ... to generate them.")
        return

    datasets_needed = sorted({ds for _, _, ds in grouped})

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Global tiering pass: cutpoints pooled across all datasets (unchanged behavior) ----
    print("\n=== Global tiering (cutpoints pooled across all datasets) ===")
    try:
        cutpoints, datasets_pooled = _compute_global_cutpoints(percentiles)
    except FileNotFoundError as e:
        print(f"  [error] {e}")
        return
    _write_thresholds(percentiles, tier_names, cutpoints, datasets_pooled)
    global_cutpoints_by_ds = {ds: cutpoints for ds in datasets_needed}
    _run_tiering_pass(tier_names, datasets_needed, grouped, global_cutpoints_by_ds, suffix="")

    # ---- Per-dataset tiering pass: cutpoints relative to each dataset's own distribution ----
    print("\n=== Per-dataset tiering (cutpoints relative to each dataset's own UIQM distribution) ===")
    per_ds_cutpoints: dict[str, list[float]] = {}
    per_ds_n_images: dict[str, int] = {}
    for ds in datasets_needed:
        try:
            cps, n_images = _compute_per_dataset_cutpoints(ds, percentiles)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip] {e}")
            continue
        per_ds_cutpoints[ds] = cps
        per_ds_n_images[ds] = n_images
    _write_thresholds_per_dataset(percentiles, tier_names, per_ds_cutpoints, per_ds_n_images)
    all_tier_rows_per_ds = _run_tiering_pass(
        tier_names, list(per_ds_cutpoints), grouped, per_ds_cutpoints, suffix="_per_dataset"
    )

    effect_rows = _tier_effect_significance(all_tier_rows_per_ds, tier_names)
    effect_path = OUT_DIR / "quality_tier_effect_per_dataset.csv"
    with open(effect_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ds", "n_tiers", "n_models", "friedman_stat", "friedman_p",
                                                "lowest_tier", "highest_tier", "mean_miou_diff_extreme_tiers"])
        writer.writeheader()
        writer.writerows(effect_rows)
    print(f"Per-dataset quality-tier effect significance -> {effect_path}")


if __name__ == "__main__":
    main()
