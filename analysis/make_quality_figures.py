"""Render the per-sample / UIQM-quality-tier figure from analysis/quality_regime_analysis.py
outputs. Reflects whatever --ds / --models / --percentiles that script was run with.

For the "all encoders" ranking figures (critical difference, Kendall tau, mIoU CI,
efficiency), see analysis/make_ranking_figures.py instead.

Reads only results/analysis/test_miou_by_quality_tier.csv + quality_tier_thresholds.csv --
run analysis.quality_regime_analysis first.

Outputs (results/analysis/figures/, saved as .pdf and .svg):
  quality_tier             mIoU (mean +/- std over seeds) per GLOBAL UIQM quality tier, faceted by
                            dataset, every (encoder, head) combo found
  quality_tier_by_regime   same, restricted to the 3 headline regimes (ViT-B linear / ViT-Largest
                            linear / ViT-B UPerNet frozen), colored by encoder family, with the
                            per-dataset dominant TRAIN-data quality tier shaded (train_quality_profile.csv)
                            and per-tier test-image counts on the x labels
  quality_tier_per_dataset same as quality_tier, but tiers are computed relative to each dataset's
                            own UIQM distribution (test_miou_by_quality_tier_per_dataset.csv) instead
                            of pooled globally -- isolates the quality effect from dataset identity.
                            Each panel's title includes the per-dataset Friedman-test p-value from
                            quality_tier_effect_per_dataset.csv.
  uiqm_distributions       diagnostic: per-dataset UIQM KDE overlaid + per-tier image counts, showing
                            the global tiers are largely dataset identity (reads *_image_stats.csv)
  uiqm_distributions_faceted  same data as uiqm_distributions, one panel per dataset instead of
                            overlaid on a shared axis -- easier to read individual dataset shapes
  size_quality_grid        mIoU heatmap over (quality tier x dominant object-size tier), per family

Usage:
  python -m analysis.make_quality_figures [--analysis-dir results/analysis]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.utils.ds_names import ALL_DATASETS
from analysis.utils.model_meta import ENCODER_TO_FAMILY, FAMILIES, LARGEST_ENCODERS, VITB_ENCODERS

sns.set_theme(style="whitegrid")

# The 3 headline evaluation regimes (see compute_rankings.py REGIMES / GROUPS) -- everything
# else (resnet50, fully fine-tuned upernet) is excluded from this figure on purpose.
REGIME_LABELS = {
    "vitb_linear": "ViT-B (linear)",
    "largest_linear": "ViT-L (linear)",
    "vitb_upernet_frozen": "ViT-B UPerNet (frozen)",
}
REGIME_ORDER = list(REGIME_LABELS)
REGIME_MARKERS = {"vitb_linear": "o", "largest_linear": "s", "vitb_upernet_frozen": "^"}
REGIME_LINESTYLES = {"vitb_linear": "-", "largest_linear": "--", "vitb_upernet_frozen": ":"}


def _assign_regime(encoder: str, head: str) -> str | None:
    if head == "linear" and encoder in VITB_ENCODERS:
        return "vitb_linear"
    if head == "linear" and encoder in LARGEST_ENCODERS:
        return "largest_linear"
    if head == "upernet_frozen" and encoder in VITB_ENCODERS:
        return "vitb_upernet_frozen"
    return None


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_dir / name}.{{pdf,svg}}")


def _tier_order(analysis_dir: Path, tiers: list[str],
                 threshold_file: str = "quality_tier_thresholds.csv", ds: str | None = None) -> list[str]:
    """Order tiers by percentile_lower (from `threshold_file`) if available. `threshold_file`
    may have a `ds` column (per-dataset thresholds, cutpoints differ per dataset) -- if so,
    pass `ds` to filter to that dataset's own thresholds before ordering."""
    path = analysis_dir / threshold_file
    if not path.exists():
        return sorted(tiers)
    tdf = pd.read_csv(path)
    if "ds" in tdf.columns and ds is not None:
        tdf = tdf[tdf["ds"] == ds]
    order = (tdf[["tier", "percentile_lower"]].drop_duplicates()
             .sort_values("percentile_lower")["tier"].tolist())
    return [t for t in order if t in tiers] or sorted(tiers)


def plot_quality_tier(analysis_dir: Path, out_dir: Path) -> None:
    path = analysis_dir / "test_miou_by_quality_tier.csv"
    if not path.exists():
        print(f"  [skip] {path} not found -- run analysis.quality_regime_analysis first")
        return
    df = pd.read_csv(path)
    df = df[df["mean_miou"].notna()]
    if df.empty:
        print("  [skip] test_miou_by_quality_tier.csv has no usable rows")
        return
    df["model"] = df["encoder"] + "__" + df["head"]

    n_train_test_by_ds: dict[str, pd.DataFrame] = {}
    profile_path = analysis_dir / "train_quality_profile.csv"
    if profile_path.exists():
        pdf = pd.read_csv(profile_path)
        for ds, gdf in pdf.groupby("ds"):
            n_train_test_by_ds[ds] = gdf.set_index("tier")[["n_train", "n_test"]]

    tiers = _tier_order(analysis_dir, sorted(df["uiqm_tier"].unique()))
    ds_order = [d for d in ALL_DATASETS if d in df["ds"].unique()]
    n_ds = len(ds_order)
    n_cols = min(3, n_ds) or 1
    n_rows = -(-n_ds // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, ds_order):
        gdf = df[df["ds"] == ds]
        models = sorted(gdf["model"].unique())
        width = 0.8 / max(len(models), 1)
        for i, model in enumerate(models):
            mdf = gdf[gdf["model"] == model].set_index("uiqm_tier").reindex(tiers)
            xs = [j + (i - len(models) / 2) * width + width / 2 for j in range(len(tiers))]
            yerr = mdf["std_miou"].fillna(0)
            ax.errorbar(xs, mdf["mean_miou"], yerr=yerr, fmt="o", markersize=4, capsize=3, label=model)
        ax.set_xticks(range(len(tiers)))
        counts = n_train_test_by_ds.get(ds)
        labels = []
        for t in tiers:
            if counts is not None and t in counts.index:
                n_tr, n_te = int(counts.loc[t, "n_train"]), int(counts.loc[t, "n_test"])
                labels.append(f"{t}\ntrain={n_tr}\ntest={n_te}")
            else:
                labels.append(t)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("mIoU")
        ax.set_title(ds)
        ax.legend(fontsize=6, loc="best")

    for ax in axes_flat[n_ds:]:
        ax.axis("off")

    fig.suptitle("mIoU by UIQM quality tier (mean +/- std over seeds)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "quality_tier")


def plot_quality_tier_per_dataset(analysis_dir: Path, out_dir: Path) -> None:
    """Same as plot_quality_tier, but tiers are computed relative to each dataset's own UIQM
    distribution (test_miou_by_quality_tier_per_dataset.csv) rather than pooled globally --
    this holds dataset identity fixed and asks the sharper question: within this dataset,
    does image quality tier affect mIoU? Each panel title shows the per-dataset Friedman-test
    p-value (quality_tier_effect_per_dataset.csv) as a direct significance readout."""
    path = analysis_dir / "test_miou_by_quality_tier_per_dataset.csv"
    if not path.exists():
        print(f"  [skip] {path} not found -- run analysis.quality_regime_analysis first")
        return
    df = pd.read_csv(path)
    df = df[df["mean_miou"].notna()]
    if df.empty:
        print("  [skip] test_miou_by_quality_tier_per_dataset.csv has no usable rows")
        return
    df["model"] = df["encoder"] + "__" + df["head"]

    friedman_p_by_ds: dict[str, str] = {}
    effect_path = analysis_dir / "quality_tier_effect_per_dataset.csv"
    if effect_path.exists():
        edf = pd.read_csv(effect_path)
        for _, row in edf.iterrows():
            p = row["friedman_p"]
            friedman_p_by_ds[row["ds"]] = f"{p:.3g}" if pd.notna(p) and str(p) != "" else "n/a"

    ds_order = [d for d in ALL_DATASETS if d in df["ds"].unique()]
    n_ds = len(ds_order)
    n_cols = min(3, n_ds) or 1
    n_rows = -(-n_ds // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, ds_order):
        gdf = df[df["ds"] == ds]
        tiers = _tier_order(analysis_dir, sorted(gdf["uiqm_tier"].unique()),
                             threshold_file="quality_tier_thresholds_per_dataset.csv", ds=ds)
        models = sorted(gdf["model"].unique())
        width = 0.8 / max(len(models), 1)
        for i, model in enumerate(models):
            mdf = gdf[gdf["model"] == model].set_index("uiqm_tier").reindex(tiers)
            xs = [j + (i - len(models) / 2) * width + width / 2 for j in range(len(tiers))]
            yerr = mdf["std_miou"].fillna(0)
            ax.errorbar(xs, mdf["mean_miou"], yerr=yerr, fmt="o", markersize=4, capsize=3, label=model)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(tiers, rotation=30, ha="right")
        ax.set_ylabel("mIoU")
        p_str = friedman_p_by_ds.get(ds, "n/a")
        ax.set_title(f"{ds} (Friedman p={p_str})")
        ax.legend(fontsize=6, loc="best")

    for ax in axes_flat[n_ds:]:
        ax.axis("off")

    fig.suptitle("mIoU by per-dataset-relative UIQM quality tier (mean +/- std over seeds)\n"
                  "tiers computed from each dataset's own UIQM distribution, not pooled globally", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "quality_tier_per_dataset")


def plot_quality_tier_by_regime(analysis_dir: Path, out_dir: Path) -> None:
    """mIoU per UIQM quality tier, restricted to the 3 headline evaluation regimes
    (ViT-B linear / ViT-Largest linear / ViT-B UPerNet frozen backbone), colored by
    encoder family, faceted by dataset. Shades the quality tier holding the most TRAIN
    images for that dataset (from train_quality_profile.csv) -- this is dataset-specific,
    not global: across the 9 datasets, p33-66 and p66-100 are each dominant for 4, p0-33
    for 1, so there's no single tier to mark once and reuse across every panel."""
    path = analysis_dir / "test_miou_by_quality_tier.csv"
    if not path.exists():
        print(f"  [skip] {path} not found -- run analysis.quality_regime_analysis first")
        return
    df = pd.read_csv(path)
    df = df[df["mean_miou"].notna()].copy()
    df["regime"] = [_assign_regime(e, h) for e, h in zip(df["encoder"], df["head"])]
    df = df[df["regime"].notna()]
    if df.empty:
        print("  [skip] no rows for the 3 headline regimes (vitb_linear/largest_linear/vitb_upernet_frozen)")
        return
    df["family"] = df["encoder"].map(ENCODER_TO_FAMILY)

    tiers = _tier_order(analysis_dir, sorted(df["uiqm_tier"].unique()))
    families = [f for f in FAMILIES if f in df["family"].unique()]
    palette = dict(zip(families, sns.color_palette("tab10", len(families))))

    dom_tier_by_ds: dict[str, str] = {}
    n_train_by_ds: dict[str, pd.Series] = {}
    dom_path = analysis_dir / "train_quality_profile.csv"
    if dom_path.exists():
        tdf = pd.read_csv(dom_path)
        for ds, gdf in tdf.groupby("ds"):
            dom = gdf.loc[gdf["is_dominant_train_tier"], "tier"]
            if len(dom):
                dom_tier_by_ds[ds] = dom.iloc[0]
            n_train_by_ds[ds] = gdf.set_index("tier")["n_train"]

    ds_order = [d for d in ALL_DATASETS if d in df["ds"].unique()]
    n_ds = len(ds_order)
    n_cols = min(3, n_ds) or 1
    n_rows = -(-n_ds // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, ds_order):
        gdf = df[df["ds"] == ds]
        ax.set_xlim(-0.5, len(tiers) - 0.5)
        dom_tier = dom_tier_by_ds.get(ds)
        if dom_tier in tiers:
            i = tiers.index(dom_tier)
            ax.axvspan(i - 0.5, i + 0.5, color="gold", alpha=0.25, zorder=0)
        for family in families:
            for regime in REGIME_ORDER:
                sdf = gdf[(gdf["family"] == family) & (gdf["regime"] == regime)]
                if sdf.empty:
                    continue
                sdf = sdf.set_index("uiqm_tier").reindex(tiers)
                xs = range(len(tiers))
                yerr = sdf["std_miou"].fillna(0)
                ax.errorbar(xs, sdf["mean_miou"], yerr=yerr,
                            fmt=REGIME_MARKERS[regime], linestyle=REGIME_LINESTYLES[regime],
                            color=palette[family], markersize=5, capsize=2, linewidth=1.2,
                            label=f"{family} / {REGIME_LABELS[regime]}")
        # per-tier TRAIN/TEST image count (same for every model on this ds) -- annotate so
        # tiny-n tiers (noisy few-image mIoU estimates) are obvious at a glance.
        n_test_by_tier = gdf.groupby("uiqm_tier")["n_samples"].max()
        n_train_by_tier = n_train_by_ds.get(ds)
        labels = []
        for t in tiers:
            n_te = int(n_test_by_tier.get(t, 0))
            n_tr = int(n_train_by_tier.get(t, 0)) if n_train_by_tier is not None else 0
            labels.append(f"{t}\ntrain={n_tr}\ntest={n_te}" if (n_te or n_tr) else f"{t}\n--")
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(labels, rotation=0, fontsize=7)
        ax.set_ylabel("mIoU")
        dom_str = f" (most train imgs: {dom_tier})" if dom_tier else ""
        ax.set_title(f"{ds}{dom_str}", fontsize=9)

    for ax in axes_flat[n_ds:]:
        ax.axis("off")

    handles_by_label: dict[str, object] = {}
    for ax in axes_flat[:n_ds]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            handles_by_label.setdefault(li, hi)
    fig.legend(handles_by_label.values(), handles_by_label.keys(),
               loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)

    fig.suptitle("mIoU by UIQM quality tier -- ViT-B linear / ViT-L linear / ViT-B UPerNet (frozen), "
                  "by family\ngold band = quality tier with the most TRAIN images for that dataset", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "quality_tier_by_regime")


def _global_cutpoints(analysis_dir: Path) -> list[float]:
    """Numeric UIQM cutpoints between tiers, from quality_tier_thresholds.csv."""
    path = analysis_dir / "quality_tier_thresholds.csv"
    if not path.exists():
        return []
    tdf = pd.read_csv(path).sort_values("percentile_lower")
    return [float(x) for x in tdf["uiqm_upper"].tolist() if pd.notna(x) and str(x) != ""]


def plot_uiqm_distributions(analysis_dir: Path, out_dir: Path, split: str = "test") -> None:
    """Per-dataset UIQM score distribution (KDE), all overlaid on one axis, with the global
    tier cutpoints drawn as vertical lines. This is the diagnostic for the quality-tier
    figures: it shows that the *global* tiers are largely dataset-membership (each dataset
    occupies a narrow, dataset-specific UIQM band), so a per-tier mIoU aggregate pooled
    across datasets is confounded with which datasets fall in that tier. It also shows,
    per tier, how many of each dataset's images land there (n annotations)."""
    cutpoints = _global_cutpoints(analysis_dir)
    ds_present = [d for d in ALL_DATASETS if (analysis_dir / f"{d}_image_stats.csv").exists()]
    if not ds_present:
        print("  [skip] no *_image_stats.csv found -- run analysis.compute_image_stats first")
        return

    palette = dict(zip(ds_present, sns.color_palette("tab10", len(ds_present))))
    fig, (ax_kde, ax_cnt) = plt.subplots(2, 1, figsize=(11, 9), height_ratios=[2, 1.4])

    tier_names = ["p0-33", "p33-66", "p66-100"]  # matches default 33/66 split
    edges = [-np.inf] + cutpoints + [np.inf]
    counts: dict[str, list[int]] = {}

    for ds in ds_present:
        sdf = pd.read_csv(analysis_dir / f"{ds}_image_stats.csv")
        s = sdf.loc[sdf["split"] == split, "uiqm_score"].to_numpy()
        if s.size == 0:
            continue
        sns.kdeplot(x=s, ax=ax_kde, color=palette[ds], label=ds, fill=False, linewidth=1.8, clip=(0, None))
        counts[ds] = [int(((s >= edges[i]) & (s < edges[i + 1])).sum()) for i in range(len(edges) - 1)]

    for cp in cutpoints:
        ax_kde.axvline(cp, color="gray", linestyle="--", linewidth=1)
        ax_kde.text(cp, ax_kde.get_ylim()[1] * 0.98, f"{cp:.2f}", rotation=90,
                    va="top", ha="right", fontsize=7, color="gray")
    ax_kde.set_xlabel("UIQM score (higher = 'better' underwater image quality)")
    ax_kde.set_ylabel("density")
    ax_kde.set_title(f"Per-dataset UIQM distribution ({split} split), with global tier cutpoints\n"
                      "datasets occupy distinct UIQM bands -> global quality tiers are largely dataset identity")
    ax_kde.legend(fontsize=8, ncol=2)

    # stacked-style grouped bars: n images per tier per dataset
    n_tiers = len(tier_names)
    ds_with_counts = [d for d in ds_present if d in counts]
    width = 0.8 / max(len(ds_with_counts), 1)
    for i, ds in enumerate(ds_with_counts):
        xs = [t + (i - len(ds_with_counts) / 2) * width + width / 2 for t in range(n_tiers)]
        ax_cnt.bar(xs, counts[ds], width=width, color=palette[ds], label=ds)
    ax_cnt.set_xticks(range(n_tiers))
    ax_cnt.set_xticklabels(tier_names)
    ax_cnt.set_ylabel(f"# {split} images")
    ax_cnt.set_xlabel("global UIQM quality tier")
    ax_cnt.set_title("Image count per global tier, per dataset (tiny bars = mIoU in that tier is a few-image estimate)")

    fig.tight_layout()
    _save(fig, out_dir, "uiqm_distributions")


def plot_uiqm_distributions_faceted(analysis_dir: Path, out_dir: Path, split: str = "test") -> None:
    """Same data/scope as plot_uiqm_distributions (per-dataset UIQM score distribution,
    `split` split, global tier cutpoints as vlines), but one panel per dataset instead of
    all datasets overlaid on a shared axis -- easier to read an individual dataset's shape
    without the others crowding it."""
    cutpoints = _global_cutpoints(analysis_dir)
    ds_present = [d for d in ALL_DATASETS if (analysis_dir / f"{d}_image_stats.csv").exists()]
    if not ds_present:
        print("  [skip] no *_image_stats.csv found -- run analysis.compute_image_stats first")
        return

    palette = dict(zip(ds_present, sns.color_palette("tab10", len(ds_present))))
    n_ds = len(ds_present)
    n_cols = min(3, n_ds) or 1
    n_rows = -(-n_ds // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, ds_present):
        sdf = pd.read_csv(analysis_dir / f"{ds}_image_stats.csv")
        s = sdf.loc[sdf["split"] == split, "uiqm_score"].to_numpy()
        if s.size == 0:
            ax.set_title(f"{ds} (no {split} data)", fontsize=9)
            ax.axis("off")
            continue
        sns.histplot(x=s, ax=ax, color=palette[ds], stat="density", bins=30, alpha=0.35, edgecolor="none")
        sns.kdeplot(x=s, ax=ax, color=palette[ds], linewidth=1.8, clip=(0, None))
        for cp in cutpoints:
            ax.axvline(cp, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{ds} (n={s.size})", fontsize=9)
        ax.set_xlabel("UIQM score", fontsize=8)
        ax.set_ylabel("density", fontsize=8)

    for ax in axes_flat[n_ds:]:
        ax.axis("off")

    fig.suptitle(f"Per-dataset UIQM distribution ({split} split)\n"
                  "gray dashed lines = global tier cutpoints (pooled over all datasets)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "uiqm_distributions_faceted")


def plot_size_quality_grid(analysis_dir: Path, out_dir: Path) -> None:
    """Heatmap of mIoU across (UIQM quality tier x dominant object-size tier), one panel
    per encoder family, ViT-B linear only. See analysis.object_size_quality_analysis for
    how cells are computed -- object-size dominance is by connected-component COUNT, not
    pixel area, so treat this as a first-pass signal."""
    path = analysis_dir / "size_quality_grid_aggregated.csv"
    if not path.exists():
        print(f"  [skip] {path} not found -- run analysis.object_size_quality_analysis first")
        return
    df = pd.read_csv(path)
    df = df[df["mean_miou"].notna()]
    if df.empty:
        print("  [skip] size_quality_grid_aggregated.csv has no usable rows")
        return

    tiers = _tier_order(analysis_dir, sorted(df["quality_tier"].unique()))
    size_order = [s for s in ("small", "medium", "large") if s in df["size_tier"].unique()]
    families = [f for f in FAMILIES if f in df["family"].unique()]

    n_fam = len(families)
    n_cols = min(3, n_fam) or 1
    n_rows = -(-n_fam // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    vmin, vmax = df["mean_miou"].min(), df["mean_miou"].max()
    for ax, family in zip(axes_flat, families):
        fdf = df[df["family"] == family]
        pivot = fdf.pivot(index="size_tier", columns="quality_tier", values="mean_miou").reindex(
            index=size_order, columns=tiers)
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="viridis", vmin=vmin, vmax=vmax,
                    cbar=False, linewidths=0.5, linecolor="white")
        ax.set_title(family)
        ax.set_xlabel("UIQM quality tier")
        ax.set_ylabel("dominant object-size tier")

    for ax in axes_flat[n_fam:]:
        ax.axis("off")

    fig.suptitle("mIoU by quality tier x dominant object-size tier, ViT-B linear\n"
                  "(size = most-frequent COCO area bucket by component count, not area-weighted)", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "size_quality_grid")


def main():
    parser = argparse.ArgumentParser(description="Render the quality-tier figure from "
                                                   "quality_regime_analysis.py outputs.")
    parser.add_argument("--analysis-dir", default="results/analysis", type=Path)
    args = parser.parse_args()

    if not (args.analysis_dir / "test_miou_by_quality_tier.csv").exists():
        print(f"No quality-tier outputs found in {args.analysis_dir}.")
        print("Run `python -m analysis.quality_regime_analysis` first.")
        return

    out_dir = args.analysis_dir / "figures"

    print("Quality-tier mIoU...")
    plot_quality_tier(args.analysis_dir, out_dir)

    print("Quality-tier mIoU (per-dataset-relative tiers)...")
    plot_quality_tier_per_dataset(args.analysis_dir, out_dir)

    print("Quality-tier mIoU by regime (ViT-B linear / ViT-L linear / ViT-B UPerNet frozen)...")
    plot_quality_tier_by_regime(args.analysis_dir, out_dir)

    print("Per-dataset UIQM distributions (diagnostic)...")
    plot_uiqm_distributions(args.analysis_dir, out_dir)

    print("Per-dataset UIQM distributions (faceted)...")
    plot_uiqm_distributions_faceted(args.analysis_dir, out_dir)

    print("Object-size x quality-tier grid (ViT-B linear)...")
    plot_size_quality_grid(args.analysis_dir, out_dir)


if __name__ == "__main__":
    main()
