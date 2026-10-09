"""Render the "all encoders" ranking figures from analysis/compute_rankings.py outputs.

Reads only {analysis-dir}/*.csv + insights.json -- run compute_rankings.py first (with the
same --metric, if not the default) so the two stay in sync.
For the per-sample / UIQM-quality-tier figure, see analysis/make_quality_figures.py instead.

Outputs ({analysis-dir}/figures/{subdir}/, each saved as .pdf and .svg):
  kendall/kendall_heatmap_{group}[_with_resnet]
                                            one 9x9 dataset-agreement heatmap per linear/upernet group,
                                            vfm_only and with_resnet (resnet50/upernet added as extra model)
  kendall/kendall_heatmap_regimes          3x3 family-rank agreement between evaluation regimes
                                            (vitb_linear/vitb_upernet_frozen/largest_linear), ResNet-50 excluded
  rank_stability/rank_stability_heads      ViT-B encoders: avg rank under linear vs UPerNet (slope chart)
  miou_ci/miou_ci                          mIoU with 95% CI per dataset, faceted by group
  efficiency/efficiency                    GPU-hours vs avg rank, bubble size = parameter count
  critical_difference/cd_diagram_{group}_{variant}
                                            Demsar (2006) critical-difference diagram, per group x
                                            {vfm_only, with_resnet}
  critical_difference/cd_diagram_combined_regimes_family
                                            same, but family-level pooled across all 3 evaluation
                                            regimes (27 blocks) -- see the caveat in
                                            compute_combined_critical_difference_family()
  family_performance/family_performance_{family}
                                            per-dataset mIoU, ViT-B linear vs ViT-B UPerNet
                                            (frozen backbone) vs ViT-Largest linear, one figure
                                            per encoder family
  family_performance/family_performance_combined
                                            same, all families faceted into one figure
  upernet_regime/upernet_regime_{family}   per-dataset mIoU, ViT-B UPerNet frozen backbone vs.
                                            fully fine-tuned, one figure per encoder family
  upernet_regime/upernet_regime_combined   same, all families faceted into one figure

Usage:
  python -m analysis.make_ranking_figures [--analysis-dir PATH] [--metric miou|miou_all]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.utils.ds_names import ALL_DATASETS
from analysis.utils.model_meta import ENCODER_TO_FAMILY, FAMILIES, PARAMS_M
from analysis.utils.paths import analysis_dir as resolve_analysis_dir

sns.set_theme(style="whitegrid")

GROUPS = ["vitb_linear", "largest_linear", "vitb_upernet_frozen", "vitb_upernet"]


def _save(fig, out_dir: Path, name: str, subdir: str = "") -> None:
    dest = out_dir / subdir if subdir else out_dir
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(dest / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {dest / name}.{{pdf,svg}}")


def plot_kendall_heatmaps(analysis_dir: Path, out_dir: Path) -> None:
    path = analysis_dir / "kendall_between_datasets.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return
    df = pd.read_csv(path)

    variants = [(group, group, "") for group in GROUPS] + \
               [(f"{group}_with_resnet", group, " (with ResNet-50)") for group in GROUPS]
    for variant, group, title_suffix in variants:
        gdf = df[df["group"] == variant]
        if gdf.empty:
            print(f"  [skip] no kendall data for group {variant}")
            continue
        mat = pd.DataFrame(1.0, index=ALL_DATASETS, columns=ALL_DATASETS)
        for _, row in gdf.iterrows():
            mat.loc[row["ds1"], row["ds2"]] = row["tau"]
            mat.loc[row["ds2"], row["ds1"]] = row["tau"]

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                    square=True, cbar_kws={"label": "Kendall's tau"}, ax=ax)
        ax.set_title(f"Dataset-agreement (Kendall tau) -- {group}{title_suffix}")
        _save(fig, out_dir, f"kendall_heatmap_{variant}", subdir="kendall")


REGIME_LABELS = {
    "vitb_linear": "ViT-B Linear",
    "vitb_upernet_frozen": "ViT-B UPerNet\n(frozen)",
    "largest_linear": "ViT-Largest\nLinear",
}
REGIME_ORDER = list(REGIME_LABELS.keys())


def plot_kendall_regime_heatmap(analysis_dir: Path, out_dir: Path) -> None:
    path = analysis_dir / "kendall_between_regimes.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return
    df = pd.read_csv(path)
    df = df[df["ds"] == "ALL_AGGREGATE"]
    if df.empty:
        print("  [skip] no aggregate kendall data between regimes")
        return

    labels = [REGIME_LABELS[r] for r in REGIME_ORDER]
    mat = pd.DataFrame(1.0, index=labels, columns=labels)
    for _, row in df.iterrows():
        l1, l2 = REGIME_LABELS[row["regime1"]], REGIME_LABELS[row["regime2"]]
        tau = row["tau"] if row["tau"] != "" else float("nan")
        mat.loc[l1, l2] = tau
        mat.loc[l2, l1] = tau

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "Kendall's tau"}, ax=ax)
    ax.set_title("Family-rank agreement across evaluation regimes\n(ResNet-50 excluded)")
    fig.tight_layout()
    _save(fig, out_dir, "kendall_heatmap_regimes", subdir="kendall")


def plot_rank_stability(analysis_dir: Path, out_dir: Path) -> None:
    path = analysis_dir / "combined_rankings.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("  [skip] combined_rankings.csv is empty")
        return

    long = df.melt(id_vars="encoder", value_vars=["avg_rank_linear", "avg_rank_upernet_frozen"],
                    var_name="head", value_name="avg_rank")
    long["head"] = long["head"].map({"avg_rank_linear": "Linear", "avg_rank_upernet_frozen": "UPerNet (frozen)"})

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.lineplot(data=long, x="head", y="avg_rank", hue="encoder", marker="o", ax=ax)
    ax.invert_yaxis()  # rank 1 = best, show at top
    ax.set_ylabel("Average rank (lower = better)")
    ax.set_xlabel("")
    ax.set_title("ViT-B rank stability: Linear vs UPerNet (frozen encoder)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    _save(fig, out_dir, "rank_stability_heads", subdir="rank_stability")


def plot_miou_ci(analysis_dir: Path, out_dir: Path) -> None:
    path = analysis_dir / "confidence_intervals.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return
    df = pd.read_csv(path)
    df = df[df["group"].isin(GROUPS)]
    if df.empty:
        print("  [skip] no CI data for the 3 groups")
        return

    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(11, 4 * len(GROUPS)), sharex=True)
    for ax, group in zip(axes, GROUPS):
        gdf = df[df["group"] == group]
        models = sorted(gdf["model"].unique())
        ds_order = [d for d in ALL_DATASETS if d in gdf["ds"].unique()]
        x_base = {ds: i for i, ds in enumerate(ds_order)}
        width = 0.8 / max(len(models), 1)

        for i, model in enumerate(models):
            mdf = gdf[gdf["model"] == model].set_index("ds").reindex(ds_order)
            xs = [x_base[ds] + (i - len(models) / 2) * width + width / 2 for ds in ds_order]
            yerr_low = mdf["mean_miou"] - mdf["ci_low"]
            yerr_high = mdf["ci_high"] - mdf["mean_miou"]
            ax.errorbar(xs, mdf["mean_miou"], yerr=[yerr_low, yerr_high], fmt="o",
                        markersize=4, capsize=3, label=model)

        ax.set_xticks(range(len(ds_order)))
        ax.set_xticklabels(ds_order, rotation=45, ha="right")
        ax.set_ylabel("mIoU")
        ax.set_title(group)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)

    fig.suptitle("mIoU with 95% CI per dataset", y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, "miou_ci", subdir="miou_ci")


def plot_efficiency(analysis_dir: Path, out_dir: Path) -> None:
    insights_path = analysis_dir / "insights.json"
    if not insights_path.exists():
        print(f"  [skip] {insights_path} not found")
        return
    insights = json.loads(insights_path.read_text())
    gpu_hours = insights.get("gpu_hours_total_by_model", {})

    avg_rank: dict[str, float] = {}
    for group in GROUPS:
        rpath = analysis_dir / f"rankings_{group}.csv"
        if not rpath.exists():
            continue
        rdf = pd.read_csv(rpath)
        for _, row in rdf.iterrows():
            avg_rank[row["model"]] = row["avg_rank"]

    rows = []
    for model, hours in gpu_hours.items():
        if model not in avg_rank:
            continue
        encoder = model.split("__")[0]
        params = PARAMS_M.get(encoder)
        if params is None:
            continue
        rows.append({"model": model, "gpu_hours": hours, "avg_rank": avg_rank[model], "params_m": params})

    if not rows:
        print("  [skip] no overlapping model data for efficiency plot")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="gpu_hours", y="avg_rank", size="params_m", sizes=(50, 1000),
                     hue="model", legend=False, alpha=0.7, ax=ax)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["gpu_hours"], row["avg_rank"]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
    ax.invert_yaxis()
    ax.set_xlabel("Total GPU hours")
    ax.set_ylabel("Average rank (lower = better)")
    ax.set_title("Efficiency: GPU cost vs. rank (bubble size = parameter count)")
    _save(fig, out_dir, "efficiency", subdir="efficiency")


VARIANT_LABELS = {
    "vitb_linear": "ViT-B (linear)",
    "vitb_upernet_frozen": "ViT-B (UPerNet, frozen)",
    "largest_linear": "ViT-Largest (linear)",
}
VARIANT_ORDER = list(VARIANT_LABELS.values())

# Frozen-backbone vs fully fine-tuned UPerNet, ViT-B tier only -- a secondary comparison
# (the frozen-backbone protocol is the main research focus; full fine-tuning is a later extra).
UPERNET_REGIME_LABELS = {
    "vitb_upernet_frozen": "UPerNet (frozen backbone)",
    "vitb_upernet": "UPerNet (fully fine-tuned)",
}
UPERNET_REGIME_ORDER = list(UPERNET_REGIME_LABELS.values())


def _variant_long_df(analysis_dir: Path, variant_labels: dict[str, str]) -> pd.DataFrame:
    path = analysis_dir / "confidence_intervals.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["group"].isin(variant_labels)]
    df["encoder"] = df["model"].str.split("__").str[0]
    df["family"] = df["encoder"].map(ENCODER_TO_FAMILY)
    df = df.dropna(subset=["family"])
    df["variant"] = df["group"].map(variant_labels)
    return df


def _draw_variant_bars(ax, df: pd.DataFrame, ds_order: list[str], full_variant_order: list[str]) -> None:
    """Grouped bar chart with 95% CI error bars (from confidence_intervals.csv's ci_low/ci_high)
    -- shows not just which variant wins per dataset but how big/reliable the gap is. Colors are
    keyed by position in full_variant_order so a variant keeps the same color across figures even
    when a family is missing one of the variants."""
    present = [v for v in full_variant_order if v in df["variant"].unique()]
    palette = dict(zip(full_variant_order, sns.color_palette(n_colors=len(full_variant_order))))
    width = 0.8 / max(len(present), 1)
    x_base = {ds: i for i, ds in enumerate(ds_order)}

    for i, variant in enumerate(present):
        vdf = df[df["variant"] == variant].set_index("ds").reindex(ds_order)
        xs = [x_base[ds] + (i - len(present) / 2) * width + width / 2 for ds in ds_order]
        means = vdf["mean_miou"]
        yerr = [means - vdf["ci_low"], vdf["ci_high"] - means]
        ax.bar(xs, means, width=width * 0.9, label=variant, color=palette[variant],
               yerr=yerr, capsize=2, error_kw={"linewidth": 0.8, "ecolor": "black"})

    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels(ds_order, rotation=45, ha="right")


def _plot_variant_family_grid(df: pd.DataFrame, variant_order: list[str], out_dir: Path,
                               name_prefix: str, subdir: str, title: str) -> None:
    ds_order = [d for d in ALL_DATASETS if d in df["ds"].unique()]
    families = [f for f in FAMILIES if f in df["family"].unique()]

    for family in families:
        fdf = df[df["family"] == family]
        if fdf.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _draw_variant_bars(ax, fdf, ds_order, variant_order)
        ax.set_xlabel("")
        ax.set_ylabel("mIoU")
        ax.set_title(f"{family}: {title} (error bars: 95% CI)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        fig.tight_layout()
        _save(fig, out_dir, f"{name_prefix}_{family}", subdir=subdir)

    n_fam = len(families)
    n_cols = min(3, n_fam) or 1
    n_rows = -(-n_fam // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, family in zip(axes_flat, families):
        fdf = df[df["family"] == family]
        _draw_variant_bars(ax, fdf, ds_order, variant_order)
        ax.set_ylabel("mIoU")
        ax.set_title(family)
        ax.legend(fontsize=7, loc="best")
    for ax in axes_flat[n_fam:]:
        ax.axis("off")
    fig.suptitle(f"{title}, all families (error bars: 95% CI)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, f"{name_prefix}_combined", subdir=subdir)


def plot_family_performance(analysis_dir: Path, out_dir: Path) -> None:
    df = _variant_long_df(analysis_dir, VARIANT_LABELS)
    if df.empty:
        print("  [skip] no data for family-performance figures")
        return
    _plot_variant_family_grid(df, VARIANT_ORDER, out_dir, "family_performance", "family_performance",
                               "per-dataset mIoU across model tiers")


def plot_upernet_regime_comparison(analysis_dir: Path, out_dir: Path) -> None:
    df = _variant_long_df(analysis_dir, UPERNET_REGIME_LABELS)
    if df.empty:
        print("  [skip] no data for UPerNet frozen-vs-fine-tuned figures")
        return
    _plot_variant_family_grid(df, UPERNET_REGIME_ORDER, out_dir, "upernet_regime", "upernet_regime",
                               "UPerNet: frozen backbone vs. fully fine-tuned")


def _cliques(ranks: list[float], cd: float) -> list[tuple[int, int]]:
    """Maximal runs of a sorted rank list whose span is within the CD threshold -- i.e.
    groups of models not significantly different from one another (Demsar 2006)."""
    k = len(ranks)
    raw = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i:
            raw.append((i, j))
    return [c for c in raw if not any(o != c and o[0] <= c[0] and c[1] <= o[1] for o in raw)]


def _draw_cd_diagram(df: pd.DataFrame, title: str, out_name: str, out_dir: Path) -> None:
    """Demsar (2006) critical-difference diagram: models on a mean-rank axis, with a thick
    bar connecting each clique of models whose mean-rank difference is within CD (i.e. not
    significantly different, per the Friedman + Nemenyi test compute_rankings.py already ran)."""
    df = df.sort_values("mean_rank").reset_index(drop=True)
    if df.empty:
        return
    models = df["model"].tolist()
    ranks = df["mean_rank"].tolist()
    cd = float(df["cd_value"].iloc[0])
    k = len(models)
    cliques = _cliques(ranks, cd)
    n_cliques = len(cliques)

    top = k + 0.3 * n_cliques + 0.9  # room for the clique bars + the CD legend
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.35 * k + 0.3 * n_cliques))
    lo, hi = min(ranks) - 0.5, max(ranks) + 0.5
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top)

    for i, (model, r) in enumerate(zip(models, ranks)):
        y = k - i
        ax.plot([r], [y], "o", color="black", zorder=3)
        ax.plot([r, hi], [y, y], color="lightgray", linewidth=0.8, zorder=1)
        ax.text(hi + 0.1, y, model, va="center", fontsize=9)

    for level, (a, b) in enumerate(cliques):
        y = k + 0.3 * (level + 1)
        ax.plot([ranks[a], ranks[b]], [y, y], color="black", linewidth=3, zorder=2)

    cd_y = k + 0.3 * n_cliques + 0.5
    ax.plot([lo + 0.1, lo + 0.1 + cd], [cd_y, cd_y], color="black", linewidth=1.5)
    ax.text(lo + 0.1 + cd / 2, cd_y + 0.1, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Average rank (lower = better)")
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    friedman_p = float(df["friedman_p"].iloc[0])
    ax.set_title(f"{title}, Friedman p={friedman_p:.4f}")
    fig.tight_layout()
    _save(fig, out_dir, out_name, subdir="critical_difference")


def plot_critical_difference(analysis_dir: Path, out_dir: Path) -> None:
    for group in GROUPS:
        for variant in ("vfm_only", "with_resnet"):
            path = analysis_dir / f"critical_difference_{group}_{variant}.csv"
            if not path.exists():
                print(f"  [skip] {path} not found")
                continue
            df = pd.read_csv(path)
            _draw_cd_diagram(df, f"Critical difference -- {group} ({variant})",
                              f"cd_diagram_{group}_{variant}", out_dir)


def plot_combined_critical_difference_family(analysis_dir: Path, out_dir: Path) -> None:
    """Family-level CD diagram pooled across all 3 evaluation regimes (vitb_linear,
    vitb_upernet_frozen, largest_linear -- 27 blocks). CAVEAT: largest_linear scores each
    family via its largest-tier encoder while the other two regimes use ViT-B, so this is a
    "best-in-class across protocols" ranking, not a fixed-scale comparison -- see
    compute_combined_critical_difference_family() in compute_rankings.py."""
    path = analysis_dir / "critical_difference_combined_regimes_family.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return
    df = pd.read_csv(path)
    _draw_cd_diagram(
        df,
        "Critical difference -- combined regimes, family-level\n"
        "(vitb_linear + vitb_upernet_frozen + largest_linear; largest_linear uses each family's largest encoder)",
        "cd_diagram_combined_regimes_family", out_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Render ranking figures from compute_rankings.py outputs.")
    parser.add_argument("--analysis-dir", default=None, type=Path,
                         help="Defaults to $RESULTS_DIR/analysis[_miou_all] (via --metric).")
    parser.add_argument("--metric", choices=["miou", "miou_all"], default="miou",
                         help="Must match the --metric used for compute_rankings.py; only used "
                              "to resolve the default --analysis-dir.")
    args = parser.parse_args()

    in_dir = args.analysis_dir if args.analysis_dir is not None else resolve_analysis_dir(args.metric)

    if not (in_dir / "all_results.csv").exists() and not (in_dir / "insights.json").exists():
        print(f"No ranking outputs found in {in_dir}.")
        print("Run `python -m analysis.compute_rankings` first.")
        return

    out_dir = in_dir / "figures"

    print("Kendall heatmaps...")
    plot_kendall_heatmaps(in_dir, out_dir)

    print("Kendall regime-agreement heatmap...")
    plot_kendall_regime_heatmap(in_dir, out_dir)

    print("Rank stability (linear vs upernet)...")
    plot_rank_stability(in_dir, out_dir)

    print("mIoU with CI...")
    plot_miou_ci(in_dir, out_dir)

    print("Efficiency scatter...")
    plot_efficiency(in_dir, out_dir)

    print("Critical-difference diagrams...")
    plot_critical_difference(in_dir, out_dir)

    print("Combined critical-difference diagram (family-level, all regimes)...")
    plot_combined_critical_difference_family(in_dir, out_dir)

    print("Per-family performance...")
    plot_family_performance(in_dir, out_dir)

    print("UPerNet frozen vs. fully fine-tuned...")
    plot_upernet_regime_comparison(in_dir, out_dir)


if __name__ == "__main__":
    main()
