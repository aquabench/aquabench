"""Per-class size vs. mIoU figure: DINOv3-B / Franca-B / ResNet-50 across
CS, LIACI, SC, TC.

Research question: which classes drive the mIoU gap between backbones -- are the
gaps concentrated in rare (small pixel-fraction) classes, or uniform across the
class-size spectrum?

Per-class test mIoU is derived by pooling intersection/union (summed over all test
images) within each seed's results/per_sample/*.csv file, then averaging across
seeds -- the same pool-then-average pattern used in quality_regime_analysis.py.
Class size is each class's share of test-split pixels, from
{ds}_class_pixel_dist.csv (analysis.dataset_summary, run separately -- needs
$DATA_ROOT, not available on every machine).

Protocols are mixed on purpose: DINOv3-base/Franca-base use the `linear` (frozen
probe) head; ResNet-50 has no linear runs in the results, so it uses `upernet`
(fully fine-tuned) instead -- labeled explicitly in the legend so the mismatch
stays visible rather than implied as an apples-to-apples comparison.

Prerequisites:
  results/analysis/{ds}_class_pixel_dist.csv for ds in {coralscapes,liaci,seaclear,trashcan}
    (python -m analysis.dataset_summary --ds cs liaci sc tc, run where $DATA_ROOT is mounted,
    then copy the CSVs + dataset_splits.csv into results/analysis/ here)
  results/analysis/dataset_splits.csv (same script; supplies metric_ignore_index)
  results/per_sample/{ds}__{encoder}__{head}__seed*__per_sample_test.csv
  results/analysis/{ds}_gt_pixels_per_sample.csv (analysis.gt_pixel_stats, same --ds/copy
    dance as dataset_summary above) -- only needed for the *_trainsize/_trainsamplefrac
    variants; the other figures skip gracefully if it's missing.

Outputs (results/analysis/figures/, saved as .pdf and .svg):
  class_size_vs_miou                     2x2 grid (CS/LIACI/SC/TC), x=test-set pixel_fraction
                                          (log), y=per-class test mIoU (mean +/- std over
                                          seeds), one series per model (DINOv3-B/Franca-B/
                                          ResNet-50). Excludes each dataset's ignored
                                          (background) class, matching test_miou.
  class_size_vs_miou_all                 same, but includes the background class --
                                          matching test_miou_all.
  class_size_vs_miou_dino                same as class_size_vs_miou, DINOv3-B only.
  class_size_vs_miou_dino_trainsize      DINOv3-B only, x=TRAIN-set pixel_fraction instead
                                          of test-set (from {ds}_class_pixel_dist.csv's
                                          split=="train" rows -- no extra prerequisite).
  class_size_vs_miou_dino_trainsamplefrac  DINOv3-B only, x=fraction of train samples that
                                          contain the class at all (not weighted by how many
                                          pixels) -- needs {ds}_gt_pixels_per_sample.csv.

Usage:
  python -m analysis.make_class_size_vs_miou_figure [--analysis-dir results/analysis]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.utils.ds_names import DS_HF_NAME
from analysis.utils.paths import analysis_dir as resolve_analysis_dir

sns.set_theme(style="whitegrid")

DATASETS = ["cs", "liaci", "sc", "tc"]
DATASET_TITLES = {
    "cs": "Coralscapes (CS)", "liaci": "LIACI", "sc": "SeaClear (SC)", "tc": "TrashCan (TC)",
}

# (encoder, head, display label) per model -- see module docstring for why the head differs.
MODELS = [
    ("dinov3_base", "linear", "DINOv3-B (linear)"),
    ("franca_base", "linear", "Franca-B (linear)"),
    ("resnet50", "upernet", "ResNet-50 (UPerNet, fine-tuned)"),
]
MODEL_COLORS = dict(zip((m[2] for m in MODELS), sns.color_palette("tab10", len(MODELS))))
MODEL_MARKERS = {MODELS[0][2]: "o", MODELS[1][2]: "s", MODELS[2][2]: "^"}

PER_SAMPLE_DIR = Path("results/per_sample")


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_dir / name}.{{pdf,svg}}")


def _per_class_miou(ds_hf: str, encoder: str, head: str) -> pd.DataFrame:
    """Per-class test mIoU (mean +/- std over seeds); each seed pools intersection/union
    across all test images before dividing (matches quality_regime_analysis.py's pattern)."""
    per_seed: dict[int, list[float]] = defaultdict(list)
    pattern = f"{ds_hf}__{encoder}__{head}__seed*__per_sample_test.csv"
    paths = sorted(PER_SAMPLE_DIR.glob(pattern))
    if not paths:
        return pd.DataFrame(columns=["class_idx", "mean_miou", "std_miou", "n_seeds"])

    for path in paths:
        intersection: dict[int, int] = defaultdict(int)
        union: dict[int, int] = defaultdict(int)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                c = int(row["class_idx"])
                intersection[c] += int(row["intersection"])
                union[c] += int(row["union"])
        for c, u in union.items():
            if u > 0:
                per_seed[c].append(intersection[c] / u)

    rows = []
    for c, ious in per_seed.items():
        s = pd.Series(ious)
        rows.append({"class_idx": c, "mean_miou": s.mean(), "std_miou": s.std(ddof=0), "n_seeds": len(ious)})
    return pd.DataFrame(rows)


def _class_pixel_fractions(analysis_dir: Path, ds: str, split: str = "test") -> pd.DataFrame:
    """Class size = share of that split's total pixels. split="test" asks 'how much of the
    *evaluated* data is this class'; split="train" asks 'how much of what the model learned
    from was this class'. Returns [class_idx, class_name, size_value]."""
    # class_pixel_dist.csv is named by the short abbreviation (analysis.dataset_summary's
    # --ds argument), unlike per_sample_test.csv which is named by the HF long name.
    df = pd.read_csv(analysis_dir / f"{ds}_class_pixel_dist.csv")
    df = df[df["split"] == split]
    return df[["class_idx", "class_name", "pixel_fraction"]].rename(columns={"pixel_fraction": "size_value"})


def _class_sample_fractions(analysis_dir: Path, ds: str, split: str = "train") -> pd.DataFrame:
    """Class size = fraction of `split` samples containing >=1 pixel of that class -- 'how
    often did the model see this class', as opposed to _class_pixel_fractions' 'how much of
    the pixels were this class' (a class can be in every image but tiny, or rare but large
    when present). Returns [class_idx, class_name, size_value].

    Needs {ds}_gt_pixels_per_sample.csv (analysis.gt_pixel_stats), which -- like
    {ds}_class_pixel_dist.csv -- needs $DATA_ROOT and must be copied in separately."""
    df = pd.read_csv(analysis_dir / f"{ds}_gt_pixels_per_sample.csv")
    df = df[df["split"] == split]
    n_total = df["sample_idx"].nunique()
    present = df[df["gt_pixels"] > 0].groupby("class_idx")["sample_idx"].nunique()
    sizes = (present / n_total).rename("size_value").reset_index()

    # gt_pixels_per_sample.csv has no class_name column -- pull it from class_pixel_dist.csv.
    names = pd.read_csv(analysis_dir / f"{ds}_class_pixel_dist.csv")
    names = names[names["split"] == split][["class_idx", "class_name"]].drop_duplicates()
    return sizes.merge(names, on="class_idx", how="left")


def _metric_ignore_index(analysis_dir: Path, ds: str) -> int:
    path = analysis_dir / "dataset_splits.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    row = df[df["ds"] == ds]
    if row.empty or pd.isna(row["metric_ignore_index"].iloc[0]):
        return 0
    return int(row["metric_ignore_index"].iloc[0])


def _build_panel_data(analysis_dir: Path, ds: str, include_ignored_class: bool = False,
                       models: list[tuple[str, str, str]] = MODELS,
                       size_fn=_class_pixel_fractions) -> pd.DataFrame:
    ds_hf = DS_HF_NAME[ds]
    sizes = size_fn(analysis_dir, ds)
    ignore_idx = _metric_ignore_index(analysis_dir, ds)

    frames = []
    for encoder, head, label in models:
        miou = _per_class_miou(ds_hf, encoder, head)
        if miou.empty:
            continue
        merged = miou.merge(sizes, on="class_idx", how="inner")
        if not include_ignored_class:
            merged = merged[merged["class_idx"] != ignore_idx]
        merged["model"] = label
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _annotate_low_miou_by_size_bin(ax, df: pd.DataFrame, n_bins: int = 5, threshold: float = 0.2) -> None:
    """Split the panel's size range into n_bins equal-width bins in log-space (matching the
    log x-axis, so the bins are evenly spaced visually) and label the worst-scoring point
    in each bin -- but only if it's actually bad (mIoU < threshold). This surfaces classes
    that stay hard across the whole size spectrum, not just the smallest ones, while
    skipping bins where every class is already handled well (no clutter)."""
    df = df[df["size_value"] > 0]
    if df.empty:
        return
    log_size = np.log10(df["size_value"])
    edges = np.linspace(log_size.min(), log_size.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(log_size, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        bin_df = df[bin_idx == b]
        low = bin_df[bin_df["mean_miou"] < threshold]
        if low.empty:
            continue
        row = low.loc[low["mean_miou"].idxmin()]
        ax.annotate(row["class_name"], (row["size_value"], row["mean_miou"]),
                    fontsize=6.5, color="firebrick", xytext=(3, 3), textcoords="offset points")


def plot_class_size_vs_miou(analysis_dir: Path, out_dir: Path, include_ignored_class: bool = False,
                             models: list[tuple[str, str, str]] = MODELS, name: str | None = None,
                             title_suffix: str = "", size_fn=_class_pixel_fractions,
                             xlabel: str = "class size (fraction of test-set pixels)") -> None:
    if name is None:
        name = "class_size_vs_miou_all" if include_ignored_class else "class_size_vs_miou"
    title = "Per-class test mIoU vs. class size (ViT-B tier)" + title_suffix
    if include_ignored_class:
        title += " -- background class included (test_miou_all)"

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, ds in zip(axes.flat, DATASETS):
        df = _build_panel_data(analysis_dir, ds, include_ignored_class=include_ignored_class,
                                models=models, size_fn=size_fn)
        if df.empty:
            ax.set_visible(False)
            continue
        for _, _, label in models:
            sub = df[df["model"] == label].sort_values("size_value")
            if sub.empty:
                continue
            ax.errorbar(sub["size_value"], sub["mean_miou"], yerr=sub["std_miou"],
                        fmt=MODEL_MARKERS[label], color=MODEL_COLORS[label], label=label,
                        markersize=5, capsize=2, elinewidth=1, alpha=0.85)
        _annotate_low_miou_by_size_bin(ax, df)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(DATASET_TITLES[ds])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("per-class test mIoU")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out_dir, name)


def main():
    parser = argparse.ArgumentParser(description="Render per-class size vs. mIoU figure.")
    parser.add_argument("--analysis-dir", default=None, type=Path,
                         help="Defaults to $RESULTS_DIR/analysis.")
    args = parser.parse_args()

    in_dir = args.analysis_dir if args.analysis_dir is not None else resolve_analysis_dir("miou")

    missing = [ds for ds in DATASETS if not (in_dir / f"{ds}_class_pixel_dist.csv").exists()]
    if missing:
        print(f"Missing class_pixel_dist.csv for: {missing}")
        print("Run `python -m analysis.dataset_summary --ds cs liaci sc tc` on a machine "
              f"with $DATA_ROOT mounted, then copy the CSVs into {in_dir}")
        return

    out_dir = in_dir / "figures"
    print("Per-class size vs. mIoU (background excluded, matches test_miou)...")
    plot_class_size_vs_miou(in_dir, out_dir, include_ignored_class=False)
    print("Per-class size vs. mIoU (background included, matches test_miou_all)...")
    plot_class_size_vs_miou(in_dir, out_dir, include_ignored_class=True)
    print("Per-class size vs. mIoU, DINOv3-B only (background excluded)...")
    plot_class_size_vs_miou(in_dir, out_dir, include_ignored_class=False, models=[MODELS[0]],
                             name="class_size_vs_miou_dino", title_suffix=" -- DINOv3-B only")
    print("Per-class TRAIN pixel-size vs. test mIoU, DINOv3-B only (background excluded)...")
    plot_class_size_vs_miou(in_dir, out_dir, include_ignored_class=False, models=[MODELS[0]],
                             name="class_size_vs_miou_dino_trainsize",
                             title_suffix=" -- DINOv3-B only, x = train-set class size",
                             size_fn=partial(_class_pixel_fractions, split="train"),
                             xlabel="class size (fraction of train-set pixels)")

    sample_frac_missing = [ds for ds in DATASETS if not (in_dir / f"{ds}_gt_pixels_per_sample.csv").exists()]
    if sample_frac_missing:
        print(f"Skipping train-sample-fraction figure -- missing gt_pixels_per_sample.csv for: {sample_frac_missing}")
    else:
        print("Fraction of TRAIN samples containing class vs. test mIoU, DINOv3-B only...")
        plot_class_size_vs_miou(in_dir, out_dir, include_ignored_class=False, models=[MODELS[0]],
                                 name="class_size_vs_miou_dino_trainsamplefrac",
                                 title_suffix=" -- DINOv3-B only, x = frac. of train samples with class",
                                 size_fn=partial(_class_sample_fractions, split="train"),
                                 xlabel="fraction of train samples containing class")


if __name__ == "__main__":
    main()
