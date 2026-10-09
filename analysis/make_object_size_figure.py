"""Per-class object-size (connected-component pixel area) distribution, train vs. test.

Research question: how big (in pixels) are individual object instances of each class,
and does that differ between train and test splits? Answers "is this class typically a
few huge blobs or many tiny specks" per dataset -- complementary to
class_size_vs_miou.py's pixel_fraction (total pixels of a class, pooled over the whole
split) which can't distinguish "one huge object" from "many small ones" summing to the
same total.

Background is excluded (metric_ignore_index from dataset_splits.csv if available,
else the class literally named "background").

Prerequisites:
  results/analysis/{ds}_object_size_dist.csv for ds in {cs, liaci, sc, tc}
    (python -m analysis.compute_object_size_dist --ds cs liaci sc tc, run where
    $DATA_ROOT is mounted, then copy the CSVs into results/analysis/ here)
  results/analysis/dataset_splits.csv (optional; supplies metric_ignore_index -- falls
    back to excluding the class named "background" if missing)

Outputs (results/analysis/figures/, saved as .pdf and .svg):
  object_size_by_class   2x2 grid (CS/LIACI/SC/TC), x=class (sorted by median test-split
                          object area), y=connected-component pixel area (log scale),
                          boxplot per class split by train/test.

Usage:
  python -m analysis.make_object_size_figure [--analysis-dir results/analysis]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

DATASETS = ["cs", "liaci", "sc", "tc"]
DATASET_TITLES = {
    "cs": "Coralscapes (CS)", "liaci": "LIACI", "sc": "SeaClear (SC)", "tc": "TrashCan (TC)",
}
SPLIT_ORDER = ["train", "test"]


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_dir / name}.{{pdf,svg}}")


def _metric_ignore_index(analysis_dir: Path, ds: str) -> int | None:
    path = analysis_dir / "dataset_splits.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["ds"] == ds]
    if row.empty or pd.isna(row["metric_ignore_index"].iloc[0]):
        return None
    return int(row["metric_ignore_index"].iloc[0])


def _load_object_sizes(analysis_dir: Path, ds: str) -> pd.DataFrame:
    df = pd.read_csv(analysis_dir / f"{ds}_object_size_dist.csv")
    df = df[df["split"].isin(SPLIT_ORDER)]
    ignore_idx = _metric_ignore_index(analysis_dir, ds)
    if ignore_idx is not None:
        df = df[df["class_idx"] != ignore_idx]
    else:
        df = df[df["class_name"].str.lower() != "background"]
    return df


def plot_object_size_by_class(analysis_dir: Path, out_dir: Path) -> None:
    missing = [ds for ds in DATASETS if not (analysis_dir / f"{ds}_object_size_dist.csv").exists()]
    if missing:
        print(f"  [skip] Missing {{ds}}_object_size_dist.csv for: {missing}")
        print("  Run `python -m analysis.compute_object_size_dist --ds cs liaci sc tc` on a "
              "machine with $DATA_ROOT mounted, then copy the CSVs into "
              f"{analysis_dir}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, ds in zip(axes.flat, DATASETS):
        df = _load_object_sizes(analysis_dir, ds)
        if df.empty:
            ax.set_visible(False)
            continue

        # Order classes by median TEST-split object area, smallest to largest.
        test_medians = (df[df["split"] == "test"].groupby("class_name")["area_px"]
                         .median().sort_values())
        class_order = test_medians.index.tolist()
        for c in df["class_name"].unique():
            if c not in class_order:
                class_order.append(c)

        sns.boxplot(data=df, x="class_name", y="area_px", hue="split", order=class_order,
                    hue_order=SPLIT_ORDER, ax=ax, fliersize=1.5, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(DATASET_TITLES[ds])
        ax.set_xlabel("")
        ax.set_ylabel("object area (px, log scale)")
        ax.tick_params(axis="x", rotation=60)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.legend(title="split", fontsize=7, loc="upper left")

    fig.suptitle("Per-class connected-component object size (train vs. test), background excluded")
    fig.tight_layout()
    _save(fig, out_dir, "object_size_by_class")


def main():
    parser = argparse.ArgumentParser(description="Render per-class object-size-by-connected-component figure.")
    parser.add_argument("--analysis-dir", default="results/analysis", type=Path)
    args = parser.parse_args()

    out_dir = args.analysis_dir / "figures"
    print("Per-class object size (connected-component area), train vs. test...")
    plot_object_size_by_class(args.analysis_dir, out_dir)


if __name__ == "__main__":
    main()
