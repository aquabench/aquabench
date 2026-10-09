"""Pivot confidence_intervals.csv into paper-ready per-table CSVs (mIoU + 95% CI, wide
format: rows=backbone, columns=dataset). Ranking tables already exist in wide format as
rankings_{group}_with_resnet.csv -- this script only fills the missing mIoU+CI shape.

Outputs (one CSV per benchmark table, all in --output-dir):
  table_vitb_linear_miou_ci.csv      ViT-B encoders, linear probe, + resnet50 reference row
  table_largest_linear_miou_ci.csv   Largest encoders, linear probe, + resnet50 reference row
  table_vitb_upernet_miou_ci.csv     ViT-B encoders, upernet (fine-tuned), + resnet50 reference row

Each row: backbone, then per dataset (canonical order from ds_names.ALL_DATASETS):
  {ds}_mean, {ds}_ci_low, {ds}_ci_high

Usage:
  python -m analysis.make_result_tables
  python -m analysis.make_result_tables --input-dir results --output-dir results/tables
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analysis.utils.ds_names import ALL_DATASETS

DATASETS = list(ALL_DATASETS)

GROUPS: dict[str, str] = {
    "vitb_linear": "table_vitb_linear_miou_ci.csv",
    "largest_linear": "table_largest_linear_miou_ci.csv",
    "vitb_upernet": "table_vitb_upernet_miou_ci.csv",
    "vitb_upernet_frozen": "table_vitb_upernet_frozen_miou_ci.csv",
}

RESNET_GROUP = "resnet50"


def load_confidence_intervals(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def backbone_name(model: str) -> str:
    # "dinov3_base__linear" -> "dinov3_base"
    return model.split("__")[0]


def pivot_group(rows: list[dict], group: str, include_resnet: bool) -> list[dict]:
    group_rows = [r for r in rows if r["group"] == group]
    models = sorted({r["model"] for r in group_rows})

    by_model_ds = {
        (r["model"], r["ds"]): r for r in group_rows
    }

    out_rows = []
    for model in models:
        row = {"backbone": backbone_name(model)}
        for ds in DATASETS:
            cell = by_model_ds.get((model, ds))
            if cell is None:
                row[f"{ds}_mean"] = ""
                row[f"{ds}_ci_low"] = ""
                row[f"{ds}_ci_high"] = ""
            else:
                row[f"{ds}_mean"] = round(float(cell["mean_miou"]), 4)
                row[f"{ds}_ci_low"] = round(float(cell["ci_low"]), 4)
                row[f"{ds}_ci_high"] = round(float(cell["ci_high"]), 4)
        out_rows.append(row)

    if include_resnet:
        resnet_rows = [r for r in rows if r["group"] == RESNET_GROUP]
        resnet_by_ds = {r["ds"]: r for r in resnet_rows}
        row = {"backbone": "resnet50 (reference)"}
        for ds in DATASETS:
            cell = resnet_by_ds.get(ds)
            if cell is None:
                row[f"{ds}_mean"] = ""
                row[f"{ds}_ci_low"] = ""
                row[f"{ds}_ci_high"] = ""
            else:
                row[f"{ds}_mean"] = round(float(cell["mean_miou"]), 4)
                row[f"{ds}_ci_low"] = round(float(cell["ci_low"]), 4)
                row[f"{ds}_ci_high"] = round(float(cell["ci_high"]), 4)
        out_rows.append(row)

    return out_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = ["backbone"] + [
        f"{ds}_{suffix}" for ds in DATASETS for suffix in ("mean", "ci_low", "ci_high")
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    ci_path = args.input_dir / "confidence_intervals.csv"
    if not ci_path.exists():
        raise SystemExit(f"missing {ci_path} -- run analysis.compute_rankings first")

    rows = load_confidence_intervals(ci_path)

    for group, filename in GROUPS.items():
        pivoted = pivot_group(rows, group, include_resnet=True)
        if not pivoted:
            print(f"[skip] no rows for group={group}")
            continue
        out_path = args.output_dir / filename
        write_csv(out_path, pivoted)
        print(f"[ok] {out_path} ({len(pivoted)} rows)")


if __name__ == "__main__":
    main()
