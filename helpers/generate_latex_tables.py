#!/usr/bin/env python3
"""Generate paper-ready LaTeX tables for AQUABENCH from results CSVs.

Modes
-----
iou          mIoU ± CI table from a wide table_*_miou_ci.csv
             Best per column bold, second-best underlined.
             ResNet-50 row placed last with * marker, excluded from bold/underline.

rank         Per-dataset rank table with colored cells + avg rank + overall rank.
             Input: rankings_*_with_resnet.csv

rank-summary Multi-method rank summary (AvgR_max / AvgR_mean / AvgR_med + OR).
             Pass up to 3 --input files: --input max.csv mean.csv med.csv
             with matching --method-labels "max" "mean" "med".

Examples
--------
  python helpers/generate_latex_tables.py iou \\
      results/tables/table_vitb_linear_miou_ci.csv

  python helpers/generate_latex_tables.py iou \\
      results/tables/table_largest_linear_miou_ci.csv \\
      --sizes dinov3_7b=7B eva02_large=L franca_giant=g radio_v4_h=H vjepa21_vitg=g \\
      --caption "Largest-variant linear probing results."

  python helpers/generate_latex_tables.py rank \\
      results/rankings_vitb_linear_with_resnet.csv

  python helpers/generate_latex_tables.py rank-summary \\
      --input results/rankings_vitb_linear_with_resnet.csv \\
      --method-labels "\\\\mu"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ── Dataset display names ────────────────────────────────────────────────────

ALL_DATASETS = ["cs", "cou", "df", "l4s", "liaci", "sc", "suim", "tc", "uiis10k"]

DS_LABEL: dict[str, str] = {
    "cs": "CS",
    "cou": "COU",
    "df": "DF",
    "l4s": "L4S",
    "liaci": "LIACI",
    "sc": "SC",
    "suim": "SUIM",
    "tc": "TC",
    "uiis10k": "UIIS10K",
    "usis16k": "USIS16K",
}

# ── Backbone display names ───────────────────────────────────────────────────

BACKBONE_DISPLAY: dict[str, str] = {
    # ViT-B scale
    "dinov3_base": "DINOv3",
    "eva02_base": "EVA02",
    "franca_base": "Franca",
    "radio_v3_base": "RADIO v3",
    "vjepa21_vitb": "V-JEPA2.1",
    # Large scale
    "dinov3_large": "DINOv3 L",
    "eva02_large": "EVA02",
    "franca_large": "Franca L",
    # Giant / Huge / 7B
    "dinov3_7b": "DINOv3",
    "franca_giant": "Franca",
    "radio_v4_h": "RADIO v4",
    "radio_v4_so400m": "RADIO v4 SO",
    "vjepa21_vitg": "V-JEPA2.1",
    # Reference baseline
    "resnet50 (reference)": "ResNet-50",
    "resnet50": "ResNet-50",
    "resnet50__upernet": "ResNet-50",
}

RESNET_KEYS = {"resnet50 (reference)", "resnet50", "resnet50__upernet"}

# Rank color definitions used in the paper (rank1 = best warm, rank10 = worst cool)
MAX_RANK_COLOR = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def display_name(raw: str, extra_map: dict[str, str] | None = None) -> str:
    if extra_map and raw in extra_map:
        return extra_map[raw]
    # Strip __linear / __upernet suffix used in rankings CSVs
    key = raw.split("__")[0].strip()
    return BACKBONE_DISPLAY.get(key, key.replace("_", " "))


def is_resnet(raw: str) -> bool:
    key = raw.split("__")[0].strip()
    return key in RESNET_KEYS


def fmt_value(v: float, decimals: int = 3) -> str:
    """Format a float as '.XXX' (leading zero suppressed)."""
    s = f"{v:.{decimals}f}"          # e.g. "0.415"
    return s.lstrip("0") or "0"      # e.g. ".415"  (keep "0" if < 0.001)


def fmt_miou_ci(mean: float, half: float) -> str:
    """Format as '.415\\pm.007' — matching the paper style."""
    return f"{fmt_value(mean)}\\pm{fmt_value(half)}"


def rank_color(r: int | str) -> str:
    try:
        ri = int(float(r))
    except (ValueError, TypeError):
        return ""
    ri = max(1, min(ri, MAX_RANK_COLOR))
    return f"\\cellcolor{{rank{ri}}}"


def bold(s: str) -> str:
    return f"$\\mathbf{{{s}}}$"


def uline(s: str) -> str:
    return f"\\underline{{${s}$}}"


def plain(s: str) -> str:
    return f"${s}$"


def rothead(label: str) -> str:
    return f"\\rotatebox{{90}}{{\\small {label}}}"


def write_table(lines: list[str], output: Path | None) -> None:
    text = "\n".join(lines)
    if output:
        output.write_text(text)
        print(f"[ok] written to {output}", file=sys.stderr)
    else:
        print(text)


# ── IoU table ────────────────────────────────────────────────────────────────

def load_iou_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Return (datasets_found, rows) from a wide miou_ci CSV."""
    rows = list(csv.DictReader(path.open()))
    # Detect which datasets are present (have a _mean column)
    cols = list(rows[0].keys()) if rows else []
    datasets = [c.removesuffix("_mean") for c in cols if c.endswith("_mean")]
    return datasets, rows


def iou_table(args: argparse.Namespace) -> None:
    path = Path(args.input[0])
    datasets, rows = load_iou_csv(path)

    # Filter / reorder datasets
    if args.datasets:
        wanted = [d.lower() for d in args.datasets]
        datasets = [d for d in datasets if d in wanted]

    # Parse size map from "key=val key=val ..."
    size_map: dict[str, str] = {}
    if args.sizes:
        for item in args.sizes:
            k, _, v = item.partition("=")
            size_map[k.strip()] = v.strip()
    has_size = bool(size_map)

    # Split resnet from VFM rows
    vfm_rows = [r for r in rows if not is_resnet(r["backbone"])]
    ref_rows = [r for r in rows if is_resnet(r["backbone"])]

    # Per-dataset: compute (mean, half_ci) for each row
    def parse_row(r: dict) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for ds in datasets:
            try:
                mean = float(r[f"{ds}_mean"])
                lo   = float(r[f"{ds}_ci_low"])
                hi   = float(r[f"{ds}_ci_high"])
                out[ds] = (mean, (hi - lo) / 2)
            except (KeyError, ValueError):
                out[ds] = (float("nan"), float("nan"))
        return out

    vfm_parsed  = [parse_row(r) for r in vfm_rows]

    # Find best and second-best per dataset across ALL rows (VFMs + reference)
    ref_parsed = [parse_row(r) for r in ref_rows]
    all_parsed = vfm_parsed + ref_parsed

    best: dict[str, float] = {}
    second: dict[str, float] = {}
    for ds in datasets:
        vals = sorted(
            (p[ds][0] for p in all_parsed if not (p[ds][0] != p[ds][0])),  # exclude NaN
            reverse=True,
        )
        best[ds]   = vals[0] if len(vals) > 0 else float("nan")
        second[ds] = vals[1] if len(vals) > 1 else float("nan")

    # Build column spec
    n_ds = len(datasets)
    size_col = "l" if has_size else ""
    col_spec = f"l{size_col}{'c' * n_ds}"

    # Dataset header row
    rot_headers = " & ".join(rothead(DS_LABEL.get(d, d.upper())) for d in datasets)
    if has_size:
        backbone_head = "\\textbf{Backbone} & \\textbf{Size}"
    else:
        backbone_head = "\\textbf{Backbone}"
    header_cols = f"{backbone_head} & {rot_headers} \\\\"

    # Render a data row
    def render_row(r: dict, parsed: dict[str, tuple[float, float]], is_ref: bool = False) -> str:
        raw_key = r["backbone"]
        name = display_name(raw_key, args.name_map)
        if is_ref:
            name = name + "\\textsuperscript{*}"

        size_cell = f" & {size_map.get(raw_key, '---')}" if has_size else ""
        cells = []
        for ds in datasets:
            mean, half = parsed[ds]
            if mean != mean:  # NaN
                cells.append("---")
                continue
            s = fmt_miou_ci(mean, half)
            if abs(mean - best[ds]) < 1e-9:
                cells.append(bold(s))
            elif abs(mean - second[ds]) < 1e-9:
                cells.append(uline(s))
            else:
                cells.append(plain(s))
        return f"    {name}{size_cell} & {' & '.join(cells)} \\\\"

    label = args.label or "tab:miou_results"
    caption = args.caption or "Mean IoU with confidence intervals per backbone and dataset."

    lines = [
        "\\begin{table}[ht]",
        f"  \\caption{{{caption}}}",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{2pt}",
        "  \\resizebox{\\linewidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header_cols}",
        "    \\midrule",
    ]

    for r, p in zip(vfm_rows, vfm_parsed):
        lines.append(render_row(r, p))

    if ref_rows:
        lines.append("    \\midrule")
        for r, p in zip(ref_rows, ref_parsed):
            lines.append(render_row(r, p, is_ref=True))

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  }",
        f"  \\label{{{label}}}",
        "\\end{table}",
    ]

    write_table(lines, Path(args.output) if args.output else None)


# ── Rank table ───────────────────────────────────────────────────────────────

def load_rank_csv(path: Path) -> tuple[list[str], list[dict]]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return [], []
    # Detect dataset columns (exclude meta columns)
    meta = {"model", "avg_rank", "avg_rank_std", "overall_rank", "encoder",
            "avg_rank_linear", "avg_rank_upernet", "combined_avg_rank", "combined_overall_rank"}
    datasets = [k for k in rows[0].keys() if k not in meta]
    return datasets, rows


def rank_table(args: argparse.Namespace) -> None:
    path = Path(args.input[0])
    datasets, rows = load_rank_csv(path)

    if args.datasets:
        wanted = [d.lower() for d in args.datasets]
        datasets = [d for d in datasets if d in wanted]

    has_avg  = "avg_rank" in rows[0] if rows else False
    has_or   = "overall_rank" in rows[0] if rows else False

    # Split resnet reference rows
    vfm_rows = [r for r in rows if not is_resnet(r.get("model", r.get("encoder", "")))]
    ref_rows = [r for r in rows if is_resnet(r.get("model", r.get("encoder", "")))]

    rot_headers = " & ".join(rothead(DS_LABEL.get(d, d.upper())) for d in datasets)
    extra_heads = ""
    if has_avg:
        extra_heads += " & \\textbf{AvgR}"
    if has_or:
        extra_heads += " & \\textbf{OR}"

    n_ds = len(datasets)
    extra_cols = ("c" * has_avg) + ("c" * has_or)
    col_spec = f"l{'c' * n_ds}{extra_cols}"

    label   = args.label   or "tab:rank_results"
    caption = args.caption or "Per-dataset ranks and overall rank per backbone."

    def render_rank_row(r: dict, is_ref: bool = False) -> str:
        key = r.get("model", r.get("encoder", ""))
        name = display_name(key, args.name_map)
        if is_ref:
            name = name + "\\textsuperscript{*}"

        cells = []
        for ds in datasets:
            val = r.get(ds, "")
            try:
                ri = int(float(val))
                cells.append(f"{rank_color(ri)}{ri}")
            except (ValueError, TypeError):
                cells.append("---")

        if has_avg:
            avg = r.get("avg_rank", "")
            try:
                cells.append(f"{float(avg):.2f}")
            except (ValueError, TypeError):
                cells.append("---")

        if has_or:
            ov = r.get("overall_rank", "")
            try:
                ov_i = int(float(ov))
                cells.append(f"{rank_color(ov_i)}{ov_i}")
            except (ValueError, TypeError):
                cells.append("---")

        return f"    {name} & {' & '.join(cells)} \\\\"

    lines = [
        "\\begin{table}[ht]",
        f"  \\caption{{{caption}}}",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\resizebox{\\linewidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    \\textbf{{Backbone}} & {rot_headers}{extra_heads} \\\\",
        "    \\midrule",
    ]

    for r in vfm_rows:
        lines.append(render_rank_row(r))

    if ref_rows:
        lines.append("    \\midrule")
        for r in ref_rows:
            lines.append(render_rank_row(r, is_ref=True))

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  }",
        f"  \\label{{{label}}}",
        "\\end{table}",
    ]

    write_table(lines, Path(args.output) if args.output else None)


# ── Rank-summary table ───────────────────────────────────────────────────────
# Combines multiple ranking CSVs (e.g. max / mean / median methods)
# Each file contributes one AvgR column and one OR column.

def rank_summary_table(args: argparse.Namespace) -> None:
    inputs = [Path(p) for p in args.input]
    method_labels = args.method_labels or [f"M{i+1}" for i in range(len(inputs))]
    if len(method_labels) != len(inputs):
        sys.exit("--method-labels must have the same count as --input files")

    # Load all files; key by backbone name (without __linear etc.)
    all_data: list[tuple[str, dict[str, dict]]] = []
    backbone_order: list[str] | None = None

    for path, label in zip(inputs, method_labels):
        _, rows = load_rank_csv(path)
        by_backbone: dict[str, dict] = {}
        order: list[str] = []
        for r in rows:
            key = r.get("model", r.get("encoder", ""))
            clean = key.split("__")[0].strip()
            by_backbone[clean] = r
            order.append(clean)
        all_data.append((label, by_backbone))
        if backbone_order is None:
            backbone_order = order

    assert backbone_order is not None

    # Build header: Backbone | AvgR_M1 OR_M1 | AvgR_M2 OR_M2 | ...
    col_spec = "l|" + "|".join("cc" for _ in inputs)
    header_parts = []
    for lbl in method_labels:
        header_parts.append(
            f"\\textbf{{AvgR\\textsubscript{{{lbl}}}}} & "
            f"\\textbf{{OR\\textsubscript{{{lbl}}}}}"
        )
    header = "\\textbf{Backbone} & " + " & ".join(header_parts) + " \\\\"

    label   = args.label   or "tab:avg_rank_summary"
    caption = args.caption or "Average per-dataset rank and overall rank per backbone across ranking methods."

    def render_summary_row(raw_key: str, is_ref: bool = False) -> str:
        name = display_name(raw_key, args.name_map)
        if is_ref:
            name = name + "\\textsuperscript{*}"
        cells = []
        for _, by_backbone in all_data:
            r = by_backbone.get(raw_key, {})
            avg = r.get("avg_rank", "")
            ov  = r.get("overall_rank", r.get("combined_overall_rank", ""))
            try:
                cells.append(f"{float(avg):.2f}")
            except (ValueError, TypeError):
                cells.append("---")
            try:
                ov_i = int(float(ov))
                cells.append(f"{rank_color(ov_i)}{ov_i}")
            except (ValueError, TypeError):
                cells.append("---")
        return f"    {name} & {' & '.join(cells)} \\\\"

    vfm_keys = [k for k in backbone_order if k not in RESNET_KEYS]
    ref_keys  = [k for k in backbone_order if k in RESNET_KEYS]

    lines = [
        "\\begin{table}[ht]",
        f"  \\caption{{{caption}}}",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header}",
        "    \\midrule",
    ]

    for k in vfm_keys:
        lines.append(render_summary_row(k))

    if ref_keys:
        lines.append("    \\midrule")
        for k in ref_keys:
            lines.append(render_summary_row(k, is_ref=True))

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\label{{{label}}}",
        "\\end{table}",
    ]

    write_table(lines, Path(args.output) if args.output else None)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_name_map(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in (items or []):
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── iou ──
    pi = sub.add_parser("iou", help="mIoU ± CI table")
    pi.add_argument("input", nargs=1, help="Wide miou_ci CSV (table_*_miou_ci.csv)")
    pi.add_argument("--datasets", nargs="+", help="Subset / reorder datasets")
    pi.add_argument(
        "--sizes", nargs="+", metavar="KEY=VAL",
        help="Add a Size column: e.g. dinov3_7b=7B franca_giant=g",
    )
    pi.add_argument("--caption", help="Table caption")
    pi.add_argument("--label", help="\\label{...} key")
    pi.add_argument("--output", "-o", help="Write to file instead of stdout")
    pi.add_argument(
        "--name-map", nargs="+", metavar="KEY=VAL",
        help="Override display names: e.g. dinov3_base='DINOv3 B'",
        type=str,
    )

    # ── rank ──
    pr = sub.add_parser("rank", help="Per-dataset rank table with colored cells")
    pr.add_argument("input", nargs=1, help="Rankings CSV (rankings_*_with_resnet.csv)")
    pr.add_argument("--datasets", nargs="+")
    pr.add_argument("--caption")
    pr.add_argument("--label")
    pr.add_argument("--output", "-o")
    pr.add_argument("--name-map", nargs="+", metavar="KEY=VAL", type=str)

    # ── rank-summary ──
    ps = sub.add_parser(
        "rank-summary",
        help="Multi-method rank summary table (AvgR + OR per method)",
    )
    ps.add_argument(
        "--input", nargs="+", required=True,
        help="One rankings CSV per method (e.g. max mean median)",
    )
    ps.add_argument(
        "--method-labels", nargs="+",
        help="LaTeX labels for each method column, e.g. 'max' '\\\\mu' 'med'",
    )
    ps.add_argument("--caption")
    ps.add_argument("--label")
    ps.add_argument("--output", "-o")
    ps.add_argument("--name-map", nargs="+", metavar="KEY=VAL", type=str)

    args = p.parse_args()
    args.name_map = parse_name_map(getattr(args, "name_map", None))

    if args.mode == "iou":
        iou_table(args)
    elif args.mode == "rank":
        rank_table(args)
    elif args.mode == "rank-summary":
        rank_summary_table(args)


if __name__ == "__main__":
    main()
