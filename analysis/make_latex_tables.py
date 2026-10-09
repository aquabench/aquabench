"""Convert confidence_intervals.csv into paper-ready LaTeX table fragments.

NAMING CONVENTION (see also analysis/compute_rankings.py's module docstring): the
frozen-backbone UPerNet protocol (group vitb_upernet_frozen, head=upernet_frozen -- encoder
locked, only the decoder trains) is the paper's main research focus, so every label/caption
this module generates uses "UPerNet" *unqualified* to mean that one. The fully fine-tuned
protocol (group vitb_upernet, head=upernet -- encoder + decoder both train) must always
carry an explicit "(fully fine-tuned)" qualifier wherever it's shown, never a bare "UPerNet".

Outputs (one .tex per benchmark table, all in --output-dir):
  table_vitb_linear.tex          ViT-B encoders, linear probe
  table_largest_linear.tex       Largest encoders, linear probe
  table_vitb_upernet_frozen.tex  ViT-B encoders, UPerNet (frozen backbone) -- the "UPerNet" default
  table_vitb_upernet.tex         ViT-B encoders, UPerNet, fully fine-tuned end-to-end

Usage:
  python -m analysis.make_latex_tables
  python -m analysis.make_latex_tables --input-dir results/analysis --output-dir results/tables
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analysis.utils.ds_names import ALL_DATASETS

DATASETS = list(ALL_DATASETS)

DS_ABBREV: dict[str, str] = {
    "cs": "CS", "cou": "COU", "df": "DF", "l4s": "L4S",
    "liaci": "LIACI", "sc": "SC", "suim": "SUIM", "tc": "TC",
    "uiis10k": "UIIS10K",
}

# model key → (display name, size label or None)
MODEL_INFO: dict[str, tuple[str, str | None]] = {
    "dinov3_base__linear":        ("DINOv3",     None),
    "eva02_base__linear":         ("EVA02",      None),
    "franca_base__linear":        ("Franca",     None),
    "radio_v3_base__linear":      ("RADIO v3",   None),
    "vjepa21_vitb__linear":       ("V-JEPA2.1",  None),
    "dinov3_7b__linear":          ("DINOv3",     "7B"),
    "eva02_large__linear":        ("EVA02",      "L"),
    "franca_giant__linear":       ("Franca",     "g"),
    "radio_v4_h__linear":         ("RADIO v4",   "H"),
    "vjepa21_vitg__linear":       ("V-JEPA2.1",  "g"),
    "dinov3_base__upernet":       ("DINOv3",     None),
    "eva02_base__upernet":        ("EVA02",      None),
    "franca_base__upernet":       ("Franca",     None),
    "radio_v3_base__upernet":     ("RADIO v3",   None),
    "vjepa21_vitb__upernet":      ("V-JEPA2.1",  None),
    "dinov3_base__upernet_frozen":    ("DINOv3",    None),
    "eva02_base__upernet_frozen":     ("EVA02",     None),
    "franca_base__upernet_frozen":    ("Franca",    None),
    "radio_v3_base__upernet_frozen":  ("RADIO v3",  None),
    "vjepa21_vitb__upernet_frozen":   ("V-JEPA2.1", None),
    "resnet50__upernet":          ("ResNet-50",  None),
}

GROUPS: dict[str, dict] = {
    "vitb_linear": {
        "filename": "table_vitb_linear.tex",
        "caption": (
            "Mean IoU with 95\\% confidence intervals per backbone and dataset "
            "under the ViT-B frozen-encoder linear probing protocol. "
            "CIs are reported as $\\pm$ half-width. "
            "Best result per dataset is \\textbf{bold}, second-best is \\underline{underlined}. "
            "ResNet-50 is shown as reference (fully fine-tuned, 5 seeds)."
        ),
        "label": "tab:miou_vitb_linear_ci",
        "header": "\\makecell{\\textbf{ViT-B}\\\\\\textbf{Backbone}}",
        "has_size": False,
    },
    "largest_linear": {
        "filename": "table_largest_linear.tex",
        "caption": (
            "Mean IoU with 95\\% confidence intervals per backbone and dataset "
            "under the largest-variant frozen-encoder linear probing protocol. "
            "CIs are reported as $\\pm$ half-width. "
            "\\textbf{Size} denotes the evaluated model scale: "
            "\\textbf{7B}~=~7 billion parameters, \\textbf{L}~=~Large, "
            "\\textbf{g}/\\textbf{G}~=~giant, \\textbf{H}~=~Huge. "
            "Best result per dataset is \\textbf{bold}, second-best is \\underline{underlined}. "
            "ResNet-50 is shown as reference (fully fine-tuned, 5 seeds)."
        ),
        "label": "tab:miou_largest_linear_ci",
        "header": "\\textbf{Backbone}",
        "has_size": True,
    },
    # NAMING CONVENTION: "vitb_upernet" (head=upernet) trains the encoder AND decoder end-to-end
    # -- it is the FULLY FINE-TUNED protocol and must always be labeled as such. "vitb_upernet_frozen"
    # (head=upernet_frozen) locks the encoder and only trains the decoder -- this is the FROZEN-backbone
    # protocol, the paper's main research focus, and everywhere else in this pipeline "UPerNet"
    # unqualified means this one. (The captions below used to have this backwards -- double check
    # the head each table is built from before touching.)
    "vitb_upernet": {
        "filename": "table_vitb_upernet.tex",
        "caption": (
            "Mean IoU with 95\\% confidence intervals per backbone and dataset "
            "under the ViT-B UPerNet protocol, \\textbf{fully fine-tuned end-to-end} "
            "(encoder and decoder both trained). "
            "CIs are reported as $\\pm$ half-width. "
            "Best result per dataset is \\textbf{bold}, second-best is \\underline{underlined}. "
            "ResNet-50 is shown as reference (fully fine-tuned end-to-end, 5 seeds)."
        ),
        "label": "tab:miou_vitb_upernet_ci",
        "header": "\\makecell{\\textbf{ViT-B}\\\\\\textbf{Backbone}}",
        "has_size": False,
    },
    "vitb_upernet_frozen": {
        "filename": "table_vitb_upernet_frozen.tex",
        "caption": (
            "Mean IoU with 95\\% confidence intervals per backbone and dataset "
            "under the ViT-B UPerNet protocol with a \\textbf{frozen encoder} "
            "(only the decoder is trained). "
            "CIs are reported as $\\pm$ half-width. "
            "Best result per dataset is \\textbf{bold}, second-best is \\underline{underlined}. "
            "ResNet-50 is shown as reference (fully fine-tuned end-to-end, 5 seeds)."
        ),
        "label": "tab:miou_vitb_upernet_frozen_ci",
        "header": "\\makecell{\\textbf{ViT-B}\\\\\\textbf{Backbone}}",
        "has_size": False,
    },
}

RESNET_MODEL = "resnet50"


def load_ci(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fmt_val(mean: float, ci_half: float) -> str:
    m = f"{mean:.3f}".lstrip("0") or "0"
    c = f"{ci_half:.3f}".lstrip("0") or "0"
    return f"{m}\\pm{c}"


def build_entries(
    all_rows: list[dict],
    group: str,
    resnet_rows: list[dict],
    has_size: bool,
) -> list[dict]:
    group_rows = [r for r in all_rows if r["group"] == group]
    models = sorted({r["model"] for r in group_rows})
    by_model_ds = {(r["model"], r["ds"]): r for r in group_rows}

    entries = []
    for model in models:
        name, size = MODEL_INFO.get(model, (model, None))
        cells: dict[str, tuple[float, float]] = {}
        for ds in DATASETS:
            cell = by_model_ds.get((model, ds))
            if cell:
                mean = float(cell["mean_miou"])
                ci_half = (float(cell["ci_high"]) - float(cell["ci_low"])) / 2
                cells[ds] = (mean, ci_half)
        entries.append({"model": model, "name": name, "size": size, "cells": cells, "is_resnet": False})

    # ResNet-50 reference row
    resnet_by_ds = {r["ds"]: r for r in resnet_rows}
    cells = {}
    for ds in DATASETS:
        cell = resnet_by_ds.get(ds)
        if cell:
            mean = float(cell["mean_miou"])
            ci_half = (float(cell["ci_high"]) - float(cell["ci_low"])) / 2
            cells[ds] = (mean, ci_half)
    entries.append({
        "model": "resnet50__upernet",
        "name": "ResNet-50",
        "size": "---" if has_size else None,
        "cells": cells,
        "is_resnet": True,
    })

    return entries


def render_table(entries: list[dict], meta: dict) -> str:
    has_size = meta["has_size"]
    n_ds = len(DATASETS)
    col_spec = ("ll" if has_size else "l") + "c" * n_ds

    # Best and second-best mean per dataset (across all rows including resnet)
    best: dict[str, float] = {}
    second: dict[str, float] = {}
    for ds in DATASETS:
        vals = sorted(
            (e["cells"][ds][0] for e in entries if ds in e["cells"]),
            reverse=True,
        )
        if vals:
            best[ds] = vals[0]
            second[ds] = vals[1] if len(vals) > 1 else float("-inf")

    def cell_tex(ds: str, mean: float, ci_half: float) -> str:
        inner = fmt_val(mean, ci_half)
        if ds in best and abs(mean - best[ds]) < 1e-8:
            return f"$\\mathbf{{{inner}}}$"
        if ds in second and abs(mean - second[ds]) < 1e-8:
            return f"\\underline{{${inner}$}}"
        return f"${inner}$"

    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append(f"  \\caption{{{meta['caption']}}}")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{2pt}")
    lines.append("  \\resizebox{\\linewidth}{!}{%")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    ds_headers = " & ".join(
        f"\\rotatebox{{90}}{{\\small {DS_ABBREV[ds]}}}" for ds in DATASETS
    )
    if has_size:
        lines.append(f"    {meta['header']} & \\textbf{{Size}} & {ds_headers} \\\\")
    else:
        lines.append(f"    {meta['header']} & {ds_headers} \\\\")
    lines.append("    \\midrule")

    for entry in entries:
        name = entry["name"]
        if entry["is_resnet"]:
            name += "\\textsuperscript{*}"

        row_cells = []
        for ds in DATASETS:
            if ds in entry["cells"]:
                mean, ci_half = entry["cells"][ds]
                row_cells.append(cell_tex(ds, mean, ci_half))
            else:
                row_cells.append("{---}")

        row = " & ".join(row_cells)
        if has_size:
            size = entry["size"] or "---"
            lines.append(f"    {name} & {size} & {row} \\\\")
        else:
            lines.append(f"    {name} & {row} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  }")
    lines.append(f"  \\label{{{meta['label']}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# Ranking summary table
# Maps model key → (family display name, size label)
RANK_MODEL_INFO: dict[str, tuple[str, str]] = {
    "dinov3_base__linear":           ("DINOv3",    "B"),
    "dinov3_base__upernet_frozen":   ("DINOv3",    "B"),
    "dinov3_7b__linear":             ("DINOv3",    "7B"),
    "franca_base__linear":           ("Franca",    "B"),
    "franca_base__upernet_frozen":   ("Franca",    "B"),
    "franca_giant__linear":          ("Franca",    "g"),
    "vjepa21_vitb__linear":          ("V-JEPA2.1", "B"),
    "vjepa21_vitb__upernet_frozen":  ("V-JEPA2.1", "B"),
    "vjepa21_vitg__linear":          ("V-JEPA2.1", "g"),
    "eva02_base__linear":            ("EVA02",     "B"),
    "eva02_base__upernet_frozen":    ("EVA02",     "B"),
    "eva02_large__linear":           ("EVA02",     "L"),
    "radio_v3_base__linear":         ("RADIO",     "B"),
    "radio_v3_base__upernet_frozen": ("RADIO",     "B"),
    "radio_v4_h__linear":            ("RADIO",     "H"),
    "resnet50__upernet":             ("ResNet-50", "---"),
}

# Family order for rows (follows vitb_linear overall rank)
FAMILY_ORDER = ["DINOv3", "Franca", "V-JEPA2.1", "EVA02", "RADIO", "ResNet-50"]

RANK_COLORS = {
    1: "rank1", 2: "rank2", 3: "rank3", 4: "rank4", 5: "rank5",
    6: "rank6", 7: "rank7", 8: "rank8", 9: "rank9", 10: "rank10",
}

RANKING_CRITERIA = [
    ("vitb_linear",         "rankings_vitb_linear_with_resnet.csv",         "ViT-B Linear"),
    ("vitb_upernet_frozen", "rankings_vitb_upernet_frozen_with_resnet.csv", "ViT-B UPerNet$^\\dagger$"),
    ("largest_linear",      "rankings_largest_linear_with_resnet.csv",      "Largest Linear"),
]


def load_rankings(path: Path) -> dict[str, dict]:
    """Return {model_key: row_dict} from a rankings CSV."""
    with path.open() as f:
        return {r["model"]: r for r in csv.DictReader(f)}


def render_ranking_summary(input_dir: Path) -> str:
    criteria_data = []
    for _group, filename, label in RANKING_CRITERIA:
        path = input_dir / filename
        if not path.exists():
            raise SystemExit(f"missing {path}")
        criteria_data.append((label, load_rankings(path)))

    # Build rows: one per family. Each cell has (avg_rank, overall_rank, size).
    # Family → criterion index → (avg_rank, overall_rank, size)
    family_rows: dict[str, list[tuple[str, str, str] | None]] = {f: [] for f in FAMILY_ORDER}

    for _label, rankings in criteria_data:
        seen: dict[str, tuple[str, str, str]] = {}
        for model_key, row in rankings.items():
            info = RANK_MODEL_INFO.get(model_key)
            if info is None:
                continue
            family, size = info
            seen[family] = (row["avg_rank"], row["overall_rank"], size)
        for family in FAMILY_ORDER:
            family_rows[family].append(seen.get(family))

    def rank_cell(overall_rank: str) -> str:
        r = int(overall_rank)
        color = RANK_COLORS.get(r, "")
        if color:
            return f"\\cellcolor{{{color}}}{r}"
        return str(r)

    n_criteria = len(RANKING_CRITERIA)
    # columns: backbone | (AvgR OR size?) × n_criteria
    # For vitb cols size is always B so we skip it; for largest we show size
    # Layout: backbone | [AvgR OR] × 2 vitb | size AvgR OR for largest
    col_spec = "l" + "".join(["rr"] * (n_criteria - 1)) + "lrr"

    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append(
        "  \\caption{Overall ranking per backbone under three evaluation protocols. "
        "\\textbf{AvgR}: average per-dataset rank (lower is better). "
        "\\textbf{OR}: overall rank (rank of AvgR across all models). "
        "$^\\dagger$UPerNet head trained on frozen backbone features. "
        "ResNet-50 is shown as reference (fully fine-tuned end-to-end).}"
    )
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Top header: criterion group spans
    vitb_lin_label  = criteria_data[0][0]
    upernet_label   = criteria_data[1][0]
    largest_label   = criteria_data[2][0]
    lines.append(
        f"    \\multirow{{2}}{{*}}{{\\textbf{{Backbone}}}} "
        f"& \\multicolumn{{2}}{{c}}{{{vitb_lin_label}}} "
        f"& \\multicolumn{{2}}{{c}}{{{upernet_label}}} "
        f"& \\multicolumn{{3}}{{c}}{{{largest_label}}} \\\\"
    )
    lines.append(
        "    & AvgR & OR & AvgR & OR & Size & AvgR & OR \\\\"
    )
    lines.append("    \\midrule")

    for family in FAMILY_ORDER:
        cells = family_rows[family]
        is_resnet = family == "ResNet-50"
        name = family + ("\\textsuperscript{*}" if is_resnet else "")

        row_parts: list[str] = []
        for i, cell in enumerate(cells):
            is_largest = (i == n_criteria - 1)
            if cell is None:
                row_parts.append("--- & ---")
                if is_largest:
                    row_parts[-1] = "--- & --- & ---"
            else:
                avg_r, or_val, size = cell
                avg_fmt = f"{float(avg_r):.2f}"
                if is_largest:
                    row_parts.append(f"{size} & {avg_fmt} & {rank_cell(or_val)}")
                else:
                    row_parts.append(f"{avg_fmt} & {rank_cell(or_val)}")

        lines.append(f"    {name} & {' & '.join(row_parts)} \\\\")

    lines.append("    \\bottomrule")
    lines.append("    \\multicolumn{8}{l}{\\textsuperscript{*}Fully fine-tuned (UPerNet), 5 seeds.}")
    lines.append("  \\end{tabular}")
    lines.append("  \\label{tab:ranking_summary}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# Family/size rank summary table -- reads compute_rankings.py's family_rank_table[.csv/_with_resnet.csv]
# (family, per-dataset ranks + avg/overall rank for ViT-B linear / ViT-B UPerNet / ViT-Largest linear)
# and renders the avg-rank + overall-rank (+ size for ViT-Largest) columns as a paper-ready table,
# same condensed layout as render_ranking_summary() above but sourced from the consolidated CSV.

FAMILY_DISPLAY: dict[str, str] = {
    "dinov3": "DINOv3", "franca": "Franca", "radio": "RADIO", "eva": "EVA02", "vjepa": "V-JEPA2.1",
}
FAMILY_ROW_ORDER = ["dinov3", "franca", "vjepa", "eva", "radio"]


def load_family_rank_table(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _resnet_rank_rows(input_dir: Path) -> dict[str, dict] | None:
    """{'vitb': row, 'vitb_upernet': row, 'vitl': row} for resnet50__upernet, each row being
    the full rankings_*_with_resnet.csv record (per-dataset ranks + avg_rank + overall_rank).
    family_rank_table has no resnet row of its own, since resnet isn't a family -- its presence
    only shifts the VFM families' ranks -- so this reads the raw rankings files directly.
    The 'vitb_upernet' block reads the *frozen*-backbone ranking pool ("UPerNet" == frozen,
    see module docstring), matching what render_family_rank_table's ViT-B UPerNet column uses."""
    sources = {"vitb": "rankings_vitb_linear_with_resnet.csv",
               "vitb_upernet": "rankings_vitb_upernet_frozen_with_resnet.csv",
               "vitl": "rankings_largest_linear_with_resnet.csv"}
    out: dict[str, dict] = {}
    for block, filename in sources.items():
        path = input_dir / filename
        if not path.exists():
            return None
        rankings = load_rankings(path)
        row = rankings.get("resnet50__upernet")
        if row is None:
            return None
        out[block] = row
    return out


def render_family_rank_table(input_dir: Path, with_resnet: bool) -> str | None:
    csv_name = "family_rank_table_with_resnet.csv" if with_resnet else "family_rank_table.csv"
    csv_path = input_dir / csv_name
    if not csv_path.exists():
        return None
    rows_by_family = {r["family"]: r for r in load_family_rank_table(csv_path)}

    def rank_cell(overall_rank: str) -> str:
        r = int(overall_rank)
        color = RANK_COLORS.get(r, "")
        return f"\\cellcolor{{{color}}}{r}" if color else str(r)

    col_spec = "l" + "rr" * 2 + "lrr"

    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    variant = "including" if with_resnet else "excluding"
    lines.append(
        "  \\caption{Overall ranking per model family, average per-dataset rank (\\textbf{AvgR}, "
        "lower is better) and overall rank (\\textbf{OR}) under three evaluation protocols "
        f"({variant} the ResNet-50 supervised reference in the ranking pool). "
        "\\textbf{Size} denotes the evaluated ViT-Largest-tier model scale.}"
    )
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")
    lines.append(
        "    \\multirow{2}{*}{\\textbf{Backbone}} "
        "& \\multicolumn{2}{c}{ViT-B Linear} "
        "& \\multicolumn{2}{c}{ViT-B UPerNet$^\\dagger$} "
        "& \\multicolumn{3}{c}{ViT-Largest Linear} \\\\"
    )
    lines.append("    & AvgR & OR & AvgR & OR & Size & AvgR & OR \\\\")
    lines.append("    \\midrule")

    for family in FAMILY_ROW_ORDER:
        row = rows_by_family.get(family)
        if row is None:
            continue
        name = FAMILY_DISPLAY.get(family, family)
        vitl_size = row["vitl_size_m"]
        size_label = f"{vitl_size}M" if vitl_size else "---"
        cells = (
            f"{float(row['vitb_avg_rank']):.2f} & {rank_cell(row['vitb_overall_rank'])} & "
            f"{float(row['vitb_upernet_avg_rank']):.2f} & {rank_cell(row['vitb_upernet_overall_rank'])} & "
            f"{size_label} & {float(row['vitl_avg_rank']):.2f} & {rank_cell(row['vitl_overall_rank'])}"
        )
        lines.append(f"    {name} & {cells} \\\\")

    resnet_found = False
    if with_resnet:
        resnet = _resnet_rank_rows(input_dir)
        if resnet is not None:
            resnet_found = True
            cells = (
                f"{float(resnet['vitb']['avg_rank']):.2f} & {rank_cell(resnet['vitb']['overall_rank'])} & "
                f"{float(resnet['vitb_upernet']['avg_rank']):.2f} & {rank_cell(resnet['vitb_upernet']['overall_rank'])} & "
                f"--- & {float(resnet['vitl']['avg_rank']):.2f} & {rank_cell(resnet['vitl']['overall_rank'])}"
            )
            lines.append(f"    ResNet-50\\textsuperscript{{*}} & {cells} \\\\")
    lines.append("    \\bottomrule")
    lines.append("    \\multicolumn{8}{l}{\\footnotesize $^\\dagger$UPerNet head trained on frozen backbone features.}")
    if resnet_found:
        lines.append("    \\multicolumn{8}{l}{\\footnotesize \\textsuperscript{*}Fully fine-tuned (UPerNet), 5 seeds.}")

    lines.append("  \\end{tabular}")
    label = "tab:family_rank_with_resnet" if with_resnet else "tab:family_rank"
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# Full family/size rank table -- same data source as render_family_rank_table(), but keeps every
# per-dataset rank column instead of collapsing to AvgR/OR only (Torben's ASCII sketch: a FAMILY
# row per encoder family, with a ViT-B block of per-dataset rank columns, etc.).

def render_family_rank_full_table(input_dir: Path, with_resnet: bool) -> str | None:
    csv_name = "family_rank_table_with_resnet.csv" if with_resnet else "family_rank_table.csv"
    csv_path = input_dir / csv_name
    if not csv_path.exists():
        return None
    rows_by_family = {r["family"]: r for r in load_family_rank_table(csv_path)}

    def rank_cell(r: str) -> str:
        ri = int(r)
        color = RANK_COLORS.get(ri, "")
        return f"\\cellcolor{{{color}}}{ri}" if color else str(ri)

    def or_cell(r: str) -> str:
        """Overall-rank cell: same rank coloring as rank_cell(), but bold so OR stands out
        from the per-dataset rank cells within each evaluation-regime block."""
        ri = int(r)
        color = RANK_COLORS.get(ri, "")
        inner = f"\\textbf{{{ri}}}"
        return f"\\cellcolor{{{color}}}{inner}" if color else inner

    def block_cells(row: dict, prefix: str) -> str:
        return " & ".join(rank_cell(row[f"{prefix}{ds}"]) for ds in DATASETS)

    n_ds = len(DATASETS)
    # thin gray vertical rule between the three evaluation-regime blocks (ViT-B Linear /
    # ViT-B UPerNet / ViT-Largest Linear), subtler than a full double rule.
    sep = "!{\\color{gray}\\vrule width 0.4pt}"
    col_spec = "l" + "c" * (n_ds + 2) + sep + "c" * (n_ds + 2) + sep + "l" + "c" * (n_ds + 2)
    ds_headers = " & ".join(f"\\rotatebox{{90}}{{\\small {DS_ABBREV[ds]}}}" for ds in DATASETS)

    # column indices (1-based, family is col 1) for the \cmidrule under each block header
    block1 = (2, 1 + (n_ds + 2))
    block2 = (block1[1] + 1, block1[1] + (n_ds + 2))
    block3 = (block2[1] + 1, block2[1] + 1 + (n_ds + 2))  # includes the Size column

    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    variant = "including" if with_resnet else "excluding"
    lines.append(
        "  \\caption{Per-dataset rank (1 = best) per model family under three evaluation "
        "protocols, with average rank (\\textbf{AvgR}) and overall rank (\\textbf{OR}, "
        "bold) "
        f"({variant} the ResNet-50 supervised reference in the ranking pool). "
        "\\textbf{Size} denotes the evaluated ViT-Largest-tier model scale.}"
    )
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{2pt}")
    lines.append("  \\resizebox{\\linewidth}{!}{%")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")
    lines.append(
        f"    \\multirow{{2}}{{*}}{{\\textbf{{Family}}}} "
        f"& \\multicolumn{{{n_ds + 2}}}{{c}}{{ViT-B Linear}} "
        f"& \\multicolumn{{{n_ds + 2}}}{{c}}{{ViT-B UPerNet$^\\dagger$}} "
        f"& \\multicolumn{{{n_ds + 3}}}{{c}}{{ViT-Largest Linear}} \\\\"
    )
    lines.append(
        f"    \\cmidrule(lr){{{block1[0]}-{block1[1]}}} "
        f"\\cmidrule(lr){{{block2[0]}-{block2[1]}}} "
        f"\\cmidrule(lr){{{block3[0]}-{block3[1]}}}"
    )
    lines.append(
        f"    & {ds_headers} & AvgR & \\textbf{{OR}} & {ds_headers} & AvgR & \\textbf{{OR}} "
        f"& \\textbf{{Size}} & {ds_headers} & AvgR & \\textbf{{OR}} \\\\"
    )
    lines.append("    \\midrule")

    for family in FAMILY_ROW_ORDER:
        row = rows_by_family.get(family)
        if row is None:
            continue
        name = FAMILY_DISPLAY.get(family, family)
        size_label = f"{row['vitl_size_m']}M" if row["vitl_size_m"] else "---"
        line = (
            f"    {name} & {block_cells(row, 'vitb_')} & {float(row['vitb_avg_rank']):.2f} "
            f"& {or_cell(row['vitb_overall_rank'])} "
            f"& {block_cells(row, 'vitb_upernet_')} & {float(row['vitb_upernet_avg_rank']):.2f} "
            f"& {or_cell(row['vitb_upernet_overall_rank'])} "
            f"& {size_label} & {block_cells(row, 'vitl_')} & {float(row['vitl_avg_rank']):.2f} "
            f"& {or_cell(row['vitl_overall_rank'])} \\\\"
        )
        lines.append(line)

    n_cols = 1 + 3 * (n_ds + 2) + 1
    resnet_found = False
    if with_resnet:
        resnet = _resnet_rank_rows(input_dir)
        if resnet is not None:
            resnet_found = True
            vitb_r, vitb_up_r, vitl_r = resnet["vitb"], resnet["vitb_upernet"], resnet["vitl"]
            vitb_cells = " & ".join(rank_cell(vitb_r[ds]) for ds in DATASETS)
            vitb_up_cells = " & ".join(rank_cell(vitb_up_r[ds]) for ds in DATASETS)
            vitl_cells = " & ".join(rank_cell(vitl_r[ds]) for ds in DATASETS)
            line = (
                f"    ResNet-50\\textsuperscript{{*}} & {vitb_cells} & {float(vitb_r['avg_rank']):.2f} "
                f"& {or_cell(vitb_r['overall_rank'])} "
                f"& {vitb_up_cells} & {float(vitb_up_r['avg_rank']):.2f} & {or_cell(vitb_up_r['overall_rank'])} "
                f"& --- & {vitl_cells} & {float(vitl_r['avg_rank']):.2f} & {or_cell(vitl_r['overall_rank'])} \\\\"
            )
            lines.append(line)
    lines.append("    \\bottomrule")
    lines.append(f"    \\multicolumn{{{n_cols}}}{{l}}{{\\footnotesize $^\\dagger$UPerNet head trained on frozen backbone features.}}")
    if resnet_found:
        lines.append(f"    \\multicolumn{{{n_cols}}}{{l}}{{\\footnotesize \\textsuperscript{{*}}Fully fine-tuned (UPerNet), 5 seeds.}}")

    lines.append("  \\end{tabular}")
    lines.append("  }")
    label = "tab:family_rank_full_with_resnet" if with_resnet else "tab:family_rank_full"
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    ci_path = args.input_dir / "confidence_intervals.csv"
    if not ci_path.exists():
        raise SystemExit(f"missing {ci_path} -- run analysis.compute_rankings first")

    all_rows = load_ci(ci_path)
    resnet_rows = [r for r in all_rows if r["group"] == RESNET_MODEL]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for group, meta in GROUPS.items():
        entries = build_entries(all_rows, group, resnet_rows, meta["has_size"])
        vfm_entries = [e for e in entries if not e["is_resnet"]]
        if not any(e["cells"] for e in vfm_entries):
            print(f"[skip] no data for group={group}")
            continue
        tex = render_table(entries, meta)
        out_path = args.output_dir / meta["filename"]
        out_path.write_text(tex + "\n")
        print(f"[ok] {out_path} ({len(entries)} rows)")

    rank_tex = render_ranking_summary(args.input_dir)
    rank_path = args.output_dir / "table_ranking_summary.tex"
    rank_path.write_text(rank_tex + "\n")
    print(f"[ok] {rank_path}")

    for with_resnet, filename in [(False, "table_family_rank.tex"), (True, "table_family_rank_with_resnet.tex")]:
        tex = render_family_rank_table(args.input_dir, with_resnet)
        if tex is None:
            print(f"[skip] {filename} -- family_rank_table{'_with_resnet' if with_resnet else ''}.csv not found "
                  f"(run analysis.compute_rankings first)")
            continue
        out_path = args.output_dir / filename
        out_path.write_text(tex + "\n")
        print(f"[ok] {out_path}")

    for with_resnet, filename in [(False, "table_family_rank_full.tex"), (True, "table_family_rank_full_with_resnet.tex")]:
        tex = render_family_rank_full_table(args.input_dir, with_resnet)
        if tex is None:
            print(f"[skip] {filename} -- family_rank_table{'_with_resnet' if with_resnet else ''}.csv not found "
                  f"(run analysis.compute_rankings first)")
            continue
        out_path = args.output_dir / filename
        out_path.write_text(tex + "\n")
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
