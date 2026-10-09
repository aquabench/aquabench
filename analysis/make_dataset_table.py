"""Generate the dataset overview table (tab:datasets) with train/val/test counts.

Per-split image counts are read from `results/analysis/{ds}_image_stats.csv` (one row
per sample, with a `split` column) rather than reloaded from the unified HF datasets --
those CSVs are already checked into results/analysis and give exact split sizes without
touching DATA_ROOT.

USIS16K is excluded: dropped from the benchmark since the ECCV submission (see
.claude/CHANGES_SINCE_SUBMISSION.md), so the current scope is the 9 datasets in
analysis.utils.ds_names.ALL_DATASETS.

Everything else (full names, #classes, citation keys, licenses) is static dataset
metadata that isn't derivable from local data and is hardcoded below.

Outputs:
  results/tables/table_datasets.tex

Usage:
  python -m analysis.make_dataset_table
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

IN_DIR = Path("results/analysis")
OUT_DIR = Path("results/tables")

# (short, full name, abbrev, #classes, cite key, license name, license url or None)
# Paper row order (not the ds_names.ALL_DATASETS order).
DATASETS: list[tuple[str, str, str, int, str, str, str | None]] = [
    ("cs", "Coralscapes", "CS", 39, "Sauder.2025",
     "Apache 2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
    ("cou", "Common Obj.\\ UW", "COU", 24, "Mukherjee.28.02.2025",
     "CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
    ("l4s", "Looking for Seagrass", "L4S", 2, "Reus.2018",
     "BSD 2-Clause", "https://opensource.org/licenses/BSD-2-Clause"),
    ("df", "DeepFish", "DF", 2, "deepfish",
     "MIT", "https://opensource.org/licenses/MIT"),
    ("liaci", "LIACI", "LIACI", 11, "liaci",
     "CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
    ("sc", "SeaClear", "SC", 41, "seaclear",
     "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
    ("suim", "SUIM", "SUIM", 8, "islam2020suim",
     "MIT", "https://opensource.org/licenses/MIT"),
    ("tc", "TrashCan", "TC", 23, "Hong.16.07.2020",
     "Non Commercial\\tnote{a}", None),
    ("uiis10k", "UW Instance Seg.", "UIIS10K", 11, "Li.2025",
     "Apache 2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
]

TRASHCAN_FOOTNOTE = (
    "Free for academic use with attribution to the dataset authors and to JAMSTEC, "
    "the original data provider. Commercial use requires separate permission from "
    "JAMSTEC (see \\url{http://www.godac.jamstec.go.jp/darwin/explain/1/e\\#condition}); "
    "attribution is still required after permission is granted."
)


def count_splits(ds: str) -> tuple[int, int, int]:
    path = IN_DIR / f"{ds}_image_stats.csv"
    counts: Counter[str] = Counter()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["split"]] += 1
    return counts["train"], counts["validation"], counts["test"]


def license_cell(name: str, url: str | None) -> str:
    if url is None:
        return name
    return f"\\href{{{url}}}{{{name}}}"


def build_table() -> str:
    rows = []
    total_train = total_val = total_test = 0
    for ds, full_name, abbrev, n_classes, cite_key, license_name, license_url in DATASETS:
        n_train, n_val, n_test = count_splits(ds)
        total_train += n_train
        total_val += n_val
        total_test += n_test
        rows.append(
            f"{full_name} & {abbrev} & {n_train} & {n_val} & {n_test} & {n_classes} "
            f"& \\cite{{{cite_key}}} & {license_cell(license_name, license_url)} \\\\"
        )

    total_row = (
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Total}}}} & {total_train} & {total_val} "
        f"& {total_test} & & & \\\\"
    )

    body = "\n\n".join(rows)

    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Overview of underwater datasets combined in the benchmark suite.}}
\\label{{tab:datasets}}
\\footnotesize

\\resizebox{{\\linewidth}}{{!}}{{
\\begin{{threeparttable}}
\\begin{{tabular}}{{llS[table-format=5.0]S[table-format=4.0]S[table-format=5.0]rl l}}
\\toprule
{{Dataset}} & {{Abbrev.}} & {{Train}} & {{Val}} & {{Test}} & {{\\#Cls}} & {{Src.}} & {{License}} \\\\
\\midrule

{body}

\\addlinespace
\\midrule
{total_row}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\item[a] {TRASHCAN_FOOTNOTE}
\\end{{tablenotes}}
\\end{{threeparttable}}
}}
\\end{{table}}
"""


def main() -> None:
    table = build_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "table_datasets.tex"
    out_path.write_text(table)
    print(table)
    print(f"\nTable -> {out_path}")


if __name__ == "__main__":
    main()
