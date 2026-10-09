"""Aggregate all per-run test results into a single results table.

Primary data source: results/analysis/summary.csv, appended to directly by
aquabench/cli.py's BenchCLI._write_summary_csv() after every `test` run -- regardless
of which Lightning logger is configured. Real sweeps compose `--config configs/wandb.yaml`
on top of configs/linear.yaml / configs/upernet.yaml, which replaces the CSVLogger with
WandbLogger, so Lightning's local metrics.csv is never written and scanning logs/ for it
(the old approach) finds nothing. summary.csv is logger-independent and therefore the only
reliable local record of test_miou per run.

summary.csv columns: ds, backbone, head, seed, test_miou, test_miou_all (append-only --
one row per `test` invocation). If a (ds, backbone, head, seed) run appears more than once
(e.g. a resubmitted job, or after_test firing twice), the FIRST row for that key wins and
later duplicates are ignored (with a printed warning).

Output:
  results/analysis/all_results.csv
  Columns: ds, encoder, head, seed, test_miou, test_miou_all, gpu_hours

gpu_hours is best-effort, parsed from SLURM .out files (logs/slurm/) if present, else left blank.

Usage:
  python -m analysis.collect_results [--summary results/analysis/summary.csv] [--slurm-dir logs/slurm]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path


OUT_DIR = Path(os.environ.get("RESULTS_DIR", "results")) / "analysis"
OUT_PATH = OUT_DIR / "all_results.csv"
SUMMARY_PATH = OUT_DIR / "summary.csv"
FIELDNAMES = ["ds", "encoder", "head", "seed", "test_miou", "test_miou_all", "gpu_hours"]


def _slurm_gpu_hours(slurm_dir: Path) -> dict[str, float]:
    """Parse SLURM .out files for GPU-hour usage.  Returns {run_name: hours}."""
    hours: dict[str, float] = {}
    if not slurm_dir.exists():
        return hours
    # Pattern: "Total time: HH:MM:SS" in SLURM stdout
    time_re = re.compile(r"Total time[:\s]+(\d+):(\d+):(\d+)")
    run_re = re.compile(r"Running (.+?) on (.+?)[\s,]")
    for out_file in slurm_dir.glob("*.out"):
        text = out_file.read_text(errors="replace")
        run_match = run_re.search(text)
        time_match = time_re.search(text)
        if run_match and time_match:
            key = f"{run_match.group(1)}_{run_match.group(2)}"
            h, m, s = int(time_match.group(1)), int(time_match.group(2)), int(time_match.group(3))
            hours[key] = round(h + m / 60 + s / 3600, 2)
    return hours


def _read_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def collect(summary_path: Path, slurm_dir: Path) -> list[dict]:
    slurm_hours = _slurm_gpu_hours(slurm_dir)
    raw_rows = _read_summary(summary_path)

    # Dedupe reruns of the same (ds, encoder, head, seed): first row wins, later
    # duplicates are ignored.
    by_key: dict[tuple[str, str, str, str], dict] = {}
    duplicate_keys: list[tuple[str, str, str, str]] = []
    for row in raw_rows:
        key = (row.get("ds", ""), row.get("backbone", ""), row.get("head", ""), row.get("seed", ""))
        if key in by_key:
            duplicate_keys.append(key)
            continue
        by_key[key] = row

    if duplicate_keys:
        print(f"  [WARN] {len(duplicate_keys)} duplicate row(s) in {summary_path} ignored "
              f"(first occurrence kept) for: {sorted(set(duplicate_keys))[:10]}"
              f"{' ...' if len(set(duplicate_keys)) > 10 else ''}")

    rows = []
    for (ds, encoder, head, seed), row in sorted(by_key.items()):
        if row.get("test_miou", "").strip() == "":
            continue
        run_key = f"{encoder}_{ds}"
        rows.append({
            "ds": ds,
            "encoder": encoder,
            "head": head,
            "seed": seed,
            "test_miou": row.get("test_miou", ""),
            "test_miou_all": row.get("test_miou_all", ""),
            "gpu_hours": slurm_hours.get(run_key, ""),
        })
    return rows


def _warn_stray_summaries(summary_path: Path) -> None:
    """Look for other summary.csv files under the repo root that aren't the one being read --
    a sign runs were submitted from different working directories and data got scattered."""
    repo_root = Path(__file__).resolve().parent.parent
    used = summary_path.resolve()
    strays = [p for p in repo_root.rglob("summary.csv")
              if p.resolve() != used and ".venv" not in p.parts]
    if strays:
        print(f"  [WARN] found {len(strays)} other summary.csv file(s) under {repo_root} "
              f"that are NOT being read (only {used} is used) -- these runs won't be "
              f"included unless you pass --summary or merge them in:")
        for p in strays:
            print(f"    {p}")


def _diagnose_empty(summary_path: Path) -> None:
    """Print filesystem breadcrumbs to explain why 0 usable rows were found."""
    cwd = Path.cwd()
    print(f"  cwd: {cwd}")
    if not summary_path.exists():
        print(f"  {summary_path} does not exist (resolved: {summary_path.resolve()}).")
        print(f"  It's written by aquabench/cli.py after every `test` run -- make sure at "
              f"least one test job has completed, and that this script runs from the same "
              f"directory as training (so the relative results/analysis/ path resolves).")
        return
    with open(summary_path, newline="") as f:
        n_rows = sum(1 for _ in csv.DictReader(f))
    print(f"  {summary_path} exists (resolved: {summary_path.resolve()}) with {n_rows} row(s), "
          f"but none had a populated test_miou.")


def main():
    parser = argparse.ArgumentParser(description="Aggregate results/analysis/summary.csv into all_results.csv.")
    parser.add_argument("--summary", default=SUMMARY_PATH, type=Path)
    parser.add_argument("--slurm-dir", default="logs/slurm", type=Path)
    args = parser.parse_args()

    print(f"Reading {args.summary} (resolved: {args.summary.resolve()})")
    rows = collect(args.summary, args.slurm_dir)

    # Always (re)write all_results.csv, even with 0 rows -- every run cleanly overrides
    # whatever was there before rather than leaving a previous run's data lying around.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print(f"No usable rows found. Wrote empty {OUT_PATH}.")
        _diagnose_empty(args.summary)
    else:
        print(f"Collected {len(rows)} runs → {OUT_PATH}")

    _warn_stray_summaries(args.summary)


if __name__ == "__main__":
    main()
