"""Compute per-dataset rankings, Kendall-tau agreement analyses, critical difference
(CD, Demsar 2006), and a vs-ResNet-50 reference table for the current sweep.

Sweep layout (the experimental protocol; reproduce with the sweep loops in the README):
  vitb_linear         : 5 ViT-B encoders x 9 datasets x 5 seeds, linear head
  largest_linear      : 5 largest-per-family encoders x 9 datasets x 5 seeds, linear head
  vitb_upernet_frozen : 5 ViT-B encoders x 9 datasets x 3 seeds, upernet_frozen head
                        (encoder locked, only the decoder trains)
  vitb_upernet        : 5 ViT-B encoders x 9 datasets x 3 seeds, upernet head
                        (encoder + decoder both train, i.e. fully fine-tuned end-to-end)
  resnet50            : resnet50 x 9 datasets x 5 seeds, upernet head (supervised reference)

NAMING CONVENTION: the frozen-backbone UPerNet protocol (vitb_upernet_frozen) is the main
research focus, so everywhere this pipeline generates a label/caption/figure, "UPerNet"
*unqualified* always means vitb_upernet_frozen. The fully fine-tuned variant (vitb_upernet)
must always carry an explicit "(fully fine-tuned)" qualifier in any output-facing text --
never leave it as a bare "UPerNet" label, that's what caused rankings in different tables
to silently disagree before this convention was written down.

Every group (including the two linear ones) gets both a vfm_only variant and a
with_resnet variant that adds resnet50/upernet as an extra reference model -- resnet is
always upernet, so it's compared against the linear groups' VFMs across a head-type gap,
same as it already was for vitb_upernet.

`ds` values in all_results.csv are the full HF dataset names used as the --ds CLI
arg (e.g. "coralscapes", "deepfish"), not the short abbreviations in
analysis/utils/ds_names.py -- we join on DS_HF_NAME to get 9 short labels for
readable output while matching rows by the long name.

Outputs (in $RESULTS_DIR/analysis/, or $RESULTS_DIR/analysis_miou_all/ for --metric=miou_all;
RESULTS_DIR defaults to "results"):
  run_completeness.csv
  confidence_intervals.csv
  rankings_{group}.csv                                  (plain and _with_resnet variant, every group)
  critical_difference_{group}_{with_resnet|vfm_only}.csv
  critical_difference_combined_regimes_family.csv        (family-level, pooled across
                                                            vitb_linear/vitb_upernet_frozen/
                                                            largest_linear, 27 blocks)
  kendall_between_datasets.csv                          (vfm_only and _with_resnet variant, every group)
  kendall_group_summary.csv                             (same variants)
  kendall_between_heads.csv
  kendall_between_scales.csv
  kendall_between_regimes.csv                           (vitb_linear/vitb_upernet_frozen/largest_linear,
                                                          family-level, ResNet-50 excluded)
  combined_rankings.csv
  family_rank_table.csv                                 (plain and _with_resnet variant)
  head_gain.csv
  vs_resnet50.csv
  insights.json

Usage:
  python -m analysis.compute_rankings [--results PATH] [--metric miou|miou_all]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.utils.ds_names import ALL_DATASETS, DS_HF_NAME
from analysis.utils.model_meta import FAMILIES, LARGEST_ENCODERS, PARAMS_M, RESNET_ENCODER, VITB_ENCODERS
from analysis.utils.paths import analysis_dir

OUT_DIR = analysis_dir("miou")
METRIC_COL = "test_miou"  # reassigned from --metric in main(); "test_miou" or "test_miou_all"

# short abbrev (e.g. "cs") -> long HF name (e.g. "coralscapes") used as `ds` in all_results.csv
DS_LONG = {short: DS_HF_NAME[short] for short in ALL_DATASETS}
# inverse: long HF name -> short abbrev, for readable output
DS_SHORT = {long: short for short, long in DS_LONG.items()}
DATASETS = list(ALL_DATASETS)  # canonical short-name order, 9 datasets

# expected seeds per group
SEEDS_5 = [42, 851, 634, 913, 1627]
SEEDS_3 = [42, 634, 1627]

GROUPS: dict[str, dict] = {
    "vitb_linear": {"encoders": VITB_ENCODERS, "head": "linear", "seeds": SEEDS_5},
    "largest_linear": {"encoders": LARGEST_ENCODERS, "head": "linear", "seeds": SEEDS_5},
    "vitb_upernet_frozen": {"encoders": VITB_ENCODERS, "head": "upernet_frozen", "seeds": SEEDS_3},
    "vitb_upernet": {"encoders": VITB_ENCODERS, "head": "upernet", "seeds": SEEDS_3},
}
RESNET_GROUP = {"encoders": [RESNET_ENCODER], "head": "upernet", "seeds": SEEDS_5}

# Nemenyi critical values q_alpha at alpha=0.05 (Demsar 2006, Table 5); index = k-2
_NEMENYI_Q = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]


def _nemenyi_q(k: int) -> float:
    idx = k - 2
    if idx < 0:
        return 0.0
    if idx < len(_NEMENYI_Q):
        return _NEMENYI_Q[idx]
    return _NEMENYI_Q[-1]


def _cd_value(k: int, n: int) -> float:
    return _nemenyi_q(k) * math.sqrt(k * (k + 1) / (6 * n))


def _load_results(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _model_key(encoder: str, head: str) -> str:
    return f"{encoder}__{head}"


def _seed_matrix(rows: list[dict], encoders: list[str], head: str, seeds: list[int]) -> dict:
    """Return {(encoder, ds_short, seed): miou} for the requested (encoders, head, seeds)."""
    out: dict[tuple[str, str, int], float] = {}
    for row in rows:
        if row["head"] != head or row["encoder"] not in encoders:
            continue
        ds_long = row["ds"]
        ds_short = DS_SHORT.get(ds_long)
        if ds_short is None:
            continue
        val = row.get(METRIC_COL, "")
        if val == "":
            continue
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError):
            continue
        if seed not in seeds:
            continue
        out[(row["encoder"], ds_short, seed)] = float(val)
    return out


def _stats_over_seeds(seed_matrix: dict, encoders: list[str], seeds: list[int]) -> dict:
    """{(encoder, ds): {mean, std, ci_low, ci_high, n_seeds}}."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (enc, ds, seed), val in seed_matrix.items():
        buckets[(enc, ds)].append(val)

    out: dict[tuple[str, str], dict] = {}
    for (enc, ds), vals in buckets.items():
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        n = len(vals)
        if n >= 2:
            lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=stats.sem(vals))
        else:
            lo, hi = mean, mean
        out[(enc, ds)] = {"mean": mean, "std": std, "ci_low": lo, "ci_high": hi, "n_seeds": n}
    return out


def _rank_from_scores(scores: dict[str, float]) -> dict[str, int]:
    """{model: rank} for a single dataset's scores, higher score = rank 1. Strict ordering, no
    ties: CI-overlap ("not significantly different") isn't transitive (A~B and B~C doesn't
    imply A~C), so collapsing it into a shared scalar rank can assert false equivalences --
    e.g. UPerNet's 3-seed CIs are wide enough that long overlap chains formed and dragged
    unrelated models to the same rank. The critical-difference diagrams (_cliques() in
    make_ranking_figures.py) are the correct place for that story: they draw (possibly
    overlapping) clique bars instead of forcing a single rank number, which handles
    non-transitivity properly. Rank columns here stay a plain "who won on average" ordering;
    check confidence_intervals.csv's raw CI bounds for any specific pairwise significance
    question."""
    valid = {m: s for m, s in scores.items() if s is not None}
    ranked = sorted(valid, key=lambda m: -valid[m])
    return {m: i + 1 for i, m in enumerate(ranked)}


def _rank_matrix(model_ds_mean: dict[tuple[str, str], float], models: list[str]) -> dict[str, dict[str, int]]:
    """{model: {ds: rank}} from mean-mIoU-per-(model,ds), ranked within each dataset."""
    ranks: dict[str, dict[str, int]] = {m: {} for m in models}
    for ds in DATASETS:
        scores = {m: model_ds_mean.get((m, ds)) for m in models}
        for m, r in _rank_from_scores(scores).items():
            ranks[m][ds] = r
    return ranks


def _safe_avg_rank(rank_dict: dict[str, int], keys: list[str]) -> float | None:
    """Mean of rank_dict[ds] over `keys` present in rank_dict; None if none present."""
    vals = [rank_dict[k] for k in keys if k in rank_dict]
    return float(np.mean(vals)) if vals else None


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# 1.0 Run completeness
# ---------------------------------------------------------------------------

def compute_completeness(rows: list[dict]) -> list[dict]:
    found = set()
    for row in rows:
        ds_short = DS_SHORT.get(row["ds"])
        if ds_short is None or row.get(METRIC_COL, "") == "":
            continue
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError):
            continue
        found.add((row["encoder"], row["head"], ds_short, seed))

    expected_groups = {**GROUPS, "resnet50": RESNET_GROUP}
    out_rows = []
    n_missing = 0
    for group_name, spec in expected_groups.items():
        for enc in spec["encoders"]:
            for ds in DATASETS:
                for seed in spec["seeds"]:
                    ok = (enc, spec["head"], ds, seed) in found
                    if not ok:
                        n_missing += 1
                    out_rows.append({
                        "group": group_name, "encoder": enc, "head": spec["head"],
                        "ds": ds, "seed": seed, "found": ok,
                    })
    if n_missing:
        print(f"  [WARN] {n_missing} / {len(out_rows)} expected runs are MISSING -- see run_completeness.csv")
    else:
        print(f"  All {len(out_rows)} expected runs present.")
    return out_rows


# ---------------------------------------------------------------------------
# 1.1 Confidence intervals (per group, per model, per dataset)
# ---------------------------------------------------------------------------

def compute_confidence_intervals(rows: list[dict]) -> dict[str, dict]:
    """Returns {group_name: {(encoder, ds): stats_dict}} including the resnet50 group."""
    all_groups = {**GROUPS, "resnet50": RESNET_GROUP}
    per_group_stats: dict[str, dict] = {}
    ci_rows = []
    for group_name, spec in all_groups.items():
        sm = _seed_matrix(rows, spec["encoders"], spec["head"], spec["seeds"])
        st = _stats_over_seeds(sm, spec["encoders"], spec["seeds"])
        per_group_stats[group_name] = st
        for (enc, ds), d in st.items():
            ci_rows.append({
                "group": group_name, "model": _model_key(enc, spec["head"]), "ds": ds,
                "mean_miou": round(d["mean"], 6), "std_miou": round(d["std"], 6),
                "ci_low": round(d["ci_low"], 6), "ci_high": round(d["ci_high"], 6),
                "n_seeds": d["n_seeds"],
            })
    ci_rows.sort(key=lambda x: (x["group"], x["model"], x["ds"]))
    _write_csv(OUT_DIR / "confidence_intervals.csv", ci_rows,
               ["group", "model", "ds", "mean_miou", "std_miou", "ci_low", "ci_high", "n_seeds"])
    return per_group_stats


# ---------------------------------------------------------------------------
# 1.2 Rankings with rank uncertainty (per group, w/ and w/o resnet50 for upernet)
# ---------------------------------------------------------------------------

def compute_rankings(rows: list[dict], group_stats: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Returns {group_variant: {model: avg_rank}} for downstream combined-rank use.

    Every group gets a plain variant (its own encoders, its own head) and a
    "_with_resnet" variant that adds the resnet50/upernet supervised baseline as an
    extra reference model -- resnet is always upernet, regardless of the group's head,
    so encoder/head pairs are tracked individually rather than assuming one head per group.
    """
    avg_ranks: dict[str, dict[str, float]] = {}
    RESNET_PAIR = (RESNET_ENCODER, "upernet")

    for group_name, spec in GROUPS.items():
        base_pairs = [(e, spec["head"]) for e in spec["encoders"]]
        variants = [
            (group_name, base_pairs),
            (f"{group_name}_with_resnet", base_pairs + [RESNET_PAIR]),
        ]

        for variant_name, encoder_head_pairs in variants:
            models = [_model_key(e, h) for e, h in encoder_head_pairs]
            mean_scores = {}
            for enc, h in encoder_head_pairs:
                st = group_stats["resnet50" if enc == RESNET_ENCODER else group_name]
                for ds in DATASETS:
                    d = st.get((enc, ds))
                    if d is not None:
                        mean_scores[(_model_key(enc, h), ds)] = d["mean"]

            rank_mat = _rank_matrix(mean_scores, models)

            # per-seed ranking, to get rank uncertainty across seeds
            per_seed_ranks: dict[str, list[float]] = {m: [] for m in models}
            for seed in spec["seeds"]:
                seed_scores_by_ds = {}
                for enc, h in encoder_head_pairs:
                    sm = _seed_matrix(rows, [enc], h, [seed])
                    for (e, ds, s), val in sm.items():
                        seed_scores_by_ds.setdefault(ds, {})[_model_key(e, h)] = val
                seed_ranks: dict[str, list[int]] = {m: [] for m in models}
                for ds in DATASETS:
                    scores = seed_scores_by_ds.get(ds, {})
                    if len(scores) < len(models):
                        continue
                    for m, r in _rank_from_scores(scores).items():
                        seed_ranks[m].append(r)
                for m in models:
                    if seed_ranks[m]:
                        per_seed_ranks[m].append(float(np.mean(seed_ranks[m])))

            ranking_rows = []
            for m in models:
                ds_ranks = {ds: rank_mat[m].get(ds, "") for ds in DATASETS}
                covered = [v for v in ds_ranks.values() if v != ""]
                avg_rank = round(float(np.mean(covered)), 4) if covered else None
                seed_vals = per_seed_ranks[m]
                avg_rank_std = round(float(np.std(seed_vals, ddof=1)), 4) if len(seed_vals) >= 2 else 0.0
                ranking_rows.append({
                    "model": m, **ds_ranks,
                    "avg_rank": avg_rank if avg_rank is not None else "",
                    "avg_rank_std": avg_rank_std,
                })
            ranking_rows.sort(key=lambda x: x["avg_rank"] if x["avg_rank"] != "" else 9999)
            for i, r in enumerate(ranking_rows, start=1):
                r["overall_rank"] = i

            _write_csv(OUT_DIR / f"rankings_{variant_name}.csv", ranking_rows,
                       ["model"] + DATASETS + ["avg_rank", "avg_rank_std", "overall_rank"])

            avg_ranks[variant_name] = {r["model"]: r["avg_rank"] for r in ranking_rows if r["avg_rank"] != ""}

    return avg_ranks


# ---------------------------------------------------------------------------
# 1.3 Kendall tau between datasets (per group), + Kendall's W summary
# ---------------------------------------------------------------------------

def _kendalls_w(rank_mat: np.ndarray) -> float:
    """rank_mat: (n_datasets, k_models) of ranks. Kendall's W coefficient of concordance."""
    n, k = rank_mat.shape
    if n < 2 or k < 2:
        return float("nan")
    rank_sums = rank_mat.sum(axis=0)
    s = np.sum((rank_sums - rank_sums.mean()) ** 2)
    w = 12 * s / (n ** 2 * (k ** 3 - k))
    return float(w)


def _kendall_between_datasets_for_variant(variant_name: str, models: list[str],
                                           mean_scores: dict, kd_rows: list, summary_rows: list) -> None:
    rank_mat = _rank_matrix(mean_scores, models)

    covered = [ds for ds in DATASETS if all(ds in rank_mat[m] for m in models)]
    taus = []
    variant_rows = []
    for ds1, ds2 in combinations(covered, 2):
        v1 = [rank_mat[m][ds1] for m in models]
        v2 = [rank_mat[m][ds2] for m in models]
        tau, p = stats.kendalltau(v1, v2)
        taus.append(tau)
        variant_rows.append({"group": variant_name, "ds1": ds1, "ds2": ds2,
                              "tau": round(float(tau), 4), "p_value": round(float(p), 6)})
    kd_rows.extend(variant_rows)

    w = float("nan")
    if covered:
        rank_arr = np.array([[rank_mat[m][ds] for m in models] for ds in covered])
        w = _kendalls_w(rank_arr)

    n_sig = sum(1 for r in variant_rows if r["p_value"] < 0.05)
    n_pairs = len(variant_rows)
    summary_rows.append({
        "group": variant_name,
        "mean_tau": round(float(np.mean(taus)), 4) if taus else "",
        "min_tau": round(float(np.min(taus)), 4) if taus else "",
        "frac_significant_p05": round(n_sig / n_pairs, 4) if n_pairs else "",
        "kendalls_w": round(w, 4) if not math.isnan(w) else "",
        "n_datasets": len(covered),
    })


def compute_kendall_between_datasets(group_stats: dict[str, dict]) -> None:
    """Every group gets a vfm_only variant plus a with_resnet variant that adds the
    resnet50/upernet supervised baseline as an extra reference model, matching the
    vfm_only/with_resnet split already used for rankings and critical difference."""
    kd_rows = []
    summary_rows = []
    resnet_model = _model_key(RESNET_ENCODER, "upernet")
    resnet_st = group_stats["resnet50"]

    for group_name, spec in GROUPS.items():
        models = [_model_key(e, spec["head"]) for e in spec["encoders"]]
        st = group_stats[group_name]
        mean_scores = {(_model_key(e, spec["head"]), ds): st[(e, ds)]["mean"]
                        for e in spec["encoders"] for ds in DATASETS if (e, ds) in st}
        _kendall_between_datasets_for_variant(group_name, models, mean_scores, kd_rows, summary_rows)

        models_wr = models + [resnet_model]
        mean_scores_wr = dict(mean_scores)
        for ds in DATASETS:
            d = resnet_st.get((RESNET_ENCODER, ds))
            if d is not None:
                mean_scores_wr[(resnet_model, ds)] = d["mean"]
        _kendall_between_datasets_for_variant(f"{group_name}_with_resnet", models_wr, mean_scores_wr,
                                               kd_rows, summary_rows)

    _write_csv(OUT_DIR / "kendall_between_datasets.csv", kd_rows,
               ["group", "ds1", "ds2", "tau", "p_value"])
    _write_csv(OUT_DIR / "kendall_group_summary.csv", summary_rows,
               ["group", "mean_tau", "min_tau", "frac_significant_p05", "kendalls_w", "n_datasets"])


# ---------------------------------------------------------------------------
# 1.4 Kendall tau between heads (linear vs upernet, ViT-B tier, VFMs only)
# ---------------------------------------------------------------------------

def compute_kendall_between_heads(group_stats: dict[str, dict]) -> None:
    lin_st = group_stats["vitb_linear"]
    up_st = group_stats.get("vitb_upernet_frozen", {})

    lin_scores = {(e, ds): lin_st[(e, ds)]["mean"] for e in VITB_ENCODERS for ds in DATASETS if (e, ds) in lin_st}
    up_scores = {(e, ds): up_st[(e, ds)]["mean"] for e in VITB_ENCODERS for ds in DATASETS if (e, ds) in up_st}

    lin_ranks = _rank_matrix({(k[0], k[1]): v for k, v in lin_scores.items()}, VITB_ENCODERS)
    up_ranks = _rank_matrix({(k[0], k[1]): v for k, v in up_scores.items()}, VITB_ENCODERS)

    rows = []
    per_ds_tau = []
    for ds in DATASETS:
        ds_encoders = [e for e in VITB_ENCODERS if ds in lin_ranks[e] and ds in up_ranks[e]]
        if len(ds_encoders) < 2:
            continue
        v1 = [lin_ranks[e][ds] for e in ds_encoders]
        v2 = [up_ranks[e][ds] for e in ds_encoders]
        tau, p = stats.kendalltau(v1, v2)
        per_ds_tau.append(tau)
        rows.append({"ds": ds, "tau": round(float(tau), 4), "p_value": round(float(p), 6),
                      "n_models": len(ds_encoders)})

    # aggregate row using avg-rank vectors, restricted to encoders covered in both heads
    covered_encoders = [e for e in VITB_ENCODERS
                         if _safe_avg_rank(lin_ranks[e], DATASETS) is not None
                         and _safe_avg_rank(up_ranks[e], DATASETS) is not None]
    dropped = set(VITB_ENCODERS) - set(covered_encoders)
    if dropped:
        print(f"  [WARN] kendall_between_heads: excluding encoders with no coverage in both heads: {sorted(dropped)}")
    if len(covered_encoders) >= 2:
        lin_avg = [_safe_avg_rank(lin_ranks[e], DATASETS) for e in covered_encoders]
        up_avg = [_safe_avg_rank(up_ranks[e], DATASETS) for e in covered_encoders]
        agg_tau, agg_p = stats.kendalltau(lin_avg, up_avg)
        rows.append({"ds": "ALL_AGGREGATE", "tau": round(float(agg_tau), 4), "p_value": round(float(agg_p), 6),
                      "n_models": len(covered_encoders)})
    else:
        rows.append({"ds": "ALL_AGGREGATE", "tau": "", "p_value": "", "n_models": len(covered_encoders)})

    _write_csv(OUT_DIR / "kendall_between_heads.csv", rows, ["ds", "tau", "p_value", "n_models"])


# ---------------------------------------------------------------------------
# 1.5 Combined ranking across stages (linear + upernet, ViT-B tier)
# ---------------------------------------------------------------------------

def compute_combined_rankings(avg_ranks: dict[str, dict[str, float]]) -> None:
    lin = avg_ranks.get("vitb_linear", {})
    up_frozen = avg_ranks.get("vitb_upernet_frozen", {})
    up = avg_ranks.get("vitb_upernet", {})

    rows = []
    for enc in VITB_ENCODERS:
        lin_key = _model_key(enc, "linear")
        up_frozen_key = _model_key(enc, "upernet_frozen")
        up_key = _model_key(enc, "upernet")
        r_lin = lin.get(lin_key)
        r_up_frozen = up_frozen.get(up_frozen_key)
        r_up = up.get(up_key)
        if r_lin is None or r_up_frozen is None:
            continue
        # Combined rank uses the two frozen-encoder paradigms (linear + frozen upernet)
        combined = round((r_lin + r_up_frozen) / 2, 4)
        rows.append({
            "encoder": enc,
            "avg_rank_linear": r_lin,
            "avg_rank_upernet_frozen": r_up_frozen,
            "avg_rank_upernet": r_up if r_up is not None else "",
            "combined_avg_rank": combined,
        })
    rows.sort(key=lambda x: x["combined_avg_rank"])
    for i, r in enumerate(rows, start=1):
        r["combined_overall_rank"] = i

    _write_csv(OUT_DIR / "combined_rankings.csv", rows,
               ["encoder", "avg_rank_linear", "avg_rank_upernet_frozen", "avg_rank_upernet",
                "combined_avg_rank", "combined_overall_rank"])


# ---------------------------------------------------------------------------
# 1.5b Family/size rank table: one row per family, ViT-B / ViT-B+UPerNet / ViT-Largest
# blocks side by side (each with its per-dataset ranks + avg/overall rank), reading the
# rankings_{group}[_with_resnet].csv files this module already wrote.
#
# NAMING CONVENTION (paper-wide, see also make_latex_tables.py): the two ViT-B UPerNet
# sweeps are group "vitb_upernet_frozen" (head=upernet_frozen -- encoder locked, only the
# decoder trains) and group "vitb_upernet" (head=upernet -- encoder + decoder both train,
# i.e. fully fine-tuned end-to-end). The frozen-backbone protocol is the main research
# focus, so in the paper and in every generated label/caption "UPerNet" *unqualified*
# always means the FROZEN variant; the fully fine-tuned variant must always carry an
# explicit "(fully fine-tuned)" qualifier wherever it's shown. This table's "ViT-B UPerNet"
# block therefore sources group "vitb_upernet_frozen", not "vitb_upernet".
# ---------------------------------------------------------------------------

def compute_family_rank_table(vfm_only: bool) -> None:
    suffix = "" if vfm_only else "_with_resnet"
    out_name = "family_rank_table.csv" if vfm_only else "family_rank_table_with_resnet.csv"

    def _load_rankings(group: str) -> dict[str, dict]:
        path = OUT_DIR / f"rankings_{group}{suffix}.csv"
        if not path.exists():
            return {}
        with open(path, newline="") as f:
            return {row["model"]: row for row in csv.DictReader(f)}

    vitb_rank = _load_rankings("vitb_linear")
    vitb_up_rank = _load_rankings("vitb_upernet_frozen")  # "UPerNet" == frozen backbone, see note above
    vitl_rank = _load_rankings("largest_linear")

    rows = []
    for family, (vitb_enc, vitl_enc) in FAMILIES.items():
        row: dict = {"family": family}

        vitb_row = vitb_rank.get(_model_key(vitb_enc, "linear"), {})
        row["vitb_encoder"] = vitb_enc
        for ds in DATASETS:
            row[f"vitb_{ds}"] = vitb_row.get(ds, "")
        row["vitb_avg_rank"] = vitb_row.get("avg_rank", "")
        row["vitb_overall_rank"] = vitb_row.get("overall_rank", "")

        vitb_up_row = vitb_up_rank.get(_model_key(vitb_enc, "upernet_frozen"), {})
        row["vitb_upernet_encoder"] = vitb_enc
        for ds in DATASETS:
            row[f"vitb_upernet_{ds}"] = vitb_up_row.get(ds, "")
        row["vitb_upernet_avg_rank"] = vitb_up_row.get("avg_rank", "")
        row["vitb_upernet_overall_rank"] = vitb_up_row.get("overall_rank", "")

        vitl_row = vitl_rank.get(_model_key(vitl_enc, "linear"), {})
        row["vitl_encoder"] = vitl_enc
        row["vitl_size_m"] = PARAMS_M.get(vitl_enc, "")
        for ds in DATASETS:
            row[f"vitl_{ds}"] = vitl_row.get(ds, "")
        row["vitl_avg_rank"] = vitl_row.get("avg_rank", "")
        row["vitl_overall_rank"] = vitl_row.get("overall_rank", "")

        rows.append(row)

    fieldnames = (
        ["family", "vitb_encoder"] + [f"vitb_{ds}" for ds in DATASETS] + ["vitb_avg_rank", "vitb_overall_rank"]
        + ["vitb_upernet_encoder"] + [f"vitb_upernet_{ds}" for ds in DATASETS]
        + ["vitb_upernet_avg_rank", "vitb_upernet_overall_rank"]
        + ["vitl_encoder", "vitl_size_m"] + [f"vitl_{ds}" for ds in DATASETS]
        + ["vitl_avg_rank", "vitl_overall_rank"]
    )
    _write_csv(OUT_DIR / out_name, rows, fieldnames)


# ---------------------------------------------------------------------------
# 1.6 Kendall tau between scales (ViT-B vs largest, linear head, family-matched)
# ---------------------------------------------------------------------------

def compute_kendall_between_scales(group_stats: dict[str, dict]) -> None:
    vitb_st = group_stats["vitb_linear"]
    largest_st = group_stats["largest_linear"]
    family_names = list(FAMILIES.keys())

    vitb_scores = {(fam, ds): vitb_st[(FAMILIES[fam][0], ds)]["mean"]
                   for fam in family_names for ds in DATASETS if (FAMILIES[fam][0], ds) in vitb_st}
    largest_scores = {(fam, ds): largest_st[(FAMILIES[fam][1], ds)]["mean"]
                       for fam in family_names for ds in DATASETS if (FAMILIES[fam][1], ds) in largest_st}

    vitb_ranks = _rank_matrix(vitb_scores, family_names)
    largest_ranks = _rank_matrix(largest_scores, family_names)

    rows = []
    for ds in DATASETS:
        ds_families = [f for f in family_names if ds in vitb_ranks[f] and ds in largest_ranks[f]]
        if len(ds_families) < 2:
            continue
        v1 = [vitb_ranks[f][ds] for f in ds_families]
        v2 = [largest_ranks[f][ds] for f in ds_families]
        tau, p = stats.kendalltau(v1, v2)
        rows.append({"ds": ds, "tau": round(float(tau), 4), "p_value": round(float(p), 6),
                      "n_families": len(ds_families)})

    covered_families = [f for f in family_names
                         if _safe_avg_rank(vitb_ranks[f], DATASETS) is not None
                         and _safe_avg_rank(largest_ranks[f], DATASETS) is not None]
    dropped = set(family_names) - set(covered_families)
    if dropped:
        print(f"  [WARN] kendall_between_scales: excluding families with no coverage in both tiers: {sorted(dropped)}")
    if len(covered_families) >= 2:
        vitb_avg = [_safe_avg_rank(vitb_ranks[f], DATASETS) for f in covered_families]
        largest_avg = [_safe_avg_rank(largest_ranks[f], DATASETS) for f in covered_families]
        agg_tau, agg_p = stats.kendalltau(vitb_avg, largest_avg)
        rows.append({"ds": "ALL_AGGREGATE", "tau": round(float(agg_tau), 4), "p_value": round(float(agg_p), 6),
                      "n_families": len(covered_families)})
    else:
        rows.append({"ds": "ALL_AGGREGATE", "tau": "", "p_value": "", "n_families": len(covered_families)})

    _write_csv(OUT_DIR / "kendall_between_scales.csv", rows, ["ds", "tau", "p_value", "n_families"])


# ---------------------------------------------------------------------------
# 1.6b Kendall tau between evaluation regimes (ViT-B linear / ViT-B UPerNet frozen /
# ViT-Largest linear), family-level (5 families), ResNet-50 excluded.
# ---------------------------------------------------------------------------

# regime name -> (group_stats key, tier index into FAMILIES[fam] tuple: 0=vitb, 1=largest)
REGIMES: dict[str, tuple[str, int]] = {
    "vitb_linear": ("vitb_linear", 0),
    "vitb_upernet_frozen": ("vitb_upernet_frozen", 0),
    "largest_linear": ("largest_linear", 1),
}


def compute_kendall_between_regimes(group_stats: dict[str, dict]) -> None:
    """Does the family ranking hold up across evaluation protocols, not just within one?
    VFM families only (no ResNet-50 -- it isn't a family, and the comparison is across
    protocols the same 5 families are evaluated under)."""
    family_names = list(FAMILIES.keys())

    regime_ranks: dict[str, dict[str, dict[str, int]]] = {}
    for regime, (group_key, tier) in REGIMES.items():
        st = group_stats.get(group_key, {})
        scores = {(fam, ds): st[(FAMILIES[fam][tier], ds)]["mean"]
                  for fam in family_names for ds in DATASETS if (FAMILIES[fam][tier], ds) in st}
        regime_ranks[regime] = _rank_matrix(scores, family_names)

    rows = []
    for r1, r2 in combinations(REGIMES.keys(), 2):
        ranks1, ranks2 = regime_ranks[r1], regime_ranks[r2]
        for ds in DATASETS:
            ds_families = [f for f in family_names if ds in ranks1[f] and ds in ranks2[f]]
            if len(ds_families) < 2:
                continue
            v1 = [ranks1[f][ds] for f in ds_families]
            v2 = [ranks2[f][ds] for f in ds_families]
            tau, p = stats.kendalltau(v1, v2)
            rows.append({"regime1": r1, "regime2": r2, "ds": ds, "tau": round(float(tau), 4),
                         "p_value": round(float(p), 6), "n_families": len(ds_families)})

        covered = [f for f in family_names
                   if _safe_avg_rank(ranks1[f], DATASETS) is not None
                   and _safe_avg_rank(ranks2[f], DATASETS) is not None]
        if len(covered) >= 2:
            avg1 = [_safe_avg_rank(ranks1[f], DATASETS) for f in covered]
            avg2 = [_safe_avg_rank(ranks2[f], DATASETS) for f in covered]
            agg_tau, agg_p = stats.kendalltau(avg1, avg2)
            rows.append({"regime1": r1, "regime2": r2, "ds": "ALL_AGGREGATE",
                         "tau": round(float(agg_tau), 4), "p_value": round(float(agg_p), 6),
                         "n_families": len(covered)})
        else:
            rows.append({"regime1": r1, "regime2": r2, "ds": "ALL_AGGREGATE",
                         "tau": "", "p_value": "", "n_families": len(covered)})

    _write_csv(OUT_DIR / "kendall_between_regimes.csv", rows,
               ["regime1", "regime2", "ds", "tau", "p_value", "n_families"])


# ---------------------------------------------------------------------------
# 1.6c Combined critical difference across evaluation regimes (family-level)
# ---------------------------------------------------------------------------

def compute_combined_critical_difference_family(group_stats: dict[str, dict]) -> None:
    """One CD diagram pooling all 3 evaluation regimes (vitb_linear, vitb_upernet_frozen,
    largest_linear) as blocks alongside the 9 datasets (27 blocks total), ranking families
    (not individual encoder/head models) by how well they do across evaluation protocols.

    CAVEAT (always caption this): largest_linear scores each family via its LARGEST encoder
    while the other two regimes use the ViT-B encoder -- this is a "best-in-class across
    protocols" ranking, not a fixed-scale, apples-to-apples comparison. Family rank here
    can be pulled up by a strong largest-tier result even if the family's ViT-B tier is
    mediocre, or vice versa. See kendall_between_regimes.csv for whether ranks actually
    agree across regimes before treating this as one clean signal.
    """
    family_names = list(FAMILIES.keys())

    mean_scores: dict[tuple[str, str], float] = {}
    for regime, (group_key, tier) in REGIMES.items():
        st = group_stats.get(group_key, {})
        for fam in family_names:
            enc = FAMILIES[fam][tier]
            for ds in DATASETS:
                d = st.get((enc, ds))
                if d is not None:
                    mean_scores[(fam, f"{ds}__{regime}")] = d["mean"]

    blocks = [f"{ds}__{regime}" for ds in DATASETS for regime in REGIMES]
    rank_mat: dict[str, dict[str, int]] = {f: {} for f in family_names}
    for block in blocks:
        scores = {f: mean_scores.get((f, block)) for f in family_names}
        for f, r in _rank_from_scores(scores).items():
            rank_mat[f][block] = r

    covered = [b for b in blocks if all(b in rank_mat[f] for f in family_names)]
    n, k = len(covered), len(family_names)
    if n == 0 or k < 2:
        print(f"  [CD/combined_regimes_family] Not enough data (k={k}, n={n}), skipping.")
        return

    rank_arr = np.array([[rank_mat[f][b] for b in covered] for f in family_names])
    mean_ranks = rank_arr.mean(axis=1)
    friedman_stat, friedman_p = stats.friedmanchisquare(*[rank_arr[i] for i in range(k)])
    cd = _cd_value(k, n)

    rows = [{"model": f, "mean_rank": round(float(mr), 4), "cd_value": round(cd, 4),
             "friedman_p": round(float(friedman_p), 6), "n_datasets": n}
            for f, mr in zip(family_names, mean_ranks)]
    _write_csv(OUT_DIR / "critical_difference_combined_regimes_family.csv", rows,
               ["model", "mean_rank", "cd_value", "friedman_p", "n_datasets"])
    print(f"    CD={cd:.3f}, Friedman p={friedman_p:.4f}, blocks={n} (9 ds x {len(REGIMES)} regimes)")


# ---------------------------------------------------------------------------
# 1.7 Head gain (linear -> upernet, ViT-B tier)
# ---------------------------------------------------------------------------

def compute_head_gain(group_stats: dict[str, dict]) -> None:
    lin_st = group_stats["vitb_linear"]
    # Compute gain for both frozen and fully fine-tuned UPerNet; callers can filter by `head`.
    upernet_groups = [
        ("upernet_frozen", group_stats.get("vitb_upernet_frozen", {})),
        ("upernet",        group_stats.get("vitb_upernet", {})),
    ]

    rows = []
    for head_name, up_st in upernet_groups:
        for enc in VITB_ENCODERS:
            for ds in DATASETS:
                lin_d = lin_st.get((enc, ds))
                up_d = up_st.get((enc, ds))
                if lin_d is None or up_d is None:
                    continue
                gain = up_d["mean"] - lin_d["mean"]
                gain_std = math.sqrt(
                    (lin_d["std"] ** 2 / max(lin_d["n_seeds"], 1)) + (up_d["std"] ** 2 / max(up_d["n_seeds"], 1))
                )
                rows.append({
                    "encoder": enc, "ds": ds, "head": head_name,
                    "linear_mean": round(lin_d["mean"], 6), "linear_std": round(lin_d["std"], 6),
                    "upernet_mean": round(up_d["mean"], 6), "upernet_std": round(up_d["std"], 6),
                    "gain_pp": round(gain * 100, 4), "gain_pp_std": round(gain_std * 100, 4),
                })
    rows.sort(key=lambda x: (x["head"], x["encoder"], x["ds"]))
    _write_csv(OUT_DIR / "head_gain.csv", rows,
               ["encoder", "ds", "head", "linear_mean", "linear_std", "upernet_mean", "upernet_std",
                "gain_pp", "gain_pp_std"])


# ---------------------------------------------------------------------------
# 1.8 Critical difference (Friedman + Nemenyi) and vs-ResNet-50 reference
# ---------------------------------------------------------------------------

def _write_cd(models: list[str], rank_mat: dict[str, dict[str, int]], out_path: Path, label: str) -> None:
    covered = [ds for ds in DATASETS if all(ds in rank_mat[m] for m in models)]
    n = len(covered)
    k = len(models)
    if n == 0 or k < 2:
        print(f"  [CD/{label}] Not enough data (k={k}, n={n}), skipping.")
        return

    rank_arr = np.array([[rank_mat[m][ds] for ds in covered] for m in models])
    mean_ranks = rank_arr.mean(axis=1)
    friedman_stat, friedman_p = stats.friedmanchisquare(*[rank_arr[i] for i in range(k)])
    cd = _cd_value(k, n)

    rows = [{"model": m, "mean_rank": round(float(mr), 4), "cd_value": round(cd, 4),
             "friedman_p": round(float(friedman_p), 6), "n_datasets": n}
            for m, mr in zip(models, mean_ranks)]
    _write_csv(out_path, rows, ["model", "mean_rank", "cd_value", "friedman_p", "n_datasets"])
    print(f"    CD={cd:.3f}, Friedman p={friedman_p:.4f}")


def compute_critical_difference(group_stats: dict[str, dict]) -> None:
    """Every group gets a vfm_only CD table plus a with_resnet CD table that adds the
    resnet50/upernet supervised baseline as an extra reference model (resnet is always
    upernet, even inside the linear groups)."""
    for group_name, spec in GROUPS.items():
        models = [_model_key(e, spec["head"]) for e in spec["encoders"]]
        st = group_stats[group_name]
        mean_scores = {(_model_key(e, spec["head"]), ds): st[(e, ds)]["mean"]
                        for e in spec["encoders"] for ds in DATASETS if (e, ds) in st}
        rank_mat = _rank_matrix(mean_scores, models)

        print(f"\n[CD] Group: {group_name}  models={models}")
        _write_cd(models, rank_mat, OUT_DIR / f"critical_difference_{group_name}_vfm_only.csv", f"{group_name}_vfm_only")

        resnet_model = _model_key(RESNET_ENCODER, "upernet")
        resnet_st = group_stats["resnet50"]
        models_wr = models + [resnet_model]
        mean_scores_wr = dict(mean_scores)
        for ds in DATASETS:
            d = resnet_st.get((RESNET_ENCODER, ds))
            if d is not None:
                mean_scores_wr[(resnet_model, ds)] = d["mean"]
        rank_mat_wr = _rank_matrix(mean_scores_wr, models_wr)
        _write_cd(models_wr, rank_mat_wr, OUT_DIR / f"critical_difference_{group_name}_with_resnet.csv",
                   f"{group_name}_with_resnet")


def compute_vs_resnet50(group_stats: dict[str, dict]) -> None:
    resnet_st = group_stats["resnet50"]
    resnet_scores = {ds: d for (enc, ds), d in resnet_st.items()}

    rows = []
    for group_name, spec in GROUPS.items():
        st = group_stats[group_name]
        for enc in spec["encoders"]:
            for ds in DATASETS:
                d = st.get((enc, ds))
                r = resnet_scores.get(ds)
                if d is None:
                    continue
                delta = round(d["mean"] - r["mean"], 6) if r else ""
                beats = (d["mean"] > r["mean"]) if r else ""
                rows.append({
                    "group": group_name, "model": _model_key(enc, spec["head"]), "ds": ds,
                    "model_miou": round(d["mean"], 6), "model_std": round(d["std"], 6),
                    "resnet_miou": round(r["mean"], 6) if r else "",
                    "resnet_std": round(r["std"], 6) if r else "",
                    "delta": delta, "beats_resnet": beats,
                })
    rows.sort(key=lambda x: (x["group"], x["model"], x["ds"]))
    _write_csv(OUT_DIR / "vs_resnet50.csv", rows,
               ["group", "model", "ds", "model_miou", "model_std", "resnet_miou", "resnet_std",
                "delta", "beats_resnet"])


# ---------------------------------------------------------------------------
# 1.9 insights.json
# ---------------------------------------------------------------------------

def compute_insights(group_stats: dict[str, dict], avg_ranks: dict[str, dict[str, float]], rows: list[dict]) -> None:
    def best_model(variant: str) -> dict:
        ranks = avg_ranks.get(variant, {})
        if not ranks:
            return {}
        best = min(ranks, key=ranks.get)
        return {"model": best, "avg_rank": ranks[best]}

    kendall_summary_path = OUT_DIR / "kendall_group_summary.csv"
    kendall_summary = list(csv.DictReader(open(kendall_summary_path))) if kendall_summary_path.exists() else []

    heads_path = OUT_DIR / "kendall_between_heads.csv"
    heads_rows = list(csv.DictReader(open(heads_path))) if heads_path.exists() else []
    heads_agg = next((r for r in heads_rows if r["ds"] == "ALL_AGGREGATE"), {})

    scales_path = OUT_DIR / "kendall_between_scales.csv"
    scales_rows = list(csv.DictReader(open(scales_path))) if scales_path.exists() else []
    scales_agg = next((r for r in scales_rows if r["ds"] == "ALL_AGGREGATE"), {})

    gain_path = OUT_DIR / "head_gain.csv"
    gain_rows = list(csv.DictReader(open(gain_path))) if gain_path.exists() else []
    gains = [float(r["gain_pp"]) for r in gain_rows if r["gain_pp"] != ""]

    gpu_hours: dict[str, float] = defaultdict(float)
    for row in rows:
        gh = row.get("gpu_hours", "")
        if gh not in ("", None):
            try:
                gpu_hours[f"{row['encoder']}__{row['head']}"] += float(gh)
            except ValueError:
                pass

    insights = {
        "best_model_per_group": {
            "vitb_linear": best_model("vitb_linear"),
            "vitb_linear_with_resnet": best_model("vitb_linear_with_resnet"),
            "largest_linear": best_model("largest_linear"),
            "largest_linear_with_resnet": best_model("largest_linear_with_resnet"),
            "vitb_upernet_vfm_only": best_model("vitb_upernet"),
            "vitb_upernet_with_resnet": best_model("vitb_upernet_with_resnet"),
        },
        "kendall_between_datasets_per_group": kendall_summary,
        "kendall_between_heads_aggregate": heads_agg,
        "kendall_between_scales_aggregate": scales_agg,
        "head_gain_pp": {
            "min": round(min(gains), 4) if gains else None,
            "max": round(max(gains), 4) if gains else None,
            "mean": round(float(np.mean(gains)), 4) if gains else None,
        },
        "gpu_hours_total_by_model": {k: round(v, 2) for k, v in sorted(gpu_hours.items())},
    }

    with open(OUT_DIR / "insights.json", "w") as f:
        json.dump(insights, f, indent=2)
    print(f"  -> {OUT_DIR / 'insights.json'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global OUT_DIR, METRIC_COL

    parser = argparse.ArgumentParser(description="Compute rankings, CIs, Kendall analyses, CD, vs-ResNet reference.")
    parser.add_argument("--results", default=None, type=Path,
                         help="Defaults to $RESULTS_DIR/analysis/all_results.csv (RESULTS_DIR defaults to 'results').")
    parser.add_argument("--metric", choices=["miou", "miou_all"], default="miou",
                         help="miou = background excluded (default, matches current behavior); "
                              "miou_all = background included. miou_all outputs go to a separate "
                              "analysis_miou_all/ dir so the two never overwrite each other.")
    args = parser.parse_args()

    METRIC_COL = "test_miou" if args.metric == "miou" else "test_miou_all"
    OUT_DIR = analysis_dir(args.metric)
    results_path = args.results if args.results is not None else analysis_dir("miou") / "all_results.csv"

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run `python -m analysis.collect_results` first.")
        return

    rows = _load_results(results_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {results_path}, ranking on {METRIC_COL!r}, writing to {OUT_DIR}/")

    print("Run completeness...")
    completeness_rows = compute_completeness(rows)
    _write_csv(OUT_DIR / "run_completeness.csv", completeness_rows, ["group", "encoder", "head", "ds", "seed", "found"])

    print("\nConfidence intervals...")
    group_stats = compute_confidence_intervals(rows)

    print("\nRankings...")
    avg_ranks = compute_rankings(rows, group_stats)

    print("\nCritical difference analysis...")
    compute_critical_difference(group_stats)

    print("\nKendall tau between datasets...")
    compute_kendall_between_datasets(group_stats)

    print("\nKendall tau between heads (linear vs upernet)...")
    compute_kendall_between_heads(group_stats)

    print("\nCombined rankings across stages...")
    compute_combined_rankings(avg_ranks)

    print("\nFamily/size rank table...")
    compute_family_rank_table(vfm_only=True)
    compute_family_rank_table(vfm_only=False)

    print("\nKendall tau between scales (ViT-B vs largest)...")
    compute_kendall_between_scales(group_stats)

    print("\nKendall tau between regimes (family-level, ResNet-50 excluded)...")
    compute_kendall_between_regimes(group_stats)

    print("\nCombined critical difference across regimes (family-level)...")
    compute_combined_critical_difference_family(group_stats)

    print("\nHead gain analysis...")
    compute_head_gain(group_stats)

    print("\nvs-ResNet-50 reference...")
    compute_vs_resnet50(group_stats)

    print("\nInsights summary...")
    compute_insights(group_stats, avg_ranks, rows)


if __name__ == "__main__":
    main()
