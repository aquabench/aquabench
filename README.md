# AQUABENCH

**A benchmark of vision foundation models for underwater semantic segmentation.**

AQUABENCH evaluates pretrained image encoders as frozen (and optionally fine-tuned)
feature extractors for dense prediction on underwater imagery, across a unified
collection of **9 underwater segmentation datasets**. Everything runs through a single
[Lightning](https://lightning.ai/) CLI — pick an **encoder × head × task × dataset** and
go.

The design is deliberately factored into four orthogonal axes so that any encoder can be
compared under an identical training/evaluation protocol:

| Axis | Flag | What it selects |
|---|---|---|
| **Encoder** | `--encoder` | the pretrained backbone under test |
| **Head** | `--head` | how backbone features become segmentation logits |
| **Task** | `--task` | the objective + metric (`semantic_seg`) |
| **Dataset** | `--ds` | which underwater dataset config to load |

> **Note** — This is the camera-ready code release. It contains everything needed to
> **train, evaluate, and reproduce the paper's tables and figures**. Cluster-specific
> job-submission scripts (SLURM) have been removed; the equivalent sweeps are documented
> below as plain shell loops so they run on any machine. For a leaner, model-only install
> (no reproduction pipeline), see the [**`main`** branch](../../tree/main).

---

## Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [The four axes](#the-four-axes)
- [Quickstart](#quickstart)
- [Configs](#configs)
- [Heads and tasks](#heads-and-tasks)
- [Logging and metrics](#logging-and-metrics)
- [sklearn probe (fast sibling)](#sklearn-probe-fast-sibling)
- [Reproducing the paper](#reproducing-the-paper)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

---

## Installation

Python ≥ 3.10 and a CUDA-capable GPU are recommended. Using
[uv](https://github.com/astral-sh/uv):

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Or with plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `aquabench` package and the `aquabench` console script. Encoder weights
(DINOv3, Franca, RADIO, EVA-02, V-JEPA 2.1) are downloaded on first use from their
respective sources (HuggingFace Hub / `torch.hub` / official URLs) and cached locally.

## Dataset

AQUABENCH is trained and evaluated on a **unified** build of 9 underwater segmentation
datasets, harmonised to a common on-disk layout so a single loader handles all of them.

**1. Download the published dataset** from the project page:

> **[aquabench.github.io](https://aquabench.github.io)**

**2. Place it at `$DATA_ROOT/unified`** and point `DATA_ROOT` at it:

```bash
export DATA_ROOT=/path/to/data          # the loader reads $DATA_ROOT/unified
```

The loader (`aquabench/data`) expects the unified directory to contain, per dataset
config, the split parquet files plus a metadata file it reads for `num_classes`,
`img_size`, and the loss/metric ignore indices:

```
$DATA_ROOT/unified/
  _metadata/
    cou.json  suim.json  l4s.json  ...     # per-config metadata
  <parquet data for each config/split>
```

Datasets are loaded by short config name via `datasets.load_dataset($DATA_ROOT/unified,
<ds>)`. The 9 configs (short name → source dataset):

| `--ds` | Dataset | | `--ds` | Dataset |
|---|---|---|---|---|
| `cs` | CoralScapes | | `sc` | SeaCLEAR |
| `cou` | COU | | `suim` | SUIM |
| `df` | DeepFish | | `tc` | TrashCan |
| `l4s` | Looking for Seagrass (L4S) | | `uiis10k` | UIIS10K |
| `liaci` | LIACi | | | |

## The four axes

Everything is a combination of the four flags.

| Axis | Flag | Choices |
|---|---|---|
| Encoder | `--encoder` | `dinov3_{small,base,large,7b}`, `franca_{base,large,giant}`, `radio_v3_{base,large}`, `radio_v4_{h,so400m}`, `eva02_{base,large}`, `vjepa21_{vitb,vitg}`, `resnet50`, `resnet101`, `dummy` |
| Head | `--head` | `linear` (frozen encoder + 1×1 conv), `upernet_frozen` (frozen encoder + UPerNet decoder), `upernet` (full end-to-end fine-tuning) |
| Task | `--task` | `semantic_seg` |
| Dataset | `--ds` | `cs`, `cou`, `df`, `l4s`, `liaci`, `sc`, `suim`, `tc`, `uiis10k` |

`--out_indices` is generic (`last`, an int *N* for *N* evenly-spaced layers, or an explicit
list); it defaults to the last layer for `linear` and 4 evenly-spaced layers for the
UPerNet heads. `num_classes`, `img_size`, and ignore indices are read from the dataset
metadata automatically and linked into the model.

The paper's **model roster** is 5 encoder families × 2 tiers — a ViT-B tier
(`dinov3_base`, `franca_base`, `radio_v3_base`, `eva02_base`, `vjepa21_vitb`) and a
largest-per-family tier (`dinov3_7b`, `franca_giant`, `radio_v4_h`, `eva02_large`,
`vjepa21_vitg`) — plus a supervised **ResNet-50** reference.

## Quickstart

```bash
export DATA_ROOT=/path/to/data

# Linear probe: frozen DINOv3-B + 1x1 conv head on COU
python -m aquabench.cli fit  --config configs/linear.yaml  --encoder dinov3_base --ds cou

# Frozen-encoder UPerNet decoder on SUIM
python -m aquabench.cli fit  --config configs/upernet.yaml --encoder dinov3_base --ds suim --head upernet_frozen

# Fully fine-tuned UPerNet (encoder + decoder) with a ResNet-50 backbone
python -m aquabench.cli fit  --config configs/upernet.yaml --encoder resnet50    --ds suim

# Evaluate a trained checkpoint on the test split
python -m aquabench.cli test --config configs/linear.yaml  --encoder dinov3_base --ds cou --ckpt_path <path/to/best.ckpt>
```

Any axis or trainer option can be overridden on the CLI, e.g.
`--trainer.max_epochs 100`, `--data.batch_size 16`, `--seed_everything 634`.

## Configs

Reusable defaults live in `configs/` and are composed with `--config`:

- `configs/default.yaml` — shared defaults (optimizer, augmentation, checkpoint callbacks).
- `configs/linear.yaml` — linear-probe protocol (frozen encoder + 1×1 conv).
- `configs/upernet.yaml` — UPerNet decoder protocol.
- `configs/wandb.yaml` — compose on top of any config to switch the logger to
  Weights & Biases: `--config configs/linear.yaml --config configs/wandb.yaml`.

## Heads and tasks

Heads return **logits only**; the **Task** owns the loss and metric (CrossEntropy + IoU
for every head under `semantic_seg`). The three heads differ in what they attach and
whether the encoder is frozen:

| `--head` | Encoder | Decoder | Role |
|---|---|---|---|
| `linear` | frozen | 1×1 conv on the last feature map | benchmark primary (linear probing) |
| `upernet_frozen` | frozen | UPerNet (4 levels) | benchmark secondary (frozen features, trained decoder) |
| `upernet` | trainable | UPerNet (4 levels) | full end-to-end fine-tuning |

The default freeze policy comes from the head but can be overridden with
`--model.freeze_encoder`. Augmentation presets are `none` / `light` / `medium` / `heavy`
(`--data.augmentation`), applied on-device via [kornia](https://kornia.readthedocs.io/).

## Logging and metrics

The Lightning logger is swappable via `--trainer.logger` (CSV by default; point at
`WandbLogger` / `TensorBoardLogger` in a config). A `MetricsCallback` (added by default)
logs per-class IoU at validation/test epoch end and best-effort sample predictions.

**mIoU is logged twice:** `val/miou` excludes the background class (the headline number)
and `val/miou_all` includes background as a class — same for `test/`. Every `test` run
also appends a row to `results/analysis/summary.csv` (`ds, backbone, head, seed,
test_miou, test_miou_all`), which is the logger-independent record the analysis pipeline
reads.

## sklearn probe (fast sibling)

A lightweight, training-free alternative to the Lightning linear head:

```bash
python -m aquabench.probe --ds l4s --encoder dinov3_base
```

Frozen encoder (last layer) → patch embeddings → `LogisticRegression` at patch
resolution → `results/`. Reports mIoU with and without background. Embeddings are
**streamed to disk** as per-batch shards under `--cache-dir` (default
`results/probe_cache/<ds>__<encoder>/`) so they never all sit in RAM: fit pulls at most
`--sklearn-max-train` rows in, and val/test are scored by streaming a confusion matrix.
The cache is deleted afterwards unless `--keep-cache`.

## Reproducing the paper

The full benchmark is a sweep over **encoder × dataset × seed** for each protocol,
followed by an analysis pipeline that turns the collected results into the paper's tables
and figures. Seeds used in the paper:

- **Linear probing** — 5 seeds: `42 851 634 913 1627`
- **UPerNet (frozen and fine-tuned)** — 3 seeds: `42 634 1627`

### 1. Train + evaluate the sweep

Example — the ViT-B linear-probing group (5 encoders × 9 datasets × 5 seeds). Run the
analogous loops for the largest-per-family encoders and for the `upernet_frozen` /
`upernet` heads (3 seeds), and a `resnet50` / `upernet` group as the supervised reference:

```bash
export DATA_ROOT=/path/to/data
DATASETS="cs cou df l4s liaci sc suim tc uiis10k"
VITB="dinov3_base franca_base radio_v3_base eva02_base vjepa21_vitb"

for ds in $DATASETS; do
  for enc in $VITB; do
    for seed in 42 851 634 913 1627; do
      # 1. Train — best/last checkpoints land under logs/linear/<run>/checkpoints/
      python -m aquabench.cli fit  --config configs/linear.yaml --encoder $enc --ds $ds --seed_everything $seed

      # 2. Evaluate the run's best checkpoint on the test split
      ckpt=$(find logs/linear -path '*checkpoints*' -name 'miou=*.ckpt' -printf '%T@ %p\n' \
             | sort -nr | head -1 | cut -d' ' -f2-)
      python -m aquabench.cli test --config configs/linear.yaml --encoder $enc --ds $ds --seed_everything $seed \
             --ckpt_path "$ckpt"
    done
  done
done
```

The `best-{epoch}-{val/miou}` checkpoint template contains a `/` (from the `val/miou`
metric name), so Lightning saves it as `.../checkpoints/best-epoch=NN-val/miou=X.ckpt`;
the `find` above matches that nested `miou=*.ckpt` file and picks the most recent, i.e.
the run that just finished. Each `test` invocation appends a row to
`results/analysis/summary.csv` (see [Logging and metrics](#logging-and-metrics)), so the
sweep's results accumulate there regardless of which logger is active.

### 2. Collect, rank, and render

```bash
# Aggregate every test run into one results table
python -m analysis.collect_results

# Per-dataset rankings, Kendall-tau agreement, Kendall's W, Friedman/Nemenyi
# critical-difference diagrams, and 95% confidence intervals
python -m analysis.compute_rankings

# Paper-ready CSV tables (mIoU +/- CI, wide format) and figures
python -m analysis.make_result_tables
python -m analysis.make_ranking_figures

# UIQM image-quality-tier analysis + figure
python -m analysis.quality_regime_analysis
python -m analysis.make_quality_figures

# Dataset statistics tables/figures (image stats, object-size distribution, sample grids)
python -m analysis.dataset_summary
python -m analysis.make_dataset_table
python -m analysis.make_sample_figures

# LaTeX tables for the paper (mIoU +/- CI, ranking tables)
python helpers/generate_latex_tables.py iou   results/tables/table_vitb_linear_miou_ci.csv
python helpers/generate_latex_tables.py rank   results/rankings_vitb_linear_with_resnet.csv
```

Most analysis scripts accept `--help` and read their inputs from `results/` (overridable
via `--input-dir` / `--output-dir`). Figures are written to `results/analysis/figures/`
as `.pdf` and `.svg`.

## Repository layout

Grouped by **role**: `models/` = the network you swap on the CLI (encoder × head),
`tasks/` = the objective + metric, `data/` = ingestion + on-device transforms, and the
flat top-level files = orchestration.

```
aquabench/
  cli.py             # LightningCLI (--encoder --head --task --ds)
  module.py          # TaskModule = compose(encoder, head, task)
  callbacks.py       # MetricsCallback (per-class IoU + sample preds)
  probe.py           # precompute + sklearn probe
  models/
    encoders/
      __init__.py    # ENCODERS catalog + build_encoder(...)
      dense_backbone.py  resnet.py  dummy.py
      extractors/    # dino / franca / radio / eva / vjepa raw feature extraction
    heads/
      __init__.py    # HEADS: linear, upernet, upernet_frozen (return logits only)
      linear.py  upernet.py
  tasks/
    __init__.py      # TASKS registry
    semantic_seg.py  # SemanticSegTask: CE loss + IoU metric
  data/
    __init__.py      # HFSegDataModule (load_dataset only)
    augment.py       # kornia presets (none/light/medium/heavy)
    tokens.py        # token-grid helpers
configs/             # composable Lightning CLI configs
analysis/            # results aggregation, rankings, statistics, figures
helpers/             # LaTeX table generation
scripts/             # standalone smoke test (V-JEPA backbone)
```

## Citation

If you use AQUABENCH, please cite:

```bibtex
<!-- TODO: add the paper's BibTeX entry -->
```

## License

Released under the terms in [LICENSE](LICENSE).
