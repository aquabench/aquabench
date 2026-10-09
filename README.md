# AQUABENCH

**Vision foundation models for underwater semantic segmentation — ready to run.**

AQUABENCH evaluates pretrained image encoders as frozen (and optionally fine-tuned)
feature extractors for dense prediction on underwater imagery, across a unified
collection of **9 underwater segmentation datasets**. Everything runs through a single
[Lightning](https://lightning.ai/) CLI — pick an **encoder × head × task × dataset** and
go.

The design is factored into four orthogonal axes so any encoder can be compared under an
identical training/evaluation protocol:

| Axis | Flag | What it selects |
|---|---|---|
| **Encoder** | `--encoder` | the pretrained backbone under test |
| **Head** | `--head` | how backbone features become segmentation logits |
| **Task** | `--task` | the objective + metric (`semantic_seg`) |
| **Dataset** | `--ds` | which underwater dataset config to load |

> **This is the lean, out-of-the-box version** — just what you need to train, evaluate,
> and run the models. To reproduce the paper's tables and figures (rankings, Kendall-τ,
> critical-difference diagrams, confidence intervals, quality-tier analysis), check out
> the [**`camera-ready` branch**](../../tree/camera-ready), which adds the `analysis/`
> and `helpers/` pipelines.

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
- [Verify your install](#verify-your-install)
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
| `linear` | frozen | 1×1 conv on the last feature map | linear probing |
| `upernet_frozen` | frozen | UPerNet (4 levels) | frozen features, trained decoder |
| `upernet` | trainable | UPerNet (4 levels) | full end-to-end fine-tuning |

The default freeze policy comes from the head but can be overridden with
`--model.freeze_encoder`. Augmentation presets are `none` / `light` / `medium` / `heavy`
(`--data.augmentation`), applied on-device via [kornia](https://kornia.readthedocs.io/).

## Logging and metrics

The Lightning logger is swappable via `--trainer.logger` (CSV by default; point at
`WandbLogger` / `TensorBoardLogger` in a config). A `MetricsCallback` (added by default)
logs per-class IoU at validation/test epoch end and best-effort sample predictions.

**mIoU is logged twice:** `val/miou` excludes the background class (the headline number)
and `val/miou_all` includes background as a class — same for `test/`.

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

## Verify your install

A standalone smoke test loads a V-JEPA 2.1 backbone and runs the full frozen-backbone
linear-probe path on a dummy batch — a quick way to confirm your environment works
without touching the dataset:

```bash
python scripts/smoke_vjepa.py                      # lightweight ViT-B (fast)
python scripts/smoke_vjepa.py --encoder vjepa21_vitg   # the 2B ViT-G (large download)
```

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
scripts/             # standalone smoke test (V-JEPA backbone)
```

## Citation

If you use AQUABENCH, please cite:

```bibtex
<!-- TODO: add the paper's BibTeX entry -->
```

## License

Released under the terms in [LICENSE](LICENSE).
