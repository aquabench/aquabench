# AQUABENCH

**Underwater semantic segmentation benchmark for vision encoders. Plain PyTorch.**

You prepare the encoder; AQUABENCH handles data, heads, the training loop and metrics
across **9 underwater segmentation datasets**, under one fixed protocol.

```python
import aquabench

result = aquabench.run(
    encoder,                     # your nn.Module, see "Encoder contract"
    logger=logger,               # anything with .log(dict, step); default prints
    run_type="linear",           # "linear" | "upernet_frozen" | "upernet_finetune"
    seed=0,
    data_root="/data/aquabench",
    datasets="all",              # or "SUIM", or ["SUIM", "COU"]
)
result["SUIM"].test_miou         # all-class test mIoU (result.test_miou for a single dataset)
```

One call = one encoder, one run type, one seed. Multi-seed and multi-model sweeps are
your own scripts (see [`examples/run_dinov2.py`](examples/run_dinov2.py)).

> v1.1 is a lean rewrite of v1 (Lightning CLI + built-in model zoo). The v1 code is on
> `main` / in the git history; the paper's analysis pipeline is on the `camera-ready` branch.

---

## Installation

```bash
pip install -e .                 # core: torch, numpy, pillow only
pip install -e '.[download]'     # + `datasets`, needed only for `aquabench download`
pip install -e '.[test]'         # + pytest
```

## Data

```bash
aquabench download --data-root /data/aquabench                        # all datasets
aquabench download --data-root /data/aquabench --datasets SUIM COU    # a subset
aquabench verify   --data-root /data/aquabench                        # check layout and masks
```

Data comes from [huggingface.co/datasets/TorbenGl/AQUABENCH](https://huggingface.co/datasets/TorbenGl/AQUABENCH)
(`--source` takes another repo id or a local copy of it). Download writes a plain layout
that training reads with Pillow only, plus each dataset's license notice:

```
/data/aquabench/
  SUIM/
    meta.json             # {"num_classes", "ignore_index", "size": [H, W], "class_names"}
    dataset_license.txt
    train/images/*.png  train/masks/*.png
    val/images/...      val/masks/...
    test/images/...     test/masks/...
  COU/ ...
```

Set `$AQUABENCH_DATA` to skip passing `data_root` to `run` / `evaluate`.

| Dataset | hub config | | Dataset | hub config |
|---|---|---|---|---|
| CoralScapes | `coralscapes` | | SeaClear | `seaclear` |
| COU | `cou` | | SUIM | `suim` |
| DeepFish | `deepfish` | | TrashCan | `trashcan` |
| L4S (Looking for Seagrass) | `l4s` | | UIIS10K | `uiis10k` |
| LIACi | `liaci` | | | |

Names are case-insensitive; hub config names and v1 short names (`cs`, `df`, `sc`, `tc`) work too. Splits are frozen as shipped.

## Encoder contract

AQUABENCH does no encoder handling: no model zoo, no weight loading, no layer selection.

| run type | encoder | `encoder.forward(x)` must return | head |
|---|---|---|---|
| `linear` | frozen | one tensor `[B, C, h, w]` | BatchNorm + 1×1 conv |
| `upernet_frozen` | frozen | a list of 4 tensors `[B, C_i, h_i, w_i]` | UPerNet (PPM + FPN) |
| `upernet_finetune` | fully trained | a list of 4 tensors `[B, C_i, h_i, w_i]` | UPerNet (PPM + FPN) |

For both UPerNet run types, `forward` returns a Python list (or tuple) of exactly 4 separate
tensors, shallow to deep: `[f1, f2, f3, f4]`. They are not stacked into one 5D tensor and not
concatenated, because a CNN's 4 maps differ in channels and size.

`upernet_finetune` is full fine-tuning. The encoder must offer three things:

- It has parameters, and all of them are trained, whatever their `requires_grad` flag. The flags
  are restored after the run; the weights are changed in place.
- Its `forward` lets gradients through: no `torch.no_grad()` and no `.detach()` on the 4 maps.
- It respects `train()` and `eval()`: it is in eval mode while frozen in the first epoch and for
  validation and testing, and in train mode otherwise.

The first two are checked before training.

The output is checked on a dummy image before training; a mismatch raises
`aquabench.EncoderContractError` with a clear message. This includes the token grid: the
last map must be `size / 16`, so set `patch_size` to the encoder's total stride (14 for a
ViT/14, 32 for a ResNet). If the 4 maps all share one
spatial size (plain ViT), the UPerNet head resizes them ×4, ×2, ×1, ×0.5 into a pyramid,
as in v1; maps that already form a pyramid (CNNs) are used as they are.

```python
class DINOv3Pyramid(nn.Module):
    def __init__(self, m, layers=(2, 5, 8, 11)):
        super().__init__(); self.m, self.layers = m, layers
    def forward(self, x):
        return list(self.m.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True))
```

[`examples/run_dinov2.py`](examples/run_dinov2.py) is a complete user script: DINOv2 from
`torch.hub` (public weights), optionally your own weights via `--weights file.pt`
(`torch.load` state dict; `module.` / `backbone.` prefixes and `{"model": ...}` wrappers
are handled), wrapped for `linear` or `upernet_*`, looped over seeds.

Inputs are ImageNet-normalized by default. Use what your encoder was trained with, because a
wrong choice is not detected and only lowers the score. `normalize` takes `"imagenet"`, `"none"`
for `[0, 1]` images, or any callable on a `[3, H, W]` image in `[0, 1]`:

```python
from torchvision.transforms import Normalize

aquabench.run(encoder, run_type="linear", normalize=Normalize(mean=(0.48, 0.46, 0.41), std=(0.27, 0.26, 0.28)))
```

A callable is stored by name in `config.json`, so pass it again to `aquabench.evaluate`. With
DataLoader workers it must be picklable, which rules out lambdas on some platforms.
The run folder is named after `encoder_name=...`, else `encoder.name`, else the class name.

## Protocol

- **Resolution.** Each dataset has a target size in `meta.json`, divisible by 16, with its
  own aspect ratio. Token grid = size / 16. The encoder input is `grid × patch_size`
  (`patch_size=16` by default; `14` gives a smaller input with the same token count).
  Masks stay at the dataset size, so metrics are comparable across patch sizes.
  A run stops at the start if a benchmark dataset does not have its expected size, class
  count and samples per split.
- **Training.** AdamW, cosine schedule, 50 epochs, effective batch 32
  (`batch_size` 16 × `grad_accum_steps` 2), bf16 autocast. Learning rate 1e-3 for
  `linear`, 1e-4 for both UPerNet run types.
- **Fine-tuning warm-up** (`upernet_finetune` only). The encoder is frozen for the first epoch
  (`freeze_encoder_epochs`), so the new head trains alone. Its learning rate then ramps up
  linearly over 2 epochs (`encoder_warmup_epochs`) to `encoder_lr_scale` = 0.1 of the head's
  (1e-5), and follows the same cosine schedule. `linear` and `upernet_frozen` train only a new
  head and have no warm-up.
- **Augmentation.** Zoom-in only: random scale in [1.0, 1.5], random crop back to size,
  random horizontal flip. No padding. Off for `linear`, on for `upernet_frozen` and
  `upernet_finetune`; override with `augment=True/False`.
- **Reported number.** Test mIoU over **all classes including background**, from the
  **last epoch**. Val is only logged, never used for checkpoint selection. The mIoU is the sum
  of the class IoUs divided by the dataset's number of classes, so every model is averaged over
  the same classes; a class that is absent from the test masks and never predicted adds 0.
- **Masks** are resized with `nearest-exact`.
- **Cached features** (`linear` only). Without augmentation the frozen encoder gives the same
  features every epoch, so each image is encoded once, stored in `cache_dir` (default: the
  system temp folder) and deleted after the run. The size is logged as `<Dataset>/cache/gb`; it
  can reach tens of GB for large encoders, so point `cache_dir` at local scratch. If the space
  is missing, the run warns and encodes every epoch instead. `"cache_features": false` turns
  it off.

Defaults ship in [`aquabench/configs/`](aquabench/configs) (one JSON per run type).
Override any key with a dict or a JSON file. `augment`, `batch_size`, `grad_accum_steps` and
`normalize` are also arguments, and these win over the config:

```python
aquabench.run(encoder, run_type="upernet_finetune", seed=0, patch_size=14,
              config="my_finetune.json", batch_size=8, grad_accum_steps=4)
```

## Outputs

Each dataset gets its own run folder under `output_dir` (default `runs/`):

```
runs/<encoder_name>/<run_type>/<dataset>_seed<seed>/
  config.json      # effective config
  history.json     # per-epoch train loss and val mIoU
  result.json      # test mIoU, per-class IoU, last val mIoU, runtime, version, run_info
  checkpoint.pt    # head (+ encoder weights for upernet_finetune)
```

- `run_info` in `result.json` records what ran: the encoder's class, parameter count and feature
  map shapes, the dataset's classes, sizes and samples per split, and the torch version, device
  and start time.
- A run folder that already holds a `result.json` is not replaced: `run` raises
  `FileExistsError` before training unless you pass `overwrite=True`.
- `checkpoint_dir=...` puts only the checkpoints under another root (e.g. scratch).
- `save_checkpoint=False` writes no checkpoint (large sweeps); `evaluate` is then not possible.

Re-test a saved run without training. You pass the encoder; the checkpoint provides the
head, and for `upernet_finetune` also the encoder weights. A warning is raised if a
frozen encoder's weights differ from the ones used in training.

```python
result = aquabench.evaluate(encoder, checkpoint_dir="runs/dinov3_vitb16/linear/SUIM_seed0")
# -> runs/dinov3_vitb16/linear/SUIM_seed0/result_eval.json
```

## Logging

Any object with `log(metrics: dict, step: int)`. Keys are prefixed with the dataset name
(`SUIM/train/loss`, `SUIM/val/miou`, `SUIM/test/miou`, `SUIM/test/iou/<class>`), and
`step` (optimizer steps) increases monotonically across the datasets of one call.

```python
class WandbLogger:
    def __init__(self, run): self.run = run
    def log(self, metrics, step): self.run.log(metrics, step=step)
```

## Tests

```bash
pytest                                   # unit tests on a tiny synthetic dataset (CPU, ~1 min)
AQUABENCH_DATA=/data/aquabench pytest    # also checks the downloaded benchmark
bash scripts/e2e_fresh_install.sh        # fresh install into a new venv + tiny run per run type
```

## Repository layout

```
aquabench/
  api.py        # run(), evaluate(), Result, run folder
  train.py      # plain PyTorch loop with gradient accumulation
  heads.py      # encoder contract, LinearHead, UPerNetHead
  data.py       # on-disk layout, token-grid sizing, augmentation
  metrics.py    # confusion-matrix mIoU
  cache.py      # frozen features computed once (linear)
  config.py     # default + user config
  configs/      # linear.json, upernet_frozen.json, upernet_finetune.json
  loggers.py    # Logger protocol, PrintLogger
  download.py   # unified build -> on-disk layout
  cli.py        # `aquabench download` / `aquabench verify`
tests/
scripts/e2e_fresh_install.sh
examples/run_dinov2.py
```

## Citation

```bibtex
<!-- TODO: add the paper's BibTeX entry -->
```

## License

Released under the terms in [LICENSE](LICENSE).
