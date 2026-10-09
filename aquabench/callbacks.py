"""Modular logging callback.

Decoupled from the core metric: logs **per-class IoU** at val/test epoch end (named
via the dataset's id2label) and, best-effort, a few prediction visualizations to
whichever Lightning logger is active (no-op if the logger can't take images).
"""

from __future__ import annotations

import csv
import os

import torch
import torch.nn.functional as F
from lightning.pytorch import Callback

from aquabench.data import IMAGENET_MEAN, IMAGENET_STD


class MetricsCallback(Callback):
    def __init__(self, log_samples: int = 4):
        self.log_samples = log_samples
        self._val = None
        self._test = None
        self._sample = None  # (image, pred, target) from first val batch

    def _metric(self, pl_module):
        return pl_module.task.per_class_metric().to(pl_module.device)

    @staticmethod
    def _id2label(trainer):
        return getattr(getattr(trainer, "datamodule", None), "id2label", {}) or {}

    def _update(self, metric, outputs, pl_module):
        p, t = outputs["preds"].flatten(), outputs["target"].flatten()
        valid = (t >= 0) & (t < pl_module.num_classes)
        metric.update(p[valid], t[valid])

    def _log_per_class(self, trainer, pl_module, metric, prefix):
        per = metric.compute().cpu().tolist()
        metric.reset()
        id2label = self._id2label(trainer)
        for c, v in enumerate(per):
            name = id2label.get(str(c), id2label.get(c, f"class_{c}"))
            pl_module.log(f"{prefix}_iou/{name}", float(v))

    # ---- validation ----
    def on_validation_epoch_start(self, trainer, pl_module):
        self._val = self._metric(pl_module)
        self._sample = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not outputs:
            return
        self._update(self._val, outputs, pl_module)
        if self._sample is None and self.log_samples:
            n = min(self.log_samples, outputs["preds"].shape[0])
            self._sample = (batch["pixel_values"][:n].detach().cpu(),
                            outputs["preds"][:n].detach().cpu(), outputs["target"][:n].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._val is not None:
            self._log_per_class(trainer, pl_module, self._val, "val")
        self._maybe_log_images(trainer, pl_module)

    # ---- test ----
    def on_test_epoch_start(self, trainer, pl_module):
        self._test = self._metric(pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs:
            self._update(self._test, outputs, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self._test is not None:
            self._log_per_class(trainer, pl_module, self._test, "test")

    # ---- best-effort image logging ----
    @staticmethod
    def _denormalize(imgs: torch.Tensor) -> torch.Tensor:
        """Undo ImageNet normalization -> uint8 RGB [N,3,H,W] for display."""
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return ((imgs * std + mean).clamp(0, 1) * 255).to(torch.uint8)

    @staticmethod
    def _match_hw(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Nearest-resize an integer mask [N,H',W'] to (h, w) so it overlays the image."""
        if mask.shape[-2:] == (h, w):
            return mask
        return F.interpolate(mask.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()

    def _maybe_log_images(self, trainer, pl_module):
        """Log the real input image with prediction + ground-truth overlays.

        W&B gets native interactive segmentation masks (named via id2label); a
        TensorBoard logger falls back to the denormalized RGB input. No-op for any
        other logger.
        """
        if self._sample is None:
            return
        logger = getattr(trainer, "logger", None)
        exp = getattr(logger, "experiment", None)
        if exp is None:
            return

        imgs, preds, target = self._sample
        rgb = self._denormalize(imgs)  # [N,3,H,W] uint8
        h, w = rgb.shape[-2:]
        preds = self._match_hw(preds, h, w)
        target = self._match_hw(target, h, w)

        id2label = self._id2label(trainer)
        class_labels = {
            c: id2label.get(str(c), id2label.get(c, f"class_{c}")) for c in range(pl_module.num_classes)
        }
        ignore = getattr(getattr(trainer, "datamodule", None), "loss_ignore_index", 255)
        class_labels[int(ignore)] = "ignore"

        if type(exp).__module__.startswith("wandb"):  # W&B: interactive mask overlays
            import wandb
            images = [
                wandb.Image(
                    rgb[i].permute(1, 2, 0).numpy(),
                    masks={
                        "prediction": {"mask_data": preds[i].numpy(), "class_labels": class_labels},
                        "ground_truth": {"mask_data": target[i].numpy(), "class_labels": class_labels},
                    },
                )
                for i in range(rgb.shape[0])
            ]
            try:
                exp.log({"val/predictions": images})
            except Exception:
                pass
        elif hasattr(exp, "add_image"):  # TensorBoard: at least show the real input
            for i in range(rgb.shape[0]):
                try:
                    exp.add_image(f"val_input/{i}", rgb[i], global_step=trainer.global_step)
                except Exception:
                    break


class PerSampleTestCallback(Callback):
    """Writes per-sample intersection/union to CSV during test for later subset analysis.

    No-op during fit/validation. Output: results/per_sample/{ds}__{encoder}__{head}__per_sample_test.csv
    Columns: sample_idx, class_idx, intersection, union, seed
    """

    _FIELDNAMES = ("sample_idx", "class_idx", "intersection", "union", "seed")

    def __init__(self, output_dir: str = os.path.join(os.environ.get("RESULTS_DIR", "results"), "per_sample")):
        self.output_dir = output_dir
        self._records: list[dict] = []
        self._sample_idx = 0
        self.seed: int | None = None  # injected by BenchCLI.before_test

    def on_test_epoch_start(self, trainer, pl_module):
        self._records = []
        self._sample_idx = 0

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not outputs:
            return
        preds = outputs["preds"]   # [B, H, W]
        target = outputs["target"] # [B, H, W]
        n_classes = pl_module.num_classes
        for i in range(preds.shape[0]):
            p = preds[i].flatten()
            t = target[i].flatten()
            valid = (t >= 0) & (t < n_classes)
            pv, tv = p[valid], t[valid]
            for c in range(n_classes):
                pred_c = pv == c
                tgt_c = tv == c
                self._records.append({
                    "sample_idx": self._sample_idx,
                    "class_idx": c,
                    "intersection": int((pred_c & tgt_c).sum()),
                    "union": int((pred_c | tgt_c).sum()),
                    "seed": self.seed,
                })
            self._sample_idx += 1

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._records:
            return
        ds = getattr(getattr(trainer, "datamodule", None), "ds_name", "unknown")
        encoder = pl_module.hparams.get("encoder", "unknown")
        head = pl_module.hparams.get("head", "unknown")
        os.makedirs(self.output_dir, exist_ok=True)
        seed_suffix = f"__seed{self.seed}" if self.seed is not None else ""
        path = os.path.join(self.output_dir, f"{ds}__{encoder}__{head}{seed_suffix}__per_sample_test.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._records)
