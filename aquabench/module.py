"""TaskModule: compose encoder x head x task into one LightningModule.

`num_classes` / `img_size` / ignore indices are injected by the CLI from the
datamodule (read from the dataset metadata), so a run only specifies
`--encoder --head --task --ds`.
"""

from __future__ import annotations

from typing import Optional, Union, List

import torch
from lightning.pytorch import LightningModule


def build_model(encoder, head, num_classes, img_size, out_indices):
    from aquabench.models.encoders import build_encoder
    from aquabench.models.heads import HEADS
    from aquabench.data.tokens import target_token_grid

    h = HEADS[head]
    enc = build_encoder(encoder, out_indices, h.scales, h.use_lateral_convs, h.out_channels, tuple(img_size))
    # Every encoder emits the same token grid per task (fixed by patch 16), so
    # encoders with a smaller patch shrink their input rather than add tokens.
    grid = target_token_grid(tuple(img_size))
    model = h.build(enc, num_classes=num_classes, n_levels=h.n_levels, token_grid=grid)
    return model, h


class TaskModule(LightningModule):
    def __init__(
        self,
        encoder: str = "dinov3_base",
        head: str = "linear",
        task: str = "semantic_seg",
        num_classes: int = 2,
        img_size: tuple = (540, 960),
        out_indices: Union[None, str, int, List[int]] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        encoder_lr_scale: float = 1.0,
        warmup_epochs: int = 0,
        freeze_encoder: Optional[bool] = None,
        scheduler: str = "cosine",
        loss_ignore_index: int = 255,
        metric_ignore_index: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model, head_spec = build_model(encoder, head, num_classes, img_size, out_indices)
        do_freeze = head_spec.freeze_encoder if freeze_encoder is None else freeze_encoder
        if do_freeze and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        from aquabench.tasks import TASKS
        self.task = TASKS[task](
            num_classes=num_classes, ignore_index=metric_ignore_index,
            loss_ignore_index=loss_ignore_index,
        )
        self.num_classes = num_classes
        # IoU logged twice: excluding background (primary) and including it (`_all`).
        self.val_metric = self.task.build_metric()
        self.val_metric_all = self.task.build_metric(include_background=True)
        self.test_metric = self.task.build_metric()
        self.test_metric_all = self.task.build_metric(include_background=True)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.task.step(self.model, batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["labels"].shape[0])
        return loss

    def _eval_step(self, batch, metrics, prefix):
        loss, preds, target = self.task.step(self.model, batch)
        if loss is not None:
            self.log(f"{prefix}/loss", loss, prog_bar=True, batch_size=batch["labels"].shape[0])
        p, t = preds.flatten(), target.flatten()
        valid = (t >= 0) & (t < self.num_classes)
        for m in metrics:
            m.update(p[valid], t[valid])
        return {"preds": preds.detach(), "target": target.detach()}

    def _log_miou(self, prefix, metric, metric_all):
        name = self.task.metric_name
        self.log(f"{prefix}/{name}", metric.compute(), prog_bar=True)          # excl. background
        self.log(f"{prefix}/{name}_all", metric_all.compute())                 # incl. background
        metric.reset()
        metric_all.reset()

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, (self.val_metric, self.val_metric_all), "val")

    def on_validation_epoch_end(self):
        self._log_miou("val", self.val_metric, self.val_metric_all)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, (self.test_metric, self.test_metric_all), "test")

    def on_test_epoch_end(self):
        self._log_miou("test", self.test_metric, self.test_metric_all)

    def on_save_checkpoint(self, checkpoint):
        # Frozen encoder weights don't need to be persisted — they are always
        # re-loaded from pretrained sources on instantiation.
        frozen = {
            f"model.{name}"
            for name, param in self.model.named_parameters()
            if not param.requires_grad
        }
        for k in list(checkpoint["state_dict"]):
            if k in frozen:
                del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint):
        # Re-populate missing encoder keys from the already-initialized model
        # (pretrained weights are loaded in __init__ before this hook fires).
        current = self.state_dict()
        for k in current:
            if k not in checkpoint["state_dict"]:
                checkpoint["state_dict"][k] = current[k]

    def configure_optimizers(self):
        base_lr = self.hparams.lr
        wd = self.hparams.weight_decay
        scale = self.hparams.encoder_lr_scale
        warmup = self.hparams.warmup_epochs
        max_epochs = getattr(self.trainer, "max_epochs", None)

        # Back-compat path: a single param group + plain cosine, byte-identical to
        # the original behavior that every existing/ViT checkpoint was trained with.
        if scale == 1.0 and warmup == 0:
            params = [p for p in self.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)
            if self.hparams.scheduler == "cosine" and max_epochs:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
                return {"optimizer": opt, "lr_scheduler": sched}
            return opt

        # Fine-tuning path: encoder trains at a reduced LR; optional LR warmup.
        enc_params = self.model._encoder_params() if hasattr(self.model, "_encoder_params") else []
        enc_ids = {id(p) for p in enc_params}
        enc, rest = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (enc if id(p) in enc_ids else rest).append(p)
        groups = [{"params": rest, "lr": base_lr}]
        if enc:
            groups.append({"params": enc, "lr": base_lr * scale})
        opt = torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)

        if self.hparams.scheduler == "cosine" and max_epochs:
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs - warmup))
            if warmup > 0:
                warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup)
                sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[warmup])
            else:
                sched = cosine
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
