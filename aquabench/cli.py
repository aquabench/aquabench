"""Single Lightning CLI for the aquabench.

    python -m aquabench.cli fit  --encoder dinov3_base --head linear --task semantic_seg --ds cou
    python -m aquabench.cli fit  --encoder dinov3_base --head upernet --ds cou --trainer.max_epochs 100
    python -m aquabench.cli test --encoder dinov3_base --head upernet --ds cou --ckpt_path <ckpt>

`--encoder/--head/--task` feed the model, `--ds` the datamodule; the dataset's
num_classes / img_size / ignore indices are linked datamodule -> model. A
`MetricsCallback` (per-class IoU + sample preds) is added by default; the Lightning
logger is swappable via `--trainer.logger` (CSV by default in the configs).
"""

import csv
import json
import os
from pathlib import Path

import aquabench  # noqa: F401
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelSummary

from aquabench.module import TaskModule
from aquabench.data import HFSegDataModule
from aquabench.callbacks import MetricsCallback, PerSampleTestCallback


class BenchCLI(LightningCLI):
    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        try:
            sub = self.config[self.subcommand]
            if "WandbLogger" not in sub.trainer.logger.class_path:
                return
            if getattr(sub.trainer.logger.init_args, "name", None):
                return  # already set explicitly — respect it
            seed = getattr(sub, "seed_everything", 0)
            sub.trainer.logger.init_args.name = (
                f"{sub.ds}_{sub.encoder}_{sub.head}_{sub.task}_{seed}"
            )
        except AttributeError:
            pass

    def add_arguments_to_parser(self, parser):
        # Bare top-level axes -> model/data init args.
        parser.add_argument("--encoder", default="dinov3_base")
        parser.add_argument("--head", default="linear")
        parser.add_argument("--task", default="semantic_seg")
        parser.add_argument("--ds", default="cou")
        parser.link_arguments("encoder", "model.encoder")
        parser.link_arguments("head", "model.head")
        parser.link_arguments("task", "model.task")
        parser.link_arguments("ds", "data.ds")
        # dataset metadata -> model
        for src, dst in [
            ("data.num_classes", "model.num_classes"),
            ("data.img_size", "model.img_size"),
            ("data.loss_ignore_index", "model.loss_ignore_index"),
            ("data.metric_ignore_index", "model.metric_ignore_index"),
        ]:
            parser.link_arguments(src, dst, apply_on="instantiate")

    def _write_run_meta(self):
        """Write run_meta.json alongside the Lightning log dir for collect_results.py."""
        try:
            log_dir = self.trainer.log_dir or self.trainer.default_root_dir
            if log_dir is None:
                return
            sub = self.config[self.subcommand]
            meta = {
                "ds": sub.ds,
                "encoder": sub.encoder,
                "head": sub.head,
                "task": sub.task,
                "seed": getattr(sub, "seed_everything", None),
            }
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "run_meta.json"), "w") as f:
                json.dump(meta, f)
        except Exception:
            pass

    def before_test(self):
        seed = getattr(self.config[self.subcommand], "seed_everything", None)
        for cb in self.trainer.callbacks:
            if isinstance(cb, PerSampleTestCallback):
                cb.seed = seed

    def after_fit(self):
        self._write_run_meta()

    def after_test(self):
        self._write_run_meta()
        self._write_summary_csv()

    def _write_summary_csv(self):
        try:
            sub = self.config[self.subcommand]
            metrics = self.trainer.callback_metrics
            miou = metrics.get("test/miou")
            miou_all = metrics.get("test/miou_all")
            row = {
                "ds": sub.ds,
                "backbone": sub.encoder,
                "head": sub.head,
                "seed": getattr(sub, "seed_everything", None),
                "test_miou": round(float(miou), 6) if miou is not None else "",
                "test_miou_all": round(float(miou_all), 6) if miou_all is not None else "",
            }
            fieldnames = ["ds", "backbone", "head", "seed", "test_miou", "test_miou_all"]
            out_path = Path(os.environ.get("RESULTS_DIR", "results")) / "analysis" / "summary.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not out_path.exists()
            with open(out_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            pass


def main():
    BenchCLI(
        TaskModule,
        HFSegDataModule,
        trainer_defaults={"callbacks": [MetricsCallback(), PerSampleTestCallback(), ModelSummary(max_depth=2)]},
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
