"""Example user script: prepare DINOv3 encoders, bring a W&B logger, loop over seeds.

AQUABENCH does not load models or manage sweeps; all of that lives in scripts like this.
Requires (not AQUABENCH dependencies): the DINOv3 repo via torch.hub, and wandb.

    python examples/dinov3_sweep.py --data-root /data/aquabench --run-type linear
"""

import argparse

import torch
import torch.nn as nn

import aquabench


class DINOv3Last(nn.Module):
    """linear: one feature map [B, C, h, w] from the last layer."""

    def __init__(self, m, name):
        super().__init__()
        self.m, self.name = m, name

    def forward(self, x):
        return self.m.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]


class DINOv3Pyramid(nn.Module):
    """upernet_*: four feature maps, layers chosen here (not by AQUABENCH)."""

    def __init__(self, m, name, layers=(2, 5, 8, 11)):
        super().__init__()
        self.m, self.name, self.layers = m, name, layers

    def forward(self, x):
        return list(self.m.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True))


class WandbLogger:
    def __init__(self, run):
        self.run = run

    def log(self, metrics, step):
        self.run.log(metrics, step=step)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-type", default="linear", choices=aquabench.RUN_TYPES)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--datasets", nargs="+", default=["all"])
    args = p.parse_args()

    import wandb

    for seed in args.seeds:
        # fresh pretrained weights per seed (fine-tuning modifies the encoder in place)
        # see the DINOv3 README for how to pass the weights (weights=<url or path>)
        m = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16", pretrained=True)
        wrap = DINOv3Last if args.run_type == "linear" else DINOv3Pyramid
        encoder = wrap(m, name="dinov3_vitb16")
        run = wandb.init(project="aquabench", name=f"dinov3_vitb16_{args.run_type}_seed{seed}")
        results = aquabench.run(
            encoder, logger=WandbLogger(run), run_type=args.run_type, seed=seed,
            data_root=args.data_root,
            datasets="all" if args.datasets == ["all"] else args.datasets,
            patch_size=16,
        )
        run.finish()
        for name, r in results.items():
            print(f"seed {seed} {name}: test mIoU {r.test_miou:.4f}")


if __name__ == "__main__":
    main()
