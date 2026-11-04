"""Example: benchmark a DINOv2 encoder, optionally with your own weights file.

The architecture (and public pretrained weights) come from ``torch.hub``, no login
needed. ``--weights`` loads a checkpoint with ``torch.load`` on top, e.g. a domain-adapted
or fine-tuned DINOv2. AQUABENCH itself never loads models; this wrapping is the user side.

    python examples/run_dinov2.py --data-root /data/aquabench --run-type linear --datasets SUIM
    python examples/run_dinov2.py --data-root /data/aquabench --model dinov2_vitb14 \
        --weights my_adapted_vitb14.pt --run-type upernet_frozen --seeds 0 1 2
"""

import argparse

import torch
import torch.nn as nn

import aquabench


class LastLayer(nn.Module):
    """linear: one feature map [B, C, h, w] from the last block."""

    def __init__(self, vit, name):
        super().__init__()
        self.vit, self.name = vit, name

    def forward(self, x):
        return self.vit.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]


class FourLayers(nn.Module):
    """upernet_*: four feature maps from evenly spaced blocks (e.g. 2, 5, 8, 11 for depth 12)."""

    def __init__(self, vit, name):
        super().__init__()
        self.vit, self.name = vit, name
        step = len(vit.blocks) // 4
        self.layers = [step * (i + 1) - 1 for i in range(4)]

    def forward(self, x):
        return list(self.vit.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True))


def load_weights(vit: nn.Module, path: str) -> None:
    """Load a state dict saved with torch.save. Unwraps {"model": ...} / {"state_dict": ...} /
    {"teacher": ...} and strips "module." / "backbone." prefixes."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "teacher"):
        if isinstance(sd, dict) and isinstance(sd.get(key), dict):
            sd = sd[key]
            break
    for prefix in ("module.", "backbone."):
        if all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    print(f"loaded {path}: {len(sd)} tensors, missing {missing}, unexpected {unexpected}")
    if missing:
        raise RuntimeError(f"{len(missing)} encoder weights missing from {path}")


def build_encoder(model: str, run_type: str, weights=None) -> nn.Module:
    vit = torch.hub.load("facebookresearch/dinov2", model, pretrained=weights is None)
    name = model
    if weights:
        load_weights(vit, weights)
        name = f"{model}_{weights.rsplit('/', 1)[-1].rsplit('.', 1)[0]}"
    wrap = LastLayer if run_type == "linear" else FourLayers
    return wrap(vit, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", default="dinov2_vits14",
                   choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    p.add_argument("--weights", default=None, help="state dict file to load with torch.load")
    p.add_argument("--run-type", default="linear", choices=aquabench.RUN_TYPES)
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=None, help="override the protocol (for quick tests only)")
    args = p.parse_args()

    config = {"epochs": args.epochs} if args.epochs else {}
    if not torch.cuda.is_available():
        config["precision"] = "fp32"  # bf16 autocast is very slow on most CPUs
    for seed in args.seeds:
        # a fresh encoder per seed: upernet_finetune changes it in place
        encoder = build_encoder(args.model, args.run_type, args.weights)
        results = aquabench.run(
            encoder, run_type=args.run_type, seed=seed, data_root=args.data_root,
            datasets="all" if args.datasets == ["all"] else args.datasets,
            patch_size=encoder.vit.patch_size, config=config,
        )
        for name, r in results.items():
            print(f"{encoder.name} {args.run_type} seed {seed} {name}: test mIoU {r.test_miou:.4f}")


if __name__ == "__main__":
    main()
