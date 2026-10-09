"""Smoke test for the V-JEPA 2.1 backbone (real pretrained weights).

Loads a V-JEPA 2.1 encoder, runs the full frozen-backbone linear-probe path
(DenseBackbone -> 1x1 head -> upsample -> mIoU) on a dummy batch, and checks the
spatial grid + that only the head is trainable. Downloads the checkpoint on first
run (cached under $VJEPA_CACHE_DIR or <repo>/checkpoints/vjepa2/).

    # lightweight ViT-B (80M, distilled) — fast smoke
    python scripts/smoke_vjepa.py
    # the 2B ViT-G (large download)
    python scripts/smoke_vjepa.py --encoder vjepa21_vitg
    # mimic a specific dataset's default image size
    python scripts/smoke_vjepa.py --img-size 540 960
"""

import argparse

import torch

from aquabench.models.encoders import build_encoder
from aquabench.models.heads.linear import LinearProbingSegmentationModel
from aquabench.data.tokens import target_token_grid
from aquabench.tasks import TASKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vjepa21_vitb", choices=["vjepa21_vitb", "vjepa21_vitg"])
    ap.add_argument("--img-size", type=int, nargs=2, default=[540, 960], metavar=("H", "W"))
    ap.add_argument("--num-classes", type=int, default=5)
    args = ap.parse_args()

    img_size = tuple(args.img_size)
    grid = target_token_grid(img_size)
    print(f"encoder={args.encoder} img_size={img_size} token_grid={grid}")

    # Linear-probe build: 1 level, native channel width (no lateral convs), last layer.
    enc = build_encoder(args.encoder, None, [1.0], False, None, img_size)
    print(f"loaded: out_channels={enc.out_channels} patch_size={enc.patch_size} "
          f"num_special_tokens={enc.num_special_tokens}")

    model = LinearProbingSegmentationModel(
        enc, num_classes=args.num_classes, hidden_size=enc.out_channels, token_grid=grid
    )
    model.train()  # head trainable, backbone frozen + eval

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("classifier" in n for n in trainable), \
        f"only the 1x1 head must be trainable, got: {trainable}"
    print(f"trainable params (head only): {trainable}")

    task = TASKS["semantic_seg"](num_classes=args.num_classes, ignore_index=0, loss_ignore_index=255)
    batch = {
        "pixel_values": torch.randn(2, 3, *img_size),
        "labels": torch.randint(0, args.num_classes, (2, *img_size)),
    }
    loss, preds, target = task.step(model, batch)
    assert preds.shape == (2, *img_size), preds.shape

    metric = task.build_metric()
    metric.update(preds.flatten(), target.flatten())
    print(f"loss={float(loss):.4f} preds={tuple(preds.shape)} (== label res) "
          f"miou(random head)={float(metric.compute()):.4f}")
    print("SMOKE OK ✅")


if __name__ == "__main__":
    main()
