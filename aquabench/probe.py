#!/usr/bin/env python3
"""
Fast sklearn linear probe: frozen encoder (last layer) -> patch embeddings ->
LogisticRegression at patch resolution. The cheap sibling to the torch linear head.

    python -m aquabench.probe --ds l4s --encoder dinov3_base
    # -> results/probe_summary.csv (+ per-class JSON)

Memory: patch embeddings are streamed to **disk** as per-batch shards under
`--cache-dir` (default `<results-dir>/probe_cache/<ds>__<encoder>/`), so they never
all sit in RAM. The sklearn fit pulls at most `--sklearn-max-train` rows into RAM;
val/test are scored by streaming a confusion matrix shard-by-shard. The cache is
deleted afterwards unless `--keep-cache`.

mIoU is reported both excluding `metric_ignore_index` (background — the headline
number) and including it (`miou_all`).

The fitted probe is always saved to `results/<ds>__<encoder>__probe.joblib` (clf +
normalization stats + metadata) so test inference can be re-run later. Pass `--wandb`
to also log metrics to Weights & Biases (the probe itself is never uploaded); on an
offline/air-gapped cluster set `WANDB_MODE=offline` (or `disabled`).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _subsample(labels: np.ndarray, cap: Optional[int], ignore_index: int, seed: int) -> np.ndarray:
    """Indices of non-ignored patches, optionally capped to `cap` (random)."""
    valid = np.flatnonzero(labels != ignore_index)
    if cap is not None and valid.size > cap:
        valid = np.sort(np.random.default_rng(seed).choice(valid, size=cap, replace=False))
    return valid


def _encode_split(encoder, loader, patch, grid, ignore, mean, std, device, cap, seed, out_dir) -> Tuple[int, int]:
    """Encode a split to per-batch .npz shards on disk; return (n_patches, feat_dim).

    The image is resized so the encoder emits exactly `grid=(gh, gw)` tokens
    (matched across encoders, no padding); labels are downsampled to that grid.
    """
    import torch
    import torch.nn.functional as F
    from aquabench.data.tokens import resize_to_tokens

    out_dir.mkdir(parents=True, exist_ok=True)
    n_total, dim, seen, shard = 0, 0, 0, 0
    for batch in loader:
        images = batch["pixel_values"].to(device)
        masks = batch["labels"].to(device)
        images = (images - mean) / std
        images = resize_to_tokens(images, patch, grid)
        with torch.no_grad():
            fmap = encoder(images).feature_maps[-1]  # [B,C,gh,gw]
        gh, gw = fmap.shape[-2:]
        fmap = fmap.permute(0, 2, 3, 1).reshape(images.shape[0], gh * gw, -1).float().cpu().numpy()
        # downsample the original labels to the patch grid (no padding tokens)
        plab = F.interpolate(masks.unsqueeze(1).float(), size=(gh, gw), mode="nearest").squeeze(1)
        plab = plab.reshape(images.shape[0], gh * gw).long().cpu().numpy()

        Xs, ys = [], []
        for b in range(images.shape[0]):
            keep = _subsample(plab[b], cap, ignore, seed + seen + b)
            if keep.size:
                Xs.append(fmap[b, keep].astype(np.float16))
                ys.append(plab[b, keep].astype(np.int16))
        seen += images.shape[0]
        if Xs:
            X, y = np.concatenate(Xs), np.concatenate(ys)
            np.savez(out_dir / f"{shard:05d}.npz", X=X, y=y)
            shard, n_total, dim = shard + 1, n_total + y.shape[0], X.shape[1]
    return n_total, dim


def _load_train_subsample(shard_dir: Path, max_train: int, seed: int):
    """Load train shards into RAM, keeping at most ~max_train rows (bounded fit memory)."""
    shards = sorted(shard_dir.glob("*.npz"))
    total = sum(int(np.load(f)["y"].shape[0]) for f in shards)
    keep_p = 1.0 if (not max_train or total <= max_train) else max_train / total
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for f in shards:
        d = np.load(f)
        X, y = d["X"], d["y"]
        if keep_p < 1.0:
            m = rng.random(y.shape[0]) < keep_p
            X, y = X[m], y[m]
        if y.shape[0]:
            Xs.append(X.astype(np.float32))
            ys.append(y.astype(np.int64))
    if not Xs:
        return np.zeros((0, 1), np.float32), np.zeros((0,), np.int64)
    return np.concatenate(Xs), np.concatenate(ys)


def _confusion_stream(clf, shard_dir: Path, mu, sd, num_classes: int) -> np.ndarray:
    """Predict each val/test shard and accumulate a confusion matrix (streamed, low RAM)."""
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for f in sorted(shard_dir.glob("*.npz")):
        d = np.load(f)
        X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
        if X.shape[0] == 0:
            continue
        pred = clf.predict((X - mu) / sd)
        k = y * num_classes + pred
        conf += np.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf


def _ious_from_confusion(conf: np.ndarray, ignore_index: int):
    """Return (per_class_iou dict over present classes, miou_excl_bg, miou_all, acc)."""
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - inter
    present = union > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(present, inter / np.maximum(union, 1), np.nan)
    miou_all = float(np.nanmean(iou[present])) if present.any() else 0.0
    mask = present.copy()
    if 0 <= ignore_index < mask.size:
        mask[ignore_index] = False
    miou = float(np.nanmean(iou[mask])) if mask.any() else 0.0
    acc = float(inter.sum() / max(1, conf.sum()))
    per_class = {int(c): float(iou[c]) for c in np.flatnonzero(present)}
    return per_class, miou, miou_all, acc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ds", required=True)
    p.add_argument("--encoder", default="dinov3_base")
    p.add_argument("--data-dir", default=os.path.join(os.environ.get("DATA_ROOT", "/workspace/data"), "unified"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-patches-per-image", type=int, default=128)
    p.add_argument("--sklearn-max-train", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cache-dir", default="", help="where embedding shards are written (default <results-dir>/probe_cache)")
    p.add_argument("--keep-cache", action="store_true", help="keep the on-disk embedding shards after the run")
    p.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases (the fitted probe is never uploaded)")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "aquabench"))
    p.add_argument("--wandb-entity", default="", help="W&B entity (default: your wandb default)")
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression

    from aquabench.models.encoders import build_encoder
    from aquabench.data import _HFSegSplit, _collate

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=f"{args.ds}_{args.encoder}_sklearn_probe_{args.seed}",
            config={"ds": args.ds, "encoder": args.encoder, "seed": args.seed,
                    "sklearn_max_train": args.sklearn_max_train,
                    "train_patches_per_image": args.train_patches_per_image, "max_iter": 300},
        )

    meta = json.loads((Path(args.data_dir) / "_metadata" / f"{args.ds}.json").read_text())
    num_classes = meta["num_classes"]
    img_size = tuple(meta["default_img_size"])
    ignore = meta.get("metric_ignore_index", 0)
    loss_ignore = meta.get("loss_ignore_index", 255) or 255
    id2label = meta.get("id2label", {})

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    encoder = build_encoder(args.encoder, out_indices="last", scales=[1.0],
                            use_lateral_convs=False, out_channels=None, img_size=img_size)
    encoder.eval().to(device)
    for prm in encoder.parameters():
        prm.requires_grad = False
    patch = int(getattr(encoder, "patch_size", 16))
    from aquabench.data.tokens import target_token_grid
    grid = target_token_grid(img_size)  # shared token count across encoders
    print(f"[{args.ds}/{args.encoder}] patch={patch} target token grid={grid[0]}x{grid[1]}")
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    cache_root = Path(args.cache_dir or (Path(args.results_dir) / "probe_cache")) / f"{args.ds}__{args.encoder}"
    if cache_root.exists():
        shutil.rmtree(cache_root)

    dsd = load_dataset(args.data_dir, args.ds)
    split_map = {"train": "train", "val": "validation", "test": "test"}
    present: Dict[str, Path] = {}
    for key, hf_split in split_map.items():
        if hf_split not in dsd:
            continue
        loader = DataLoader(_HFSegSplit(dsd[hf_split]), batch_size=args.batch_size,
                            num_workers=args.num_workers, collate_fn=_collate)
        cap = args.train_patches_per_image if key == "train" else None
        out_dir = cache_root / key
        t0 = time.time()
        n, _ = _encode_split(encoder, loader, patch, grid, loss_ignore, mean, std, device, cap, args.seed, out_dir)
        present[key] = out_dir
        print(f"[{args.ds}/{args.encoder}] {key}: {n} patches -> {out_dir} ({time.time()-t0:.0f}s)")

    Xtr, ytr = _load_train_subsample(present["train"], args.sklearn_max_train, args.seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    clf = LogisticRegression(max_iter=300)
    t0 = time.time()
    clf.fit((Xtr - mu) / sd, ytr)

    run = {"ds": args.ds, "encoder": args.encoder, "seconds": round(time.time() - t0, 1),
           "n_train": int(Xtr.shape[0])}
    for key in ("val", "test"):
        if key not in present:
            continue
        conf = _confusion_stream(clf, present[key], mu, sd, num_classes)
        per_class, miou, miou_all, acc = _ious_from_confusion(conf, ignore)
        run[key] = {"miou": miou, "miou_all": miou_all, "acc": acc,
                    "per_class_iou": {id2label.get(str(c), id2label.get(c, f"class_{c}")): round(v, 4)
                                      for c, v in per_class.items()}}
        print(f"[{args.ds}/{args.encoder}] {key} mIoU={miou:.4f} (all={miou_all:.4f}) acc={acc:.4f}")

    if not args.keep_cache:
        shutil.rmtree(cache_root, ignore_errors=True)

    rd = Path(args.results_dir)
    rd.mkdir(parents=True, exist_ok=True)

    # Persist the fitted probe so test inference can be re-run later for analysis.
    # Self-contained bundle: everything _confusion_stream / a fresh predict needs.
    import joblib
    model_path = rd / f"{args.ds}__{args.encoder}__probe.joblib"
    joblib.dump({
        "clf": clf, "mu": mu, "sd": sd,
        "num_classes": num_classes, "ignore_index": ignore, "id2label": id2label,
        "encoder": args.encoder, "ds": args.ds, "seed": args.seed,
        "patch_size": patch, "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD,
    }, model_path)
    run["model_path"] = str(model_path)
    print(f"-> {model_path}")

    summary = rd / "probe_summary.csv"
    new = not summary.exists()
    fields = ["timestamp", "ds", "encoder", "val_miou", "val_miou_all", "test_miou", "test_miou_all", "n_train", "seconds"]
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"), "ds": args.ds, "encoder": args.encoder,
            "val_miou": round(run.get("val", {}).get("miou", float("nan")), 4),
            "val_miou_all": round(run.get("val", {}).get("miou_all", float("nan")), 4),
            "test_miou": round(run.get("test", {}).get("miou", float("nan")), 4),
            "test_miou_all": round(run.get("test", {}).get("miou_all", float("nan")), 4),
            "n_train": run["n_train"], "seconds": run["seconds"],
        })
    (rd / f"{args.ds}__{args.encoder}__probe.json").write_text(json.dumps(run, indent=2))
    print(f"-> {summary}")

    if wb is not None:
        # Metrics only — the fitted probe (results/*.joblib) is never uploaded.
        metrics = {"n_train": run["n_train"], "fit/seconds": run["seconds"]}
        for key in ("val", "test"):
            if key not in run:
                continue
            metrics[f"{key}/miou"] = run[key]["miou"]
            metrics[f"{key}/miou_all"] = run[key]["miou_all"]
            metrics[f"{key}/acc"] = run[key]["acc"]
            for name, v in run[key]["per_class_iou"].items():
                metrics[f"{key}_iou/{name}"] = v
        wb.log(metrics)
        wb.finish()


if __name__ == "__main__":
    main()
