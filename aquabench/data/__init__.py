"""HF-only segmentation DataModule.

Loads one config of the unified parquet dataset (built by the conversion project)
with `datasets.load_dataset` — no shared-memory cache, no preprocessor pipeline.
Augmentation + ImageNet normalization happen on-device in `on_after_batch_transfer`.
`num_classes` / `img_size` / ignore indices come from `_metadata/<ds>.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

#: Default root for the unified parquet dataset; overridable via $DATA_ROOT so the
#: same configs work across machines (matches aquabench.probe's --data-dir default).
DEFAULT_DATA_DIR = os.path.join(os.environ.get("DATA_ROOT", "/workspace/data"), "unified")

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_HF_SPLITS = {"train": "train", "val": "validation", "test": "test"}


class _HFSegSplit(Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = np.array(ex["image"].convert("RGB"), dtype=np.uint8)
        mask = np.array(ex["mask"], dtype=np.int64)
        return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0), torch.from_numpy(mask)


def _collate(batch):
    return {
        "pixel_values": torch.stack([b[0] for b in batch], 0),
        "labels": torch.stack([b[1] for b in batch], 0),
    }


class HFSegDataModule(LightningDataModule):
    def __init__(
        self,
        ds: str = "cou",
        data_dir: str = DEFAULT_DATA_DIR,
        batch_size: int = 8,
        num_workers: int = 4,
        augmentation: str = "medium",
    ):
        super().__init__()
        self.ds_name = ds
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation

        meta_path = Path(data_dir) / "_metadata" / f"{ds}.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing {meta_path}. Build it first: (in ../conversion) "
                f"python -m conversion.build --configs {ds}"
            )
        meta = json.loads(meta_path.read_text())
        self.num_classes: int = meta["num_classes"]
        self.img_size = tuple(meta["default_img_size"])
        self.loss_ignore_index: int = meta.get("loss_ignore_index", 255) or 255
        self.metric_ignore_index: int = meta.get("metric_ignore_index", 0)
        self.id2label: Dict = meta.get("id2label", {})

        self._splits: Dict[str, object] = {}
        self._aug = None
        self._mean = None
        self._std = None

    def setup(self, stage: Optional[str] = None):
        from datasets import load_dataset

        ds = load_dataset(self.data_dir, self.ds_name)
        self._splits = {k: ds[v] for k, v in _HF_SPLITS.items() if v in ds}
        from aquabench.data.augment import SegAugmentation, PRESETS
        if PRESETS.get(self.augmentation):
            self._aug = SegAugmentation(img_size=self.img_size, preset=self.augmentation)

    def _loader(self, split, shuffle):
        return DataLoader(
            _HFSegSplit(self._splits[split]), batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, collate_fn=_collate, pin_memory=True,
            persistent_workers=self.num_workers > 0, drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False)

    def test_dataloader(self):
        return self._loader("test", False)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        pv, labels = batch["pixel_values"], batch["labels"]
        if self._aug is not None and getattr(self, "trainer", None) is not None and self.trainer.training:
            self._aug = self._aug.to(pv.device)
            pv, labels = self._aug(pv, labels)
            pv = pv.clamp(0, 1)
        if self._mean is None or self._mean.device != pv.device:
            self._mean = torch.tensor(IMAGENET_MEAN, device=pv.device).view(1, 3, 1, 1)
            self._std = torch.tensor(IMAGENET_STD, device=pv.device).view(1, 3, 1, 1)
        batch["pixel_values"] = (pv - self._mean) / self._std
        batch["labels"] = labels
        return batch
