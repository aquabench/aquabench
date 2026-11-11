"""Download the benchmark and write it in the plain on-disk layout of :mod:`aquabench.data`.

The source is the unified parquet build on the Hugging Face hub
(https://huggingface.co/datasets/TorbenGl/AQUABENCH: one config per dataset with frozen
train / validation / test splits, ``_metadata/<config>.json`` and
``<config>/dataset_license.txt``), or a local copy of it. Reading parquet needs the optional ``datasets``
package (``pip install 'aquabench[download]'``); training never does.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from aquabench.data import DATASETS, REFERENCE_PATCH_SIZE, canonical_name

#: Hugging Face dataset repo of the unified build.
HF_REPO_ID = "TorbenGl/AQUABENCH"

_SOURCE_SPLITS = {"train": "train", "val": "validation", "test": "test"}


def convert_meta(name: str, src: dict) -> dict:
    """v1 unified metadata -> meta.json. The size is snapped to a multiple of 16,
    the same token grid v1 used (``round(side / 16)``)."""
    h, w = src["default_img_size"]
    size = [max(1, round(s / REFERENCE_PATCH_SIZE)) * REFERENCE_PATCH_SIZE for s in (h, w)]
    id2label = src.get("id2label") or {}
    class_names = [id2label.get(str(i), id2label.get(i, str(i))) for i in range(src["num_classes"])]
    return {
        "name": name,
        "num_classes": src["num_classes"],
        "ignore_index": src.get("loss_ignore_index") or 255,
        "size": size,
        "class_names": class_names,
    }


def write_dataset(out_dir: Union[str, Path], meta: dict,
                  splits: Mapping[str, Iterable[Tuple[Image.Image, Union[Image.Image, np.ndarray]]]]) -> Path:
    """Write ``meta.json`` and ``<split>/{images,masks}/<id>.png``. Written to a
    ``.partial`` folder first and renamed at the end, so a crash leaves no half dataset."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    for split, samples in splits.items():
        (tmp / split / "images").mkdir(parents=True)
        (tmp / split / "masks").mkdir(parents=True)
        for i, (img, mask) in enumerate(samples):
            mask = np.asarray(mask)
            if mask.max(initial=0) > 255:
                raise ValueError(f"{meta['name']}/{split}/{i}: mask value > 255 cannot be stored as uint8 PNG")
            img.convert("RGB").save(tmp / split / "images" / f"{i:06d}.png")
            Image.fromarray(mask.astype(np.uint8)).save(tmp / split / "masks" / f"{i:06d}.png")
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    return out_dir


def _source_file(source: str, rel: str) -> Optional[Path]:
    """``rel`` from a local unified dir or the hub repo; None if it does not exist."""
    local = Path(source) / rel
    if local.exists():
        return local
    if Path(source).is_dir():
        return None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        return Path(hf_hub_download(source, rel, repo_type="dataset"))
    except EntryNotFoundError:
        return None


def download(data_root: Union[str, Path], datasets: Union[str, Sequence[str]] = "all",
             source: Optional[str] = None, force: bool = False) -> None:
    """Fetch ``datasets`` from ``source`` (HF repo id or local unified dir) into ``data_root``."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Downloading needs the optional `datasets` package: "
                          "pip install 'aquabench[download]'") from e
    source = source or HF_REPO_ID

    names = list(DATASETS) if datasets == "all" else ([datasets] if isinstance(datasets, str) else list(datasets))
    unknown = [n for n in names if canonical_name(n) is None]
    if unknown:
        raise ValueError(f"Unknown datasets {unknown}; choose from {list(DATASETS)}")

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    for n in names:
        name = canonical_name(n)
        out = root / name
        if (out / "meta.json").exists() and not force:
            print(f"{name}: already present, skipping (use --force to re-download)")
            continue
        config = DATASETS[name]
        print(f"{name}: downloading config {config!r} from {source}")
        meta_file = _source_file(source, f"_metadata/{config}.json")
        if meta_file is None:
            raise FileNotFoundError(f"_metadata/{config}.json not found in {source}")
        meta = convert_meta(name, json.loads(meta_file.read_text()))
        hf = load_dataset(source, config)
        splits = {
            split: ((ex["image"], ex["mask"]) for ex in hf[src])
            for split, src in _SOURCE_SPLITS.items() if src in hf
        }
        write_dataset(out, meta, splits)
        license_file = _source_file(source, f"{config}/dataset_license.txt")
        if license_file is not None:
            shutil.copyfile(license_file, out / "dataset_license.txt")
        print(f"{name}: written to {out}")
