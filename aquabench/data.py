"""Datasets on disk, token-grid sizing and augmentation.

On-disk layout (written by ``aquabench download``)::

    <data_root>/<Dataset>/
        meta.json                       # num_classes, ignore_index, size, class_names
        train/images/<id>.png   train/masks/<id>.png
        val/images/...          val/masks/...
        test/images/...         test/masks/...

``meta.json["size"]`` is the dataset's target size ``[H, W]`` at patch size 16, so it
must be divisible by 16. The token grid is ``size / 16``; the encoder input is
``grid * patch_size`` (patch 14 -> smaller input, same token count). Masks always stay
at ``size`` so metrics are computed at the same resolution for every patch size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

#: The patch size the dataset sizes in meta.json are defined for.
REFERENCE_PATCH_SIZE = 16

#: Canonical dataset name -> config name on the Hugging Face hub.
DATASETS: Dict[str, str] = {
    "CoralScapes": "coralscapes",
    "COU": "cou",
    "DeepFish": "deepfish",
    "L4S": "l4s",
    "LIACi": "liaci",
    "SeaClear": "seaclear",
    "SUIM": "suim",
    "TrashCan": "trashcan",
    "UIIS10K": "uiis10k",
}
#: v1 short names that differ from the hub config names.
ALIASES: Dict[str, str] = {"cs": "CoralScapes", "df": "DeepFish", "sc": "SeaClear", "tc": "TrashCan"}
#: Expected size ``(H, W)``, class count and samples per split of each benchmark dataset, as on
#: the hub. Other folder names are not checked.
DATASET_SPECS: Dict[str, dict] = {
    "CoralScapes": {"size": (512, 1024), "num_classes": 39, "samples": {"train": 1517, "val": 166, "test": 392}},
    "COU": {"size": (576, 1024), "num_classes": 24, "samples": {"train": 6753, "val": 1952, "test": 958}},
    "DeepFish": {"size": (576, 1024), "num_classes": 2, "samples": {"train": 310, "val": 124, "test": 186}},
    "L4S": {"size": (576, 1024), "num_classes": 2, "samples": {"train": 4223, "val": 610, "test": 1204}},
    "LIACi": {"size": (576, 1024), "num_classes": 11, "samples": {"train": 1233, "val": 137, "test": 191}},
    "SeaClear": {"size": (576, 1024), "num_classes": 41, "samples": {"train": 6071, "val": 674, "test": 1865}},
    "SUIM": {"size": (480, 640), "num_classes": 8, "samples": {"train": 1297, "val": 228, "test": 110}},
    "TrashCan": {"size": (576, 1024), "num_classes": 23, "samples": {"train": 5459, "val": 606, "test": 1147}},
    "UIIS10K": {"size": (480, 640), "num_classes": 11, "samples": {"train": 7234, "val": 804, "test": 2010}},
}

_REDOWNLOAD = "Wrong or outdated data? Re-download with `aquabench download --force`."

SPLITS = ("train", "val", "test")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
META_KEYS = ("num_classes", "ignore_index", "size")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: "imagenet", "none" ([0, 1] images), or a callable mapping a [3, H, W] image in [0, 1] to the
#: encoder's input. With DataLoader workers it must be picklable (a module-level function or an
#: object such as torchvision's Normalize).
Normalize = Union[str, Callable[[torch.Tensor], torch.Tensor]]

#: Zoom-in range for augmentation. Scales < 1 are never used, so nothing is padded.
AUGMENT_SCALE = (1.0, 1.5)


# --------------------------------------------------------------------------- meta / sizes

def load_meta(ds_dir: Union[str, Path]) -> dict:
    """Read and validate ``meta.json`` of one dataset folder."""
    path = Path(ds_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `aquabench download --data-root <path>` first.")
    meta = json.loads(path.read_text())
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"{path} is missing keys {missing}")
    h, w = meta["size"]
    if h % REFERENCE_PATCH_SIZE or w % REFERENCE_PATCH_SIZE:
        raise ValueError(f"{path}: size {meta['size']} is not divisible by {REFERENCE_PATCH_SIZE}")
    spec = _spec(ds_dir)
    if spec:
        canonical = canonical_name(Path(ds_dir).name)
        if (h, w) != spec["size"]:
            raise ValueError(f"{path}: {canonical} must have size {list(spec['size'])}, got {meta['size']}. {_REDOWNLOAD}")
        if meta["num_classes"] != spec["num_classes"]:
            raise ValueError(f"{path}: {canonical} must have {spec['num_classes']} classes, "
                             f"got {meta['num_classes']}. {_REDOWNLOAD}")
    meta.setdefault("name", Path(ds_dir).name)
    return meta


def _spec(ds_dir: Union[str, Path]) -> Optional[dict]:
    """Expected properties if ``ds_dir`` is a benchmark dataset, else None."""
    return DATASET_SPECS.get(canonical_name(Path(ds_dir).name))


def check_dataset(ds_dir: Union[str, Path]) -> dict:
    """``load_meta`` plus, for a benchmark dataset, the sample count of every split."""
    meta = load_meta(ds_dir)
    spec = _spec(ds_dir)
    for split, expected in (spec["samples"] if spec else {}).items():
        sd = Path(ds_dir) / split
        found = len(list_pairs(sd)) if (sd / "images").is_dir() and (sd / "masks").is_dir() else 0
        if found != expected:
            raise ValueError(f"{sd}: expected {expected} samples, found {found}. {_REDOWNLOAD}")
    return meta


def token_grid(size: Sequence[int]) -> Tuple[int, int]:
    """Token grid ``(gh, gw)`` of a dataset size defined at patch 16."""
    h, w = size
    if h % REFERENCE_PATCH_SIZE or w % REFERENCE_PATCH_SIZE:
        raise ValueError(f"size {tuple(size)} is not divisible by {REFERENCE_PATCH_SIZE}")
    return h // REFERENCE_PATCH_SIZE, w // REFERENCE_PATCH_SIZE


def input_size(size: Sequence[int], patch_size: int) -> Tuple[int, int]:
    """Encoder input size: same token grid as at patch 16, rescaled to ``patch_size``."""
    gh, gw = token_grid(size)
    return gh * patch_size, gw * patch_size


# --------------------------------------------------------------------------- dataset lookup

def canonical_name(name: str) -> Optional[str]:
    """Canonical dataset name for a canonical, hub config or v1 short name (case-insensitive)."""
    n = name.lower()
    for canonical, config in DATASETS.items():
        if n in (canonical.lower(), config):
            return canonical
    return ALIASES.get(n)


def find_dataset(data_root: Union[str, Path], name: str) -> Path:
    """Folder of dataset ``name`` (case-insensitive; hub config and v1 short names work too)."""
    root = Path(data_root)
    canonical = canonical_name(name)
    wanted = {name.lower()} | ({canonical.lower()} if canonical else set())
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name.lower() in wanted and (d / "meta.json").exists():
                return d
    raise FileNotFoundError(
        f"Dataset {name!r} not found in {root}. "
        f"Run `aquabench download --data-root {root} --datasets {name}`."
    )


def resolve_datasets(data_root: Union[str, Path], datasets: Union[str, Sequence[str]] = "all") -> List[Path]:
    """Dataset folders for ``"all"``, a single name, or a list of names."""
    if isinstance(datasets, str):
        datasets = list(DATASETS) if datasets == "all" else [datasets]
    return [find_dataset(data_root, n) for n in datasets]


def list_pairs(split_dir: Path) -> List[Tuple[Path, Path]]:
    """(image, mask) pairs of one split, matched by file stem."""
    images = {p.stem: p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS}
    masks = {p.stem: p for p in (split_dir / "masks").iterdir() if p.suffix.lower() == ".png"}
    if images.keys() != masks.keys():
        only_img = sorted(images.keys() - masks.keys())[:5]
        only_mask = sorted(masks.keys() - images.keys())[:5]
        raise ValueError(f"{split_dir}: unmatched images {only_img} / masks {only_mask}")
    return [(images[k], masks[k]) for k in sorted(images)]


def verify_data_root(data_root: Union[str, Path], datasets: Union[str, Sequence[str]] = "all",
                     check_masks: bool = True) -> List[str]:
    """Return a list of problems with the data root (empty list = all good)."""
    problems = []
    try:
        dirs = resolve_datasets(data_root, datasets)
    except FileNotFoundError as e:
        return [str(e)]
    for d in dirs:
        try:
            meta = load_meta(d)
        except (FileNotFoundError, ValueError) as e:
            problems.append(str(e))
            continue
        for split in SPLITS:
            sd = d / split
            if not (sd / "images").is_dir() or not (sd / "masks").is_dir():
                problems.append(f"{sd}: missing images/ or masks/")
                continue
            try:
                pairs = list_pairs(sd)
            except ValueError as e:
                problems.append(str(e))
                continue
            if not pairs:
                problems.append(f"{sd}: no samples")
            spec = _spec(d)
            if spec and len(pairs) != spec["samples"][split]:
                problems.append(f"{sd}: expected {spec['samples'][split]} samples, found {len(pairs)}")
            if check_masks:
                for _, mp in pairs:
                    bad = invalid_mask_values(np.array(Image.open(mp)), meta)
                    if bad:
                        problems.append(f"{mp}: values {bad} outside [0, {meta['num_classes']}) / ignore")
                        break
    return problems


def invalid_mask_values(mask: np.ndarray, meta: dict) -> List[int]:
    vals = np.unique(mask)
    ok = (vals < meta["num_classes"]) | (vals == meta["ignore_index"])
    return vals[~ok].tolist()


# --------------------------------------------------------------------------- transforms

def normalize_label(normalize: Normalize) -> str:
    """The JSON-safe name of a normalization, for ``config.json`` and checkpoints."""
    if callable(normalize):
        return f"callable:{getattr(normalize, '__qualname__', None) or repr(normalize)}"
    if normalize not in ("imagenet", "none"):
        raise ValueError(f"normalize must be 'imagenet', 'none' or a callable, got {normalize!r}")
    return normalize


def _resize_image(img: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if tuple(img.shape[-2:]) == tuple(size):
        return img
    return F.interpolate(img[None], size=tuple(size), mode="bilinear", align_corners=False, antialias=True)[0]


def _resize_mask(mask: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask
    return F.interpolate(mask[None, None].float(), size=tuple(size), mode="nearest-exact")[0, 0].long()


def augment_pair(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zoom-in only: random scale in [1.0, 1.5], random crop back to size, random h-flip.

    Output has exactly the input size; since the scale is >= 1 nothing is padded.
    Uses torch's RNG, so it is reproducible under a seeded DataLoader.
    """
    h, w = mask.shape
    lo, hi = AUGMENT_SCALE
    s = lo + (hi - lo) * torch.rand(()).item()
    nh, nw = max(h, round(h * s)), max(w, round(w * s))
    img, mask = _resize_image(img, (nh, nw)), _resize_mask(mask, (nh, nw))
    top = int(torch.randint(0, nh - h + 1, ()))
    left = int(torch.randint(0, nw - w + 1, ()))
    img, mask = img[:, top:top + h, left:left + w], mask[top:top + h, left:left + w]
    if torch.rand(()).item() < 0.5:
        img, mask = img.flip(-1), mask.flip(-1)
    return img, mask


class SegDataset(Dataset):
    """One split of one dataset. Returns ``(image [3, h, w] float, mask [H, W] long)``.

    The mask is at the dataset size from meta.json; the image is at
    ``input_size(size, patch_size)``.
    """

    def __init__(self, ds_dir: Union[str, Path], split: str, patch_size: int = 16,
                 augment: bool = False, normalize: Normalize = "imagenet"):
        self.dir = Path(ds_dir)
        self.meta = load_meta(self.dir)
        self.size = tuple(self.meta["size"])
        self.input_size = input_size(self.size, patch_size)
        self.augment = augment
        normalize_label(normalize)  # validates
        self.normalize = normalize
        self.pairs = list_pairs(self.dir / split)
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        ip, mp = self.pairs[i]
        img = torch.from_numpy(np.array(Image.open(ip).convert("RGB"), dtype=np.uint8))
        img = img.permute(2, 0, 1).float().div_(255.0)
        mask = torch.from_numpy(np.array(Image.open(mp), dtype=np.int64))
        img, mask = _resize_image(img, self.size), _resize_mask(mask, self.size)
        if self.augment:
            img, mask = augment_pair(img, mask)
        img = _resize_image(img, self.input_size)
        if callable(self.normalize):
            out = self.normalize(img)
            if not isinstance(out, torch.Tensor) or out.shape != img.shape:
                raise ValueError(f"normalize callable must return a tensor of shape {list(img.shape)}, "
                                 f"got {list(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__}")
            img = out
        elif self.normalize == "imagenet":
            img = (img - self._mean) / self._std
        return img, mask


def has_split(ds_dir: Union[str, Path], split: str) -> bool:
    d = Path(ds_dir) / split
    return (d / "images").is_dir() and any((d / "images").iterdir())


def maybe_dataset(ds_dir, split, **kw) -> Optional[SegDataset]:
    return SegDataset(ds_dir, split, **kw) if has_split(ds_dir, split) else None
