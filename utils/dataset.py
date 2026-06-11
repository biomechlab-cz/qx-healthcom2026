"""Torch Dataset wrappers for chromosome crops.

`CropDataset` — one chromosome per item (instance-level).
`BagDataset`  — one image (full metaphase) per item, returning the stack of all its crops.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.config import PATHS


def _read_crops_manifest() -> list[dict]:
    with PATHS.crops_manifest.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_fold(fold: int) -> tuple[list[str], list[str]]:
    """Return (train_image_ids, val_image_ids) for fold k."""
    payload = json.loads((PATHS.splits / f"fold_{fold}.json").read_text())
    return payload["train"], payload["val"]


def load_instance_split(fold: int) -> tuple[list[str], list[str]]:
    """Return (instance_train_crop_ids, instance_val_crop_ids) for the per-fold pretraining split."""
    payload = json.loads((PATHS.splits / "instance_pretrain.json").read_text())
    p = payload["per_fold"][f"fold_{fold}"]
    return p["instance_train"], p["instance_val"]


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class CropDataset(Dataset):
    """Per-chromosome dataset. __getitem__ returns (image_chw_tensor, label_int, crop_id)."""

    def __init__(
        self,
        crop_ids: Sequence[str] | None = None,
        image_ids: Sequence[str] | None = None,
        transform: Callable | None = None,
    ):
        all_rows = _read_crops_manifest()
        by_id = {r["crop_id"]: r for r in all_rows}

        if crop_ids is not None:
            rows = [by_id[cid] for cid in crop_ids if cid in by_id]
        elif image_ids is not None:
            ids_set = set(image_ids)
            rows = [r for r in all_rows if r["image_id"] in ids_set]
        else:
            rows = all_rows

        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        r = self.rows[idx]
        img = _load_rgb(PATHS.root / r["crop_path"])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img, int(r["class"]), r["crop_id"]

    def class_counts(self) -> tuple[int, int]:
        n0 = sum(1 for r in self.rows if int(r["class"]) == 0)
        n1 = sum(1 for r in self.rows if int(r["class"]) == 1)
        return n0, n1


class BagDataset(Dataset):
    """One bag = one image. Returns (instances_KxCxHxW, bag_label, count, image_id).

    bag_label = 1 if any crop is DC, else 0.
    count = number of DC crops in the image.
    """

    def __init__(
        self,
        image_ids: Sequence[str],
        transform: Callable | None = None,
    ):
        all_rows = _read_crops_manifest()
        by_img: dict[str, list[dict]] = defaultdict(list)
        for r in all_rows:
            by_img[r["image_id"]].append(r)

        self.image_ids = [img for img in image_ids if img in by_img]
        self.by_img = by_img
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        img_id = self.image_ids[idx]
        rows = self.by_img[img_id]
        crops = []
        for r in rows:
            arr = _load_rgb(PATHS.root / r["crop_path"])
            if self.transform is not None:
                t = self.transform(image=arr)["image"]
            else:
                t = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
            crops.append(t)
        x = torch.stack(crops, dim=0)
        count = sum(1 for r in rows if int(r["class"]) == 1)
        bag_label = 1 if count > 0 else 0
        return x, torch.tensor(bag_label, dtype=torch.long), torch.tensor(count, dtype=torch.float32), img_id

    def bag_labels(self) -> list[int]:
        return [1 if any(int(r["class"]) == 1 for r in self.by_img[i]) else 0 for i in self.image_ids]


def make_weighted_sampler(dataset: CropDataset) -> torch.utils.data.WeightedRandomSampler:
    """Class-balanced sampler for instance pretraining (DC is rare)."""
    n0, n1 = dataset.class_counts()
    w0 = 1.0 / max(1, n0)
    w1 = 1.0 / max(1, n1)
    weights = [w1 if int(r["class"]) == 1 else w0 for r in dataset.rows]
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
