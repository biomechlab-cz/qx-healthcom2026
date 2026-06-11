"""Training-time augmentation for chromosome crops.

Train transform uses rotation, flips, mild brightness/contrast, blur, and an elastic
deformation (chromosomes are non-rigid). Val transform only normalizes.

Normalization stats are loaded from data/preprocessed/normalization_stats.json
which scripts/02_extract_crops.py produced. The file stores RGB-order mean/std.

API:
    from utils.augmentation import build_transforms
    train_tf, val_tf = build_transforms()
"""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from utils.config import PATHS


def _load_norm_stats() -> tuple[list[float], list[float]]:
    if not PATHS.norm_stats.exists():
        raise FileNotFoundError(
            f"{PATHS.norm_stats} not found; run scripts/02_extract_crops.py first."
        )
    payload = json.loads(PATHS.norm_stats.read_text())
    if payload.get("channel_order") != "RGB":
        raise ValueError(f"Expected RGB normalization stats, got {payload.get('channel_order')}")
    return payload["mean"], payload["std"]


def build_transforms(stats_path: Path | None = None) -> tuple[A.Compose, A.Compose]:
    """Return (train_transform, val_transform). Both expect HWC uint8 RGB and emit CHW float tensors."""
    mean, std = _load_norm_stats()

    train = A.Compose(
        [
            A.RandomRotate90(p=1.0),
            A.Rotate(limit=180, p=0.8, border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ElasticTransform(alpha=20, sigma=4, p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    val = A.Compose(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return train, val
