"""Synthetic-bag generator for MIL training.

The real n=34 dataset is too small for bag-level loss to converge. We multiply the
effective bag count by sampling new bags from each fold's training-crop pool, with
controlled DC composition.

A synthetic bag is a list of crops drawn from the fold-k training set:
  - choose a target DC count from a configurable distribution
  - sample K crops total: target DC crops from the fold's DC-positive pool, K-target from MC pool
  - the bag's true_count = target DC, true_label = 1 if target >= 1 else 0
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from utils.config import PATHS


@dataclass(frozen=True)
class SyntheticBagConfig:
    n_bags: int = 200          # synthetic bags per fold
    K_min: int = 38            # min crops per bag
    K_max: int = 46            # max crops per bag
    dc_count_weights: tuple = (1.0, 2.0, 2.0, 2.0, 1.5, 1.0)  # P(target DC count = 0..5)
    seed: int = 42


def _load_crops_manifest() -> list[dict]:
    with PATHS.crops_manifest.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def crop_pool_for_fold(fold_train_image_ids: list[str]) -> tuple[list[dict], list[dict]]:
    """Return (mc_crops, dc_crops) from the fold's training image set."""
    train_set = set(fold_train_image_ids)
    rows = [r for r in _load_crops_manifest() if r["image_id"] in train_set]
    mc = [r for r in rows if int(r["class"]) == 0]
    dc = [r for r in rows if int(r["class"]) == 1]
    return mc, dc


def generate_synthetic_bags(
    mc_pool: list[dict], dc_pool: list[dict], cfg: SyntheticBagConfig | None = None,
) -> list[dict]:
    """Return a list of synthetic bag specs.

    Each entry: {"crop_paths": [...], "labels": [0/1...], "true_count": int, "true_label": int, "K": int}.
    """
    cfg = cfg or SyntheticBagConfig()
    rng = np.random.default_rng(cfg.seed)
    if len(dc_pool) == 0:
        raise ValueError("DC pool is empty — cannot generate positive synthetic bags")

    weights = np.asarray(cfg.dc_count_weights, dtype=float)
    weights = weights / weights.sum()
    n_dc_options = np.arange(len(weights))

    bags = []
    for _ in range(cfg.n_bags):
        K = int(rng.integers(cfg.K_min, cfg.K_max + 1))
        target_dc = int(rng.choice(n_dc_options, p=weights))
        target_dc = min(target_dc, K, len(dc_pool))  # cap by availability

        # Sample WITHOUT replacement when possible; with replacement otherwise.
        dc_idx = rng.choice(len(dc_pool), size=target_dc, replace=(target_dc > len(dc_pool)))
        n_mc = K - target_dc
        mc_idx = rng.choice(len(mc_pool), size=n_mc, replace=(n_mc > len(mc_pool)))

        sampled = [dc_pool[int(i)] for i in dc_idx] + [mc_pool[int(i)] for i in mc_idx]
        # Shuffle order within bag so DC isn't always first.
        rng.shuffle(sampled)

        paths = [s["crop_path"] for s in sampled]
        labels = [int(s["class"]) for s in sampled]
        bags.append({
            "crop_paths": paths,
            "labels": labels,
            "true_count": target_dc,
            "true_label": 1 if target_dc > 0 else 0,
            "K": K,
        })
    return bags


def precompute_phi_for_crops(
    crop_rows: list[dict], backbone: torch.nn.Module, transform,
    device: torch.device, batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Run ResNet-18 backbone on every crop ONCE and cache phi by crop_path.

    Speeds up synthetic-bag training: subsequent epochs reuse phi instead of re-running the backbone.
    """
    import cv2

    backbone = backbone.to(device).eval()
    cache: dict[str, np.ndarray] = {}

    def load_rgb(p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    paths = [r["crop_path"] for r in crop_rows]
    n = len(paths)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_paths = paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                arr = load_rgb(PATHS.root / p)
                t = transform(image=arr)["image"]
                tensors.append(t)
            x = torch.stack(tensors).to(device)
            phi = backbone(x).cpu().numpy()
            for p, vec in zip(batch_paths, phi):
                cache[p] = vec
    return cache
