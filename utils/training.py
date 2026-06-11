"""Small training helpers: early stopping, best-model tracker, checkpoint I/O.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class EarlyStopper:
    patience: int = 30
    mode: str = "max"  # "max" for AUROC, "min" for loss
    best: float = field(init=False)
    bad_epochs: int = 0
    should_stop: bool = False

    def __post_init__(self):
        self.best = -float("inf") if self.mode == "max" else float("inf")

    def step(self, value: float) -> bool:
        """Return True if `value` is the new best."""
        if self.mode == "max":
            improved = value > self.best
        else:
            improved = value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return improved


def save_ckpt(path: Path, model: torch.nn.Module, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)


def load_ckpt_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def lap(self) -> float:
        t = time.time()
        dt = t - self.t0
        self.t0 = t
        return dt


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
