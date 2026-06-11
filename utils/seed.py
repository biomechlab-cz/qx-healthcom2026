"""Reproducibility utilities. See CLAUDE.md §2.4."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed numpy, random, and torch (if installed). Optionally lock deterministic algos."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_sha() -> str:
    """Short HEAD SHA, or 'no-git' if not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "no-git"


def config_hash(cfg: Any) -> str:
    """Stable 8-char hash of a dataclass (or dict) for provenance."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        payload = asdict(cfg)
    elif isinstance(cfg, dict):
        payload = cfg
    else:
        payload = {"repr": repr(cfg)}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def provenance(seed: int, cfg: Any = None) -> dict:
    """Standard provenance block to attach to every results file."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "seed": seed,
        "config_hash": config_hash(cfg) if cfg is not None else None,
    }


def write_json(path: Path, data: dict) -> None:
    """Write JSON with parents/ensured and stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
