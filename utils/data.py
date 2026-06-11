"""Shared dataset helpers: YOLO parsing, bbox geometry, manifest construction.

Conventions:
  - YOLO label rows: `cls cx cy w h`, all normalized to [0, 1].
  - Class 0 = monocentric, Class 1 = dicentric (verified at inventory time).
  - Absolute pixel boxes are (x1, y1, x2, y2) ints, no rounding bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class YoloBox:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy_abs(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        x1 = (self.cx - self.w / 2.0) * img_w
        y1 = (self.cy - self.h / 2.0) * img_h
        x2 = (self.cx + self.w / 2.0) * img_w
        y2 = (self.cy + self.h / 2.0) * img_h
        return x1, y1, x2, y2


def parse_yolo_label(label_path: Path) -> list[YoloBox]:
    """Read one YOLO .txt file. Skips blank lines; raises ValueError on malformed rows."""
    boxes: list[YoloBox] = []
    for lineno, raw in enumerate(label_path.read_text().splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{lineno}: expected 5 fields (cls cx cy w h), got {len(parts)}"
            )
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:])
        except ValueError as e:
            raise ValueError(f"{label_path}:{lineno}: parse error: {e}") from e
        boxes.append(YoloBox(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        for cand in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def iou_matrix(boxes_xyxy: Sequence[Sequence[float]]) -> np.ndarray:
    """Pairwise IoU. Returns (n, n) symmetric float array with 1.0 on the diagonal."""
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 0), dtype=float)
    b = np.asarray(boxes_xyxy, dtype=float)
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area[:, None] + area[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def max_neighbor_iou(boxes_xyxy: Sequence[Sequence[float]]) -> np.ndarray:
    """Per-box maximum IoU with any *other* box. Returns shape (n,)."""
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), dtype=float)
    m = iou_matrix(boxes_xyxy)
    np.fill_diagonal(m, 0.0)
    return m.max(axis=1)


def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) without loading full pixel data when possible."""
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # PIL returns (W, H)


def pad_to_square(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Extend the shorter side equally on both ends, then clamp to image bounds."""
    w = x2 - x1
    h = y2 - y1
    if w == h:
        side = w
    else:
        side = max(w, h)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        x1 = cx - side / 2.0
        y1 = cy - side / 2.0
        x2 = cx + side / 2.0
        y2 = cy + side / 2.0
    # Clamp; note: clamping can re-introduce non-squareness near borders. Acceptable for this dataset.
    x1c = max(0.0, x1)
    y1c = max(0.0, y1)
    x2c = min(float(img_w), x2)
    y2c = min(float(img_h), y2)
    return x1c, y1c, x2c, y2c


def pad_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pad_frac: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Expand each side by `pad_frac` of that side's length. Clamps to image."""
    w = x2 - x1
    h = y2 - y1
    return (
        max(0.0, x1 - pad_frac * w),
        max(0.0, y1 - pad_frac * h),
        min(float(img_w), x2 + pad_frac * w),
        min(float(img_h), y2 + pad_frac * h),
    )


def class_name(cls: int) -> str:
    return {0: "MC", 1: "DC"}.get(cls, f"cls{cls}")


def summarize_classes(class_counts: dict[int, int]) -> dict:
    """Return basic stats used in inventory + sanity-check reporting."""
    total = sum(class_counts.values())
    n_dc = class_counts.get(1, 0)
    n_mc = class_counts.get(0, 0)
    other = {k: v for k, v in class_counts.items() if k not in (0, 1)}
    return {
        "total_boxes": total,
        "n_mc": n_mc,
        "n_dc": n_dc,
        "n_other": sum(other.values()),
        "other_classes": other,
        "dc_fraction": (n_dc / total) if total else 0.0,
    }


def assert_class_semantics(class_counts: dict[int, int], threshold: float = 0.10) -> str | None:
    """Return a warning string if class-1 fraction exceeds threshold, else None.

    Per CLAUDE.md §7.3 we don't crash here — the inventory script decides what to do.
    """
    stats = summarize_classes(class_counts)
    if stats["dc_fraction"] > threshold:
        return (
            f"Dicentric fraction {stats['dc_fraction']:.1%} exceeds {threshold:.0%}. "
            "Class IDs may be swapped — verify before training."
        )
    if stats["n_other"]:
        return f"Found unexpected class IDs (not 0 or 1): {stats['other_classes']}"
    return None
