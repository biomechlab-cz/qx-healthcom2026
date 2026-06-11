"""Per-chromosome crop extraction.

For every (image, bbox) in the manifest:
  1. Convert YOLO -> absolute (x1, y1, x2, y2)
  2. Pad by source-specific fraction (15% web, 5% web2)
  3. Square it (extend shorter side)
  4. Crop, resize to intermediate_crop_size, then resize to crop_size
  5. Save under data/preprocessed/crops/{source}/{class_name}/{image_id}_{chrom_idx}.png
  6. Append a row to data/preprocessed/crops/crops_manifest.csv
  7. Accumulate channel statistics -> data/preprocessed/normalization_stats.json
  8. Render a 5-per-class sanity grid -> results/02_crop_samples.png
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import DATA, PATHS, TRAIN, ensure_dirs  # noqa: E402
from utils.data import (  # noqa: E402
    class_name,
    image_size,
    max_neighbor_iou,
    pad_box,
    pad_to_square,
    parse_yolo_label,
)
from utils.seed import provenance, set_seed, write_json  # noqa: E402

CROPS_COLS = [
    "crop_id",
    "image_id",
    "source",
    "class",
    "chrom_idx",
    "orig_x1", "orig_y1", "orig_x2", "orig_y2",
    "padded_x1", "padded_y1", "padded_x2", "padded_y2",
    "max_iou_with_neighbor",
    "is_potentially_noisy",
    "crop_path",
]

PAD_BY_SOURCE = {"web": DATA.web_padding, "web2": DATA.web2_padding, "web3": DATA.web3_padding}


def read_manifest_rows() -> list[dict]:
    with PATHS.manifest.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def process_image(row: dict) -> tuple[list[dict], np.ndarray, np.ndarray, int]:
    """Crop every chromosome in one image. Returns (crop rows, sum, sum_sq, n_pixels)."""
    source = row["source"]
    image_id = row["image_id"]
    stem = image_id.split("/", 1)[1]
    img_path = PATHS.root / row["image_path"]
    label_path = PATHS.root / row["label_path"]

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read {img_path}")
    img_h, img_w = img.shape[:2]

    boxes = parse_yolo_label(label_path)
    xyxy = [b.to_xyxy_abs(img_w, img_h) for b in boxes]
    max_ious = max_neighbor_iou(xyxy) if xyxy else []
    pad = PAD_BY_SOURCE[source]

    crop_rows: list[dict] = []
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_n = 0

    out_dir_root = PATHS.crops / source
    for idx, (box, (x1, y1, x2, y2)) in enumerate(zip(boxes, xyxy)):
        cls = box.cls
        cls_dir = out_dir_root / class_name(cls).lower()
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Pad, then square.
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, pad, img_w, img_h)
        sx1, sy1, sx2, sy2 = pad_to_square(px1, py1, px2, py2, img_w, img_h)
        ix1, iy1 = int(round(sx1)), int(round(sy1))
        ix2, iy2 = int(round(sx2)), int(round(sy2))
        if ix2 - ix1 < 4 or iy2 - iy1 < 4:
            # Degenerate box; skip but warn via the manifest (noisy flag will be True too).
            continue

        crop = img[iy1:iy2, ix1:ix2]
        crop = cv2.resize(crop, (DATA.intermediate_crop_size, DATA.intermediate_crop_size),
                          interpolation=cv2.INTER_AREA)
        # Resize once more to the final size; center crop not needed because we already squared.
        crop = cv2.resize(crop, (DATA.crop_size, DATA.crop_size), interpolation=cv2.INTER_AREA)

        crop_path = cls_dir / f"{stem}_{idx:03d}.png"
        cv2.imwrite(str(crop_path), crop)

        # Accumulate normalization stats in BGR -> we'll save as RGB below.
        f = crop.astype(np.float64) / 255.0
        pixel_sum += f.sum(axis=(0, 1))
        pixel_sum_sq += (f ** 2).sum(axis=(0, 1))
        pixel_n += f.shape[0] * f.shape[1]

        max_iou = float(max_ious[idx]) if len(max_ious) else 0.0
        crop_rows.append(
            {
                "crop_id": f"{image_id}#{idx:03d}",
                "image_id": image_id,
                "source": source,
                "class": cls,
                "chrom_idx": idx,
                "orig_x1": x1, "orig_y1": y1, "orig_x2": x2, "orig_y2": y2,
                "padded_x1": sx1, "padded_y1": sy1, "padded_x2": sx2, "padded_y2": sy2,
                "max_iou_with_neighbor": max_iou,
                "is_potentially_noisy": max_iou > DATA.iou_noisy_threshold,
                "crop_path": str(crop_path.relative_to(PATHS.root)),
            }
        )

    return crop_rows, pixel_sum, pixel_sum_sq, pixel_n


def write_crops_manifest(rows: list[dict]) -> Path:
    PATHS.crops.mkdir(parents=True, exist_ok=True)
    with PATHS.crops_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CROPS_COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return PATHS.crops_manifest


def sample_grid(rows: list[dict], n_per_class: int = 5) -> Path:
    """Render an n_per_class x 2 grid for a quick eyeball pass."""
    import matplotlib.pyplot as plt

    by_class: dict[int, list[dict]] = {0: [], 1: []}
    for r in rows:
        by_class[int(r["class"])].append(r)

    rng = np.random.default_rng(TRAIN.seed)
    picks = {
        cls: list(rng.choice(by_class[cls], size=min(n_per_class, len(by_class[cls])), replace=False))
        for cls in (0, 1)
    }

    fig, axes = plt.subplots(2, n_per_class, figsize=(2 * n_per_class, 4.5))
    for col in range(n_per_class):
        for row_idx, cls in enumerate((0, 1)):
            ax = axes[row_idx, col] if n_per_class > 1 else axes[row_idx]
            if col < len(picks[cls]):
                p = picks[cls][col]
                img = cv2.cvtColor(cv2.imread(str(PATHS.root / p["crop_path"])), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                noisy = " *" if str(p["is_potentially_noisy"]) == "True" else ""
                ax.set_title(f"{p['crop_id']}{noisy}", fontsize=7)
            ax.axis("off")
        axes[0, 0].set_ylabel("MC", fontsize=10)
        axes[1, 0].set_ylabel("DC", fontsize=10)

    fig.suptitle("Crop samples (top: MC, bottom: DC; * = potentially noisy)", fontsize=10)
    out_path = PATHS.results / "02_crop_samples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> int:
    set_seed(TRAIN.seed)
    ensure_dirs()
    PATHS.crops.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest_rows()
    if not manifest_rows:
        print("Manifest empty. Run scripts/01_inventory_data.py first.")
        return 1

    all_crop_rows: list[dict] = []
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_n = 0

    for i, row in enumerate(manifest_rows, 1):
        crops, s, ssq, n = process_image(row)
        all_crop_rows.extend(crops)
        pixel_sum += s
        pixel_sum_sq += ssq
        pixel_n += n
        print(f"  [{i:>2}/{len(manifest_rows)}] {row['image_id']}: {len(crops)} crops")

    manifest_path = write_crops_manifest(all_crop_rows)

    # Channel stats: stored in RGB order (cv2 reads BGR, but channel-mean is symmetric per-channel,
    # so we just need to swap the order at save time to match how torchvision normalizes).
    if pixel_n == 0:
        raise RuntimeError("No crops produced — every bbox was degenerate.")
    mean_bgr = (pixel_sum / pixel_n).tolist()
    var_bgr = (pixel_sum_sq / pixel_n - (pixel_sum / pixel_n) ** 2)
    std_bgr = np.sqrt(np.clip(var_bgr, 0, None)).tolist()
    mean_rgb = list(reversed(mean_bgr))
    std_rgb = list(reversed(std_bgr))

    cls_counts = Counter(int(r["class"]) for r in all_crop_rows)
    write_json(
        PATHS.norm_stats,
        {
            "provenance": provenance(seed=TRAIN.seed, cfg=DATA),
            "n_crops": len(all_crop_rows),
            "n_class0": cls_counts.get(0, 0),
            "n_class1": cls_counts.get(1, 0),
            "n_potentially_noisy": sum(1 for r in all_crop_rows if str(r["is_potentially_noisy"]) == "True"),
            "channel_order": "RGB",
            "mean": mean_rgb,
            "std": std_rgb,
            "crop_size": DATA.crop_size,
        },
    )

    grid_path = sample_grid(all_crop_rows, n_per_class=5)

    print()
    print(f"crops:        {len(all_crop_rows)}  (MC={cls_counts.get(0,0)}  DC={cls_counts.get(1,0)})")
    print(f"manifest:     {manifest_path}")
    print(f"norm stats:   {PATHS.norm_stats}")
    print(f"sample grid:  {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
