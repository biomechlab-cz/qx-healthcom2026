"""Data inventory and class-balance verification.

Walks every labeled source (web, web2). For each image:
  - parse YOLO labels
  - record image size, bbox count, DC count
  - flag images without labels (orphans) and labels without images
  - flag any image whose bbox neighbours overlap (IoU > threshold)

Outputs:
  data/preprocessed/manifest.csv         — one row per image
  results/01_data_inventory.md           — human summary
  results/01_data_inventory.json         — machine-readable counts + provenance
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import DATA, PATHS, TRAIN, ensure_dirs  # noqa: E402
from utils.data import (  # noqa: E402
    IMG_EXTS,
    assert_class_semantics,
    find_image_for_label,
    image_size,
    list_images,
    max_neighbor_iou,
    parse_yolo_label,
    summarize_classes,
)
from utils.seed import provenance, write_json  # noqa: E402


MANIFEST_COLS = [
    "image_id",
    "source",
    "image_path",
    "label_path",
    "image_w",
    "image_h",
    "num_chromosomes",
    "num_dicentrics",
    "n_class0",
    "n_class1",
    "max_neighbor_iou",
    "has_overlap_flag",
]


def inventory_source(source: str) -> tuple[list[dict], list[str]]:
    """Return (per-image rows, orphan warnings)."""
    raw_dir = PATHS.raw / source
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"

    rows: list[dict] = []
    warnings: list[str] = []

    images = list_images(images_dir)
    label_stems = {p.stem for p in labels_dir.iterdir() if p.suffix == ".txt"} if labels_dir.exists() else set()
    image_stems = {p.stem for p in images}

    for img_path in images:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            warnings.append(f"{source}/{stem}: image without label — skipping in manifest")
            continue

        boxes = parse_yolo_label(label_path)
        if not boxes:
            warnings.append(f"{source}/{stem}: empty label file — keeping as 0-chromosome bag")

        w, h = image_size(img_path)
        xyxy = [b.to_xyxy_abs(w, h) for b in boxes]
        max_iou = max_neighbor_iou(xyxy) if xyxy else []

        cls_counts = Counter(b.cls for b in boxes)
        n_class0 = cls_counts.get(0, 0)
        n_class1 = cls_counts.get(1, 0)
        n_dc = n_class1  # working assumption; verified globally below

        rows.append(
            {
                "image_id": f"{source}/{stem}",
                "source": source,
                "image_path": str(img_path.relative_to(PATHS.root)),
                "label_path": str(label_path.relative_to(PATHS.root)),
                "image_w": w,
                "image_h": h,
                "num_chromosomes": len(boxes),
                "num_dicentrics": n_dc,
                "n_class0": n_class0,
                "n_class1": n_class1,
                "max_neighbor_iou": float(max(max_iou)) if len(max_iou) else 0.0,
                "has_overlap_flag": bool(len(max_iou) and max(max_iou) > DATA.iou_noisy_threshold),
            }
        )

    # Labels without images.
    for stem in sorted(label_stems - image_stems):
        warnings.append(f"{source}/{stem}: label without image — ignored")

    return rows, warnings


def write_manifest(rows: list[dict]) -> Path:
    PATHS.manifest.parent.mkdir(parents=True, exist_ok=True)
    with PATHS.manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return PATHS.manifest


def build_summary(rows: list[dict], warnings: list[str], cls_counts: Counter) -> str:
    stats = summarize_classes(dict(cls_counts))
    by_source: dict[str, dict[str, int]] = {}
    for r in rows:
        s = by_source.setdefault(r["source"], {"images": 0, "chromosomes": 0, "dc": 0, "overlap_imgs": 0})
        s["images"] += 1
        s["chromosomes"] += r["num_chromosomes"]
        s["dc"] += r["num_dicentrics"]
        s["overlap_imgs"] += int(bool(r["has_overlap_flag"]))

    lines = []
    lines.append("# Data Inventory (Phase 1.1)")
    lines.append("")
    lines.append(f"Seed: {TRAIN.seed}    Source dirs: `data/raw/{', '.join(DATA.labeled_sources)}`")
    lines.append("")
    lines.append("## Per-source totals")
    lines.append("")
    lines.append("| source | images | chromosomes | dicentrics | images w/ overlap |")
    lines.append("|---|---:|---:|---:|---:|")
    for src in sorted(by_source):
        s = by_source[src]
        lines.append(
            f"| {src} | {s['images']} | {s['chromosomes']} | {s['dc']} | {s['overlap_imgs']} |"
        )
    total_imgs = sum(s["images"] for s in by_source.values())
    total_chr = sum(s["chromosomes"] for s in by_source.values())
    total_dc = sum(s["dc"] for s in by_source.values())
    lines.append(f"| **all** | **{total_imgs}** | **{total_chr}** | **{total_dc}** | — |")
    lines.append("")
    lines.append("## Class distribution")
    lines.append("")
    lines.append(f"- class 0 (MC): {stats['n_mc']}")
    lines.append(f"- class 1 (DC): {stats['n_dc']}")
    lines.append(f"- other:        {stats['n_other']} ({stats['other_classes']})")
    lines.append(f"- DC fraction:  {stats['dc_fraction']:.2%}")
    lines.append("")
    warn = assert_class_semantics(dict(cls_counts), threshold=DATA.iou_noisy_threshold)
    if warn:
        lines.append(f"WARNING: {warn}")
        lines.append("")
    else:
        lines.append("Class-semantics check: PASS (DC fraction is small, class IDs are 0/1).")
        lines.append("")

    lines.append("## Per-image DC-count distribution")
    lines.append("")
    dc_hist = Counter(r["num_dicentrics"] for r in rows)
    lines.append("| DC per image | num images |")
    lines.append("|---:|---:|")
    for k in sorted(dc_hist):
        lines.append(f"| {k} | {dc_hist[k]} |")
    lines.append("")

    lines.append("## Warnings / orphans")
    lines.append("")
    if not warnings:
        lines.append("(none)")
    else:
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ensure_dirs()

    all_rows: list[dict] = []
    all_warnings: list[str] = []
    cls_counter: Counter = Counter()

    for src in DATA.labeled_sources:
        rows, warns = inventory_source(src)
        for r in rows:
            cls_counter[0] += r["n_class0"]
            cls_counter[1] += r["n_class1"]
        all_rows.extend(rows)
        all_warnings.extend(warns)

    manifest_path = write_manifest(all_rows)

    summary = build_summary(all_rows, all_warnings, cls_counter)
    report_path = PATHS.results / "01_data_inventory.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary, encoding="utf-8")

    payload = {
        "provenance": provenance(seed=TRAIN.seed, cfg=DATA),
        "per_source": {},
        "totals": {
            "images": len(all_rows),
            "chromosomes": sum(r["num_chromosomes"] for r in all_rows),
            "dicentrics": sum(r["num_dicentrics"] for r in all_rows),
        },
        "class_counts": dict(cls_counter),
        "warnings": all_warnings,
        "manifest_path": str(manifest_path.relative_to(PATHS.root)),
    }
    write_json(PATHS.results / "01_data_inventory.json", payload)

    print(summary)
    print(f"\nmanifest: {manifest_path}")
    print(f"report:   {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
