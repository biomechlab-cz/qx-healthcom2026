"""Cross-validation splits.

Image-level 5-fold stratified CV, stratified on DC-count tertile:
  T0: 0 dicentrics, T1: exactly 1, T2: >=2.

Outputs:
  data/preprocessed/splits/fold_{0..4}.json   — {"train": [...], "val": [...]}
  data/preprocessed/splits/instance_pretrain.json  — per-fold 80/20 instance-level split
                                                    over crops of that fold's training images
  results/03_splits_summary.md                — table of per-fold counts
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import DATA, PATHS, TRAIN, ensure_dirs  # noqa: E402
from utils.seed import provenance, set_seed, write_json  # noqa: E402


def dc_tertile(num_dc: int) -> int:
    if num_dc == 0:
        return 0
    if num_dc == 1:
        return 1
    return 2


def stratified_kfold(image_ids: list[str], tertiles: list[int], n_folds: int, seed: int) -> list[dict]:
    """Hand-rolled stratified k-fold. Splits each tertile into folds of as-equal-as-possible size."""
    rng = np.random.default_rng(seed)
    folds: list[dict] = [{"train": [], "val": []} for _ in range(n_folds)]

    by_t: dict[int, list[str]] = defaultdict(list)
    for img, t in zip(image_ids, tertiles):
        by_t[t].append(img)

    for t, ids in by_t.items():
        ids = list(ids)
        rng.shuffle(ids)
        # Round-robin assign to folds so spread is as even as possible.
        per_fold = [[] for _ in range(n_folds)]
        for i, img_id in enumerate(ids):
            per_fold[i % n_folds].append(img_id)
        for k in range(n_folds):
            folds[k]["val"].extend(per_fold[k])

    all_ids = set(image_ids)
    for k in range(n_folds):
        val_set = set(folds[k]["val"])
        folds[k]["train"] = sorted(all_ids - val_set)
        folds[k]["val"] = sorted(folds[k]["val"])

    return folds


def instance_pretrain_split(
    folds: list[dict],
    image_to_crops: dict[str, list[str]],
    seed: int,
    train_frac: float = 0.8,
) -> dict:
    """For each fold, split that fold's training-image crops 80/20 for pretraining."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, list[str]]] = {}
    for k, fold in enumerate(folds):
        crops = [c for img in fold["train"] for c in image_to_crops.get(img, [])]
        rng.shuffle(crops)
        n_train = int(round(train_frac * len(crops)))
        out[f"fold_{k}"] = {
            "instance_train": sorted(crops[:n_train]),
            "instance_val": sorted(crops[n_train:]),
        }
    return out


def main() -> int:
    set_seed(TRAIN.seed)
    ensure_dirs()
    PATHS.splits.mkdir(parents=True, exist_ok=True)

    with PATHS.manifest.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    if not manifest:
        print("Manifest empty. Run scripts/01_inventory_data.py first.")
        return 1

    image_ids = [r["image_id"] for r in manifest]
    tertiles = [dc_tertile(int(r["num_dicentrics"])) for r in manifest]
    dc_per_img = {r["image_id"]: int(r["num_dicentrics"]) for r in manifest}

    folds = stratified_kfold(image_ids, tertiles, n_folds=DATA.n_folds, seed=TRAIN.seed)

    # Sanity: every image appears in val exactly once.
    val_counter = Counter(img for fold in folds for img in fold["val"])
    bad = [img for img, n in val_counter.items() if n != 1]
    if bad:
        raise RuntimeError(f"Each image must be in val exactly once; offenders: {bad[:5]}")

    # Per-fold artifacts.
    for k, fold in enumerate(folds):
        payload = {
            "fold": k,
            "n_folds": DATA.n_folds,
            "stratify_key": "dc_tertile",
            "provenance": provenance(seed=TRAIN.seed, cfg=DATA),
            "train": fold["train"],
            "val": fold["val"],
            "val_dc_counts": {img: dc_per_img[img] for img in fold["val"]},
        }
        write_json(PATHS.splits / f"fold_{k}.json", payload)

    # Instance pretraining split — needs crops_manifest.
    crops_csv = PATHS.crops_manifest
    image_to_crops: dict[str, list[str]] = defaultdict(list)
    if crops_csv.exists():
        with crops_csv.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                image_to_crops[r["image_id"]].append(r["crop_id"])
        ip = instance_pretrain_split(folds, image_to_crops, seed=TRAIN.seed)
        write_json(
            PATHS.splits / "instance_pretrain.json",
            {
                "provenance": provenance(seed=TRAIN.seed, cfg=DATA),
                "train_frac": 0.8,
                "per_fold": ip,
            },
        )
    else:
        print("(crops_manifest.csv not found — skipping instance pretrain split)")

    # Summary.
    lines = []
    lines.append("# CV Splits (Phase 1.3)")
    lines.append("")
    lines.append(f"Seed: {TRAIN.seed}    Folds: {DATA.n_folds}    Stratify: DC-count tertile (T0=0, T1=1, T2>=2)")
    lines.append("")
    lines.append("| fold | n_train | n_val | val T0 | val T1 | val T2 | val DC sum |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for k, fold in enumerate(folds):
        val_t = Counter(dc_tertile(dc_per_img[img]) for img in fold["val"])
        val_dc = sum(dc_per_img[img] for img in fold["val"])
        lines.append(
            f"| {k} | {len(fold['train'])} | {len(fold['val'])} | "
            f"{val_t[0]} | {val_t[1]} | {val_t[2]} | {val_dc} |"
        )
    lines.append("")
    if image_to_crops:
        lines.append("Instance pretrain split: per-fold 80/20 of crops from training images only.")
    lines.append("")
    report_path = PATHS.results / "03_splits_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nsplits dir:    {PATHS.splits}")
    print(f"report:        {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
