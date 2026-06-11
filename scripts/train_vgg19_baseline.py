"""Modified-VGG-19 baseline (Shin et al. 2024).

Instance-level binary classifier (per crop). For bag-level comparison, aggregates the
per-crop DC probabilities of each image with max-pooling: "any DC crop -> DC bag".

VGG-19 features are kept trainable to allow head + last block to adapt; learning rate
is conservative to avoid catastrophic forgetting of the ImageNet features.

Outputs:
  results/models/vgg19_fold_{k}.pt
  results/06_vgg19_per_fold.csv
  results/06_vgg19_summary.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.augmentation import build_transforms  # noqa: E402
from utils.config import DATA, PATHS, TRAIN, ensure_dirs  # noqa: E402
from utils.dataset import (  # noqa: E402
    BagDataset,
    CropDataset,
    load_fold,
    load_instance_split,
    make_weighted_sampler,
)
from utils.metrics import aggregate_folds, count_mae, summarize  # noqa: E402
from utils.models import build_vgg19_modified, count_params  # noqa: E402
from utils.seed import provenance, set_seed, write_json  # noqa: E402
from utils.training import EarlyStopper, Timer, device, save_ckpt  # noqa: E402


def train_one_fold(fold: int, epochs: int, batch_size: int) -> dict:
    dev = device()
    train_tf, val_tf = build_transforms()

    it_ids, iv_ids = load_instance_split(fold)
    _, val_img_ids = load_fold(fold)

    train_ds = CropDataset(crop_ids=it_ids, transform=train_tf)
    val_ds = CropDataset(crop_ids=iv_ids, transform=val_tf)
    bag_ds = BagDataset(image_ids=val_img_ids, transform=val_tf)

    n0, n1 = train_ds.class_counts()
    print(f"  fold {fold}: inst train n0={n0} n1={n1}  inst val={len(val_ds)}  bag val={len(bag_ds)}")

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = build_vgg19_modified(pretrained=True, num_classes=2).to(dev)

    cw = torch.tensor([1.0, max(1.0, n0 / max(1, n1))], dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    optim = torch.optim.AdamW(model.parameters(), lr=TRAIN.instance_lr, weight_decay=TRAIN.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    stopper = EarlyStopper(patience=TRAIN.early_stop_patience, mode="max")

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_auroc = -1.0
    t = Timer()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for x, y, _ in train_loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        sched.step()
        train_loss /= max(1, n_seen)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(dev, non_blocking=True)
                p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
                ps.append(p); ys.append(y.numpy())
        y_inst = np.concatenate(ys); p_inst = np.concatenate(ps)
        from sklearn.metrics import roc_auc_score
        inst_auroc = float(roc_auc_score(y_inst, p_inst)) if len(np.unique(y_inst)) > 1 else float("nan")

        improved = stopper.step(inst_auroc if not np.isnan(inst_auroc) else -1.0)
        if improved:
            best_auroc = inst_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 10 == 0 or stopper.should_stop:
            print(f"    ep {ep:3d}  train_loss={train_loss:.4f}  inst_val_auroc={inst_auroc:.4f}"
                  f"  best={best_auroc:.4f}  ({t.lap():.1f}s)")
        if stopper.should_stop:
            print(f"    early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    model.eval()

    bag_y, bag_score, ct_t, ct_p = [], [], [], []
    with torch.no_grad():
        for i in range(len(bag_ds)):
            x, y, c, _ = bag_ds[i]
            x = x.to(dev)
            p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            bag_score.append(float(p.max()))
            bag_y.append(int(y.item()))
            ct_t.append(float(c.item()))
            ct_p.append(float((p >= 0.5).sum()))
    bag_y = np.array(bag_y); bag_s = np.array(bag_score)
    bag_metrics = summarize(bag_y, bag_s, n_resamples=1000, seed=TRAIN.seed)
    mae = count_mae(np.array(ct_t), np.array(ct_p))

    ckpt_path = PATHS.models / f"vgg19_fold_{fold}.pt"
    save_ckpt(
        ckpt_path,
        model,
        meta={
            "fold": fold,
            "epochs_trained": ep,
            "best_inst_auroc": best_auroc,
            "trainable_params": count_params(model),
            "provenance": provenance(seed=TRAIN.seed, cfg={"model": "vgg19", "fold": fold}),
        },
    )

    return {
        "fold": fold,
        "epochs_trained": ep,
        "best_inst_auroc": best_auroc,
        "bag": bag_metrics,
        "count_mae": mae,
        "trainable_params": count_params(model),
        "checkpoint": str(ckpt_path.relative_to(PATHS.root)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=TRAIN.instance_pretrain_epochs)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(TRAIN.seed)
    ensure_dirs()

    folds = [args.fold] if args.fold is not None else list(range(DATA.n_folds))
    results = []
    for k in folds:
        print(f"\n=== Modified VGG-19 baseline — fold {k} ===")
        r = train_one_fold(k, epochs=args.epochs, batch_size=args.batch_size)
        results.append(r)

    csv_path = PATHS.results / "06_vgg19_per_fold.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", "epochs", "inst_auroc",
                    "bag_auroc", "bag_auroc_lo", "bag_auroc_hi",
                    "bag_auprc", "bag_f1", "bag_bacc", "count_mae", "trainable_params"])
        for r in results:
            b = r["bag"]
            w.writerow([
                r["fold"], r["epochs_trained"], f"{r['best_inst_auroc']:.4f}",
                f"{b['auroc']['point']:.4f}", f"{b['auroc']['ci_low']:.4f}", f"{b['auroc']['ci_high']:.4f}",
                f"{b['auprc']['point']:.4f}", f"{b['f1']['point']:.4f}", f"{b['balanced_acc']['point']:.4f}",
                f"{r['count_mae']:.4f}", r["trainable_params"],
            ])

    write_json(
        PATHS.results / "06_vgg19_summary.json",
        {
            "provenance": provenance(seed=TRAIN.seed, cfg={"epochs": args.epochs}),
            "per_fold": results,
            "aggregate": {
                "inst_auroc_mean": float(np.mean([r["best_inst_auroc"] for r in results])),
                "bag_auroc": aggregate_folds([r["bag"] for r in results], "auroc"),
                "bag_auprc": aggregate_folds([r["bag"] for r in results], "auprc"),
                "bag_f1":    aggregate_folds([r["bag"] for r in results], "f1"),
            },
        },
    )

    inst = [r["best_inst_auroc"] for r in results]
    auroc_pts = [r["bag"]["auroc"]["point"] for r in results]
    print("\n=== Aggregate ===")
    print(f"  instance AUROC : mean={np.mean(inst):.4f}")
    print(f"  bag AUROC      : mean={np.nanmean(auroc_pts):.4f}  per-fold={['%.3f' % x for x in auroc_pts]}")
    print(f"\nper-fold CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
