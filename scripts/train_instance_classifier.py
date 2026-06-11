"""ResNet-18 instance (per-chromosome) classifier.

For each fold k:
  - train set: instance-pretrain train crops from fold k's training images
  - val set:   instance-pretrain val crops from fold k's training images
              (held-out instance val; the FOLD val images are kept untouched for bag eval)
  - eval:      bag-level metrics on fold k's val images using crop-level scores aggregated
              by max ("any DC crop -> DC bag") to mirror modified-VGG-19 style aggregation
  - save model -> results/models/resnet18_fold_{k}.pt

Outputs:
  results/04_resnet18_per_fold.csv
  results/04_resnet18_summary.json
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
from utils.config import DATA, MODEL, PATHS, TRAIN, ensure_dirs  # noqa: E402
from utils.dataset import (  # noqa: E402
    BagDataset,
    CropDataset,
    load_fold,
    load_instance_split,
    make_weighted_sampler,
)
from utils.metrics import aggregate_folds, count_mae, summarize  # noqa: E402
from utils.models import build_resnet18_instance, count_params  # noqa: E402
from utils.seed import provenance, set_seed, write_json  # noqa: E402
from utils.training import EarlyStopper, Timer, device, save_ckpt  # noqa: E402


def train_one_fold(fold: int, epochs: int, batch_size: int = 64, num_workers: int = 0) -> dict:
    dev = device()
    train_tf, val_tf = build_transforms()

    it_ids, iv_ids = load_instance_split(fold)
    _, val_img_ids = load_fold(fold)

    train_ds = CropDataset(crop_ids=it_ids, transform=train_tf)
    val_ds = CropDataset(crop_ids=iv_ids, transform=val_tf)
    bag_ds = BagDataset(image_ids=val_img_ids, transform=val_tf)

    n0_tr, n1_tr = train_ds.class_counts()
    print(f"  fold {fold}: instance train n0={n0_tr} n1={n1_tr}  inst val={len(val_ds)}  bag val={len(bag_ds)}")

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = build_resnet18_instance(
        pretrained=MODEL.backbone_pretrained,
        freeze_until="layer3",  # train layer4 + fc
        num_classes=2,
    ).to(dev)

    # Class weights for CE: rare class up-weighted.
    cw = torch.tensor([1.0, max(1.0, n0_tr / max(1, n1_tr))], dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=TRAIN.instance_lr, weight_decay=TRAIN.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    stopper = EarlyStopper(patience=TRAIN.early_stop_patience, mode="max")

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_inst_auroc = -1.0
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

        # Instance val.
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(dev, non_blocking=True)
                p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
                ps.append(p)
                ys.append(y.numpy())
        y_inst = np.concatenate(ys)
        p_inst = np.concatenate(ps)
        from sklearn.metrics import roc_auc_score
        inst_auroc = float(roc_auc_score(y_inst, p_inst)) if len(np.unique(y_inst)) > 1 else float("nan")

        improved = stopper.step(inst_auroc)
        if improved:
            best_inst_auroc = inst_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 10 == 0 or stopper.should_stop:
            print(f"    ep {ep:3d}  train_loss={train_loss:.4f}  inst_val_auroc={inst_auroc:.4f}"
                  f"  best={best_inst_auroc:.4f}  ({t.lap():.1f}s)")
        if stopper.should_stop:
            print(f"    early stop at epoch {ep}")
            break

    # Restore best.
    model.load_state_dict(best_state)
    model.eval()

    # Bag-level eval via max-pool aggregation.
    bag_y, bag_score, bag_count_true, bag_count_pred = [], [], [], []
    with torch.no_grad():
        for i in range(len(bag_ds)):
            x, y, c, _ = bag_ds[i]
            x = x.to(dev)
            p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            bag_score.append(float(p.max()))
            bag_y.append(int(y.item()))
            bag_count_true.append(float(c.item()))
            bag_count_pred.append(float((p >= 0.5).sum()))
    bag_y = np.array(bag_y)
    bag_score = np.array(bag_score)
    bag_metrics = summarize(bag_y, bag_score, n_resamples=1000, seed=TRAIN.seed)
    mae = count_mae(np.array(bag_count_true), np.array(bag_count_pred))

    # Save model.
    ckpt_path = PATHS.models / f"resnet18_fold_{fold}.pt"
    save_ckpt(
        ckpt_path,
        model,
        meta={
            "fold": fold,
            "epochs_trained": ep,
            "best_inst_auroc": best_inst_auroc,
            "bag_auroc_point": bag_metrics["auroc"]["point"],
            "trainable_params": count_params(model),
            "provenance": provenance(seed=TRAIN.seed, cfg={"model": "resnet18", "fold": fold}),
        },
    )

    return {
        "fold": fold,
        "epochs_trained": ep,
        "best_inst_auroc": best_inst_auroc,
        "bag": bag_metrics,
        "count_mae": mae,
        "checkpoint": str(ckpt_path.relative_to(PATHS.root)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="single fold; default = all 5")
    parser.add_argument("--epochs", type=int, default=TRAIN.instance_pretrain_epochs)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(TRAIN.seed)
    ensure_dirs()

    folds = [args.fold] if args.fold is not None else list(range(DATA.n_folds))
    results = []
    for k in folds:
        print(f"\n=== ResNet-18 instance classifier — fold {k} ===")
        r = train_one_fold(k, epochs=args.epochs, batch_size=args.batch_size,
                           num_workers=args.num_workers)
        results.append(r)

    # Per-fold CSV
    csv_path = PATHS.results / "04_resnet18_per_fold.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["fold", "epochs", "inst_auroc",
                "bag_auroc", "bag_auroc_lo", "bag_auroc_hi",
                "bag_auprc", "bag_auprc_lo", "bag_auprc_hi",
                "bag_f1", "bag_bacc", "count_mae"]
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            b = r["bag"]
            w.writerow([
                r["fold"], r["epochs_trained"], f"{r['best_inst_auroc']:.4f}",
                f"{b['auroc']['point']:.4f}", f"{b['auroc']['ci_low']:.4f}", f"{b['auroc']['ci_high']:.4f}",
                f"{b['auprc']['point']:.4f}", f"{b['auprc']['ci_low']:.4f}", f"{b['auprc']['ci_high']:.4f}",
                f"{b['f1']['point']:.4f}", f"{b['balanced_acc']['point']:.4f}",
                f"{r['count_mae']:.4f}",
            ])

    write_json(
        PATHS.results / "04_resnet18_summary.json",
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

    print("\n=== Aggregate ===")
    print(f"  instance AUROC : mean={np.mean([r['best_inst_auroc'] for r in results]):.4f}")
    auroc_pts = [r["bag"]["auroc"]["point"] for r in results]
    print(f"  bag AUROC      : mean={np.mean(auroc_pts):.4f}  per-fold={['%.3f' % x for x in auroc_pts]}")
    print(f"\nper-fold CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
