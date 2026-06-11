"""Classical gated-attention MIL baseline.

Per Ilse et al. 2018 over a frozen ResNet-18 instance backbone (loaded from
results/models/resnet18_fold_{k}.pt). Trains attention + bag head + count head only.

Per fold k:
  - bags = images, 1 bag per step with gradient accumulation over `accumulation_steps` bags
  - loss = CE(bag) + lambda * MSE(count)
  - val on fold-k val images, bag-level AUROC + count MAE

Outputs:
  results/models/classical_mil_fold_{k}.pt
  results/05_classical_mil_per_fold.csv
  results/05_classical_mil_summary.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.augmentation import build_transforms  # noqa: E402
from utils.config import DATA, PATHS, TRAIN, ensure_dirs  # noqa: E402
from utils.dataset import BagDataset, load_fold  # noqa: E402
from utils.metrics import aggregate_folds, count_mae, summarize  # noqa: E402
from utils.models import build_classical_mil, count_params  # noqa: E402
from utils.seed import provenance, set_seed, write_json  # noqa: E402
from utils.training import EarlyStopper, Timer, device, save_ckpt  # noqa: E402


def train_one_fold(fold: int, epochs: int, count_loss_weight: float, accumulation_steps: int) -> dict:
    dev = device()
    train_tf, val_tf = build_transforms()

    train_imgs, val_imgs = load_fold(fold)
    train_bag = BagDataset(image_ids=train_imgs, transform=train_tf)
    val_bag = BagDataset(image_ids=val_imgs, transform=val_tf)

    backbone_ckpt = PATHS.models / f"resnet18_fold_{fold}.pt"
    if not backbone_ckpt.exists():
        raise FileNotFoundError(f"{backbone_ckpt} not found — run scripts/04_train_instance_classifier.py first")

    model = build_classical_mil(state_dict_path=str(backbone_ckpt)).to(dev)
    print(f"  fold {fold}: train bags={len(train_bag)} val bags={len(val_bag)}  "
          f"trainable={count_params(model)}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=TRAIN.classical_lr, weight_decay=TRAIN.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    stopper = EarlyStopper(patience=TRAIN.early_stop_patience, mode="max")

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_auroc = -float("inf")
    t = Timer()

    rng = np.random.default_rng(TRAIN.seed + fold)

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_bag))
        train_loss = 0.0
        optim.zero_grad(set_to_none=True)
        for step, bag_idx in enumerate(order):
            x, y, c, _ = train_bag[int(bag_idx)]
            x = x.to(dev, non_blocking=True)
            y = y.to(dev)
            c = c.to(dev)
            bag_logits, count_pred, _ = model(x)
            loss = ce(bag_logits.unsqueeze(0), y.unsqueeze(0)) + count_loss_weight * mse(count_pred, c)
            loss = loss / accumulation_steps
            loss.backward()
            train_loss += loss.item() * accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
        # Flush any remaining gradients.
        if (len(order) % accumulation_steps) != 0:
            optim.step()
            optim.zero_grad(set_to_none=True)
        sched.step()
        train_loss /= max(1, len(order))

        # Val.
        model.eval()
        ys, ss, ct_t, ct_p = [], [], [], []
        with torch.no_grad():
            for i in range(len(val_bag)):
                x, y, c, _ = val_bag[i]
                x = x.to(dev)
                bag_logits, count_pred, _ = model(x)
                p = torch.softmax(bag_logits, dim=0)[1].item()
                ss.append(p)
                ys.append(int(y.item()))
                ct_t.append(float(c.item()))
                ct_p.append(float(count_pred.item()))
        ys_a = np.array(ys); ss_a = np.array(ss)
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(ys_a, ss_a)) if len(np.unique(ys_a)) > 1 else float("nan")

        improved = stopper.step(auroc if not np.isnan(auroc) else -1.0)
        if improved:
            best_auroc = auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 20 == 0 or stopper.should_stop:
            print(f"    ep {ep:3d}  train_loss={train_loss:.4f}  val_auroc={auroc:.4f}"
                  f"  best={best_auroc:.4f}  ({t.lap():.1f}s)")
        if stopper.should_stop:
            print(f"    early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation with bootstrap CIs.
    ys, ss, ct_t, ct_p = [], [], [], []
    with torch.no_grad():
        for i in range(len(val_bag)):
            x, y, c, _ = val_bag[i]
            x = x.to(dev)
            bag_logits, count_pred, _ = model(x)
            p = torch.softmax(bag_logits, dim=0)[1].item()
            ss.append(p)
            ys.append(int(y.item()))
            ct_t.append(float(c.item()))
            ct_p.append(float(count_pred.item()))
    bag_y = np.array(ys); bag_s = np.array(ss)
    bag_metrics = summarize(bag_y, bag_s, n_resamples=1000, seed=TRAIN.seed)
    mae = count_mae(np.array(ct_t), np.array(ct_p))

    ckpt_path = PATHS.models / f"classical_mil_fold_{fold}.pt"
    save_ckpt(
        ckpt_path,
        model,
        meta={
            "fold": fold,
            "epochs_trained": ep,
            "best_val_auroc": best_auroc,
            "trainable_params": count_params(model),
            "provenance": provenance(seed=TRAIN.seed, cfg={"model": "classical_mil", "fold": fold}),
        },
    )

    return {
        "fold": fold,
        "epochs_trained": ep,
        "best_val_auroc": best_auroc,
        "bag": bag_metrics,
        "count_mae": mae,
        "checkpoint": str(ckpt_path.relative_to(PATHS.root)),
        "trainable_params": count_params(model),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=TRAIN.mil_epochs)
    parser.add_argument("--count-loss-weight", type=float, default=TRAIN.count_loss_weight)
    parser.add_argument("--accumulation-steps", type=int, default=TRAIN.accumulation_steps)
    args = parser.parse_args()

    set_seed(TRAIN.seed)
    ensure_dirs()

    folds = [args.fold] if args.fold is not None else list(range(DATA.n_folds))
    results = []
    for k in folds:
        print(f"\n=== Classical gated-attention MIL — fold {k} ===")
        r = train_one_fold(
            k, epochs=args.epochs,
            count_loss_weight=args.count_loss_weight,
            accumulation_steps=args.accumulation_steps,
        )
        results.append(r)

    csv_path = PATHS.results / "05_classical_mil_per_fold.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", "epochs", "bag_auroc", "bag_auroc_lo", "bag_auroc_hi",
                    "bag_auprc", "bag_f1", "bag_bacc", "count_mae", "trainable_params"])
        for r in results:
            b = r["bag"]
            w.writerow([
                r["fold"], r["epochs_trained"],
                f"{b['auroc']['point']:.4f}", f"{b['auroc']['ci_low']:.4f}", f"{b['auroc']['ci_high']:.4f}",
                f"{b['auprc']['point']:.4f}", f"{b['f1']['point']:.4f}", f"{b['balanced_acc']['point']:.4f}",
                f"{r['count_mae']:.4f}", r["trainable_params"],
            ])

    write_json(
        PATHS.results / "05_classical_mil_summary.json",
        {
            "provenance": provenance(seed=TRAIN.seed, cfg={"epochs": args.epochs}),
            "per_fold": results,
            "aggregate": {
                "bag_auroc": aggregate_folds([r["bag"] for r in results], "auroc"),
                "bag_auprc": aggregate_folds([r["bag"] for r in results], "auprc"),
                "bag_f1":    aggregate_folds([r["bag"] for r in results], "f1"),
            },
        },
    )

    auroc_pts = [r["bag"]["auroc"]["point"] for r in results]
    print("\n=== Aggregate ===")
    print(f"  bag AUROC : mean={np.nanmean(auroc_pts):.4f}  per-fold={['%.3f' % x for x in auroc_pts]}")
    print(f"\nper-fold CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
