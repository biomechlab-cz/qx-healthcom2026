"""Train the quantum aggregator as a per-CROP classifier.

Architecture for this stage:
  frozen ResNet-18 -> projector(512->64->8, ReLU, Tanh) -> scale(pi/2)
    -> quantum circuit (alpha in [-1, 1]) -> Linear(1, 1) -> BCE
  (effectively: quantum output is used as logit for DC vs MC at the crop level)

Per fold:
  - train on instance_pretrain.json train crops (with class-balanced sampler)
  - val on instance_pretrain.json val crops
  - save best by val instance AUROC

Output checkpoints: results/models/qinst_fold_{k}.pt

Evaluation (no extra training): use the trained quantum as attention. For each
val bag, bag_score = max(alpha_k). Compare bag AUROC.
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

from utils.augmentation import build_transforms
from utils.config import DATA, MODEL, PATHS, TRAIN, ensure_dirs
from utils.dataset import (
    BagDataset, CropDataset, load_fold, load_instance_split, make_weighted_sampler,
)
from utils.models import build_resnet18_feature_extractor
from utils.quantum_aggregator import QuantumAttention, QuantumConfig
from utils.seed import provenance, set_seed, write_json
from utils.training import EarlyStopper, Timer, device, save_ckpt


class QInstance(nn.Module):
    """Quantum binary instance classifier. Same projector+quantum stack as HQ-MIL, plus a 1-d head."""

    def __init__(self, backbone: nn.Module, qcfg: QuantumConfig,
                 projector_hidden: int = 64):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, projector_hidden), nn.ReLU(),
            nn.Linear(projector_hidden, qcfg.n_qubits), nn.Tanh(),
        )
        self.quantum = QuantumAttention(qcfg)
        # alpha (per-crop) is in [-1, 1]; map through a tiny head so we can scale + bias.
        self.head = nn.Linear(1, 1)
        import math as _math
        self._scale = _math.pi / 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W). Backbone+projector run on whatever device x is on; quantum runs on CPU.
        with torch.no_grad():
            phi = self.backbone(x)
        z = self.projector(phi) * self._scale
        alpha = self.quantum(z)              # (B,) on CPU
        logits = self.head(alpha.unsqueeze(-1)).squeeze(-1)  # (B,)
        return logits, alpha


def train_one_fold(fold: int, epochs: int, batch_size: int, qcfg: QuantumConfig) -> dict:
    dev = device()
    train_tf, val_tf = build_transforms()
    it_ids, iv_ids = load_instance_split(fold)
    train_ds = CropDataset(crop_ids=it_ids, transform=train_tf)
    val_ds = CropDataset(crop_ids=iv_ids, transform=val_tf)

    n0, n1 = train_ds.class_counts()
    print(f"  fold {fold}: inst train n0={n0} n1={n1}  val n={len(val_ds)}")
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    backbone = build_resnet18_feature_extractor(str(PATHS.models / f"resnet18_fold_{fold}.pt"))
    backbone = backbone.to(dev)

    model = QInstance(backbone, qcfg).to(dev)
    # Move quantum to CPU explicitly (its forward expects CPU).
    model.quantum = model.quantum.cpu()

    # Class-balanced BCE.
    pos_weight = torch.tensor([n0 / max(1, n1)], dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Two parameter groups: classical (small lr), quantum (larger lr).
    classical_params = list(model.projector.parameters()) + list(model.head.parameters())
    quantum_params = list(model.quantum.parameters())
    optim = torch.optim.AdamW(
        [{"params": classical_params, "lr": TRAIN.classical_lr, "weight_decay": TRAIN.weight_decay},
         {"params": quantum_params,   "lr": TRAIN.quantum_lr,   "weight_decay": 0.0}],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    stopper = EarlyStopper(patience=10, mode="max")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_auroc = -1.0
    t = Timer()

    from sklearn.metrics import roc_auc_score
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0; n_seen = 0
        for x, y, _ in train_loader:
            x = x.to(dev); y = y.to(dev).float()
            logits, _ = model(x)
            logits = logits.to(dev)
            loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_loss += loss.item() * x.size(0); n_seen += x.size(0)
        sched.step()
        train_loss /= max(1, n_seen)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(dev)
                logits, _ = model(x)
                p = torch.sigmoid(logits).cpu().numpy()
                ps.append(p); ys.append(y.numpy())
        y_inst = np.concatenate(ys); p_inst = np.concatenate(ps)
        auroc = float(roc_auc_score(y_inst, p_inst)) if len(np.unique(y_inst)) > 1 else float("nan")

        improved = stopper.step(auroc if not np.isnan(auroc) else -1.0)
        if improved:
            best_auroc = auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    ep {ep:3d}  loss={train_loss:.4f}  inst_val_auroc={auroc:.4f}  best={best_auroc:.4f}  ({t.lap():.1f}s)")
        if stopper.should_stop:
            print(f"    early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    save_ckpt(PATHS.models / f"qinst_fold_{fold}.pt", model, meta={
        "fold": fold, "best_inst_auroc": best_auroc,
        "quantum_cfg": qcfg.__dict__,
        "provenance": provenance(seed=TRAIN.seed, cfg={"model": "qinst", "fold": fold}),
    })

    # Stage-2 evaluation: aggregate per-instance alphas across val bags via max/top-k.
    _, val_imgs = load_fold(fold)
    bag_ds = BagDataset(image_ids=val_imgs, transform=val_tf)
    model.eval()
    bag_y, bag_max, bag_top3, bag_mean = [], [], [], []
    with torch.no_grad():
        for i in range(len(bag_ds)):
            x, y, _c, _ = bag_ds[i]
            x = x.to(dev)
            _logits, alpha = model(x)
            a = alpha.cpu().numpy()
            bag_y.append(int(y.item()))
            bag_max.append(float(a.max()))
            top3 = np.sort(a)[-3:]
            bag_top3.append(float(top3.mean()))
            bag_mean.append(float(a.mean()))
    bag_y = np.array(bag_y)
    out = {"fold": fold, "best_inst_auroc": best_auroc,
           "n_val_bags": len(bag_y), "bag_y": bag_y.tolist(),
           "bag_score_max": bag_max, "bag_score_top3": bag_top3, "bag_score_mean": bag_mean}
    if len(np.unique(bag_y)) > 1:
        out["bag_auroc_max"] = float(roc_auc_score(bag_y, bag_max))
        out["bag_auroc_top3"] = float(roc_auc_score(bag_y, bag_top3))
        out["bag_auroc_mean"] = float(roc_auc_score(bag_y, bag_mean))
        print(f"  fold {fold}: inst AUROC={best_auroc:.3f}  "
              f"bag AUROC max={out['bag_auroc_max']:.3f}  "
              f"top3={out['bag_auroc_top3']:.3f}  mean={out['bag_auroc_mean']:.3f}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(TRAIN.seed)
    ensure_dirs()
    qcfg = QuantumConfig()
    folds = [args.fold] if args.fold is not None else list(range(DATA.n_folds))

    results = []
    for k in folds:
        print(f"\n=== Quantum instance pretrain (stage 1) — fold {k} ===")
        results.append(train_one_fold(k, args.epochs, args.batch_size, qcfg))

    write_json(PATHS.results / "07c_qinst_summary.json", {
        "provenance": provenance(seed=TRAIN.seed, cfg={"epochs": args.epochs}),
        "per_fold": results,
    })

    if all("bag_auroc_max" in r for r in results):
        import numpy as np
        print("\n=== Aggregate ===")
        for key in ("max", "top3", "mean"):
            mean = float(np.mean([r[f"bag_auroc_{key}"] for r in results]))
            std = float(np.std([r[f"bag_auroc_{key}"] for r in results], ddof=1) if len(results) > 1 else 0)
            print(f"  bag AUROC ({key}): mean={mean:.3f} +/- {std:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
