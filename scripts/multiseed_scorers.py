"""Multi-seed robustness for the contested per-crop comparison.

Retrains QInstance, classical-48, and classical-33k across several seeds on the
frozen ResNet-18 features and reports strict val-fold per-crop AUROC
(mean +/- std across seeds, per fold and overall). Does NOT overwrite the
seed-42 checkpoints used by the headline tables — nothing is saved to disk
except the summary JSON.

Usage: python scripts/multiseed_scorers.py [--seeds 42 1 2 3 4] [--epochs 30]
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.augmentation import build_transforms
from utils.config import DATA, PATHS, TRAIN
from utils.dataset import CropDataset, load_fold, load_instance_split, make_weighted_sampler
from utils.models import build_resnet18_feature_extractor
from utils.quantum_aggregator import QuantumConfig
from utils.seed import provenance, set_seed, write_json
from utils.training import EarlyStopper, device


def _imp(name: str, fname: str):
    spec = _ilu.spec_from_file_location(name, Path(__file__).resolve().parent / fname)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_07c = _imp("_07c", "07c_pretrain_quantum_instance.py")
_17 = _imp("_17", "17_ablation_classical_control.py")


def _build(model_name: str, fold: int, dev):
    bb = build_resnet18_feature_extractor(str(PATHS.models / f"resnet18_fold_{fold}.pt")).to(dev)
    if model_name == "QInstance":
        m = _07c.QInstance(bb, QuantumConfig()).to(dev)
        m.quantum = m.quantum.cpu()
        is_q = True
    else:
        hidden = _17.BUDGETS[model_name]
        m = _17.ClassicalInstance(bb, hidden).to(dev)
        is_q = False
    return m, is_q


def _train_and_eval(model_name: str, fold: int, epochs: int, batch_size: int, dev) -> float:
    train_tf, val_tf = build_transforms()
    it_ids, iv_ids = load_instance_split(fold)
    _, val_imgs = load_fold(fold)
    train_ds = CropDataset(crop_ids=it_ids, transform=train_tf)
    val_ds = CropDataset(crop_ids=iv_ids, transform=val_tf)
    n0, n1 = train_ds.class_counts()
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    strict_loader = DataLoader(CropDataset(image_ids=val_imgs, transform=val_tf),
                               batch_size=batch_size, shuffle=False, num_workers=0)

    model, is_q = _build(model_name, fold, dev)
    pos_weight = torch.tensor([n0 / max(1, n1)], dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if is_q:
        classical = list(model.projector.parameters()) + list(model.head.parameters())
        quantum = list(model.quantum.parameters())
        optim = torch.optim.AdamW(
            [{"params": classical, "lr": TRAIN.classical_lr, "weight_decay": TRAIN.weight_decay},
             {"params": quantum, "lr": TRAIN.quantum_lr, "weight_decay": 0.0}])
    else:
        params = list(model.projector.parameters()) + list(model.head.parameters())
        optim = torch.optim.AdamW(params, lr=TRAIN.classical_lr, weight_decay=TRAIN.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    stopper = EarlyStopper(patience=10, mode="max")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best = -1.0

    def _fwd(x):
        out = model(x)
        return out[0] if is_q else out

    for _ep in range(1, epochs + 1):
        model.train()
        for x, y, _ in train_loader:
            x = x.to(dev); y = y.to(dev).float()
            loss = loss_fn(_fwd(x).to(dev), y)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        sched.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                ps.append(torch.sigmoid(_fwd(x.to(dev)).cpu()).numpy()); ys.append(y.numpy())
        y_i = np.concatenate(ys); p_i = np.concatenate(ps)
        auc = float(roc_auc_score(y_i, p_i)) if len(np.unique(y_i)) > 1 else -1.0
        if stopper.step(auc):
            best = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.should_stop:
            break
    model.load_state_dict(best_state)

    # strict val-fold per-crop AUROC
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in strict_loader:
            ps.append(torch.sigmoid(_fwd(x.to(dev)).cpu()).numpy()); ys.append(y.numpy())
    y_s = np.concatenate(ys); p_s = np.concatenate(ps)
    return float(roc_auc_score(y_s, p_s)) if len(np.unique(y_s)) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    dev = device()
    models = ["QInstance", "classical48", "classical33k"]

    # strict_auroc[model][seed][fold]
    store: dict = {m: {} for m in models}
    for seed in args.seeds:
        for m in models:
            store[m][seed] = {}
        for fold in range(DATA.n_folds):
            for m in models:
                set_seed(seed)
                auc = _train_and_eval(m, fold, args.epochs, args.batch_size, dev)
                store[m][seed][fold] = auc
                print(f"seed={seed} fold={fold} {m:12s} strict_auroc={auc:.4f}", flush=True)

    summary: dict = {}
    print("\n" + "=" * 70)
    print(f"{'model':14s} {'per-seed fold-mean':>40}")
    print("=" * 70)
    for m in models:
        seed_means = []
        per_seed = {}
        for seed in args.seeds:
            fold_vals = [store[m][seed][f] for f in range(DATA.n_folds) if not np.isnan(store[m][seed][f])]
            sm = float(np.mean(fold_vals))
            seed_means.append(sm); per_seed[str(seed)] = {"fold_mean": sm, "per_fold": fold_vals}
        summary[m] = {
            "per_seed": per_seed,
            "seed_mean": float(np.mean(seed_means)),
            "seed_std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
        }
        sm_str = " ".join(f"{x:.3f}" for x in seed_means)
        print(f"{m:14s} mean={summary[m]['seed_mean']:.3f}+/-{summary[m]['seed_std']:.3f}  seeds=[{sm_str}]")

    out = PATHS.results / "20_multiseed_scorers.json"
    write_json(out, {
        "provenance": provenance(seed=TRAIN.seed, cfg={"seeds": args.seeds, "epochs": args.epochs}),
        "strict_auroc_raw": {m: {str(s): store[m][s] for s in args.seeds} for m in models},
        "summary": summary,
    })
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
