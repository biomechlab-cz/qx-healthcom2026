"""Stacked bag classifier: train a tiny LogReg on per-bag aggregation features.

For each fold, for each training and val bag:
  Compute features from per-crop probabilities from THREE instance classifiers:
    - ResNet-18 instance      (fold-k model, results/models/resnet18_fold_{k}.pt)
    - Modified VGG-19         (fold-k model, results/models/vgg19_fold_{k}.pt)
    - QInstance (quantum)     (fold-k model, results/models/qinst_fold_{k}.pt)
  Per scorer: max, mean, top-3 mean, num crops with P >= 0.5  -> 4 features each
  Total: 12 features per bag.

Train LogReg (L2 regularization) on the fold's training bags; evaluate on val bags.
Output:
  results/07f_stacked_per_fold.csv
  results/07f_stacked_summary.json
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils.augmentation import build_transforms
from utils.config import DATA, PATHS, TRAIN
from utils.dataset import BagDataset, load_fold
from utils.models import build_resnet18_feature_extractor, build_resnet18_instance, build_vgg19_modified
from utils.quantum_aggregator import QuantumConfig
from utils.seed import provenance, set_seed, write_json
from utils.training import device

# Load QInstance class from script 07c.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_07c", Path(__file__).resolve().parent / "07c_pretrain_quantum_instance.py")
_07c = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_07c)


def load_resnet(fold: int, dev):
    state = torch.load(PATHS.models / f"resnet18_fold_{fold}.pt", map_location="cpu", weights_only=False)
    m = build_resnet18_instance(pretrained=False, freeze_until="", num_classes=2).to(dev)
    m.load_state_dict(state["model_state"]); m.eval(); return m


def load_vgg(fold: int, dev):
    state = torch.load(PATHS.models / f"vgg19_fold_{fold}.pt", map_location="cpu", weights_only=False)
    m = build_vgg19_modified(pretrained=False, num_classes=2).to(dev)
    m.load_state_dict(state["model_state"]); m.eval(); return m


def load_qinst(fold: int, dev):
    backbone = build_resnet18_feature_extractor(str(PATHS.models / f"resnet18_fold_{fold}.pt")).to(dev)
    qcfg = QuantumConfig()
    m = _07c.QInstance(backbone, qcfg).to(dev)
    m.quantum = m.quantum.cpu()
    state = torch.load(PATHS.models / f"qinst_fold_{fold}.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(state["model_state"]); m.eval(); return m


def per_crop_probs(model, kind: str, x: torch.Tensor, dev) -> np.ndarray:
    """Return per-crop P(DC), shape (K,)."""
    with torch.no_grad():
        if kind == "resnet" or kind == "vgg":
            p = torch.softmax(model(x.to(dev)), dim=1)[:, 1].cpu().numpy()
        elif kind == "qinst":
            logits, _alpha = model(x.to(dev))
            p = torch.sigmoid(logits.cpu()).numpy()
        else:
            raise ValueError(kind)
    return p


def bag_features(p: np.ndarray) -> tuple[float, float, float, float]:
    """4 features: max, mean, top-3 mean, num >= 0.5."""
    return float(p.max()), float(p.mean()), float(np.sort(p)[-3:].mean()), float((p >= 0.5).sum())


def collect_bag_features(fold: int, img_ids: list[str], dev, val_tf):
    """Build feature matrix X (n_bags x 12) and labels y for given image IDs in fold k."""
    resnet = load_resnet(fold, dev)
    vgg    = load_vgg(fold, dev)
    qinst  = load_qinst(fold, dev)

    ds = BagDataset(image_ids=img_ids, transform=val_tf)
    X = []; y = []; ids = []
    for i in range(len(ds)):
        x, label, _c, bag_id = ds[i]
        p_r = per_crop_probs(resnet, "resnet", x, dev)
        p_v = per_crop_probs(vgg, "vgg", x, dev)
        p_q = per_crop_probs(qinst, "qinst", x, dev)
        feats = list(bag_features(p_r)) + list(bag_features(p_v)) + list(bag_features(p_q))
        X.append(feats); y.append(int(label.item())); ids.append(bag_id)
    return np.array(X, dtype=float), np.array(y, dtype=int), ids


def main() -> int:
    set_seed(TRAIN.seed)
    dev = device()
    _, val_tf = build_transforms()

    per_fold = []
    all_val_y, all_val_p, all_val_ids = [], [], []
    for k in range(DATA.n_folds):
        train_imgs, val_imgs = load_fold(k)
        print(f"\n=== fold {k}: extracting features ===")
        Xtr, ytr, _ = collect_bag_features(k, train_imgs, dev, val_tf)
        Xva, yva, val_ids = collect_bag_features(k, val_imgs, dev, val_tf)
        print(f"  train bags: {len(ytr)} (pos={int(ytr.sum())})  val bags: {len(yva)} (pos={int(yva.sum())})")

        # Strong L2 regularization given 25 bags, 12 features.
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr); Xva_s = scaler.transform(Xva)
        clf = LogisticRegression(C=0.5, class_weight="balanced", solver="lbfgs", max_iter=1000)
        clf.fit(Xtr_s, ytr)
        p_va = clf.predict_proba(Xva_s)[:, 1]

        if len(np.unique(yva)) > 1:
            au = float(roc_auc_score(yva, p_va))
        else:
            au = float("nan")
        print(f"  val AUROC: {au:.3f}")
        per_fold.append({"fold": k, "n_train": int(len(ytr)), "n_val": int(len(yva)),
                         "val_auroc": au,
                         "feature_names": ["rn_max","rn_mean","rn_top3","rn_n50",
                                           "vgg_max","vgg_mean","vgg_top3","vgg_n50",
                                           "qi_max","qi_mean","qi_top3","qi_n50"],
                         "coefs": clf.coef_[0].tolist(),
                         "per_bag": [{"bag_id": b, "y": int(yi), "p": float(pi)}
                                     for b, yi, pi in zip(val_ids, yva, p_va)]})
        all_val_y.extend(yva.tolist()); all_val_p.extend(p_va.tolist()); all_val_ids.extend(val_ids)

    print("\n=== Pooled across 5 folds ===")
    all_val_y = np.array(all_val_y); all_val_p = np.array(all_val_p)
    pooled = float(roc_auc_score(all_val_y, all_val_p))
    print(f"  pooled bag AUROC: {pooled:.3f}  (n={len(all_val_y)}, pos={int(all_val_y.sum())}, neg={int((all_val_y==0).sum())})")

    fold_aurocs = [f["val_auroc"] for f in per_fold if not np.isnan(f["val_auroc"])]
    print(f"  per-fold mean AUROC: {np.mean(fold_aurocs):.3f} +/- "
          f"{np.std(fold_aurocs, ddof=1):.3f}  per-fold={['%.3f' % x for x in fold_aurocs]}")

    write_json(PATHS.results / "07f_stacked_summary.json",
               {"provenance": provenance(seed=TRAIN.seed, cfg={"method": "stacked_logreg"}),
                "per_fold": per_fold,
                "pooled_bag_auroc": pooled})
    csv_path = PATHS.results / "07f_stacked_per_fold.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold","n_train","n_val","val_auroc"])
        for r in per_fold:
            w.writerow([r["fold"], r["n_train"], r["n_val"], f"{r['val_auroc']:.4f}"])
    print(f"\ncsv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
