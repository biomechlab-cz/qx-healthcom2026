"""DC-count regression from per-crop probabilities.

Per-image features (8 per scorer × 3 scorers + image-level K = 25 features):
  - max(P_DC), mean(P_DC), top-3 mean(P_DC)
  - #{P_DC >= 0.3}, #{P_DC >= 0.5}, #{P_DC >= 0.7}
  - sum(P_DC) over all crops
  - max - mean
plus K (#crops in the image).

Target: integer DC count per image (0-8 in our data). Treated as regression.

Models: Ridge (L2 linear), RandomForestRegressor, GradientBoostingRegressor.
Per-fold CV; report mean absolute error vs the constant-mean baseline (predict the
training-set mean count for every image).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.augmentation import build_transforms
from utils.config import DATA, PATHS, TRAIN
from utils.dataset import BagDataset, load_fold
from utils.models import build_resnet18_feature_extractor, build_resnet18_instance, build_vgg19_modified
from utils.quantum_aggregator import QuantumConfig
from utils.seed import provenance, set_seed, write_json
from utils.training import device

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_07c", Path(__file__).resolve().parent / "07c_pretrain_quantum_instance.py")
_07c = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_07c)


def _features_from_probs(p: np.ndarray) -> np.ndarray:
    if len(p) == 0:
        return np.zeros(8)
    return np.array([
        float(p.max()),
        float(p.mean()),
        float(np.sort(p)[-3:].mean()),
        float((p >= 0.3).sum()),
        float((p >= 0.5).sum()),
        float((p >= 0.7).sum()),
        float(p.sum()),
        float(p.max() - p.mean()),
    ])


def _load_resnet(fold, dev):
    state = torch.load(PATHS.models / f"resnet18_fold_{fold}.pt", map_location="cpu", weights_only=False)
    m = build_resnet18_instance(pretrained=False, freeze_until="", num_classes=2).to(dev)
    m.load_state_dict(state["model_state"]); m.eval(); return m


def _load_vgg(fold, dev):
    state = torch.load(PATHS.models / f"vgg19_fold_{fold}.pt", map_location="cpu", weights_only=False)
    m = build_vgg19_modified(pretrained=False, num_classes=2).to(dev)
    m.load_state_dict(state["model_state"]); m.eval(); return m


def _load_qinst(fold, dev):
    bb = build_resnet18_feature_extractor(str(PATHS.models / f"resnet18_fold_{fold}.pt")).to(dev)
    m = _07c.QInstance(bb, QuantumConfig()).to(dev)
    m.quantum = m.quantum.cpu()
    state = torch.load(PATHS.models / f"qinst_fold_{fold}.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(state["model_state"]); m.eval(); return m


def collect_features_one_fold(fold: int, img_ids: list[str], dev,
                              qinst_hw_by_bag: dict[str, np.ndarray] | None = None,
                              use_hw_for_qinst: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, image_ids) where X has 25 features per image and y is true DC count."""
    _, val_tf = build_transforms()
    resnet = _load_resnet(fold, dev)
    vgg = _load_vgg(fold, dev)
    qinst = _load_qinst(fold, dev)

    ds = BagDataset(image_ids=img_ids, transform=val_tf)
    X, y, ids = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, _y, c, bag_id = ds[i]
            x_dev = x.to(dev)
            p_r = torch.softmax(resnet(x_dev), dim=1)[:, 1].cpu().numpy()
            p_v = torch.softmax(vgg(x_dev), dim=1)[:, 1].cpu().numpy()
            if use_hw_for_qinst and qinst_hw_by_bag is not None and bag_id in qinst_hw_by_bag:
                p_q = qinst_hw_by_bag[bag_id]
            else:
                logits, _alpha = qinst(x_dev)
                p_q = torch.sigmoid(logits.cpu()).numpy()
            feats = np.concatenate([
                _features_from_probs(p_r),
                _features_from_probs(p_v),
                _features_from_probs(p_q),
                [float(len(p_r))],  # K
            ])
            X.append(feats); y.append(int(c.item())); ids.append(bag_id)
    return np.array(X, dtype=float), np.array(y, dtype=int), ids


def load_qinst_hw_predictions() -> dict[int, dict[str, np.ndarray]]:
    """Return per-fold mapping of bag_id -> per-crop hardware P(DC)."""
    out: dict[int, dict[str, np.ndarray]] = {}
    base = PATHS.hardware / "ibm_kingston"
    for k in range(DATA.n_folds):
        p = base / f"qinst_fold_{k}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        from collections import defaultdict
        by = defaultdict(list)
        for r in d["per_crop"]:
            by[r["bag_id"]].append(r["p_hw"])
        out[k] = {bid: np.array(v) for bid, v in by.items()}
    return out


def make_regressors():
    return {
        "Ridge (L2)": Ridge(alpha=5.0),
        "RandomForest (shallow)": RandomForestRegressor(
            n_estimators=200, max_depth=4, random_state=TRAIN.seed),
        "GBM (shallow)": GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=TRAIN.seed),
    }


def main() -> int:
    set_seed(TRAIN.seed)
    dev = device()
    qinst_hw = load_qinst_hw_predictions()

    print("Collecting per-image features (sim path)...")
    feats_sim = {}
    feats_hw = {}
    for k in range(DATA.n_folds):
        train_imgs, val_imgs = load_fold(k)
        Xtr, ytr, _ = collect_features_one_fold(k, train_imgs, dev, qinst_hw.get(k), use_hw_for_qinst=False)
        Xva, yva, vid = collect_features_one_fold(k, val_imgs, dev, qinst_hw.get(k), use_hw_for_qinst=False)
        feats_sim[k] = {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "ids": vid}

        # For hardware variant: replace QInstance features with hardware-derived ones in VALIDATION only
        # (training stays on sim since hw predictions exist only for val).
        Xva_hw, yva_hw, vid_hw = collect_features_one_fold(
            k, val_imgs, dev, qinst_hw.get(k), use_hw_for_qinst=True)
        feats_hw[k] = {"Xtr": Xtr, "ytr": ytr, "Xva": Xva_hw, "yva": yva_hw, "ids": vid_hw}
        print(f"  fold {k}: train={len(ytr)} val={len(yva)}  true count range val: "
              f"[{yva.min()}, {yva.max()}]  mean={yva.mean():.2f}")

    print("\n" + "=" * 80)
    print(f"{'Path':6s}  {'Regressor':22s}  {'mean MAE':>9}  {'std':>6}  {'baseline':>9}  per-fold MAE")
    print("=" * 80)

    overall = {}
    for path_name, feats in [("sim", feats_sim), ("hw", feats_hw)]:
        for name, _ in make_regressors().items():
            per_fold_mae, per_fold_baseline = [], []
            preds_all, truth_all = [], []
            for k in range(DATA.n_folds):
                f = feats[k]
                scaler = StandardScaler().fit(f["Xtr"])
                Xtr_s = scaler.transform(f["Xtr"]); Xva_s = scaler.transform(f["Xva"])
                reg = make_regressors()[name]
                reg.fit(Xtr_s, f["ytr"])
                pred = reg.predict(Xva_s)
                mae = float(np.mean(np.abs(pred - f["yva"])))
                baseline = float(np.mean(np.abs(f["yva"] - f["ytr"].mean())))
                per_fold_mae.append(mae); per_fold_baseline.append(baseline)
                preds_all.append(pred); truth_all.append(f["yva"])

            preds_all = np.concatenate(preds_all); truth_all = np.concatenate(truth_all)
            corr = float(np.corrcoef(truth_all, preds_all)[0, 1])
            mean_mae = float(np.mean(per_fold_mae))
            std_mae = float(np.std(per_fold_mae, ddof=1))
            mean_baseline = float(np.mean(per_fold_baseline))
            per_fold_str = " ".join(f"{m:.2f}" for m in per_fold_mae)
            print(f"{path_name:6s}  {name:22s}  {mean_mae:>9.3f}  {std_mae:>6.3f}  "
                  f"{mean_baseline:>9.3f}  {per_fold_str}  (pooled r={corr:+.3f})")
            overall[f"{path_name}:{name}"] = {
                "per_fold_mae": per_fold_mae, "mean_mae": mean_mae, "std_mae": std_mae,
                "per_fold_baseline": per_fold_baseline, "baseline_mean": mean_baseline,
                "pooled_pearson_r": corr,
                "preds": preds_all.tolist(), "truth": truth_all.tolist(),
            }

    write_json(PATHS.results / "14_count_regressor_summary.json",
               {"provenance": provenance(seed=TRAIN.seed, cfg={"method": "count_regressor"}),
                "results": overall})
    print(f"\nsaved: {PATHS.results / '14_count_regressor_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
