"""Metrics + bootstrap CIs for instance- and bag-level evaluation.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

# In small bootstrap resamples the val label vector can be all-one-class — that's expected
# noise, not a code problem. Suppress the warning at the source.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="A single label was found")


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def basic_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": _safe_auprc(y_true, y_score),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) at (1-alpha) confidence using a percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    samples: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            samples.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            samples.append(float("nan"))
    samples_arr = np.array(samples, dtype=float)
    valid = samples_arr[~np.isnan(samples_arr)]
    if len(valid) == 0:
        return metric_fn(y_true, y_score), float("nan"), float("nan")
    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    point = float(metric_fn(y_true, y_score))
    return point, lo, hi


def summarize(y_true: np.ndarray, y_score: np.ndarray, n_resamples: int = 1000, seed: int = 42) -> dict:
    """Return point estimate + 95% bootstrap CI for AUROC, AUPRC, F1, balanced acc."""
    pe, lo, hi = bootstrap_ci(y_true, y_score, _safe_auroc, n_resamples, seed)
    auroc = {"point": pe, "ci_low": lo, "ci_high": hi}
    pe, lo, hi = bootstrap_ci(y_true, y_score, _safe_auprc, n_resamples, seed)
    auprc = {"point": pe, "ci_low": lo, "ci_high": hi}

    def f1_at_05(y_t, y_s):
        return f1_score(y_t, (y_s >= 0.5).astype(int), zero_division=0)

    pe, lo, hi = bootstrap_ci(y_true, y_score, f1_at_05, n_resamples, seed)
    f1 = {"point": pe, "ci_low": lo, "ci_high": hi}

    def bacc_at_05(y_t, y_s):
        return balanced_accuracy_score(y_t, (y_s >= 0.5).astype(int))

    pe, lo, hi = bootstrap_ci(y_true, y_score, bacc_at_05, n_resamples, seed)
    bacc = {"point": pe, "ci_low": lo, "ci_high": hi}

    return {"auroc": auroc, "auprc": auprc, "f1": f1, "balanced_acc": bacc}


def aggregate_folds(per_fold: Iterable[dict], key: str) -> dict:
    vals = [f[key]["point"] for f in per_fold if not np.isnan(f[key]["point"])]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), "n": len(vals)}


def count_mae(y_true_counts: np.ndarray, y_pred_counts: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true_counts - y_pred_counts)))
