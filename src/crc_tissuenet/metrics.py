"""Evaluation metrics and statistical comparison (proposal Section 12).

* Point metrics: accuracy, balanced accuracy, macro F1, macro one-vs-rest AUC, log-loss,
  expected calibration error (ECE), per-class precision/recall/F1, confusion matrix.
* Uncertainty: percentile bootstrap 95% CIs over test patches.
* Model comparison: paired bootstrap of the macro-F1 difference and McNemar's test.

Caveat: patches are clustered within patients, but PathMNIST has no patient IDs, so
patch-level bootstrap CIs are likely somewhat too narrow. We report this as a limitation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_fscore_support, roc_auc_score)

from .data import CLASS_NAMES, NUM_CLASSES


def macro_auc(y: np.ndarray, probs: np.ndarray) -> float:
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    if len(present) == probs.shape[1]:
        return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    # Some classes absent (e.g. tiny bootstrap/debug sets): average AUC over present classes.
    return float(np.mean([roc_auc_score(y == c, probs[:, c]) for c in present]))


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


SCALAR_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": lambda y, p: float(accuracy_score(y, p.argmax(1))),
    "balanced_accuracy": lambda y, p: float(balanced_accuracy_score(y, p.argmax(1))),
    "macro_f1": lambda y, p: float(f1_score(y, p.argmax(1), average="macro",
                                            labels=np.arange(p.shape[1]), zero_division=0)),
    "macro_auc": macro_auc,
}


def compute_metrics(y: np.ndarray, probs: np.ndarray, class_names=CLASS_NAMES) -> dict:
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(1, keepdims=True)  # float32 softmax outputs may not sum exactly to 1
    k = probs.shape[1]
    out = {name: fn(y, probs) for name, fn in SCALAR_METRICS.items()}
    out["log_loss"] = float(log_loss(y, np.clip(probs, 1e-12, 1), labels=np.arange(k)))
    out["ece"] = expected_calibration_error(y, probs)
    prec, rec, f1, sup = precision_recall_fscore_support(y, probs.argmax(1), labels=np.arange(k),
                                                         zero_division=0)
    out["per_class"] = {class_names[c]: {"precision": float(prec[c]), "recall": float(rec[c]),
                                         "f1": float(f1[c]), "support": int(sup[c])} for c in range(k)}
    out["confusion_matrix"] = confusion_matrix(y, probs.argmax(1), labels=np.arange(k)).tolist()
    return out


def clinical_error_summary(y: np.ndarray, probs: np.ndarray, class_names=CLASS_NAMES) -> dict:
    """Errors that matter clinically: tumour (TUM) missed as normal mucosa or stroma, etc."""
    pred = probs.argmax(1)
    idx = {n: i for i, n in enumerate(class_names)}
    tum = y == idx["TUM"]
    out = {"tum_recall": float((pred[tum] == idx["TUM"]).mean()) if tum.any() else float("nan")}
    for other in ("NORM", "STR", "MUS", "DEB"):
        out[f"tum_as_{other.lower()}"] = int(((pred == idx[other]) & tum).sum())
    out["false_tumour_calls"] = int(((pred == idx["TUM"]) & ~tum).sum())
    return out


def bootstrap_ci(y: np.ndarray, probs: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0, metrics=("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")) -> dict:
    """Percentile bootstrap CIs (resampling test patches with replacement)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = {m: np.empty(n_boot) for m in metrics}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m in metrics:
            samples[m][b] = SCALAR_METRICS[m](y[idx], probs[idx])
    return {m: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for m, v in samples.items()}


def paired_bootstrap(y: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, metric: str = "macro_f1",
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Difference ``metric(A) - metric(B)`` on the same resampled patches.

    The two-sided p-value is the bootstrap estimate of P(difference has the opposite sign),
    doubled (a standard percentile-bootstrap test).
    """
    fn = SCALAR_METRICS[metric]
    rng = np.random.default_rng(seed)
    n = len(y)
    observed = fn(y, probs_a) - fn(y, probs_b)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = fn(y[idx], probs_a[idx]) - fn(y[idx], probs_b[idx])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"metric": metric, "difference": float(observed),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p_value": float(min(1.0, max(p, 1.0 / n_boot)))}


def mcnemar_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test on paired correctness (exact binomial when discordant pairs < 25)."""
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int((a_ok & ~b_ok).sum())  # A right, B wrong
    c = int((~a_ok & b_ok).sum())  # A wrong, B right
    if b + c == 0:
        return {"a_only_correct": b, "b_only_correct": c, "statistic": 0.0, "p_value": 1.0, "exact": True}
    if b + c < 25:
        p = float(stats.binomtest(b, b + c, 0.5).pvalue)
        return {"a_only_correct": b, "b_only_correct": c, "statistic": float(min(b, c)), "p_value": p, "exact": True}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"a_only_correct": b, "b_only_correct": c, "statistic": float(stat),
            "p_value": float(stats.chi2.sf(stat, 1)), "exact": False}


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (many models are compared with one reference)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


__all__ = ["compute_metrics", "bootstrap_ci", "paired_bootstrap", "mcnemar_test", "holm_correction",
           "expected_calibration_error", "clinical_error_summary", "macro_auc", "NUM_CLASSES"]
