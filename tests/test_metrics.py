import numpy as np

from crc_tissuenet.metrics import (bootstrap_ci, compute_metrics, expected_calibration_error, holm_correction,
                                   mcnemar_test, paired_bootstrap)


def _probs(y, k=9, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    p = np.full((len(y), k), 0.01) + rng.random((len(y), k)) * noise
    p[np.arange(len(y)), y] += 1
    return p / p.sum(1, keepdims=True)


def test_perfect_predictions():
    y = np.repeat(np.arange(9), 20)
    m = compute_metrics(y, _probs(y))
    assert m["accuracy"] == m["macro_f1"] == m["balanced_accuracy"] == m["macro_auc"] == 1.0
    assert np.trace(np.array(m["confusion_matrix"])) == len(y)


def test_bootstrap_ci_contains_point():
    y = np.repeat(np.arange(9), 30)
    p = _probs(y, noise=3.0, seed=1)
    acc = compute_metrics(y, p)["accuracy"]
    lo, hi = bootstrap_ci(y, p, n_boot=200)["accuracy"]
    assert lo <= acc <= hi


def test_paired_tests():
    y = np.repeat(np.arange(9), 50)
    good, bad = _probs(y, noise=0.5, seed=2), _probs(y, noise=4.0, seed=3)
    pb = paired_bootstrap(y, good, bad, n_boot=200)
    assert pb["difference"] > 0 and pb["p_value"] < 0.05
    same = mcnemar_test(y, good.argmax(1), good.argmax(1))
    assert same["p_value"] == 1.0


def test_ece_and_holm():
    y = np.zeros(1000, dtype=int)
    p = np.tile([1.0] + [0.0] * 8, (1000, 1))
    assert expected_calibration_error(y, p) < 1e-9
    assert holm_correction([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
