"""End-to-end smoke tests on synthetic data (CPU, no downloads)."""
import json

import numpy as np

from crc_tissuenet.ablation import candidate_config, run_ablation
from crc_tissuenet.config import apply_overrides
from crc_tissuenet.engine import load_trained_model, run_experiment
from crc_tissuenet.explain.shap_explain import channel_importance, explain, randomization_sanity_check


def test_train_learns_and_writes_artifacts(synthetic_cfg, tmp_path):
    cfg = apply_overrides(synthetic_cfg, {"eval.test": True})
    res = run_experiment(cfg, tmp_path / "run")
    assert res["val"]["macro_f1"] > 0.5            # far above chance (0.11) on learnable data
    for f in ("config.yaml", "history.csv", "metrics.json", "predictions.npz", "best.pt"):
        assert (tmp_path / "run" / f).exists()
    assert "test_ci95" in res and "clinical_errors" in res["test"]
    again = run_experiment(cfg, tmp_path / "run")  # finished runs are skipped
    assert again["val"]["macro_f1"] == res["val"]["macro_f1"]


def test_reproducible_with_same_seed(synthetic_cfg, tmp_path):
    a = run_experiment(synthetic_cfg, tmp_path / "a")
    b = run_experiment(synthetic_cfg, tmp_path / "b")
    assert a["val"]["accuracy"] == b["val"]["accuracy"]


def test_test_set_not_used_by_default(synthetic_cfg, tmp_path):
    res = run_experiment(synthetic_cfg, tmp_path / "r")
    assert "test" not in res
    assert "test_probs" not in np.load(tmp_path / "r" / "predictions.npz")


def test_scale_override():
    base = {"train": {"lr": 0.01}}
    cfg, inc = candidate_config(base, {}, {"__scale__": {"train.lr": 0.1}})
    assert abs(cfg["train"]["lr"] - 0.001) < 1e-12 and abs(inc["train.lr"] - 0.001) < 1e-12


def test_mini_ablation(tmp_path, synthetic_cfg):
    from crc_tissuenet.config import save_config
    import yaml

    save_config(apply_overrides(synthetic_cfg, {"train.epochs": 1}), tmp_path / "base.yaml")
    plan = {"name": "mini", "base": "base.yaml", "seeds": [0],
            "stages": [{"name": "activation", "candidates": {"relu": {"model.params.activation": "relu"},
                                                             "tanh": {"model.params.activation": "tanh"}}},
                       {"name": "lr", "candidates": {"x1": {}, "x0.1": {"__scale__": {"train.lr": 0.1}}}}]}
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))
    state = run_ablation(tmp_path / "plan.yaml", tmp_path / "out", best_config_out=tmp_path / "best.yaml")
    assert state["completed"] == ["activation", "lr"]
    assert (tmp_path / "best.yaml").exists() and (tmp_path / "out" / "ablation_results.csv").exists()
    # The 'x1' candidate equals the incumbent from stage 1 -> cached, not re-trained
    assert len(json.load(open(tmp_path / "out" / "state.json"))["cache"]) == 3


def test_shap_gradient_and_stain(synthetic_cfg, tmp_path):
    cfg = apply_overrides(synthetic_cfg, {"eval.test": True})
    run_experiment(cfg, tmp_path / "run")
    model, data, _, _ = load_trained_model(tmp_path / "run", device="cpu")
    imgs, labels = data.arrays["test"].images[:6], data.arrays["test"].labels[:6]
    bg = data.arrays["train"].images[:20]
    res = explain(model, imgs, labels, bg, method="gradient", nsamples=20)
    assert res.values.shape == (6, 28, 28, 3)
    # SHAP completeness: attributions sum ~ f(x) - E[f(background)] for the explained class
    assert np.isfinite(res.values).all()
    stain = explain(model, imgs, labels, bg, method="gradient", nsamples=20, stain_space=True)
    assert stain.channel_names[0] == "Haematoxylin"
    imp = channel_importance(stain)
    assert all(abs(sum(v.values()) - 1) < 1e-6 for v in imp.values())
    sanity = randomization_sanity_check(model, imgs[:3], bg, nsamples=10)
    assert -1 <= sanity["spearman_mean"] <= 1


def test_shap_partition(synthetic_cfg, tmp_path):
    cfg = apply_overrides(synthetic_cfg, {"train.epochs": 1})
    run_experiment(cfg, tmp_path / "run")
    model, data, _, _ = load_trained_model(tmp_path / "run", device="cpu")
    res = explain(model, data.arrays["val"].images[:2], data.arrays["val"].labels[:2], None,
                  method="partition", max_evals=64)
    assert res.values.shape == (2, 28, 28, 3)


def test_observations_written(tmp_path, synthetic_cfg):
    import yaml
    from crc_tissuenet.config import save_config
    from crc_tissuenet.observations import stage_observations, stage_table, write_observations
    import pandas as pd

    save_config(apply_overrides(synthetic_cfg, {"train.epochs": 3}), tmp_path / "base.yaml")
    plan = {"name": "obs", "base": "base.yaml", "seeds": [0],
            "stages": [{"name": "activation", "concept": "c", "expect": "sigmoid vanishes",
                        "candidates": {"relu": {"model.params.activation": "relu"},
                                       "sigmoid": {"model.params.activation": "sigmoid"}}}]}
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))
    run_ablation(tmp_path / "plan.yaml", tmp_path / "out")
    md = write_observations(tmp_path / "out", plan).read_text()
    assert "**Hypothesis.** sigmoid vanishes" in md and "Ranking by validation macro F1" in md
    t = stage_table(pd.read_csv(tmp_path / "out" / "ablation_results.csv"), "activation")
    assert set(t.candidate) == {"relu", "sigmoid"} and t.best_f1.notna().all()
    assert any("relu" in o for o in stage_observations(t))


def test_gpu_log_plot(tmp_path):
    from crc_tissuenet.viz import plot_gpu_log

    rows = ["2026/11/10 10:00:00.000, 0, 80, 9000", "2026/11/10 10:00:00.000, 1, 70, 8000",
            "2026/11/10 10:01:00.000, 0, 90, 9100", "2026/11/10 10:01:00.000, 1, 60, 8100"]
    (tmp_path / "g.csv").write_text("\n".join(rows))
    assert plot_gpu_log(tmp_path / "g.csv", tmp_path / "g.png") == {0: 85.0, 1: 65.0}
