#!/usr/bin/env python
"""Optuna hyper-parameter search (TPE sampler + median pruning) on validation macro F1.

    python scripts/tune.py --space configs/tuning/optuna_cnn.yaml
"""
import argparse
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401

from crc_tissuenet.config import apply_overrides, cli_overrides, load_config, save_config
from crc_tissuenet.engine import run_experiment


def suggest(trial, name, spec):
    t = spec["type"]
    if t == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if t == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(t)


def main():
    import optuna

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", required=True)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--output-dir", default="runs/optuna")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--platform", default=None, help="kaggle | colab | local | path/to/profile.yaml")
    ap.add_argument("-o", "--override", nargs="*", default=[], help="dotted overrides for every trial")
    args = ap.parse_args()
    space_path = Path(args.space)
    spec = yaml.safe_load(open(space_path))
    base = apply_overrides(load_config(space_path.parent / spec["base"]),
                           {**spec.get("overrides", {}), **cli_overrides(args.platform, args.override)})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        params = {k: suggest(trial, k, v) for k, v in spec["search_space"].items()}
        cfg = apply_overrides(base, {**params, "experiment.seed": args.seed})

        def callback(epoch, row):
            trial.report(row["val_macro_f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        res = run_experiment(cfg, out / f"trial{trial.number:03d}", epoch_callback=callback)
        return res["val"]["macro_f1"]

    study = optuna.create_study(
        study_name="crc_tissuenet", direction=spec.get("direction", "maximize"),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=f"sqlite:///{out / 'study.db'}", load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials or spec.get("n_trials", 20))
    print("Best value:", study.best_value, "\nBest params:", study.best_params)
    best = apply_overrides(load_config(space_path.parent / spec["base"]), study.best_params)
    best["experiment"]["name"] = "B1_cnn_optuna"
    save_config(best, out / "best_config.yaml")
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)


if __name__ == "__main__":
    main()
