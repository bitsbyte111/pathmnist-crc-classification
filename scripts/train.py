#!/usr/bin/env python
"""Train and evaluate a single configuration.

    python scripts/train.py --config configs/models/b1_cnn.yaml
    python scripts/train.py --config configs/models/p1_resnet50.yaml -o eval.test=true train.epochs=5
    python scripts/train.py --config configs/models/p7_mixer_b16.yaml --platform kaggle -o train.data_parallel=true
"""
import argparse

import _bootstrap  # noqa: F401

from crc_tissuenet.config import cli_overrides, load_config
from crc_tissuenet.engine import run_dir_for, run_experiment
from crc_tissuenet.viz import plot_confusion_matrix, plot_history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", nargs="*", default=[], help="dotted overrides, e.g. train.lr=3e-4")
    ap.add_argument("--platform", default=None, help="kaggle | colab | local | path/to/profile.yaml")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, cli_overrides(args.platform, args.override))
    run_dir = args.run_dir or run_dir_for(cfg)
    res = run_experiment(cfg, run_dir, overwrite=args.overwrite)
    from pathlib import Path
    rd = Path(run_dir)
    if (rd / "history.csv").exists():
        plot_history(rd / "history.csv", rd / "learning_curves.png", title=cfg["experiment"]["name"])
    split = "test" if "test" in res else "val"
    if "confusion_matrix" in res.get(split, {}):
        plot_confusion_matrix(res[split]["confusion_matrix"], rd / f"confusion_{split}.png",
                              title=f"{cfg['experiment']['name']} ({split})")


if __name__ == "__main__":
    main()
