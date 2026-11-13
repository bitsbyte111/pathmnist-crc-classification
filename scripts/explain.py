#!/usr/bin/env python
"""SHAP (and Grad-CAM) explanations for a trained run - proposal SO2.

    python scripts/explain.py --run runs/e1_e3_benchmark/P4_convnext_tiny/seed42
    python scripts/explain.py --run ... --method partition --per-class 2
    python scripts/explain.py --run ... --stain-space            # haematoxylin / eosin attributions

Writes to <run>/explain/: shap_*.png, channel_importance.{png,json}, sanity_check.json, gradcam.png.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from crc_tissuenet.data import CLASS_NAMES
from crc_tissuenet.engine import load_trained_model, predict
from crc_tissuenet.explain.shap_explain import (channel_importance, explain, randomization_sanity_check,
                                                select_examples)
from crc_tissuenet.utils import get_logger, save_json
from crc_tissuenet.viz import plot_channel_importance, plot_shap_grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory containing best.pt and config.yaml")
    ap.add_argument("--method", choices=["gradient", "partition"], default="gradient")
    ap.add_argument("--stain-space", action="store_true", help="attribute to H&E stains (gradient method)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=2, help="correct (and misclassified) examples per class")
    ap.add_argument("--background", type=int, default=100, help="training images used as SHAP background")
    ap.add_argument("--nsamples", type=int, default=200)
    ap.add_argument("--max-evals", type=int, default=500)
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    log = get_logger()
    run = Path(args.run)
    out = run / "explain"
    out.mkdir(exist_ok=True)

    model, data, cfg, device = load_trained_model(run)
    pred_file = run / "predictions.npz"
    if pred_file.exists() and f"{args.split}_probs" in np.load(pred_file):
        p = np.load(pred_file)
        probs, labels = p[f"{args.split}_probs"], p[f"{args.split}_labels"]
    else:
        probs, labels = predict(model, data.loaders[args.split], device)
    idx = select_examples(labels, probs, args.per_class, include_errors=True, seed=args.seed)
    images = np.asarray(data.arrays[args.split].images[idx])
    rng = np.random.default_rng(args.seed)
    bg_idx = np.sort(rng.choice(len(data.arrays["train"]), args.background, replace=False))
    background = np.asarray(data.arrays["train"].images[bg_idx])
    log.info("Explaining %d %s images with %s SHAP (%d background)", len(idx), args.split, args.method, len(bg_idx))

    res = explain(model, images, labels[idx], background, method=args.method, nsamples=args.nsamples,
                  max_evals=args.max_evals, stain_space=args.stain_space)
    tag = res.method
    np.savez_compressed(out / f"shap_{tag}.npz", values=res.values, indices=idx, labels=res.labels, preds=res.preds)
    correct = res.labels == res.preds
    plot_shap_grid(_subset(res, correct), out / f"shap_{tag}_correct.png", max_rows=len(CLASS_NAMES) * args.per_class)
    if (~correct).any():
        plot_shap_grid(_subset(res, ~correct), out / f"shap_{tag}_errors.png", max_rows=len(CLASS_NAMES) * args.per_class,
                       title=f"SHAP ({tag}) on misclassified patches")
    imp = channel_importance(res)
    save_json(imp, out / f"channel_importance_{tag}.json")
    plot_channel_importance(imp, out / f"channel_importance_{tag}.png")

    if not args.no_sanity and args.method == "gradient":
        k = min(12, len(images))
        sanity = randomization_sanity_check(model, images[:k], background[:50], nsamples=min(100, args.nsamples))
        save_json(sanity, out / "sanity_check.json")
        log.info("Randomisation sanity check: mean Spearman(trained, random) = %.3f (low is good)",
                 sanity["spearman_mean"])

    try:  # Grad-CAM for CNN backbones
        from crc_tissuenet.explain.gradcam import grad_cam
        import matplotlib.pyplot as plt
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255).to(device)
        cams = grad_cam(model, x)
        n = min(len(images), 18)
        fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4), squeeze=False)
        for i in range(n):
            axes[0, i].imshow(images[i])
            axes[0, i].set_title(f"{CLASS_NAMES[res.labels[i]]}->{CLASS_NAMES[res.preds[i]]}", fontsize=6)
            axes[1, i].imshow(images[i])
            axes[1, i].imshow(cams[i], cmap="jet", alpha=0.45)
            axes[0, i].axis("off")
            axes[1, i].axis("off")
        fig.tight_layout()
        fig.savefig(out / "gradcam.png", dpi=150)
        plt.close(fig)
    except ValueError as e:
        log.info("Grad-CAM skipped: %s", e)
    log.info("Explanations written to %s", out)


def _subset(res, mask):
    from dataclasses import replace
    return replace(res, images=res.images[mask], labels=res.labels[mask], preds=res.preds[mask],
                   probs=res.probs[mask], values=res.values[mask])


if __name__ == "__main__":
    main()
