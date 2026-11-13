#!/usr/bin/env python
"""Build the benchmark report for a suite: tables, statistical tests and figures.

    python scripts/report.py --suite configs/suites/e1_e3_benchmark.yaml

Writes to reports/<suite>/: summary.csv, per_seed.csv, comparisons.csv, REPORT.md, figures/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from crc_tissuenet.data import CLASS_NAMES
from crc_tissuenet.metrics import compute_metrics
from crc_tissuenet.suite import collect_suite, compare_to_reference, ensemble_predictions, load_suite, summarise
from crc_tissuenet.utils import load_json
from crc_tissuenet.viz import PUBLISHED, plot_benchmark, plot_confusion_matrix, plot_reliability

TARGET_ACC, TARGET_AUC = 0.911, 0.990  # best published PathMNIST (MedMNIST v2) - PO1 / H1


def fmt(mean, sd):
    if pd.isna(mean):
        return "-"
    return f"{mean:.4f}" if pd.isna(sd) else f"{mean:.4f} ± {sd:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    suite = load_suite(args.suite)
    out = Path(args.out or Path("reports") / suite["name"])
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    df = collect_suite(args.suite, args.runs_dir)
    if df.empty:
        raise SystemExit("No finished runs found.")
    df.to_csv(out / "per_seed.csv", index=False)
    s = summarise(df)
    s.to_csv(out / "summary.csv", index=False)
    has_test = s["test_accuracy_mean"].notna().any()

    lines = [f"# {suite['name']} - results", "",
             f"Seeds per model: mean ± SD over seeds; 95% CI = patch-level bootstrap ({args.n_boot} resamples) "
             "for the first seed. Model selection used validation data only.", ""]
    if has_test:
        lines += ["| ID | Family | Params (M) | Input | Test accuracy | 95% CI | Macro F1 | Macro AUC | "
                  "Balanced acc | ECE | TUM recall | Train (min) | Infer (ms/img) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        ci_col = next((c for c in s.columns if c.startswith("test_accuracy_ci")), None)
        for _, r in s.iterrows():
            lines.append(
                f"| {r.id} | {r.family} | {r.params / 1e6:.2f} | {int(r.input_size)} | "
                f"{fmt(r.test_accuracy_mean, r.test_accuracy_std)} | {r[ci_col] if ci_col else '-'} | "
                f"{fmt(r.test_macro_f1_mean, r.test_macro_f1_std)} | {fmt(r.test_macro_auc_mean, r.test_macro_auc_std)} | "
                f"{fmt(r.test_balanced_accuracy_mean, r.test_balanced_accuracy_std)} | {r.test_ece_mean:.3f} | "
                f"{r.tum_recall_mean:.3f} | {r.train_time_s_mean / 60:.1f} | {r.inference_ms_mean:.3f} |")
        lines += ["", "Published MedMNIST v2 PathMNIST results (Yang et al. 2023): " +
                  "; ".join(f"{k}: acc {v[0]}, AUC {v[1]}" for k, v in PUBLISHED.items()), ""]
        best = s.sort_values("test_macro_f1_mean", ascending=False).iloc[0]
        met = best.test_accuracy_mean >= TARGET_ACC and best.test_macro_auc_mean >= TARGET_AUC
        lines += [f"**PO1 / H1** - best model by test macro F1: **{best.id}** "
                  f"(accuracy {best.test_accuracy_mean:.4f}, macro AUC {best.test_macro_auc_mean:.4f}). "
                  f"Target accuracy ≥ {TARGET_ACC} and AUC ≥ {TARGET_AUC}: **{'met' if met else 'not met'}** "
                  "(point estimates; see CIs).", ""]
        # Figures for the best model
        rd = Path(df[(df.id == best.id)].sort_values("seed").run_dir.iloc[0])
        m = load_json(rd / "metrics.json")
        plot_confusion_matrix(m["test"]["confusion_matrix"], figs / "confusion_best.png", f"{best.id} - test")
        p = np.load(rd / "predictions.npz")
        plot_reliability(p["test_labels"], p["test_probs"], figs / "reliability_best.png", title=f"{best.id} - test")
        plot_benchmark(s, figs / "accuracy_vs_params.png", "test_accuracy")
        plot_benchmark(s, figs / "macro_f1_vs_params.png", "test_macro_f1")
        lines += ["![acc vs params](figures/accuracy_vs_params.png)", "",
                  "![confusion](figures/confusion_best.png) ![reliability](figures/reliability_best.png)", ""]
        # Per-class F1 across models (first seed)
        per_class = {}
        for rid, sub in df.groupby("id", sort=False):
            mm = load_json(Path(sub.sort_values("seed").run_dir.iloc[0]) / "metrics.json")
            per_class[rid] = {c: mm["test"]["per_class"][c]["f1"] for c in CLASS_NAMES}
        pc = pd.DataFrame(per_class).T
        pc.to_csv(out / "per_class_f1.csv")
        lines += ["## Per-class test F1 (first seed)", "", pc.round(3).to_markdown(), ""]
        # Seed ensemble of the best model
        if (df.id == best.id).sum() > 1:
            ens_p, ens_y = ensemble_predictions(df, best.id)
            em = compute_metrics(ens_y, ens_p)
            lines += [f"Seed-ensemble of {best.id}: accuracy {em['accuracy']:.4f}, macro F1 {em['macro_f1']:.4f}, "
                      f"macro AUC {em['macro_auc']:.4f}.", ""]
        ref = args.reference or suite.get("reference")
        if ref and (df.id == ref).any():
            cmp = compare_to_reference(df, ref, args.n_boot)
            cmp.to_csv(out / "comparisons.csv", index=False)
            lines += [f"## Paired comparisons vs {ref} (same seed, test set)", "",
                      "Paired bootstrap of the macro-F1 difference and McNemar's test; p-values Holm-adjusted.", "",
                      cmp.round(4).to_markdown(index=False), ""]
    else:
        lines += [s[["id", "family", "params", "val_macro_f1_mean", "train_time_s_mean"]].round(4).to_markdown(index=False)]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
