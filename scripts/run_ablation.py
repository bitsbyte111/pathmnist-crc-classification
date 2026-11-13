#!/usr/bin/env python
"""Run the staged CNN ablation and write the tuned baseline config + figures + summary.

    python scripts/run_ablation.py --plan configs/ablation/cnn_ablation.yaml
    python scripts/run_ablation.py --plan ... --stages optimizer learning_rate   # subset
    python scripts/run_ablation.py --plan ... --report-only                      # figures + observations only
    python scripts/run_ablation.py --plan configs/ablation/transfer_ablation.yaml --platform kaggle --gpus auto
"""
import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from crc_tissuenet.ablation import load_plan, run_ablation
from crc_tissuenet.config import cli_overrides
from crc_tissuenet.parallel import resolve_devices
from crc_tissuenet.observations import write_observations
from crc_tissuenet.viz import plot_ablation_progress, plot_ablation_stage


def write_report(out: Path, plan: dict) -> None:
    rows = pd.read_csv(out / "ablation_results.csv")
    figs = out / "figures"
    for stage in rows.stage.unique():
        plot_ablation_stage(rows, stage, figs / f"stage_{stage}.png")
    plot_ablation_progress(rows, figs / "progress.png")
    concepts = {s["name"]: s.get("concept", "") for s in plan["stages"]}
    lines = [f"# {plan.get('title', 'Ablation')} - results", "",
             "Selection metric: validation macro F1 (mean over seeds). Test data not used.", "",
             "![progress](figures/progress.png)", ""]
    for stage in rows.stage.unique():
        d = rows[rows.stage == stage]
        g = d.groupby("candidate", sort=False).agg(
            val_macro_f1=("val_macro_f1", "mean"), val_acc=("val_accuracy", "mean"),
            val_loss=("val_log_loss", "mean"), epochs=("epochs_run", "mean"), params=("params", "first"),
            time_s=("train_time_s", "mean"), status=("status", "first"),
            selected=("selected", "first")).reset_index()
        concept = concepts.get(stage, "")
        lines += [f"## {stage}", ""] + ([f"*{concept}*", ""] if concept else []) + [
                  "| candidate | val macro F1 | val acc | val log-loss | epochs | params | time (s) | status | |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            mark = "**selected**" if str(r.selected) == "True" else ""
            lines.append(f"| {r.candidate} | {r.val_macro_f1:.4f} | {r.val_acc:.4f} | {r.val_loss:.4f} | "
                         f"{r.epochs:.0f} | {int(r.params):,} | {r.time_s:.0f} | {r.status} | {mark} |")
        lines += ["", f"![{stage}](figures/stage_{stage}.png)", "",
                  "Observations and interpretation: see [OBSERVATIONS.md](OBSERVATIONS.md).", ""]
    (out / "ABLATION_REPORT.md").write_text("\n".join(lines))
    write_observations(out, plan, plan.get("title"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="configs/ablation/cnn_ablation.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("-o", "--override", nargs="*", default=[], help="applied to every run, e.g. data.subset_fraction=0.2")
    ap.add_argument("--base", default=None, help="run the plan on another base config (e.g. configs/models/p5_deit_small.yaml)")
    ap.add_argument("--best-config-out", default=None, help="default: the plan's best_config_out")
    ap.add_argument("--platform", default=None, help="kaggle | colab | local | path/to/profile.yaml")
    ap.add_argument("--gpus", default="auto", help="auto (all GPUs if >1) | 0,1 | none - one run per GPU in parallel")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    plan = load_plan(args.plan)
    out = Path(args.output_dir or Path("runs") / plan.get("name", "ablation"))
    if not args.report_only:
        extra = cli_overrides(args.platform, args.override)
        best_out = args.best_config_out or plan.get("best_config_out")
        run_ablation(args.plan, out, args.stages, args.seeds, extra, best_out, args.dry_run,
                     devices=resolve_devices(args.gpus), base=args.base)
    if not args.dry_run and (out / "ablation_results.csv").exists():
        write_report(out, plan)
        print(f"Report: {out / 'ABLATION_REPORT.md'}\nObservations: {out / 'OBSERVATIONS.md'}")


if __name__ == "__main__":
    main()
