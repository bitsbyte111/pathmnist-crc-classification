"""Turn ablation results into written, data-backed observations.

For every stage (one hyperparameter) and every candidate (one setting), the per-epoch
``history.csv`` is summarised into diagnostics a deep-learning course cares about:

* **performance** - best validation macro F1, and whether gaps exceed single-seed noise;
* **convergence** - the epoch at which a run first reached 98 % of its own best F1;
* **generalisation** - train-minus-validation accuracy at the best epoch, and how far the
  validation loss rose after its minimum (over-fitting);
* **gradient flow** - first-layer gradient norm in the first epoch (vanishing/exploding);
* **stability** - diverged or failed runs;
* **cost** - parameters and seconds per epoch, against the F1 they buy.

Each stage is written as: hypothesis (from the plan's ``expect``) -> measured facts ->
a prompt for the student's own interpretation. Every number comes from the runs on disk.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

NOISE_F1 = 0.005  # typical seed-to-seed spread of val macro F1 for one model at this scale


def _num(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def run_diagnostics(run_dir: str | Path) -> dict:
    """Diagnostics from one run's ``history.csv`` (NaNs when the file is missing)."""
    h_path = Path(run_dir) / "history.csv"
    out = {k: float("nan") for k in ("best_f1", "best_epoch", "epochs", "conv_epoch", "gap_at_best",
                                     "val_loss_rise", "grad_first_epoch0", "grad_all_epoch0",
                                     "sec_per_epoch", "final_train_loss")}
    if not h_path.exists():
        return out
    h = pd.read_csv(h_path)
    if h.empty:
        return out
    f1 = h.val_macro_f1.to_numpy(dtype=float)
    best_i = int(np.nanargmax(f1))
    best = float(f1[best_i])
    reached = np.flatnonzero(f1 >= 0.98 * best)
    vloss = h.val_loss.to_numpy(dtype=float)
    out.update(
        best_f1=best,
        best_epoch=best_i + 1,
        epochs=len(h),
        conv_epoch=int(reached[0]) + 1 if len(reached) else float("nan"),
        gap_at_best=float(h.train_acc.iloc[best_i] - h.val_acc.iloc[best_i]),
        val_loss_rise=float(vloss[-1] - np.nanmin(vloss)),
        grad_first_epoch0=_num(h.get("grad_norm_first_layer", pd.Series([np.nan])).iloc[0]),
        grad_all_epoch0=_num(h.grad_norm.iloc[0]),
        sec_per_epoch=float(h.epoch_time_s.mean()),
        final_train_loss=float(h.train_loss.iloc[-1]),
    )
    return out


def stage_table(rows: pd.DataFrame, stage: str) -> pd.DataFrame:
    """One row per candidate: mean over seeds of the result metrics and diagnostics."""
    d = rows[rows.stage == stage]
    records = []
    for cand, sub in d.groupby("candidate", sort=False):
        diags = [run_diagnostics(rd) for rd in sub.run_dir]
        rec = {"candidate": cand,
               "val_macro_f1": pd.to_numeric(sub.val_macro_f1, errors="coerce").mean(),
               "val_macro_f1_sd": pd.to_numeric(sub.val_macro_f1, errors="coerce").std(),
               "val_log_loss": pd.to_numeric(sub.val_log_loss, errors="coerce").mean(),
               "n_seeds": len(sub), "params": _num(sub.params.iloc[0]),
               "status": ",".join(sorted(set(sub.status.astype(str)))),
               "selected": str(sub.selected.iloc[0]) == "True"}
        for k in diags[0]:
            rec[k] = np.nanmean([x[k] for x in diags]) if any(np.isfinite(x[k]) for x in diags) else float("nan")
        records.append(rec)
    return pd.DataFrame(records)


def _fmt(x, nd=4):
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"


def stage_observations(t: pd.DataFrame, noise: float = NOISE_F1) -> list[str]:
    """Plain-language findings for one stage table (see :func:`stage_table`)."""
    obs: list[str] = []
    ok = t[np.isfinite(t.val_macro_f1)].sort_values("val_macro_f1", ascending=False)
    bad = t[~t.status.str.contains("completed|early_stopped")]
    if not bad.empty:
        obs.append("Unstable settings: " + ", ".join(f"**{r.candidate}** ({r.status})" for r in bad.itertuples())
                   + " - the run produced non-finite loss or crashed, so no usable model.")
    if len(ok) == 0:
        return obs or ["No completed runs."]
    ranking = " > ".join(f"{r.candidate} ({r.val_macro_f1:.4f})" for r in ok.itertuples())
    obs.append(f"Ranking by validation macro F1: {ranking}.")
    if len(ok) >= 2:
        first, second, last = ok.iloc[0], ok.iloc[1], ok.iloc[-1]
        d12 = first.val_macro_f1 - second.val_macro_f1
        spread = first.val_macro_f1 - last.val_macro_f1
        if d12 < noise:
            obs.append(f"**{first.candidate}** and **{second.candidate}** differ by only {d12:.4f} F1, below the "
                       f"~{noise} seed-to-seed noise: treat them as tied (re-run with 3 seeds to separate them).")
        else:
            obs.append(f"**{first.candidate}** leads the runner-up **{second.candidate}** by {d12:.4f} F1, "
                       f"more than single-seed noise (~{noise}).")
        obs.append(f"Spread between best and worst setting: {spread:.4f} F1 - this hyperparameter "
                   + ("matters a lot here." if spread > 0.05 else "matters moderately." if spread > 3 * noise
                      else "has little effect at this scale."))
    # Convergence speed only means something between settings that end up competitive.
    leader = ok.val_macro_f1.iloc[0]
    conv = ok[np.isfinite(ok.conv_epoch) & (ok.val_macro_f1 >= leader - 0.02)]
    if len(conv) >= 2 and conv.conv_epoch.max() > conv.conv_epoch.min():
        fast, slow = conv.loc[conv.conv_epoch.idxmin()], conv.loc[conv.conv_epoch.idxmax()]
        obs.append(f"Convergence (among settings within 0.02 F1 of the best): **{fast.candidate}** reached 98% of "
                   f"its best F1 by epoch {fast.conv_epoch:.0f}, **{slow.candidate}** only by epoch {slow.conv_epoch:.0f}.")
    gap = ok[np.isfinite(ok.gap_at_best)]
    if len(gap):
        worst = gap.loc[gap.gap_at_best.idxmax()]
        if worst.gap_at_best > 0.02:
            obs.append(f"Over-fitting signal: **{worst.candidate}** has a train-validation accuracy gap of "
                       f"{worst.gap_at_best:.3f} at its best epoch"
                       + (f" and its validation loss rose {worst.val_loss_rise:.3f} after its minimum."
                          if worst.val_loss_rise > 0.02 else "."))
        elif gap.gap_at_best.max() <= 0:
            obs.append("No over-fitting: for every setting, training accuracy (measured on augmented batches during "
                       "the epoch) is not above validation accuracy at the best epoch.")
        else:
            obs.append(f"No clear over-fitting: train-validation accuracy gaps stay at or below "
                       f"{gap.gap_at_best.max():.3f} (augmentation, and train and validation patches coming from "
                       "the same slides, both keep this gap small).")
    g = t[np.isfinite(t.grad_first_epoch0) & (t.grad_first_epoch0 > 0)]
    if len(g) >= 2:
        lo, hi = g.loc[g.grad_first_epoch0.idxmin()], g.loc[g.grad_first_epoch0.idxmax()]
        ratio = hi.grad_first_epoch0 / lo.grad_first_epoch0
        if ratio >= 3:
            obs.append(f"Gradient flow: in epoch 1 the first-layer gradient norm of **{lo.candidate}** "
                       f"({lo.grad_first_epoch0:.3g}) is {ratio:.1f}x smaller than **{hi.candidate}** "
                       f"({hi.grad_first_epoch0:.3g}) - weak gradients reach the early layers.")
    cost = ok[np.isfinite(ok.sec_per_epoch)]
    if len(cost) >= 2:
        cheap = cost.loc[cost.sec_per_epoch.idxmin()]
        best = ok.iloc[0]
        if best.candidate != cheap.candidate and cheap.sec_per_epoch > 0:
            tr = best.sec_per_epoch / cheap.sec_per_epoch
            pr = best.params / cheap.params if cheap.params else float("nan")
            params_differ = np.isfinite(pr) and abs(pr - 1) > 0.05
            if params_differ or tr >= 1.3:
                obs.append(f"Cost: the best setting **{best.candidate}** takes {tr:.1f}x the time per epoch"
                           + (f" and {pr:.1f}x the parameters" if params_differ else "")
                           + f" of the cheapest (**{cheap.candidate}**) for {best.val_macro_f1 - cheap.val_macro_f1:+.4f} F1"
                           + ("." if params_differ else " (same architecture: with parallel runs sharing CPUs, "
                              "treat time differences under ~30% as noise)."))
    return obs


def write_observations(out_dir: str | Path, plan: dict, title: str | None = None) -> Path:
    """Write ``OBSERVATIONS.md`` + ``observations_<stage>.csv`` tables for an ablation run directory."""
    out_dir = Path(out_dir)
    rows = pd.read_csv(out_dir / "ablation_results.csv")
    stages = {s["name"]: s for s in plan["stages"]}
    tables = out_dir / "tables"
    tables.mkdir(exist_ok=True)
    lines = [f"# {title or plan.get('name', 'Ablation')}: observations", "",
             "Generated from the training histories of every run (validation data only; test set unused). "
             f"Differences below ~{NOISE_F1} macro F1 are within single-seed noise.", ""]
    winners = []
    for name in rows.stage.unique():
        t = stage_table(rows, name)
        t.to_csv(tables / f"{name}.csv", index=False)
        spec = stages.get(name, {})
        sel = t[t.selected]
        winners.append((name, sel.candidate.iloc[0] if len(sel) else "?",
                        sel.val_macro_f1.iloc[0] if len(sel) else float("nan")))
        lines += [f"## {name}", ""]
        if spec.get("concept"):
            lines += [f"**Concept.** {spec['concept']}", ""]
        if spec.get("expect"):
            lines += [f"**Hypothesis.** {spec['expect']}", ""]
        lines += ["**Measured.**", "",
                  "| setting | val macro F1 | best epoch | 98% by epoch | train-val gap | 1st-layer grad (ep 1) | s/epoch | params | |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in t.sort_values("val_macro_f1", ascending=False).itertuples():
            lines.append(f"| {r.candidate} | {_fmt(r.val_macro_f1)} | {_fmt(r.best_epoch, 0)} | {_fmt(r.conv_epoch, 0)} | "
                         f"{_fmt(r.gap_at_best, 3)} | {_fmt(r.grad_first_epoch0, 3)} | {_fmt(r.sec_per_epoch, 1)} | "
                         f"{int(r.params):,} | {'**selected**' if r.selected else ''} |" if np.isfinite(r.params) else
                         f"| {r.candidate} | failed | | | | | | | |")
        lines += ["", f"![{name}](figures/stage_{name}.png)", "", "**Observations.**", ""]
        lines += [f"- {o}" for o in stage_observations(t)]
        lines += ["", "**Interpretation (write your own).** Did the measurement match the hypothesis? "
                  "If not, why might that be here (image size, batch norm, epoch budget, noise)? "
                  "What does it teach about the concept?", "", "> ", ""]
    lines += ["## Summary: the incumbent after each stage", "", "| stage | selected | val macro F1 |", "|---|---|---|"]
    lines += [f"| {s} | {c} | {_fmt(f)} |" for s, c, f in winners]
    first, last = winners[0][2], winners[-1][2]
    if np.isfinite(first) and np.isfinite(last):
        lines += ["", f"Validation macro F1 moved from {first:.4f} after the first stage to {last:.4f} after the last "
                  f"({last - first:+.4f})."]
    path = out_dir / "OBSERVATIONS.md"
    path.write_text("\n".join(lines) + "\n")
    return path
