"""Figures for the report: learning curves, ablation stages, benchmark, calibration, SHAP."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import CLASS_NAMES  # noqa: E402

FAMILY_COLORS = {"Scratch": "#7f7f7f", "CNN": "#1f77b4", "Transformer": "#d62728",
                 "MLP-Mixer": "#2ca02c", "Other": "#9467bd"}
# MedMNIST v2 published PathMNIST test results (Yang et al., Sci Data 2023, Table/Fig. 1 of proposal).
PUBLISHED = {"ResNet-18 (28)": (0.907, 0.983), "ResNet-18 (224)": (0.909, 0.989),
             "ResNet-50 (28)": (0.911, 0.990), "ResNet-50 (224)": (0.892, 0.989)}


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_history(history_csv: str | Path, out: str | Path, title: str = "") -> None:
    h = pd.read_csv(history_csv)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[1].plot(h.epoch, h.train_acc, label="train acc")
    ax[1].plot(h.epoch, h.val_acc, label="val acc")
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro F1")
    ax[1].set(title="Accuracy / F1", xlabel="epoch")
    ax[2].semilogy(h.epoch, h.grad_norm, label="all params")
    if "grad_norm_first_layer" in h:
        ax[2].semilogy(h.epoch, h.grad_norm_first_layer, label="first layer")
    ax[2].set(title="Gradient norm", xlabel="epoch")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(title)
    _save(fig, out)


def plot_confusion_matrix(cm, out: str | Path, title: str = "", normalize: bool = True) -> None:
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = shown[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else int(v), ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 * shown.max() else "black")
    ax.set(xlabel="Predicted", ylabel="True", title=title or "Confusion matrix (row-normalised)")
    fig.colorbar(im, fraction=0.046)
    _save(fig, out)


def plot_ablation_stage(rows: pd.DataFrame, stage: str, out: str | Path) -> None:
    """Bar chart (mean +/- SD over seeds) plus validation learning curves for one stage."""
    d = rows[rows.stage == stage].copy()
    d["val_macro_f1"] = pd.to_numeric(d.val_macro_f1, errors="coerce")
    order = list(dict.fromkeys(d.candidate))
    g = d.groupby("candidate", sort=False).val_macro_f1
    mean, sd = g.mean().reindex(order), g.std().reindex(order).fillna(0)
    selected = d[d.selected.astype(str) == "True"].candidate.unique()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    colors = ["#d62728" if c in selected else "#1f77b4" for c in order]
    ax[0].bar(range(len(order)), mean.fillna(0), yerr=sd, color=colors, capsize=3)
    ax[0].set_xticks(range(len(order)), order, rotation=30, ha="right")
    finite = mean.dropna()
    if len(finite):
        ax[0].set_ylim(max(0, finite.min() - 0.05), min(1, finite.max() + 0.02))
    ax[0].set(title=f"{stage}: val macro F1 (red = selected)")
    for cand in order:
        run = d[d.candidate == cand].run_dir.iloc[0]
        hist = Path(run) / "history.csv"
        if hist.exists():
            h = pd.read_csv(hist)
            ax[1].plot(h.epoch, h.val_macro_f1, label=cand)
            ax[2].semilogy(h.epoch, h.grad_norm_first_layer if "grad_norm_first_layer" in h else h.grad_norm,
                           label=cand)
    ax[1].set(title="Validation macro F1", xlabel="epoch")
    ax[2].set(title="First-layer gradient norm", xlabel="epoch")
    for a in ax[1:]:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    _save(fig, out)


def plot_ablation_progress(rows: pd.DataFrame, out: str | Path) -> None:
    d = rows[rows.selected.astype(str) == "True"].copy()
    d["val_macro_f1"] = pd.to_numeric(d.val_macro_f1, errors="coerce")
    g = d.groupby("stage", sort=False).agg(f1=("val_macro_f1", "mean"), cand=("candidate", "first"))
    fig, ax = plt.subplots(figsize=(max(6, len(g) * 0.9), 3.8))
    ax.plot(range(len(g)), g.f1, marker="o")
    for i, (s, r) in enumerate(g.iterrows()):
        ax.annotate(str(r.cand), (i, r.f1), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(range(len(g)), g.index, rotation=30, ha="right")
    ax.set(ylabel="val macro F1 of stage winner", title="Greedy ablation: incumbent after each stage")
    ax.grid(alpha=0.3)
    _save(fig, out)


def plot_benchmark(df: pd.DataFrame, out: str | Path, metric: str = "test_accuracy") -> None:
    """Accuracy vs parameters (log) by model family, with MedMNIST v2 published reference lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for fam, sub in df.groupby("family"):
        ax.errorbar(sub.params / 1e6, sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"].fillna(0), fmt="o",
                    color=FAMILY_COLORS.get(fam, "k"), label=fam, capsize=3)
        for _, r in sub.iterrows():
            ax.annotate(r.id, (r.params / 1e6, r[f"{metric}_mean"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
    if metric == "test_accuracy":
        best = max(v[0] for v in PUBLISHED.values())
        ax.axhline(best, ls="--", c="k", lw=1, label=f"Best published MedMNIST v2 ({best})")
    ax.set_xscale("log")
    ax.set(xlabel="Parameters (millions, log scale)", ylabel=metric.replace("_", " "),
           title="PathMNIST test performance vs model size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out)


def plot_reliability(y: np.ndarray, probs: np.ndarray, out: str | Path, n_bins: int = 15, title: str = "") -> None:
    conf, correct = probs.max(1), (probs.argmax(1) == y)
    edges = np.linspace(0, 1, n_bins + 1)
    accs, confs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            accs.append(correct[m].mean())
            confs.append(conf[m].mean())
            counts.append(m.sum())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(confs, accs, marker="o")
    ax.set(xlabel="Confidence", ylabel="Accuracy", title=title or "Reliability diagram")
    ax.grid(alpha=0.3)
    _save(fig, out)


def plot_shap_grid(result, out: str | Path, max_rows: int = 18, title: str = "") -> None:
    """One row per image: original | signed SHAP (summed over channels) | per-channel maps."""
    n = min(max_rows, len(result.images))
    n_ch = result.values.shape[-1]
    fig, axes = plt.subplots(n, 2 + n_ch, figsize=(2.1 * (2 + n_ch), 2.1 * n), squeeze=False)
    for i in range(n):
        v = result.values[i]
        total = v.sum(-1)
        lim = np.abs(v).max() + 1e-12
        t, p = CLASS_NAMES[result.labels[i]], CLASS_NAMES[result.preds[i]]
        axes[i, 0].imshow(result.images[i])
        axes[i, 0].set_title(f"true {t} | pred {p}\np={result.probs[i].max():.2f}", fontsize=8,
                             color="black" if t == p else "red")
        axes[i, 1].imshow(result.images[i].mean(-1), cmap="gray", alpha=0.35)
        axes[i, 1].imshow(total, cmap="bwr", vmin=-np.abs(total).max(), vmax=np.abs(total).max(), alpha=0.8)
        axes[i, 1].set_title(f"SHAP for '{p}'", fontsize=8)
        for c in range(n_ch):
            axes[i, 2 + c].imshow(v[..., c], cmap="bwr", vmin=-lim, vmax=lim)
            axes[i, 2 + c].set_title(result.channel_names[c], fontsize=8)
        for a in axes[i]:
            a.axis("off")
    fig.suptitle(title or f"SHAP ({result.method}); red = pushes towards predicted class, blue = against",
                 fontsize=10)
    _save(fig, out)


def plot_channel_importance(importance: dict, out: str | Path, title: str = "") -> None:
    df = pd.DataFrame(importance).T
    fig, ax = plt.subplots(figsize=(8, 3.8))
    df.plot(kind="bar", stacked=True, ax=ax, color=["#5b2c83", "#e377c2", "#bbbbbb"][: df.shape[1]])
    ax.set(ylabel="Share of |SHAP|", title=title or "Attribution by channel, per predicted class")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1))
    _save(fig, out)


def plot_gpu_log(csv_path: str | Path, out: str | Path) -> dict:
    """Plot ``nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used
    --format=csv,noheader,nounits -l <sec>`` output; returns mean utilisation per GPU."""
    df = pd.read_csv(csv_path, header=None, names=["time", "gpu", "util", "mem"], skipinitialspace=True)
    df["time"] = pd.to_datetime(df.time, errors="coerce")
    df = df.dropna(subset=["time"])
    for col in ("util", "mem"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for g, sub in df.groupby("gpu"):
        hours = (sub.time - df.time.min()).dt.total_seconds() / 3600
        ax[0].plot(hours, sub.util, label=f"GPU {g}", lw=0.8)
        ax[1].plot(hours, sub.mem / 1024, label=f"GPU {g}", lw=0.8)
    ax[0].set(ylabel="utilisation (%)", ylim=(0, 105), title="GPU usage during this notebook version")
    ax[1].set(ylabel="memory used (GB)", xlabel="hours since start")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    _save(fig, out)
    return {int(g): round(float(sub.util.mean()), 1) for g, sub in df.groupby("gpu")}
