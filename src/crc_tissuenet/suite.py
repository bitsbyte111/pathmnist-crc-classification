"""Experiment suites (E1-E5) and their reports.

A suite YAML lists runs; each run = a model config + optional overrides, repeated over seeds::

    name: e3_architectures
    seeds: [42, 43, 44]
    reference: B1_tuned            # comparisons are made against this run id
    common_overrides: {eval.test: true}
    runs:
      - {id: B0_mlp, config: ../models/b0_mlp.yaml}
      - {id: P1_resnet50, config: ../models/p1_resnet50.yaml, overrides: {train.epochs: 10}}

Runs are written to ``runs/<suite>/<id>/seed<k>/`` and skipped if already finished, so a
suite can be resumed after a Colab disconnect or split across team members' GPU quotas
(``--only``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import apply_overrides, load_config
from .parallel import run_jobs
from .metrics import holm_correction, mcnemar_test, paired_bootstrap
from .utils import get_logger, load_json

TEST_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "ece")


def load_suite(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        suite = yaml.safe_load(f)
    suite["_dir"] = path.parent
    return suite


def _run_cfg(suite: dict, run: dict, seed: int, output_dir: Path) -> tuple[dict, Path]:
    cfg_path = Path(run["config"])
    if not cfg_path.is_absolute():
        cfg_path = suite["_dir"] / cfg_path
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, {**suite.get("common_overrides", {}), **run.get("overrides", {}),
                                "experiment.seed": seed, "experiment.name": run["id"]})
    return cfg, output_dir / run["id"] / f"seed{seed}"


def run_suite(path: str | Path, output_dir: str | Path | None = None, only: list[str] | None = None,
              seeds: list[int] | None = None, extra_overrides: dict | None = None, dry_run: bool = False,
              devices: list[str] | None = None) -> list[dict]:
    """Run every (run id, seed) of a suite; with several ``devices`` they run in parallel."""
    suite = load_suite(path)
    out = Path(output_dir or Path("runs") / suite["name"])
    log = get_logger()
    jobs = []
    for run in suite["runs"]:
        if only and run["id"] not in only:
            continue
        cfg_path = Path(run["config"]) if Path(run["config"]).is_absolute() else suite["_dir"] / run["config"]
        if not cfg_path.exists():
            log.warning("Skipping %s: %s not found (it is written by a hyperparameter study - run that first)",
                        run["id"], cfg_path.name)
            continue
        for seed in seeds or run.get("seeds") or suite.get("seeds", [42]):
            cfg, run_dir = _run_cfg(suite, run, seed, out)
            if extra_overrides:
                cfg = apply_overrides(cfg, extra_overrides)
            if (run_dir / "metrics.json").exists():
                log.info("Skipping %s (already finished)", run_dir)
                continue
            log.info("%s %s seed %d -> %s", "[dry-run]" if dry_run else "queued", run["id"], seed, run_dir)
            jobs.append((cfg, run_dir))
    if dry_run or not jobs:
        return []
    return run_jobs(jobs, devices or [])


def collect_suite(path: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    """One row per finished (run id, seed) with test metrics, cost and model info."""
    suite = load_suite(path)
    out = Path(output_dir or Path("runs") / suite["name"])
    rows = []
    for run in suite["runs"]:
        for run_dir in sorted((out / run["id"]).glob("seed*")):
            mf = run_dir / "metrics.json"
            if not mf.exists():
                continue
            m = load_json(mf)
            row = {"id": run["id"], "seed": int(run_dir.name[4:]), "run_dir": str(run_dir),
                   "status": m.get("status"), "family": m["model"].get("family", "Scratch"),
                   "params": m["model"]["params_total"], "trainable_params": m["model"]["params_trainable"],
                   "input_size": m["model"]["input_size"], "train_time_s": m.get("train_time_s"),
                   "inference_ms": m.get("inference_ms_per_image"), "best_epoch": m.get("best_epoch"),
                   "val_macro_f1": m["val"].get("macro_f1"), "val_accuracy": m["val"].get("accuracy")}
            for k in TEST_METRICS:
                row[f"test_{k}"] = m.get("test", {}).get(k, np.nan)
            ci = m.get("test_ci95", {})
            for k in ("accuracy", "macro_f1", "macro_auc"):
                if k in ci:
                    row[f"test_{k}_ci"] = f"{ci[k][0]:.3f}-{ci[k][1]:.3f}"
            ce = m.get("test", {}).get("clinical_errors", {})
            row["tum_recall"] = ce.get("tum_recall", np.nan)
            row["tum_missed_as_norm_or_str"] = ce.get("tum_as_norm", 0) + ce.get("tum_as_str", 0)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over seeds per run id (+ CI of the first seed)."""
    agg = {f"test_{k}": ["mean", "std"] for k in TEST_METRICS}
    agg.update({"val_macro_f1": ["mean"], "tum_recall": ["mean"], "params": ["first"], "family": ["first"],
                "input_size": ["first"], "train_time_s": ["mean"], "inference_ms": ["mean"], "seed": ["count"]})
    s = df.groupby("id", sort=False).agg(agg)
    s.columns = [f"{a}_{b}" if b not in ("first",) else a for a, b in s.columns]
    s = s.rename(columns={"seed_count": "n_seeds"})
    first = df.sort_values("seed").groupby("id", sort=False).first()
    for k in ("accuracy", "macro_f1", "macro_auc"):
        col = f"test_{k}_ci"
        if col in first:
            s[f"{col}_seed{int(first.seed.iloc[0])}"] = first[col]
    return s.reset_index()


def compare_to_reference(df: pd.DataFrame, reference: str, n_boot: int = 1000) -> pd.DataFrame:
    """Paired bootstrap (macro F1) and McNemar tests vs ``reference``, on the shared seed."""
    ref_rows = df[df.id == reference]
    if ref_rows.empty:
        raise ValueError(f"Reference run '{reference}' has no finished seeds")
    out = []
    for rid, sub in df[df.id != reference].groupby("id", sort=False):
        shared = sorted(set(sub.seed) & set(ref_rows.seed))
        if not shared:
            continue
        seed = shared[0]
        a = np.load(Path(sub[sub.seed == seed].run_dir.iloc[0]) / "predictions.npz")
        b = np.load(Path(ref_rows[ref_rows.seed == seed].run_dir.iloc[0]) / "predictions.npz")
        if "test_probs" not in a or "test_probs" not in b:
            continue
        y = a["test_labels"]
        if not np.array_equal(y, b["test_labels"]):
            raise ValueError(f"{rid} and {reference} were evaluated on different test sets")
        pb = paired_bootstrap(y, a["test_probs"], b["test_probs"], "macro_f1", n_boot=n_boot)
        mc = mcnemar_test(y, a["test_probs"].argmax(1), b["test_probs"].argmax(1))
        out.append({"id": rid, "vs": reference, "seed": seed, "delta_macro_f1": pb["difference"],
                    "delta_ci95": f"{pb['ci_low']:+.4f} to {pb['ci_high']:+.4f}",
                    "p_bootstrap": pb["p_value"], "p_mcnemar": mc["p_value"],
                    "only_model_correct": mc["a_only_correct"], "only_reference_correct": mc["b_only_correct"]})
    res = pd.DataFrame(out)
    if not res.empty:
        res["p_bootstrap_holm"] = holm_correction(res.p_bootstrap.tolist())
        res["p_mcnemar_holm"] = holm_correction(res.p_mcnemar.tolist())
        res["significant_0.05"] = res.p_bootstrap_holm < 0.05
    return res


def ensemble_predictions(df: pd.DataFrame, run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Average test probabilities of all seeds of one run id (a cheap seed ensemble)."""
    probs, y = [], None
    for rd in df[df.id == run_id].run_dir:
        p = np.load(Path(rd) / "predictions.npz")
        if y is not None and not np.array_equal(y, p["test_labels"]):
            raise ValueError(f"Seeds of {run_id} were evaluated on different test sets")
        probs.append(p["test_probs"])
        y = p["test_labels"]
    return np.mean(probs, 0), y
