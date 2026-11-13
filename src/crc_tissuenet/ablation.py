"""Staged one-factor-at-a-time ablation for the from-scratch CNN.

How it works
------------
1. Start from a base config (the "incumbent").
2. Each stage varies one design factor (optimiser, learning rate, batch size, activation,
   kernel size, width, depth, ...). Every candidate = incumbent + the candidate's overrides.
3. The candidate with the best mean validation macro-F1 (over seeds) wins and its overrides
   are merged into the incumbent; ties go to the simpler candidate (fewer parameters).
4. After the last stage the incumbent is written out as the tuned baseline config.

This is a greedy coordinate search: cheap and easy to interpret (each stage isolates one
concept), but it can miss interactions between factors - e.g. the best learning rate depends
on the batch size. Stage order (optimisation -> architecture -> regularisation) is chosen to
limit this, and the final Optuna search (scripts/tune.py) can probe interactions jointly.

The test set is never used here (``eval.test`` is forced to ``false``).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .config import apply_overrides, get, load_config, save_config
from .parallel import run_jobs
from .utils import get_logger, load_json, save_json

SCALE_KEY = "__scale__"
FORCED = {"eval.test": False, "eval.bootstrap": 0}


def _config_hash(cfg: dict) -> str:
    c = copy.deepcopy(cfg)
    c.get("experiment", {}).pop("name", None)
    c.get("experiment", {}).pop("output_dir", None)
    return hashlib.sha1(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]


def candidate_config(base: dict, incumbent: dict, candidate: dict) -> tuple[dict, dict]:
    """Return ``(resolved config, new incumbent overrides if this candidate wins)``."""
    overrides = dict(incumbent)
    cand = dict(candidate or {})
    scales = cand.pop(SCALE_KEY, {}) or {}
    overrides.update(cand)
    cfg = apply_overrides(base, overrides)
    for key, factor in scales.items():  # relative changes, e.g. lr x 0.3 of the current best
        value = float(f"{get(cfg, key) * float(factor):.6g}")  # avoid 0.15000000000000002
        overrides[key] = value
        cfg = apply_overrides(cfg, {key: value})
    return cfg, overrides


def load_plan(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        plan = yaml.safe_load(f)
    plan["_dir"] = path.parent
    return plan


def run_ablation(plan_path: str | Path, output_dir: str | Path | None = None, stages: list[str] | None = None,
                 seeds: list[int] | None = None, extra_overrides: dict | None = None,
                 best_config_out: str | Path | None = None, dry_run: bool = False,
                 devices: list[str] | None = None, base: str | Path | None = None) -> dict:
    """Run the plan. ``devices`` (e.g. ["cuda:0", "cuda:1"]) trains a stage's candidates in parallel."""
    devices = devices or []
    plan = load_plan(plan_path)
    log = get_logger()
    out = Path(output_dir or Path("runs") / plan.get("name", "ablation"))
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(base or (plan["_dir"] / plan["base"]))
    base = apply_overrides(base, {**plan.get("common_overrides", {}), **(extra_overrides or {}), **FORCED})
    metric = plan.get("selection_metric", "val_macro_f1").replace("val_", "")
    seeds = seeds or plan.get("seeds", [42])

    state_file = out / "state.json"
    state = load_json(state_file) if state_file.exists() else {"incumbent": {}, "completed": [], "cache": {}}
    rows_file = out / "ablation_results.csv"
    rows: list[dict] = []
    if rows_file.exists():
        with open(rows_file) as f:
            rows = list(csv.DictReader(f))

    for stage in plan["stages"]:
        name = stage["name"]
        if stages and name not in stages:
            continue
        if name in state["completed"]:
            log.info("Stage '%s' already completed - skipping", name)
            continue
        log.info("=" * 70)
        log.info("STAGE %s - %s", name, stage.get("concept", ""))
        scores: dict[str, list[float]] = {}
        params: dict[str, int] = {}
        winners_overrides: dict[str, dict] = {}
        stage_rows = []
        # 1) plan this stage's runs (identical configs are trained once and shared)
        planned, jobs, job_hashes = [], [], []
        for label, cand in stage["candidates"].items():
            cfg, new_inc = candidate_config(base, state["incumbent"], cand)
            winners_overrides[label] = new_inc
            for seed in seeds:
                cfg_s = apply_overrides(cfg, {"experiment.seed": seed, "experiment.name": f"{name}/{label}"})
                h = _config_hash(cfg_s)
                run_dir = out / name / str(label) / f"seed{seed}"
                planned.append((label, seed, h, run_dir, cand))
                if dry_run:
                    log.info("[dry-run] %s/%s seed %d -> %s", name, label, seed, run_dir)
                    continue
                cached = h in state["cache"] and Path(state["cache"][h]).joinpath("metrics.json").exists()
                if cached:
                    log.info("%s/%s seed %d identical to %s - reusing result", name, label, seed, state["cache"][h])
                elif h not in job_hashes:
                    jobs.append((cfg_s, run_dir))
                    job_hashes.append(h)
        if dry_run:
            continue
        # 2) train (in parallel when several GPUs are available)
        for h, res, (_, run_dir) in zip(job_hashes, run_jobs(jobs, devices), jobs):
            if res.get("status") != "failed":
                state["cache"][h] = str(run_dir)
        save_json(state, state_file)
        # 3) collect results
        for label, seed, h, run_dir, cand in planned:
            src = Path(state["cache"].get(h, run_dir))
            res = load_json(src / "metrics.json") if (src / "metrics.json").exists() else {"status": "failed"}
            val = res.get("val", {})
            score = val.get(metric, float("nan"))
            score = float("nan") if score is None else float(score)
            scores.setdefault(label, []).append(score)
            params[label] = res.get("model", {}).get("params_total", 0)
            stage_rows.append({
                "stage": name, "candidate": label, "seed": seed, "status": res.get("status"),
                "val_macro_f1": val.get("macro_f1"), "val_accuracy": val.get("accuracy"),
                "val_balanced_accuracy": val.get("balanced_accuracy"), "val_macro_auc": val.get("macro_auc"),
                "val_log_loss": val.get("log_loss"), "best_epoch": res.get("best_epoch"),
                "epochs_run": res.get("epochs_run"), "train_time_s": res.get("train_time_s"),
                "params": params[label], "run_dir": str(src),
                "overrides": json.dumps({k: v for k, v in (cand or {}).items()}), "selected": False,
            })
        means = {k: (np.nanmean(v) if np.isfinite(v).any() else -np.inf) for k, v in scores.items()}
        best_label = max(means, key=lambda k: (round(means[k], 4), -params[k]))
        state["incumbent"] = winners_overrides[best_label]
        state["completed"].append(name)
        for r in stage_rows:
            r["selected"] = r["candidate"] == best_label
        rows.extend(stage_rows)
        _write_rows(rows, rows_file)
        save_json(state, state_file)
        log.info("STAGE %s winner: %s (val macro F1 %.4f)", name, best_label, means[best_label])
        for k in sorted(means, key=means.get, reverse=True):
            log.info("    %-14s %.4f", k, means[k])

    if not dry_run:
        best_cfg = apply_overrides(base, state["incumbent"])
        best_cfg = apply_overrides(best_cfg, {"eval.test": True, "eval.bootstrap": 1000,
                                              "eval.save_predictions": True,
                                              "train.save_checkpoint": "best",
                                              "experiment.name": "B1_cnn_tuned",
                                              **plan.get("final_overrides", {})})
        save_config(best_cfg, out / "best_config.yaml")
        if best_config_out:
            save_config(best_cfg, best_config_out)
        save_json({"incumbent_overrides": state["incumbent"]}, out / "best_overrides.json")
    return state


def _write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
