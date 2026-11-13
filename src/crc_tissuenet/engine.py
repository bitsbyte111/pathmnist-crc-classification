"""Training and evaluation engine.

One call to :func:`run_experiment` = one fully reproducible run. A run directory contains::

    config.yaml        resolved configuration
    train.log          console log
    history.csv        per-epoch train/val loss, accuracy, macro F1, LR, gradient norms, time
    metrics.json       final validation (and, if enabled, test) metrics + CIs + model info + environment
    predictions.npz    class probabilities and labels for val (and test) - used by report/stats
    best.pt / last.pt  checkpoints (optional; last.pt allows resuming after a Colab disconnect)

Leakage control: model selection (early stopping, ablation choices, tuning) uses the
validation split only. The test split is evaluated once, after training, and only when
``eval.test: true``.
"""

from __future__ import annotations

import copy
import csv
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .augment import build_augment
from .config import get, save_config
from .data import CLASS_NAMES, NUM_CLASSES, build_data
from .metrics import bootstrap_ci, clinical_error_summary, compute_metrics
from .models import build_model
from .optim import build_optimizer, build_scheduler
from .utils import environment_info, get_device, get_logger, load_json, save_json, seed_everything

EpochCallback = Callable[[int, dict], None]


def _to_input(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True).float().div_(255.0)


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device, amp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probabilities [N, K], labels [N])`` for a loader."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(_to_input(x, device))
        probs.append(F.softmax(logits.float(), dim=1).cpu())
        labels.append(torch.as_tensor(y).view(-1))
    return torch.cat(probs).numpy(), torch.cat(labels).numpy().astype(int)


@torch.no_grad()
def measure_inference_ms(model: nn.Module, size: int, device: torch.device, batch: int = 64,
                         repeats: int = 10, amp: bool = False) -> float:
    """Average milliseconds per image for a forward pass (after warm-up)."""
    model.eval()
    x = torch.rand(batch, 3, size, size, device=device)
    for _ in range(3):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / (repeats * batch)


class Trainer:
    def __init__(self, cfg: dict, run_dir: str | Path, epoch_callback: EpochCallback | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_callback = epoch_callback
        self.log = get_logger(self.run_dir / "train.log")

    # ------------------------------------------------------------------ setup
    def _setup(self) -> None:
        cfg = self.cfg
        seed_everything(int(get(cfg, "experiment.seed", 42)), bool(get(cfg, "experiment.deterministic", True)))
        self.device = get_device(str(get(cfg, "experiment.device", "auto")))
        self.amp = bool(get(cfg, "train.amp", True)) and self.device.type == "cuda"
        save_config(cfg, self.run_dir / "config.yaml")

        self.data = build_data(cfg)
        self.model, self.model_info = build_model(cfg, NUM_CLASSES, self.data.size, self.data.mean, self.data.std)
        self.model.to(self.device)
        # Optional: split each batch over all visible GPUs (one large model). For many runs,
        # running one experiment per GPU (crc_tissuenet.parallel) is faster.
        self.net: nn.Module = self.model
        if (get(cfg, "train.data_parallel", False) and self.device.type == "cuda"
                and torch.cuda.device_count() > 1):
            self.net = nn.DataParallel(self.model)
            self.log.info("DataParallel over %d GPUs", torch.cuda.device_count())
        self.augment = build_augment(get(cfg, "augment", {})).to(self.device)

        weight = None
        if get(cfg, "train.class_weights", False):
            counts = np.asarray(self.data.class_counts["train"], dtype=float)
            weight = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=weight.to(self.device) if weight is not None else None,
                                             label_smoothing=float(get(cfg, "train.label_smoothing", 0.0)))
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, len(self.data.loaders["train"]))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.grad_clip = get(cfg, "train.grad_clip", None)
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.log.info("Model %s | %s params (%s trainable) | input %spx | device %s | AMP %s",
                      self.model_info["name"], f"{self.model_info['params_total']:,}",
                      f"{self.model_info['params_trainable']:,}", self.model_info["input_size"],
                      self.device, self.amp)

    # ------------------------------------------------------------------ epochs
    def _train_epoch(self) -> dict:
        self.net.train()
        self.augment.train()
        tot_loss, tot_correct, n, gnorms, gfirst = 0.0, 0, 0, [], []
        for x, y in self.data.loaders["train"]:
            x = self.augment(_to_input(x, self.device))
            y = torch.as_tensor(y).view(-1).to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                logits = self.net(x)
                loss = self.criterion(logits.float(), y)
            if not torch.isfinite(loss):
                return {"train_loss": float("nan"), "diverged": True}
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            first = self.trainable[0].grad
            if first is not None:
                gfirst.append(float(first.norm()))
            gn = torch.nn.utils.clip_grad_norm_(self.trainable, float(self.grad_clip) if self.grad_clip else math.inf)
            gnorms.append(float(gn))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            bs = y.shape[0]
            tot_loss += float(loss.detach()) * bs
            tot_correct += int((logits.argmax(1) == y).sum())
            n += bs
        finite = [g for g in gnorms if math.isfinite(g)]
        return {"train_loss": tot_loss / n, "train_acc": tot_correct / n,
                "grad_norm": float(np.mean(finite)) if finite else float("nan"),
                "grad_norm_first_layer": float(np.mean(gfirst)) if gfirst else float("nan"),
                "diverged": False}

    def _validate(self) -> tuple[dict, np.ndarray, np.ndarray]:
        probs, y = predict(self.net, self.data.loaders["val"], self.device, self.amp)
        m = compute_metrics(y, probs)
        return ({"val_loss": m["log_loss"], "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
                 "val_balanced_accuracy": m["balanced_accuracy"], "val_macro_auc": m["macro_auc"]}, probs, y)

    # ------------------------------------------------------------------ checkpoints
    def _save_last(self, epoch: int, history: list, best: dict, patience_left: int) -> None:
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                    "history": history, "best_score": best["score"], "best_epoch": best["epoch"],
                    "patience_left": patience_left, "torch_rng": torch.get_rng_state()},
                   self.run_dir / "last.pt")

    # ------------------------------------------------------------------ main
    def fit(self) -> dict:
        self._setup()
        cfg = self.cfg
        epochs = int(get(cfg, "train.epochs", 30))
        patience = int(get(cfg, "train.early_stopping_patience", 8))
        monitor = str(get(cfg, "train.monitor", "val_macro_f1"))
        mode_max = monitor != "val_loss"
        save_ckpt = str(get(cfg, "train.save_checkpoint", "best"))  # best | last | none
        history: list[dict] = []
        best = {"score": -math.inf if mode_max else math.inf, "epoch": -1, "state": None}
        start_epoch, patience_left, status = 0, patience, "completed"

        last = self.run_dir / "last.pt"
        if bool(get(cfg, "train.resume", True)) and last.exists():
            ck = torch.load(last, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.optimizer.load_state_dict(ck["optimizer"])
            self.scheduler.load_state_dict(ck["scheduler"])
            self.scaler.load_state_dict(ck["scaler"])
            torch.set_rng_state(ck["torch_rng"])
            history, start_epoch, patience_left = ck["history"], ck["epoch"] + 1, ck["patience_left"]
            best.update(score=ck["best_score"], epoch=ck["best_epoch"])
            if (self.run_dir / "best.pt").exists():
                best["state"] = torch.load(self.run_dir / "best.pt", map_location="cpu", weights_only=True)
            self.log.info("Resumed from epoch %d", start_epoch)

        t0 = time.perf_counter()
        for epoch in range(start_epoch, epochs):
            te = time.perf_counter()
            tr = self._train_epoch()
            if tr["diverged"]:
                status = "diverged"
                self.log.warning("Loss became non-finite at epoch %d - stopping (learning rate too high?)", epoch)
                break
            val, _, _ = self._validate()
            row = {"epoch": epoch, **{k: v for k, v in tr.items() if k != "diverged"}, **val,
                   "lr": self.optimizer.param_groups[0]["lr"], "epoch_time_s": time.perf_counter() - te}
            history.append(row)
            self.log.info("ep %02d | loss %.4f acc %.4f | val loss %.4f acc %.4f F1 %.4f | gnorm %.3g | %.1fs",
                          epoch, row["train_loss"], row["train_acc"], row["val_loss"], row["val_acc"],
                          row["val_macro_f1"], row["grad_norm"], row["epoch_time_s"])
            score = row[monitor]
            improved = score > best["score"] if mode_max else score < best["score"]
            if improved:
                best.update(score=score, epoch=epoch,
                            state={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()})
                patience_left = patience
                if save_ckpt in ("best", "last"):
                    torch.save(best["state"], self.run_dir / "best.pt")
            else:
                patience_left -= 1
            if save_ckpt == "last":
                self._save_last(epoch, history, best, patience_left)
            if self.epoch_callback is not None:
                self.epoch_callback(epoch, row)  # e.g. Optuna pruning (may raise)
            if patience_left <= 0:
                status = "early_stopped"
                self.log.info("Early stopping at epoch %d (best epoch %d)", epoch, best["epoch"])
                break
        train_time = time.perf_counter() - t0 + sum(h["epoch_time_s"] for h in history[:start_epoch])
        self._write_history(history)

        if best["state"] is not None:
            self.model.load_state_dict(best["state"])
        return self._finalise(status, best, history, train_time)

    def _write_history(self, history: list[dict]) -> None:
        if not history:
            return
        with open(self.run_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)

    def _finalise(self, status: str, best: dict, history: list, train_time: float) -> dict:
        cfg = self.cfg
        result: dict = {"status": status, "best_epoch": best["epoch"], "epochs_run": len(history),
                        "train_time_s": train_time, "model": self.model_info,
                        "class_counts": self.data.class_counts, "environment": environment_info()}
        if best["state"] is None:  # diverged before any validation
            result["val"] = {"accuracy": float("nan"), "macro_f1": float("nan")}
            save_json(result, self.run_dir / "metrics.json")
            return result
        val_probs, val_y = predict(self.net, self.data.loaders["val"], self.device, self.amp)
        result["val"] = compute_metrics(val_y, val_probs)
        arrays = {"val_probs": val_probs.astype(np.float32), "val_labels": val_y}
        if get(cfg, "eval.test", False):
            test_probs, test_y = predict(self.net, self.data.loaders["test"], self.device, self.amp)
            result["test"] = compute_metrics(test_y, test_probs)
            result["test"]["clinical_errors"] = clinical_error_summary(test_y, test_probs)
            n_boot = int(get(cfg, "eval.bootstrap", 1000))
            if n_boot > 0:
                result["test_ci95"] = bootstrap_ci(test_y, test_probs, n_boot=n_boot,
                                                   seed=int(get(cfg, "experiment.seed", 42)))
            arrays.update(test_probs=test_probs.astype(np.float32), test_labels=test_y)
            self.log.info("TEST | acc %.4f | macro F1 %.4f | macro AUC %.4f | bal acc %.4f",
                          result["test"]["accuracy"], result["test"]["macro_f1"],
                          result["test"]["macro_auc"], result["test"]["balanced_accuracy"])
        result["inference_ms_per_image"] = measure_inference_ms(self.model, self.data.size, self.device, amp=self.amp)
        if get(cfg, "eval.save_predictions", True):
            np.savez_compressed(self.run_dir / "predictions.npz", class_names=np.array(CLASS_NAMES), **arrays)
        save_json(result, self.run_dir / "metrics.json")
        last = self.run_dir / "last.pt"
        if last.exists():
            last.unlink()  # finished: the resume checkpoint is no longer needed
        self.log.info("VAL | acc %.4f | macro F1 %.4f | best epoch %d | %.0fs | status %s",
                      result["val"]["accuracy"], result["val"]["macro_f1"], best["epoch"], train_time, status)
        return result


def run_dir_for(cfg: dict) -> Path:
    return Path(get(cfg, "experiment.output_dir", "runs")) / str(get(cfg, "experiment.name", "run")) / \
        f"seed{get(cfg, 'experiment.seed', 42)}"


def run_experiment(cfg: dict, run_dir: str | Path | None = None, overwrite: bool = False,
                   epoch_callback: EpochCallback | None = None) -> dict:
    """Train + evaluate one config. Skips (and returns cached results) if already finished."""
    run_dir = Path(run_dir) if run_dir else run_dir_for(cfg)
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists() and not overwrite:
        get_logger().info("Skipping %s (already finished)", run_dir)
        return load_json(metrics_file)
    cfg = copy.deepcopy(cfg)
    return Trainer(cfg, run_dir, epoch_callback).fit()


def load_trained_model(run_dir: str | Path, device: str = "auto"):
    """Rebuild a model from a run directory and load its best weights (for SHAP / Grad-CAM)."""
    from .config import load_config

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    data = build_data(cfg)
    model, _ = build_model(cfg, NUM_CLASSES, data.size, data.mean, data.std)
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    dev = get_device(device)
    return model.to(dev).eval(), data, cfg, dev
