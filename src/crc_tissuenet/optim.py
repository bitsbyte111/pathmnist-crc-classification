"""Optimisers (gradient-descent variants) and learning-rate schedules.

Optimisers compared in the ablation
-----------------------------------
* ``sgd``       - plain mini-batch stochastic gradient descent.
* ``momentum``  - SGD with heavy-ball momentum (Polyak).
* ``nesterov``  - SGD with Nesterov accelerated gradient.
* ``adagrad``   - per-parameter rates scaled by the sum of past squared gradients.
* ``adadelta``  - Adagrad variant with a decaying window, almost LR-free.
* ``rmsprop``   - exponentially-weighted squared-gradient scaling (Hinton).
* ``adam``      - momentum + RMSprop with bias correction; weight decay is *coupled* L2.
* ``adamw``     - Adam with *decoupled* weight decay (Loshchilov & Hutter 2019).
* ``nadam`` / ``radam`` - Nesterov-Adam and rectified Adam.

Weight decay is not applied to biases and normalisation parameters (standard practice).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import get

OPTIMIZERS = ("sgd", "momentum", "nesterov", "adagrad", "adadelta", "rmsprop",
              "adam", "adamw", "nadam", "radam")
SCHEDULERS = ("none", "cosine", "step")


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = str(get(cfg, "train.optimizer", "adamw")).lower()
    lr = float(get(cfg, "train.lr", 1e-3))
    wd = float(get(cfg, "train.weight_decay", 0.0))
    mom = float(get(cfg, "train.momentum", 0.9))
    layer_decay = get(cfg, "train.finetune.layer_decay", None)
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Options: {OPTIMIZERS}")

    backbone = getattr(model, "backbone", model)
    if layer_decay and hasattr(backbone, "group_matcher"):
        # Layer-wise LR decay (proposal S3): earlier layers get lr * decay^depth.
        import timm.optim

        timm_name = {"sgd": "momentum", "momentum": "momentum", "nesterov": "sgd"}.get(name, name)
        kwargs = {"momentum": mom if name != "sgd" else 0.0} if timm_name in ("sgd", "momentum") else {}
        opt = timm.optim.create_optimizer_v2(backbone, opt=timm_name, lr=lr, weight_decay=wd,
                                             layer_decay=float(layer_decay), **kwargs)
        for g in opt.param_groups:  # bake timm's lr_scale into lr so torch schedulers respect it
            g["lr"] = lr * g.pop("lr_scale", 1.0)
        return opt

    groups = _param_groups(model, wd)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.0)
    if name == "momentum":
        return torch.optim.SGD(groups, lr=lr, momentum=mom)
    if name == "nesterov":
        return torch.optim.SGD(groups, lr=lr, momentum=mom, nesterov=True)
    if name == "adagrad":
        return torch.optim.Adagrad(groups, lr=lr)
    if name == "adadelta":
        return torch.optim.Adadelta(groups, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=lr, alpha=0.99, momentum=0.0)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "nadam":
        return torch.optim.NAdam(groups, lr=lr)
    return torch.optim.RAdam(groups, lr=lr)


def lr_lambda_factory(schedule: str, total_steps: int, warmup_steps: int, min_ratio: float = 0.01):
    """Per-step multiplier: linear warm-up, then constant / cosine / step decay."""
    schedule = schedule.lower()
    if schedule not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{schedule}'. Options: {SCHEDULERS}")

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return 0.01 + 0.99 * step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if schedule == "cosine":
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        if schedule == "step":
            return 0.1 ** (int(t >= 0.5) + int(t >= 0.75))
        return 1.0

    return fn


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    epochs = int(get(cfg, "train.epochs", 30))
    warmup = int(float(get(cfg, "train.warmup_epochs", 0)) * steps_per_epoch)
    fn = lr_lambda_factory(str(get(cfg, "train.scheduler", "cosine")), epochs * steps_per_epoch, warmup)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
