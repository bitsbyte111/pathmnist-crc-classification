"""``ClassifierWrapper`` - one module that maps raw pixels in [0, 1] to class logits.

Resizing and normalisation live *inside* the model, so that

* training, evaluation and deployment cannot disagree about preprocessing, and
* SHAP / Grad-CAM attributions are expressed in the original pixel space a pathologist sees.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import get
from ..data import IMAGENET_MEAN, IMAGENET_STD
from .baselines import MLP, ConfigurableCNN, resnet18_small_input
from .pretrained import (FAMILY, TIMM_ALIASES, apply_finetune_strategy, build_timm_model,
                         freeze_frozen_batchnorm, model_input_spec)

SCRATCH_MODELS = ("mlp", "simple_cnn", "resnet18_scratch")


class ClassifierWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, mean, std, input_size: int | None,
                 keep_frozen_bn_eval: bool = False):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.input_size = input_size
        self.keep_frozen_bn_eval = keep_frozen_bn_eval

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size is not None and x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear",
                              align_corners=False, antialias=x.shape[-1] > self.input_size)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.preprocess(x))

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.keep_frozen_bn_eval:
            freeze_frozen_batchnorm(self.backbone)
        return self


def build_model(cfg: dict, num_classes: int, data_size: int, data_mean, data_std) -> tuple[ClassifierWrapper, dict]:
    """Build the model described by ``cfg['model']``; returns ``(model, info)``."""
    name = get(cfg, "model.name")
    params = dict(get(cfg, "model.params", {}) or {})
    input_size = get(cfg, "model.input_size", None)
    norm = get(cfg, "model.normalization", "dataset")
    info: dict = {"name": name}
    keep_bn_eval = False

    if name in SCRATCH_MODELS:
        size = input_size or data_size
        if name == "mlp":
            backbone = MLP(size, num_classes, **params)
        elif name == "simple_cnn":
            backbone = ConfigurableCNN(size, num_classes, **params)
        else:
            backbone = resnet18_small_input(num_classes, size)
        mean, std = (data_mean, data_std) if norm == "dataset" else (IMAGENET_MEAN, IMAGENET_STD)
        info["family"] = "Scratch"
    else:
        pretrained = bool(get(cfg, "model.pretrained", True))
        backbone = build_timm_model(name, num_classes, pretrained=pretrained,
                                    drop_rate=float(params.get("drop_rate", 0.0)),
                                    drop_path_rate=float(params.get("drop_path_rate", 0.0)),
                                    weights_file=get(cfg, "model.weights_file", None))
        spec = model_input_spec(backbone)
        if norm == "model":
            mean, std = spec["mean"], spec["std"]
        elif norm == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = data_mean, data_std
        if input_size is None:
            input_size = spec["input_size"]
        strategy = get(cfg, "train.finetune.strategy", "full")
        info.update(apply_finetune_strategy(backbone, strategy,
                                            float(get(cfg, "train.finetune.trainable_fraction", 0.25))))
        info["timm_name"] = TIMM_ALIASES.get(name, name)
        info["pretrained"] = pretrained
        info["family"] = FAMILY.get(name, "Other")
        keep_bn_eval = strategy != "full"
    model = ClassifierWrapper(backbone, mean, std, input_size, keep_frozen_bn_eval=keep_bn_eval)
    info["input_size"] = input_size or data_size
    info["params_total"] = sum(p.numel() for p in model.parameters())
    info["params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, info
