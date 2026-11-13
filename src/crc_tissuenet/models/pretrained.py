"""ImageNet-pretrained backbones from ``timm`` and the transfer-learning strategies S1-S3.

S1 ``linear_probe``  : freeze the backbone (including BatchNorm statistics), train the head.
S2 ``partial``       : train the head plus the top ``trainable_fraction`` of layer groups.
S3 ``full``          : train everything (optionally with layer-wise LR decay, see optim.py).

Layer groups come from timm's own ``group_matcher`` (stem, stages/blocks, head), so the
same code works for CNNs, vision transformers and MLP-Mixer.
"""

from __future__ import annotations

import timm
from torch import nn

# Short aliases -> pinned timm weights (proposal Table 11, P1-P7).
TIMM_ALIASES = {
    "resnet50": "resnet50.a1_in1k",
    "densenet121": "densenet121.tv_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "mixer_b16": "mixer_b16_224.goog_in21k_ft_in1k",
}
FAMILY = {
    "resnet50": "CNN", "densenet121": "CNN", "efficientnet_b0": "CNN", "convnext_tiny": "CNN",
    "deit_small": "Transformer", "swin_tiny": "Transformer", "mixer_b16": "MLP-Mixer",
}
STRATEGIES = ("linear_probe", "partial", "full")


def build_timm_model(name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.0,
                     drop_path_rate: float = 0.0, weights_file: str | None = None) -> nn.Module:
    """Create a timm model. ``weights_file`` loads pretrained weights from a local file
    (``.safetensors`` / ``.bin`` / ``.pth``) instead of the Hugging Face Hub - needed when
    the Kaggle notebook runs with Internet off."""
    timm_name = TIMM_ALIASES.get(name, name)
    kwargs = {"num_classes": num_classes, "drop_rate": drop_rate}
    if drop_path_rate > 0:
        kwargs["drop_path_rate"] = drop_path_rate
    if pretrained and weights_file:
        kwargs["pretrained_cfg_overlay"] = {"file": str(weights_file)}
    return timm.create_model(timm_name, pretrained=pretrained, **kwargs)


def model_input_spec(model: nn.Module) -> dict:
    """Mean/std/input size the pretrained weights expect (from timm's pretrained_cfg)."""
    cfg = timm.data.resolve_data_config({}, model=model)
    return {"mean": list(cfg["mean"]), "std": list(cfg["std"]), "input_size": int(cfg["input_size"][-1])}


def layer_groups(model: nn.Module) -> list[list[str]]:
    """Parameter names grouped from input (stem) to output (head) using timm's group matcher."""
    from timm.models import group_parameters

    groups = group_parameters(model, model.group_matcher(coarse=False))
    return [names for _, names in sorted(groups.items(), key=lambda kv: kv[0])]


def head_parameter_names(model: nn.Module) -> set[str]:
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    return {n for n, p in model.named_parameters() if id(p) in head_ids}


def apply_finetune_strategy(model: nn.Module, strategy: str = "full",
                            trainable_fraction: float = 0.25) -> dict:
    """Freeze parameters according to ``strategy``; returns a summary for logging."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    params = dict(model.named_parameters())
    head = head_parameter_names(model)
    if strategy == "full":
        trainable = set(params)
    elif strategy == "linear_probe":
        trainable = set(head)
    else:
        groups = [g for g in layer_groups(model) if not set(g) <= head]
        n_train = max(1, round(len(groups) * trainable_fraction))
        trainable = set(head).union(*[set(g) for g in groups[-n_train:]])
    for n, p in params.items():
        p.requires_grad = n in trainable
    n_total = sum(p.numel() for p in params.values())
    n_trainable = sum(p.numel() for n, p in params.items() if n in trainable)
    return {"strategy": strategy, "trainable_params": n_trainable, "total_params": n_total,
            "trainable_pct": 100.0 * n_trainable / n_total}


def freeze_frozen_batchnorm(model: nn.Module) -> None:
    """Keep BatchNorm layers whose parameters are frozen in eval mode (fixed running stats).

    Without this, a "frozen" backbone would still change through its BN running averages.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()
