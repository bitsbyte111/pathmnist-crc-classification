"""Grad-CAM (Selvaraju et al., ICCV 2017) for convolutional backbones.

Complements SHAP: Grad-CAM shows *where* the last convolutional features responded,
SHAP shows how each input pixel / stain changed the predicted probability.
For transformers and MLP-Mixer use SHAP (Grad-CAM needs a spatial feature map).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def last_conv_layer(model: nn.Module) -> nn.Module:
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("Grad-CAM needs a model with Conv2d layers; use SHAP for transformers")
    return convs[-1]


def grad_cam(model: nn.Module, x: torch.Tensor, target: torch.Tensor | None = None,
             layer: nn.Module | None = None) -> np.ndarray:
    """Return heatmaps (N, H, W) in [0, 1] for ``target`` classes (default: predicted)."""
    layer = layer or last_conv_layer(model)
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.eval()
        x = x.clone().requires_grad_(True)
        logits = model(x)
        target = logits.argmax(1) if target is None else target
        model.zero_grad(set_to_none=True)
        logits.gather(1, target.view(-1, 1)).sum().backward()
        weights = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam.detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
