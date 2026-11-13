"""SHAP explanations for the best model.

Methods
-------
``gradient``  : ``shap.GradientExplainer`` (expected gradients; Lundberg & Lee 2017, Erion et al.
                2021). Pixel-level attributions, fast on GPU, works with any differentiable model
                (CNNs, ViTs, MLP-Mixer).
``partition`` : ``shap.Explainer`` with an image blur masker (Owen values over a hierarchy of
                superpixels). Model-agnostic and gives smooth region-level maps that are easier
                for a pathologist to read; slower.

Stain space
-----------
With ``stain_space=True`` the model is re-parameterised as f(HED) = model(hed_to_rgb(HED)),
so SHAP values are attributed to the haematoxylin (nuclei), eosin (cytoplasm, stroma,
muscle) and residual channels instead of R, G, B.

Sanity check
------------
:func:`randomization_sanity_check` re-computes attributions after randomising the weights
(Adebayo et al., NeurIPS 2018). Explanations that barely change are not explaining the model.

All models here are :class:`~crc_tissuenet.models.ClassifierWrapper` instances, which take
float images in [0, 1], so attributions live in the original pixel space.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from ..augment import hed_to_rgb, rgb_to_hed
from ..data import CLASS_NAMES

STAIN_CHANNELS = ("Haematoxylin", "Eosin", "Residual")


class _HEDInput(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, hed):
        return self.model(hed_to_rgb(hed))


class _Softmax(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


@dataclass
class ShapResult:
    images: np.ndarray        # (N, H, W, 3) float in [0, 1]
    labels: np.ndarray        # (N,) true class
    preds: np.ndarray         # (N,) predicted class
    probs: np.ndarray         # (N, K)
    values: np.ndarray        # (N, H, W, C) SHAP values for the predicted class
    channel_names: tuple      # RGB or stain channel names
    method: str


def select_examples(labels: np.ndarray, probs: np.ndarray, per_class: int = 2, include_errors: bool = True,
                    seed: int = 0) -> np.ndarray:
    """Pick indices: ``per_class`` correct examples per class (+ up to as many errors per class)."""
    rng = np.random.default_rng(seed)
    preds = probs.argmax(1)
    chosen = []
    for c in range(probs.shape[1]):
        ok = np.flatnonzero((labels == c) & (preds == c))
        chosen += list(rng.choice(ok, min(per_class, len(ok)), replace=False))
        if include_errors:
            bad = np.flatnonzero((labels == c) & (preds != c))
            chosen += list(rng.choice(bad, min(per_class, len(bad)), replace=False))
    return np.array(sorted(chosen), dtype=int)


def _to_tensor(images_uint8: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(images_uint8)).permute(0, 3, 1, 2).float().div(255).to(device)


def gradient_shap(model: nn.Module, background: torch.Tensor, x: torch.Tensor, nsamples: int = 200,
                  stain_space: bool = False, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Expected-gradients SHAP for each image's top predicted class.

    Returns ``(values [N, H, W, C], explained_class [N])``. Attributions are computed on the
    softmax output, so they sum (approximately) to p(class | x) - E[p(class | background)].
    """
    import shap

    model = _Softmax(model).eval()
    if stain_space:
        model, background, x = _HEDInput(model).eval(), rgb_to_hed(background), rgb_to_hed(x)
    explainer = shap.GradientExplainer(model, background, batch_size=batch_size)
    values, idx = explainer.shap_values(x, nsamples=nsamples, ranked_outputs=1)
    values = np.asarray(values)[..., 0]                      # (N, C, H, W)
    return values.transpose(0, 2, 3, 1), np.asarray(idx.cpu() if torch.is_tensor(idx) else idx)[:, 0]


def partition_shap(model: nn.Module, x: torch.Tensor, max_evals: int = 500, blur: int = 8,
                   batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Model-agnostic Partition SHAP with a blur masker, for each image's top class."""
    import shap

    device = next(model.parameters()).device
    model.eval()

    @torch.no_grad()
    def f(batch: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float().to(device)
        return torch.softmax(model(t), 1).cpu().numpy()

    imgs = x.permute(0, 2, 3, 1).cpu().numpy()
    k = max(3, blur if blur % 2 else blur + 1)
    masker = shap.maskers.Image(f"blur({k},{k})", imgs.shape[1:])
    explainer = shap.Explainer(f, masker, output_names=CLASS_NAMES)
    expl = explainer(imgs, max_evals=max_evals, batch_size=batch_size,
                     outputs=shap.Explanation.argsort.flip[:1], silent=True)
    explained = f(imgs).argmax(1)
    return np.asarray(expl.values)[..., 0], explained


def explain(model: nn.Module, images_uint8: np.ndarray, labels: np.ndarray, background_uint8: np.ndarray,
            method: str = "gradient", nsamples: int = 200, max_evals: int = 500,
            stain_space: bool = False) -> ShapResult:
    device = next(model.parameters()).device
    x = _to_tensor(images_uint8, device)
    with torch.no_grad():
        probs = torch.softmax(model.eval()(x), 1).cpu().numpy()
    if method == "gradient":
        bg = _to_tensor(background_uint8, device)
        values, _ = gradient_shap(model, bg, x, nsamples=nsamples, stain_space=stain_space)
        channels = STAIN_CHANNELS if stain_space else ("R", "G", "B")
    elif method == "partition":
        values, _ = partition_shap(model, x, max_evals=max_evals)
        channels = ("R", "G", "B")
    else:
        raise ValueError("method must be 'gradient' or 'partition'")
    return ShapResult(images_uint8.astype(np.float32) / 255.0, np.asarray(labels), probs.argmax(1), probs,
                      values, channels, method + ("-stain" if stain_space else ""))


def channel_importance(result: ShapResult) -> dict:
    """Share of absolute SHAP mass per channel, averaged per predicted class."""
    mass = np.abs(result.values).sum(axis=(1, 2))           # (N, C)
    share = mass / np.maximum(mass.sum(1, keepdims=True), 1e-12)
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        m = result.preds == c
        if m.any():
            out[name] = {ch: float(share[m, i].mean()) for i, ch in enumerate(result.channel_names)}
    return out


def _reset_parameters(model: nn.Module, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    for m in model.modules():
        if hasattr(m, "reset_parameters") and m is not model:
            m.reset_parameters()
    return model


def randomization_sanity_check(model: nn.Module, images_uint8: np.ndarray, background_uint8: np.ndarray,
                               nsamples: int = 100) -> dict:
    """Spearman correlation of |SHAP| maps between the trained and a weight-randomised model.

    Low correlation (e.g. < 0.3) means the explanation depends on what the model learned.
    """
    device = next(model.parameters()).device
    x, bg = _to_tensor(images_uint8, device), _to_tensor(background_uint8, device)
    trained, _ = gradient_shap(model, bg, x, nsamples=nsamples)
    rand_model = _reset_parameters(copy.deepcopy(model)).to(device)
    randomised, _ = gradient_shap(rand_model, bg, x, nsamples=nsamples)
    rhos = [spearmanr(np.abs(a).sum(-1).ravel(), np.abs(b).sum(-1).ravel())[0]
            for a, b in zip(trained, randomised)]
    return {"spearman_mean": float(np.nanmean(rhos)), "spearman_per_image": [float(r) for r in rhos]}
