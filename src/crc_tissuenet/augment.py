"""Batch-wise data augmentation on the GPU for float images in [0, 1], shape (N, 3, H, W).

* Geometric: horizontal/vertical flips and 90-degree rotations. Tissue has no natural
  orientation, so these are label-preserving.
* Colour: brightness / contrast / saturation jitter.
* H&E stain jitter (Tellez et al., Med Image Anal 2019): convert RGB to optical density in
  haematoxylin-eosin-DAB space (Ruifrok & Johnston deconvolution), scale and shift each stain
  channel randomly, convert back. Simulates staining differences between laboratories.

Augmentation is applied to the training split only; evaluation never sees it.
"""

from __future__ import annotations

import torch
from torch import nn

# Stain vectors from scikit-image (skimage.color.rgb_from_hed), Ruifrok & Johnston 2001.
_RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29],
                              [0.07, 0.99, 0.11],
                              [0.27, 0.57, 0.78]])
_RGB_FROM_HED = _RGB_FROM_HED / _RGB_FROM_HED.norm(dim=1, keepdim=True)
_RGB_FROM_HED[2] = torch.linalg.cross(_RGB_FROM_HED[0], _RGB_FROM_HED[1])
_HED_FROM_RGB = torch.linalg.inv(_RGB_FROM_HED)
_EPS = 1e-6


def rgb_to_hed(x: torch.Tensor) -> torch.Tensor:
    od = -torch.log(x.clamp_min(_EPS))  # optical density, (N, 3, H, W)
    return torch.einsum("nchw,cd->ndhw", od, _HED_FROM_RGB.to(x))


def hed_to_rgb(h: torch.Tensor) -> torch.Tensor:
    od = torch.einsum("ndhw,dc->nchw", h, _RGB_FROM_HED.to(h))
    return torch.exp(-od).clamp(0, 1)


class GPUAugment(nn.Module):
    def __init__(self, flips: bool = True, rot90: bool = True, color_jitter: float = 0.0,
                 hed_jitter: float = 0.0):
        super().__init__()
        self.flips, self.rot90 = flips, rot90
        self.color_jitter, self.hed_jitter = float(color_jitter), float(hed_jitter)

    @property
    def is_identity(self) -> bool:
        return not (self.flips or self.rot90 or self.color_jitter > 0 or self.hed_jitter > 0)

    @staticmethod
    def _rand(n: int, x: torch.Tensor) -> torch.Tensor:
        return torch.rand(n, 1, 1, 1, device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.is_identity:
            return x
        n = x.shape[0]
        if self.flips:
            x = torch.where(self._rand(n, x) < 0.5, x.flip(-1), x)
            x = torch.where(self._rand(n, x) < 0.5, x.flip(-2), x)
        if self.rot90 and x.shape[-1] == x.shape[-2]:
            k = torch.randint(0, 4, (n,), device=x.device)
            out = x.clone()
            for r in (1, 2, 3):
                m = k == r
                if m.any():
                    out[m] = torch.rot90(x[m], r, dims=(-2, -1))
            x = out
        if self.hed_jitter > 0:
            s = self.hed_jitter
            hed = rgb_to_hed(x)
            alpha = 1 + (torch.rand(n, 3, 1, 1, device=x.device) * 2 - 1) * s
            beta = (torch.rand(n, 3, 1, 1, device=x.device) * 2 - 1) * s
            x = hed_to_rgb(hed * alpha + beta)
        if self.color_jitter > 0:
            s = self.color_jitter
            b = 1 + (self._rand(n, x) * 2 - 1) * s
            c = 1 + (self._rand(n, x) * 2 - 1) * s
            sat = 1 + (self._rand(n, x) * 2 - 1) * s
            x = x * b
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) * c + mean
            gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
            x = (x - gray) * sat + gray
            x = x.clamp(0, 1)
        return x


def build_augment(cfg_aug: dict | None) -> GPUAugment:
    cfg_aug = cfg_aug or {}
    if not cfg_aug.get("enabled", True):
        return GPUAugment(False, False, 0.0, 0.0)
    return GPUAugment(
        flips=cfg_aug.get("flips", True),
        rot90=cfg_aug.get("rot90", True),
        color_jitter=cfg_aug.get("color_jitter", 0.0),
        hed_jitter=cfg_aug.get("hed_jitter", 0.0),
    )
