"""Activation functions and weight initialisation schemes studied in the ablations."""

from __future__ import annotations

import math

from torch import nn

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=False),
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "elu": lambda: nn.ELU(),
    "selu": lambda: nn.SELU(),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(),        # a.k.a. Swish
    "mish": lambda: nn.Mish(),
    "tanh": lambda: nn.Tanh(),
    "sigmoid": lambda: nn.Sigmoid(),
}

# Gain used by Kaiming init for each activation (sigmoid/tanh are better served by Xavier).
_KAIMING_NONLINEARITY = {"relu": "relu", "leaky_relu": "leaky_relu", "elu": "relu",
                         "selu": "linear", "gelu": "relu", "silu": "relu", "mish": "relu",
                         "tanh": "tanh", "sigmoid": "sigmoid"}


def get_activation(name: str) -> nn.Module:
    try:
        return ACTIVATIONS[name.lower()]()
    except KeyError as e:
        raise ValueError(f"Unknown activation '{name}'. Options: {sorted(ACTIVATIONS)}") from e


def init_weights(model: nn.Module, scheme: str = "kaiming", activation: str = "relu") -> None:
    """Initialise conv/linear weights.

    * ``kaiming`` (He et al. 2015): variance 2/fan_in, keeps ReLU-family activations stable.
    * ``xavier`` (Glorot & Bengio 2010): variance 2/(fan_in+fan_out), designed for tanh/sigmoid.
    * ``normal``: N(0, 0.01^2) - a deliberately naive scheme to show why scaling matters.
    * ``default``: PyTorch's own default (Kaiming-uniform with a=sqrt(5)).
    """
    scheme = scheme.lower()
    if scheme == "default":
        return
    nonlin = _KAIMING_NONLINEARITY.get(activation.lower(), "relu")
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if scheme == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlin)
            elif scheme == "xavier":
                gain = nn.init.calculate_gain(nonlin) if nonlin in ("tanh", "sigmoid", "relu") else 1.0
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif scheme == "normal":
                nn.init.normal_(m.weight, 0.0, 0.01)
            else:
                raise ValueError(f"Unknown init scheme '{scheme}'")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def receptive_field(num_blocks: int, convs_per_block: int, kernel_size: int) -> int:
    """Theoretical receptive field (pixels) of stacked conv + 2x2 pool blocks."""
    rf, jump = 1, 1
    for _ in range(num_blocks):
        for _ in range(convs_per_block):
            rf += (kernel_size - 1) * jump
        rf += (2 - 1) * jump
        jump *= 2
    return rf


def same_padding(kernel_size: int) -> int:
    return int(math.floor(kernel_size / 2))
