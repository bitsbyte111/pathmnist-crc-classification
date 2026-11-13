"""Models trained from scratch (proposal Table 11: B0, B1, B2)."""

from __future__ import annotations

from typing import Sequence

from torch import nn

from .layers import get_activation, init_weights, same_padding


class MLP(nn.Module):
    """B0 - flattened pixels -> hidden layers -> classes. No notion of spatial structure."""

    def __init__(self, in_size: int, num_classes: int, hidden: Sequence[int] = (512, 128),
                 activation: str = "relu", dropout: float = 0.2, batch_norm: bool = False,
                 init: str = "kaiming", in_channels: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_channels * in_size * in_size
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)
        init_weights(self, init, activation)

    def forward(self, x):
        return self.head(self.features(x))


class ConvBlock(nn.Sequential):
    """[conv -> (BN) -> activation] x n  ->  downsample (max/avg pool or strided conv)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, convs: int, activation: str,
                 batch_norm: bool, pooling: str):
        layers: list[nn.Module] = []
        for i in range(convs):
            stride = 2 if (pooling == "stride" and i == convs - 1) else 1
            layers.append(nn.Conv2d(c_in if i == 0 else c_out, c_out, kernel_size, stride=stride,
                                    padding=same_padding(kernel_size), bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(get_activation(activation))
        if pooling == "max":
            layers.append(nn.MaxPool2d(2))
        elif pooling == "avg":
            layers.append(nn.AvgPool2d(2))
        elif pooling != "stride":
            raise ValueError(f"Unknown pooling '{pooling}'")
        super().__init__(*layers)


class ConfigurableCNN(nn.Module):
    """B1 - the small CNN whose design choices are studied one by one in the ablation.

    With the defaults (base_channels=32, depth=3, kernel 3) this is the proposal's
    3-block 32-64-128 network (~0.1 M parameters).

    Args:
        base_channels: filters in the first block; doubled at every block (width).
        depth: number of conv blocks (each halves the spatial size).
        kernel_size: convolution filter size (3, 5, 7 ...).
        convs_per_block: stacked convolutions per block.
        activation: name from :data:`~crc_tissuenet.models.layers.ACTIVATIONS`.
        batch_norm: use batch normalisation after each convolution.
        pooling: ``max``, ``avg`` or ``stride`` (strided convolution).
        head: ``gap`` (global average pooling) or ``flatten`` (dense classifier on all features).
        dropout: dropout before the classifier.
        init: weight initialisation scheme.
    """

    def __init__(self, in_size: int, num_classes: int, base_channels: int = 32, depth: int = 3,
                 kernel_size: int = 3, convs_per_block: int = 1, activation: str = "relu",
                 batch_norm: bool = True, pooling: str = "max", head: str = "gap",
                 dropout: float = 0.25, init: str = "kaiming", in_channels: int = 3,
                 channels: Sequence[int] | None = None):
        super().__init__()
        channels = list(channels) if channels else [base_channels * 2 ** i for i in range(depth)]
        if in_size // 2 ** len(channels) < 1:
            raise ValueError(f"depth {len(channels)} too large for {in_size}px input")
        blocks, prev = [], in_channels
        for c in channels:
            blocks.append(ConvBlock(prev, c, kernel_size, convs_per_block, activation, batch_norm, pooling))
            prev = c
        self.features = nn.Sequential(*blocks)
        if head == "gap":
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            feat = prev
        elif head == "flatten":
            self.pool = nn.Flatten()
            feat = prev * (in_size // 2 ** len(channels)) ** 2
        else:
            raise ValueError(f"Unknown head '{head}'")
        self.head = nn.Sequential(nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                                  nn.Linear(feat, num_classes))
        init_weights(self, init, activation)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def resnet18_small_input(num_classes: int, in_size: int = 28):
    """B2 - ResNet-18 as used for 28 px images in the MedMNIST v2 benchmark code.

    The ImageNet stem (7x7 stride-2 conv + max-pool) would shrink 28 px images to 7 px
    before the first residual stage; following the official MedMNIST experiments we use a
    3x3 stride-1 stem and drop the max-pool for small inputs.
    """
    import timm

    model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    if in_size < 128:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model
