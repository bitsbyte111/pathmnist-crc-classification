import pytest
import torch

from crc_tissuenet.models import (ACTIVATIONS, MLP, ConfigurableCNN, apply_finetune_strategy, build_timm_model,
                                  receptive_field, resnet18_small_input)


def test_mlp_and_cnn_shapes():
    x = torch.rand(2, 3, 28, 28)
    assert MLP(28, 9)(x).shape == (2, 9)
    cnn = ConfigurableCNN(28, 9)
    assert cnn(x).shape == (2, 9)
    assert 80_000 < sum(p.numel() for p in cnn.parameters()) < 120_000  # proposal: ~0.1 M


@pytest.mark.parametrize("act", sorted(ACTIVATIONS))
def test_every_activation(act):
    assert ConfigurableCNN(28, 9, activation=act)(torch.rand(2, 3, 28, 28)).shape == (2, 9)


@pytest.mark.parametrize("kw", [dict(kernel_size=7), dict(depth=4), dict(convs_per_block=2), dict(pooling="avg"),
                                dict(pooling="stride"), dict(head="flatten"), dict(batch_norm=False),
                                dict(init="xavier"), dict(init="normal"), dict(base_channels=64)])
def test_cnn_variants(kw):
    assert ConfigurableCNN(28, 9, **kw)(torch.rand(2, 3, 28, 28)).shape == (2, 9)


def test_depth_too_large_raises():
    with pytest.raises(ValueError):
        ConfigurableCNN(28, 9, depth=5)


def test_receptive_field():
    assert receptive_field(1, 1, 3) == 4
    assert receptive_field(3, 2, 3) > receptive_field(3, 1, 3)


def test_resnet18_small_input():
    assert resnet18_small_input(9, 28)(torch.rand(2, 3, 28, 28)).shape == (2, 9)


@pytest.mark.parametrize("name", ["resnet18", "deit_tiny_patch16_224"])
def test_finetune_strategies(name):
    m = build_timm_model(name, 9, pretrained=False)
    lp = apply_finetune_strategy(m, "linear_probe")
    part = apply_finetune_strategy(m, "partial", 0.25)
    full = apply_finetune_strategy(m, "full")
    assert lp["trainable_params"] < part["trainable_params"] < full["trainable_params"] == full["total_params"]
