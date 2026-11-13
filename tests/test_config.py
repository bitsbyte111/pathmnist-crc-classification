from conftest import ROOT

from crc_tissuenet.config import apply_overrides, deep_merge, get, load_config, parse_value


def test_inheritance_chain():
    cfg = load_config(ROOT / "configs/models/p5_deit_small.yaml")
    assert cfg["model"]["name"] == "deit_small"
    assert get(cfg, "train.finetune.layer_decay") == 0.75       # transformer_base
    assert get(cfg, "data.size") == 224                          # pretrained_base
    assert get(cfg, "eval.test") is False                        # default


def test_overrides_and_parsing():
    assert parse_value("1e-3") == 1e-3
    assert parse_value("[1, 2]") == [1, 2]
    assert parse_value("relu") == "relu"
    assert parse_value("true") is True
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c.d=x"])
    assert cfg == {"a": {"b": 2, "c": {"d": "x"}}}


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    out = deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}} and out == {"a": {"b": 1, "c": 2}}


def test_all_configs_load():
    for p in (ROOT / "configs/models").glob("*.yaml"):
        load_config(p)
