"""YAML configuration with inheritance (``_base_``) and dotted command-line overrides.

Every run is fully described by one resolved config, which is saved next to its results.
Example::

    cfg = load_config("configs/models/b1_cnn.yaml", overrides=["train.lr=3e-4"])
    get(cfg, "train.lr")  # -> 0.0003
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

BASE_KEY = "_base_"
_MISSING = object()


def deep_merge(base: Mapping, update: Mapping) -> dict:
    """Recursively merge ``update`` into a copy of ``base`` (``update`` wins)."""
    out = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping at top level")
    return data


def _resolve(path: Path, _seen: tuple = ()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular _base_ inheritance at {path}")
    data = _load_yaml(path)
    bases = data.pop(BASE_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, _resolve((path.parent / b), _seen + (path,)))
    return deep_merge(merged, data)


def parse_value(text: str) -> Any:
    """Parse a command-line value with YAML rules (``1e-3`` -> float, ``[1,2]`` -> list)."""
    value = yaml.safe_load(text)
    # PyYAML 1.1 reads '1e-3' as a string; coerce scientific notation explicitly.
    if isinstance(value, str):
        try:
            return float(value) if any(c in value.lower() for c in "e.") else int(value)
        except ValueError:
            return value
    return value


def get(cfg: Mapping, dotted: str, default: Any = _MISSING) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif default is not _MISSING:
            return default
        else:
            raise KeyError(f"Config key not found: {dotted}")
    return node


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: Iterable[str] | Mapping[str, Any] | None) -> dict:
    """Apply ``key.sub=value`` strings or a ``{dotted_key: value}`` mapping."""
    cfg = copy.deepcopy(cfg)
    if not overrides:
        return cfg
    items = overrides.items() if isinstance(overrides, Mapping) else (
        (o.split("=", 1)[0], parse_value(o.split("=", 1)[1])) for o in overrides
    )
    for key, value in items:
        set_dotted(cfg, key.strip(), value)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | Mapping | None = None) -> dict:
    """Load a YAML config, resolving ``_base_`` chains, then apply overrides."""
    cfg = _resolve(Path(path))
    return apply_overrides(cfg, overrides)


def save_config(cfg: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)


PLATFORM_DIR = Path(__file__).resolve().parents[2] / "configs" / "platforms"


def platform_overrides(platform: str | Path | None) -> dict:
    """Flat ``{dotted.key: value}`` overrides for a platform profile (``kaggle``, ``colab``,
    ``local``) or a path to such a YAML file. ``None`` -> no overrides."""
    if not platform:
        return {}
    path = Path(platform)
    if not path.suffix:
        path = PLATFORM_DIR / f"{platform}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Platform profile not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {str(k): v for k, v in data.items()}


def cli_overrides(platform: str | None, overrides: Iterable[str] | None) -> dict:
    """Platform profile first, then ``key=value`` strings from the command line (which win)."""
    out = platform_overrides(platform)
    for o in overrides or []:
        key, _, value = o.partition("=")
        out[key.strip()] = parse_value(value)
    return out
