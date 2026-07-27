"""Instrument config loading (CLAUDE.md §1, §4).

Every Instrument is a YAML *configuration* of the fixed 4-layer stack, never a
separate codebase. This loads the YAML, merges a named preset over the defaults,
then applies CLI dotted overrides (e.g. ``layerC.netd_mk=40``). The engine reads
the resulting plain dict.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

INSTRUMENT_DIR = Path(__file__).parent


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _apply_override(cfg: dict, dotted: str) -> None:
    key, _, raw = dotted.partition("=")
    if not raw and "=" not in dotted:
        raise ValueError(f"override must be key=value, got '{dotted}'")
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into '{p}' in override '{dotted}'")
    node[parts[-1]] = _coerce(raw)


def available_instruments() -> list[str]:
    return sorted(p.stem for p in INSTRUMENT_DIR.glob("*.yaml"))


def load_instrument(
    name: str,
    preset: str | None = None,
    overrides: list[str] | None = None,
) -> dict:
    """Load an instrument config with an optional preset and dotted overrides."""
    path = INSTRUMENT_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"unknown instrument '{name}'. available: {available_instruments()}"
        )
    data = yaml.safe_load(path.read_text())
    presets = data.pop("presets", {}) or {}
    cfg = data

    if preset:
        if preset not in presets:
            raise ValueError(
                f"instrument '{name}' has no preset '{preset}'. available: {sorted(presets)}"
            )
        cfg = _deep_merge(cfg, presets[preset])
        cfg["_preset"] = preset

    for ov in overrides or []:
        _apply_override(cfg, ov)

    cfg["_instrument"] = name
    cfg["_presets_available"] = sorted(presets)
    return cfg


def list_presets(name: str) -> list[str]:
    path = INSTRUMENT_DIR / f"{name}.yaml"
    data = yaml.safe_load(path.read_text())
    return sorted((data.get("presets") or {}).keys())
