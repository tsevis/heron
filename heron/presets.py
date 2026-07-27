"""Shareable, self-contained instrument presets (Phase 5 §5.3).

A preset inside the repo is a YAML fragment that only means something next to
the instrument file it belongs to and the palette file it names. A *shareable*
preset has to survive leaving this machine, so it carries everything needed to
reproduce the render: every resolved layer value, the palette ramp inlined, and
a provenance block saying what produced it.

Two rules shape the design.

**Resolve, do not reference.** The exported layers are the fully merged config —
instrument defaults, then the named preset, then any overrides — so a later edit
to the instrument YAML cannot silently change what a shared preset renders.

**Reject, do not ignore.** Import validates and fails loudly on unknown keys. A
preset that quietly drops the half of itself this build does not understand is
worse than one that refuses to load, because the render still produces an image
and nobody notices it is the wrong one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from heron.color import palettes
from heron.instruments import available_instruments, load_instrument

FORMAT = "heron-preset/1"
LAYER_KEYS = ("layerA", "layerB", "layerC", "layerD")

# Everything a bundle may contain. Anything else is a typo or a newer format.
_TOP_LEVEL = {"format", "engine_version", "instrument", "engine", "name",
              "description", "layers", "palette", "provenance"}
_PROVENANCE = {"author", "source_image", "mode", "seed", "created", "notes",
               "neural_model", "neural_strength"}


def _engine_version() -> str:
    try:
        from importlib.metadata import version

        return version("heron")
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class PresetBundle:
    """One shareable preset. ``layers`` is the fully resolved config."""

    instrument: str
    name: str
    layers: dict[str, dict[str, Any]]
    engine: str
    palette: dict | None = None
    description: str = ""
    provenance: dict = field(default_factory=dict)
    format: str = FORMAT
    engine_version: str = ""

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "engine_version": self.engine_version or _engine_version(),
            "instrument": self.instrument,
            "engine": self.engine,
            "name": self.name,
            "description": self.description,
            "layers": self.layers,
            "palette": self.palette,
            "provenance": self.provenance,
        }

    def config(self) -> dict:
        """A config dict the engine can render directly."""
        cfg: dict[str, Any] = {"engine": self.engine, "description": self.description}
        for layer, values in self.layers.items():
            cfg[layer] = dict(values)
        return cfg


def export_preset(
    instrument: str,
    preset: str | None = None,
    overrides: list[str] | None = None,
    *,
    name: str | None = None,
    author: str = "",
    source_image: str = "",
    mode: str = "physics-only",
    seed: int | None = None,
    notes: str = "",
) -> PresetBundle:
    """Resolve an instrument + preset + overrides into a shareable bundle."""
    cfg = load_instrument(instrument, preset=preset, overrides=overrides)
    layers = {k: dict(cfg[k]) for k in LAYER_KEYS if isinstance(cfg.get(k), dict)}

    palette_name = layers.get("layerC", {}).get("palette")
    palette = None
    if isinstance(palette_name, str):
        # Inlined so the preset renders the right colours on a machine that has
        # never seen this ramp — §5.3's "survives without the palette file".
        palette = palettes.palette_source(palette_name)

    return PresetBundle(
        instrument=instrument,
        name=name or preset or f"{instrument}_custom",
        layers=layers,
        engine=str(cfg["engine"]),
        palette=palette,
        description=str(cfg.get("description", "")),
        engine_version=_engine_version(),
        provenance={
            "author": author,
            "source_image": source_image,
            "mode": mode,                       # physics-only | neural
            "seed": seed,
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "notes": notes,
        },
    )


def write_preset(path: str | Path, bundle: PresetBundle) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=False) + "\n")
    return path


class PresetError(ValueError):
    """Raised with an actionable message; never swallowed into a default."""


def _check_unknown(got: set[str], allowed: set[str], where: str) -> None:
    unknown = sorted(got - allowed)
    if unknown:
        raise PresetError(
            f"{where}: unknown key(s) {unknown}. Known keys are "
            f"{sorted(allowed)}. Refusing to import rather than silently "
            f"dropping them."
        )


def load_preset(path_or_data: str | Path | dict, *, register: bool = True) -> PresetBundle:
    """Validate and load a shareable preset.

    ``register`` installs the inlined palette under its name so the normal
    ``load_palette`` path resolves it — that is what makes a shared preset
    reproduce its own colours instead of a local ramp with the same name.
    """
    if isinstance(path_or_data, dict):
        data = path_or_data
        origin = "<dict>"
    else:
        origin = str(path_or_data)
        try:
            data = json.loads(Path(path_or_data).read_text())
        except json.JSONDecodeError as e:
            raise PresetError(f"{origin}: not valid JSON ({e})") from e

    if not isinstance(data, dict):
        raise PresetError(f"{origin}: expected an object at the top level")

    fmt = data.get("format")
    if fmt != FORMAT:
        raise PresetError(
            f"{origin}: unsupported format {fmt!r}; this build reads {FORMAT!r}."
        )
    _check_unknown(set(data), _TOP_LEVEL, origin)

    for required in ("instrument", "engine", "layers"):
        if required not in data:
            raise PresetError(f"{origin}: missing required key '{required}'")

    instrument = str(data["instrument"])
    if instrument not in available_instruments():
        raise PresetError(
            f"{origin}: unknown instrument '{instrument}'. This build has "
            f"{sorted(available_instruments())}."
        )

    layers = data["layers"]
    if not isinstance(layers, dict):
        raise PresetError(f"{origin}: 'layers' must be an object")
    _check_unknown(set(layers), set(LAYER_KEYS), f"{origin}: layers")
    for layer, values in layers.items():
        if not isinstance(values, dict):
            raise PresetError(f"{origin}: layers.{layer} must be an object")

    provenance = data.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise PresetError(f"{origin}: 'provenance' must be an object")
    _check_unknown(set(provenance), _PROVENANCE, f"{origin}: provenance")

    palette = data.get("palette")
    if palette is not None:
        if not isinstance(palette, dict) or "stops" not in palette:
            raise PresetError(f"{origin}: 'palette' must be an object with 'stops'")
        if register:
            declared = layers.get("layerC", {}).get("palette")
            if isinstance(declared, str):
                palettes.register_palette(declared, palette)

    return PresetBundle(
        instrument=instrument,
        name=str(data.get("name", instrument)),
        layers={k: dict(v) for k, v in layers.items()},
        engine=str(data["engine"]),
        palette=palette,
        description=str(data.get("description", "")),
        provenance=provenance,
        format=fmt,
        engine_version=str(data.get("engine_version", "")),
    )
