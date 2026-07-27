"""The dial map is one shared asset, so it must not drift from the engine.

`webapp/dials.json` describes which controls each instrument exposes. Both the
web UI and the Photoshop panel read it over `/api/meta`, which removes the
copy-paste drift the two frontends would otherwise have — but it introduces a
different risk: an instrument, preset or engine key can be added to the engine
and forgotten here. These tests are that guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from heron.instruments import available_instruments, list_presets, load_instrument

DIALS_FILE = Path(__file__).resolve().parents[1] / "webapp" / "dials.json"
DIALS = json.loads(DIALS_FILE.read_text())


def test_every_instrument_has_dials():
    """A camera with no dial entry renders no controls in either frontend."""
    assert set(DIALS) == set(available_instruments())


@pytest.mark.parametrize("instrument", sorted(DIALS))
def test_default_preset_exists(instrument):
    """`defPreset` is selected on load — a stale name silently picks another."""
    default = DIALS[instrument]["defPreset"]
    assert default in list_presets(instrument), (
        f"{instrument}: defPreset '{default}' is not one of {list_presets(instrument)}"
    )


# Layer B parameters live on the engine's params dataclass, which is the real
# contract — the instrument YAML only sets a subset, and a renderer may read a
# key that appears in no YAML at all (fluoroscope's `headroom` is `b.get(...)`
# with a default). Checking the dataclass is what actually catches a typo.
PARAMS_BY_ENGINE = {
    "thermal": "heron.physics.thermal:ThermalParams",
    "xray": "heron.physics.xray:XrayParams",
    "photon": "heron.physics.photon:PhotonParams",
    "nir": "heron.physics.nir:NIRParams",
    "corona": "heron.physics.kirlian:KirlianParams",
    "schlieren": "heron.physics.schlieren:SchlierenParams",
    "lumen": "heron.physics.lumen:LumenParams",
    "spectral": "heron.physics.spectral:SpectralParams",
}


def _params_fields(engine: str) -> set[str]:
    import dataclasses
    import importlib

    module_name, _, class_name = PARAMS_BY_ENGINE[engine].partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return {f.name for f in dataclasses.fields(cls)}


@pytest.mark.parametrize("instrument", sorted(DIALS))
def test_layer_b_dials_are_real_physics_parameters(instrument):
    """A layerB dial must name a field its engine's params dataclass declares.

    A typo produces a slider that silently does nothing: the override is
    accepted, applied to a key no engine reads, and the image never changes.
    """
    cfg = load_instrument(instrument)
    fields = _params_fields(cfg["engine"])
    unknown = [
        dial["key"]
        for dials in DIALS[instrument]["groups"].values()
        for dial in dials
        if dial["key"].startswith("layerB.") and dial["key"].split(".", 1)[1] not in fields
    ]
    assert not unknown, f"{instrument}: dials name unknown physics params {unknown}"


@pytest.mark.parametrize("instrument", sorted(DIALS))
def test_layer_b_params_are_actually_wired(instrument):
    """Every physics field the dataclass declares must be readable from config.

    This is the bug that motivated the test: `_thermal_params` mapped only 9 of
    ThermalParams' 22 fields, so six shipped dials — and the preset values for
    them — were silently ignored. The engine's params builder must mention every
    field, or a dial bound to it is decoration.
    """
    import inspect

    cfg = load_instrument(instrument)
    engine = cfg["engine"]
    module_name, _, class_name = PARAMS_BY_ENGINE[engine].partition(":")

    import importlib

    renderer = importlib.import_module(f"heron.instruments.{instrument}")
    source = inspect.getsource(renderer)
    dialled = {
        dial["key"].split(".", 1)[1]
        for dials in DIALS[instrument]["groups"].values()
        for dial in dials
        if dial["key"].startswith("layerB.")
    }
    unwired = sorted(f for f in dialled if f'"{f}"' not in source)
    assert not unwired, (
        f"{instrument}: {module_name}:{class_name} fields {unwired} have dials but the "
        f"renderer never reads them from cfg — the sliders do nothing"
    )


@pytest.mark.parametrize("instrument", sorted(DIALS))
def test_dial_defaults_sit_inside_their_range(instrument):
    """A default outside min/max snaps on load and quietly changes the render."""
    bad = []
    for dials in DIALS[instrument]["groups"].values():
        for dial in dials:
            if not (dial["min"] <= dial["def"] <= dial["max"]):
                bad.append(f"{dial['key']}={dial['def']} outside [{dial['min']},{dial['max']}]")
    assert not bad, f"{instrument}: {bad}"


@pytest.mark.parametrize("instrument", sorted(DIALS))
def test_palette_is_a_real_palette_or_null(instrument):
    """Null means the instrument synthesizes colour and bypasses the 1-D LUT."""
    from heron.color import palettes

    declared = DIALS[instrument]["palette"]
    assert declared is None or declared in palettes.available_palettes()


def test_every_dial_is_labelled():
    """The label is all the user sees; the key is only a tooltip."""
    for instrument, cfg in DIALS.items():
        for group, dials in cfg["groups"].items():
            assert group.strip(), f"{instrument}: unnamed dial group"
            for dial in dials:
                assert dial["label"].strip(), f"{instrument}: {dial['key']} has no label"
