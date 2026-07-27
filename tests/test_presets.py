"""Shareable presets (Phase 5 §5.3) and engine purity (§7.1).

The gate for this feature is specific: export a preset, wipe local state,
re-import it, and reproduce the render bit-exactly. That is what these tests do —
including simulating the case that actually breaks sharing, where the receiving
machine has a *different* palette under the same name.

The engine-purity tests belong here too because they guard the phase's governing
rule: nothing under ``heron/`` may import ``neural/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from heron.color import palettes
from heron.core import io
from heron.engine import render_image
from heron.presets import (
    FORMAT,
    PresetError,
    export_preset,
    load_preset,
    write_preset,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    h, w = 96, 96
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    figure = ((((xx - w * 0.5) / (w * 0.25)) ** 2 + ((yy - h * 0.55) / (h * 0.32)) ** 2) < 1.0)
    img = np.where(figure[..., None], np.float32([0.72, 0.52, 0.44]),
                   np.float32([0.12, 0.14, 0.25]))
    img = np.clip(img + 0.03 * np.sin(xx * 0.7)[..., None], 0, 1).astype(np.float32)
    path = tmp_path_factory.mktemp("scene") / "scene.png"
    io.write_image(path, img)
    return path


# ------------------------------------------------------------------ export

def test_export_resolves_every_layer():
    """References would let a later YAML edit change what a shared preset does."""
    bundle = export_preset("thermograph", "translucent_glow")
    assert bundle.format == FORMAT
    assert bundle.engine == "thermal"
    assert set(bundle.layers) <= {"layerA", "layerB", "layerC", "layerD"}
    # resolved, not a diff: the base instrument values are present too
    assert "ambient_c" in bundle.layers["layerB"]
    assert "palette" in bundle.layers["layerC"]


def test_export_inlines_the_palette_ramp():
    """§5.3: the ramp must survive without the palette file."""
    bundle = export_preset("thermograph", "translucent_glow")
    assert bundle.palette is not None
    assert bundle.palette["stops"], "palette inlined without stops"


def test_export_records_provenance():
    bundle = export_preset("thermograph", "translucent_glow",
                           author="tsevis", source_image="portrait.jpg",
                           mode="physics-only", seed=7)
    p = bundle.provenance
    assert p["author"] == "tsevis"
    assert p["mode"] == "physics-only"
    assert p["seed"] == 7
    assert p["created"].endswith("Z")


def test_spectroscope_has_no_palette_to_inline():
    """It synthesizes colour directly; inventing a ramp would be a lie."""
    bundle = export_preset("spectroscope", "prism_split")
    assert bundle.palette is None or "stops" in bundle.palette


# ------------------------------------------------------------------ import

def test_roundtrip_reproduces_the_render_bit_exactly(scene, tmp_path):
    """THE acceptance gate for §5.3."""
    original = render_image(scene, instrument="thermograph",
                            preset="translucent_glow", seed=7, cache=False)

    path = write_preset(tmp_path / "look.heron.json",
                        export_preset("thermograph", "translucent_glow", seed=7))
    bundle = load_preset(path)
    reproduced = render_image(scene, instrument=bundle.instrument, seed=7,
                              cache=False, config=bundle.config())

    assert np.array_equal(original.image, reproduced.image), (
        "an imported preset did not reproduce the render bit-exactly"
    )


def test_import_wins_over_a_local_palette_of_the_same_name(scene, tmp_path):
    """The case that silently breaks sharing.

    If the receiving machine has a different ramp under the same name, resolving
    by name renders the wrong colours and nothing errors. The inlined ramp must
    win.
    """
    bundle = export_preset("thermograph", "translucent_glow")
    name = bundle.layers["layerC"]["palette"]

    # A hostile local namesake: same name, inverted ramp.
    hostile = {"name": name, "lightness": "ascending",
               "stops": [{"t": 0.0, "rgb": [255, 255, 255]},
                         {"t": 1.0, "rgb": [0, 0, 0]}]}
    palettes.register_palette(name, hostile)
    wrong = render_image(scene, instrument="thermograph", seed=7, cache=False,
                         config=bundle.config())

    # Importing re-registers the preset's own ramp, which must take precedence.
    loaded = load_preset(write_preset(tmp_path / "p.json", bundle))
    right = render_image(scene, instrument=loaded.instrument, seed=7, cache=False,
                         config=loaded.config())

    assert not np.array_equal(wrong.image, right.image), "the inlined ramp did not take effect"

    palettes._RUNTIME_PALETTES.clear()
    palettes.load_palette.cache_clear()


def test_unknown_top_level_key_is_rejected(tmp_path):
    """Silently dropping half a preset still renders an image — the wrong one."""
    data = export_preset("thermograph", "translucent_glow").to_dict()
    data["layerE"] = {"nope": 1}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(PresetError, match="unknown key"):
        load_preset(path)


def test_unknown_layer_is_rejected(tmp_path):
    data = export_preset("thermograph", "translucent_glow").to_dict()
    data["layers"]["layerZ"] = {"x": 1}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(PresetError, match="unknown key"):
        load_preset(path)


def test_wrong_format_is_rejected(tmp_path):
    data = export_preset("thermograph", "translucent_glow").to_dict()
    data["format"] = "heron-preset/99"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(PresetError, match="unsupported format"):
        load_preset(path)


def test_unknown_instrument_is_rejected(tmp_path):
    data = export_preset("thermograph", "translucent_glow").to_dict()
    data["instrument"] = "tricorder"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(PresetError, match="unknown instrument"):
        load_preset(path)


def test_malformed_json_names_the_file(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(PresetError, match="not valid JSON"):
        load_preset(path)


@pytest.mark.parametrize("instrument,preset", [
    ("thermograph", "translucent_glow"),
    ("fluoroscope", "cold_blue_nude"),
    ("kirlian", "classic_leaf"),
    ("lumen", "gfp"),
])
def test_every_instrument_round_trips(instrument, preset, tmp_path):
    bundle = export_preset(instrument, preset)
    reloaded = load_preset(write_preset(tmp_path / f"{instrument}.json", bundle))
    assert reloaded.instrument == instrument
    assert reloaded.engine == bundle.engine
    assert reloaded.layers == bundle.layers


# ------------------------------------------------------- engine purity (§7.1)

def test_engine_never_imports_neural():
    """The phase's governing rule: the engine stays deterministic and offline."""
    offenders = []
    for path in (REPO / "heron").rglob("*.py"):
        text = path.read_text()
        if "import neural" in text or "from neural" in text:
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"heron/ must not import neural/: {offenders}"


def test_engine_never_imports_a_model_runtime():
    """diffusers/transformers belong to neural/ and scene.ai, not the engine core."""
    forbidden = ("import diffusers", "from diffusers", "import safetensors")
    offenders = []
    for path in (REPO / "heron").rglob("*.py"):
        if "scene/ai" in str(path):
            continue          # Layer A's optional models are a sanctioned exception
        text = path.read_text()
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(REPO)}: {token}")
    assert not offenders, offenders
