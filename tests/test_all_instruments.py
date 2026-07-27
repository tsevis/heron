"""The all-techniques matrix test (CLAUDE.md §8).

Exercises **every Instrument × every preset** end-to-end through the real engine
on the classical (``--no-ai``) path, plus the Layer D 1.6 options and the shared
Radiance Graph. This is the regression net for the whole rack: if any instrument,
preset, engine, palette, or Layer C/D module breaks, one of these fails.

Guarantees checked per technique:
  * renders to a valid linear-light RGB image in [0,1] with no NaN/Inf
  * output is not degenerate (some structure, not a flat field)
  * identical seed -> identical pixels (§2.4 determinism)
  * works with no AI dependencies (§2.6 Layer A is skippable)
"""

from __future__ import annotations

import numpy as np
import pytest

from heron.core import io
from heron.engine import render_image
from heron.instruments import ENGINES, available_instruments, list_presets

# every (instrument, preset) pair in the rack
ALL_TECHNIQUES = [
    (inst, preset)
    for inst in available_instruments()
    for preset in list_presets(inst)
]
TECHNIQUE_IDS = [f"{i}:{p}" for i, p in ALL_TECHNIQUES]


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """A small synthetic scene with a figure, texture, sky and a bright light.

    Deliberately varied so each engine has something to bite on: a warm body
    (thermal/lumen/kirlian), fine texture (Poisson fusion), a hot highlight
    (bloom/grating), and a gradient background (schlieren/spectral).
    """
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    bg = np.stack([0.10 + 0.10 * xx / w, 0.14 + 0.08 * yy / h, 0.30 - 0.08 * xx / w], -1)
    figure = ((((xx - w * 0.5) / (w * 0.22)) ** 2 + ((yy - h * 0.55) / (h * 0.34)) ** 2) < 1.0)
    texture = 0.04 * np.sin(xx * 0.9) * np.cos(yy * 0.8)
    skin = np.stack([0.72, 0.52, 0.44], -1) + texture[..., None]

    img = np.where(figure[..., None], skin, bg)
    # a bright light source (drives bloom, source halo, grating smear)
    light = np.exp(-(((xx - w * 0.82) ** 2 + (yy - h * 0.18) ** 2) / (2 * 5.0 ** 2)))
    img = img + light[..., None] * 0.9
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    path = tmp_path_factory.mktemp("scene") / "scene.png"
    io.write_image(path, img)
    return path


def _assert_valid_render(result, label):
    img = result.image
    assert img.ndim == 3 and img.shape[2] == 3, f"{label}: expected (H,W,3), got {img.shape}"
    assert img.dtype == np.float32, f"{label}: expected float32, got {img.dtype}"
    assert np.isfinite(img).all(), f"{label}: produced NaN/Inf"
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0, f"{label}: out of [0,1]"
    assert float(img.std()) > 1e-3, f"{label}: degenerate flat output"
    assert result.meta.get("engine") in ENGINES, f"{label}: unknown engine in meta"
    # honesty guardrail (§2.8) travels with every render
    assert "simulation" in result.meta.get("disclaimer", "").lower(), f"{label}: missing disclaimer"


@pytest.mark.parametrize("instrument,preset", ALL_TECHNIQUES, ids=TECHNIQUE_IDS)
def test_technique_renders(scene, instrument, preset):
    """Every instrument × preset renders a valid image on the classical path."""
    result = render_image(scene, instrument=instrument, preset=preset, seed=7, cache=False)
    _assert_valid_render(result, f"{instrument}:{preset}")


@pytest.mark.parametrize("instrument,preset", ALL_TECHNIQUES, ids=TECHNIQUE_IDS)
def test_technique_deterministic(scene, instrument, preset):
    """Same seed + same params -> identical pixels, for every technique (§2.4)."""
    a = render_image(scene, instrument=instrument, preset=preset, seed=3, cache=False)
    b = render_image(scene, instrument=instrument, preset=preset, seed=3, cache=False)
    assert np.array_equal(a.image, b.image), f"{instrument}:{preset} is non-deterministic"


def test_every_instrument_has_presets_and_a_registered_engine():
    """The rack is wired: each instrument declares presets and a known engine."""
    from heron.instruments import load_instrument

    for inst in available_instruments():
        presets = list_presets(inst)
        assert presets, f"{inst} declares no presets"
        cfg = load_instrument(inst)
        assert cfg["engine"] in ENGINES, f"{inst} references unregistered engine {cfg['engine']}"


def test_all_eight_instruments_present():
    """The full eight-instrument rack is built (CLAUDE.md §1)."""
    expected = {
        "thermograph", "fluoroscope", "intensifier", "nir",
        "kirlian", "schlieren", "lumen", "spectroscope",
    }
    assert expected == set(available_instruments())


def test_default_preset_renders_for_every_instrument(scene):
    """Each instrument also works with no preset (its YAML defaults)."""
    for inst in available_instruments():
        result = render_image(scene, instrument=inst, preset=None, seed=1, cache=False)
        _assert_valid_render(result, f"{inst}:<defaults>")


@pytest.mark.parametrize("instrument", ["thermograph", "fluoroscope", "lumen"])
def test_layer_d_options_apply(scene, instrument):
    """Layer D 1.6 (directional bloom + Poisson materiality) changes the output."""
    plain = render_image(
        scene, instrument=instrument, seed=5, cache=False,
        overrides=["layerD.dir_bloom=0.0", "layerD.materiality=0.0"],
    )
    arted = render_image(
        scene, instrument=instrument, seed=5, cache=False,
        overrides=["layerD.dir_bloom=0.5", "layerD.materiality=0.6"],
    )
    _assert_valid_render(arted, f"{instrument}:layerD")
    assert not np.array_equal(plain.image, arted.image), f"{instrument}: Layer D had no effect"


def test_seed_changes_stochastic_instruments(scene):
    """Instruments with stochastic modules respond to the seed."""
    for inst in ("thermograph", "intensifier", "kirlian"):
        a = render_image(scene, instrument=inst, seed=1, cache=False)
        b = render_image(scene, instrument=inst, seed=2, cache=False)
        assert not np.array_equal(a.image, b.image), f"{inst}: seed had no effect"


def test_classical_path_needs_no_ai_dependencies(scene):
    """The whole rack renders without importing torch (§2.6)."""
    import sys

    torch_preloaded = "torch" in sys.modules
    for inst in available_instruments():
        render_image(scene, instrument=inst, seed=0, cache=False)
    if not torch_preloaded:
        assert "torch" not in sys.modules, "classical path pulled in an AI dependency"


def test_graph_is_shared_across_instruments(scene):
    """One Radiance Graph feeds every instrument — the architecture's core claim."""
    results = {
        inst: render_image(scene, instrument=inst, seed=0, cache=False)
        for inst in ("thermograph", "fluoroscope", "lumen", "kirlian")
    }
    graphs = list(results.values())
    ref = graphs[0].graph
    for r in graphs[1:]:
        # same Layer A contract (shape + channels) regardless of instrument
        assert r.graph.depth.shape == ref.depth.shape
        assert r.graph.matte.shape == ref.matte.shape
        assert r.graph.normals.shape == ref.normals.shape
