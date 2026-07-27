"""Layered output for the Photoshop bridge (Phase 3, task 3.0).

The panel's whole product is an editable layer group, so the service must hand
back every renderer stage as its own image plus a manifest describing it. These
tests are the contract the bridge codes against: every instrument yields at
least one layer, the stage names are the ones the panel expects, every declared
URL resolves to a real readable image, and the Radiance Graph channels travel
along as artifacts (CLAUDE.md §2.7).
"""

from __future__ import annotations

import numpy as np
import pytest

from heron.core import io
from heron.engine import render_image
from heron.instruments import available_instruments
from webapp import layers as weblayers

# The stage names the Photoshop panel maps to layers, per instrument.
EXPECTED_STAGES: dict[str, set[str]] = {
    "thermograph": {"emission", "processed_signal", "temperature_c"},
    "fluoroscope": {"emission", "processed_signal", "transmission"},
    "intensifier": {"emission", "processed_signal", "sources"},
    "nir": {"nir", "processed_signal"},
    "kirlian": {"intensity", "aura", "streamers"},
    "schlieren": {"knife", "density"},
    "lumen": {"emission", "emitter"},
    "spectroscope": {"spectral"},
}

GRAPH_CHANNELS = {"depth", "matte", "saliency", "albedo", "normals", "materials"}


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """A small synthetic figure — enough structure for every engine."""
    h, w = 96, 96
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([0.10 + 0.10 * xx / w, 0.14 + 0.08 * yy / h, 0.30 - 0.08 * xx / w], -1)
    figure = ((((xx - w * 0.5) / (w * 0.22)) ** 2 + ((yy - h * 0.55) / (h * 0.34)) ** 2) < 1.0)
    skin = np.stack([0.72, 0.52, 0.44], -1) + (0.04 * np.sin(xx * 0.9) * np.cos(yy * 0.8))[..., None]
    img = np.clip(np.where(figure[..., None], skin, bg), 0.0, 1.0).astype(np.float32)
    path = tmp_path_factory.mktemp("scene") / "scene.png"
    io.write_image(path, img)
    return path


@pytest.fixture(scope="module")
def manifests(scene, tmp_path_factory):
    """Render every instrument once and build its manifest (module-scoped: slow)."""
    out_dir = tmp_path_factory.mktemp("outputs")
    built = {}
    for inst in available_instruments():
        result = render_image(scene, instrument=inst, seed=7, cache=False)
        io.write_image(out_dir / f"{inst}.png", result.image)
        built[inst] = weblayers.build_manifest(
            f"/outputs/{inst}.png", result.stages, result.graph, out_dir, inst,
        )
    return built, out_dir


@pytest.mark.parametrize("instrument", sorted(EXPECTED_STAGES))
def test_every_instrument_yields_layers(manifests, instrument):
    """A flattened result is a failure mode — every instrument decomposes."""
    manifest = manifests[0][instrument]
    assert manifest["layers"], f"{instrument} produced no layers"
    names = {layer["name"] for layer in manifest["layers"]}
    assert names == EXPECTED_STAGES[instrument], f"{instrument}: unexpected stage names {names}"


@pytest.mark.parametrize("instrument", sorted(EXPECTED_STAGES))
def test_declared_urls_resolve_to_valid_images(manifests, instrument):
    """Every URL in the manifest opens — the panel cannot handle a dead link."""
    manifest, out_dir = manifests[0][instrument], manifests[1]
    for entry in manifest["layers"] + manifest["graph"] + [{"url": manifest["composite"]}]:
        path = out_dir / entry["url"].rsplit("/", 1)[-1]
        assert path.exists(), f"{instrument}: {entry['url']} was never written"
        image = io.read_image(path) if path.suffix == ".png" else None
        assert image is not None and image.size > 0, f"{instrument}: {path.name} is not readable"


@pytest.mark.parametrize("instrument", sorted(EXPECTED_STAGES))
def test_graph_channels_travel_with_the_layers(manifests, instrument):
    """Radiance Graph channels are artifacts, exportable as layers (§2.7)."""
    manifest = manifests[0][instrument]
    assert {entry["name"] for entry in manifest["graph"]} == GRAPH_CHANNELS


@pytest.mark.parametrize("instrument", sorted(EXPECTED_STAGES))
def test_entries_carry_what_the_panel_needs(manifests, instrument):
    """Blend mode, pixel size and encoding kind — enough to place a layer."""
    manifest = manifests[0][instrument]
    for entry in manifest["layers"] + manifest["graph"]:
        assert entry["mode"] in {"normal", "screen", "overlay"}, entry
        assert entry["kind"] in {"gray", "rgb"}, entry
        w, h = entry["size"]
        assert w > 0 and h > 0, entry


def test_glow_stages_blend_as_screen(manifests):
    """Aura and streamers are additive — normal would hide the layer below."""
    kirlian = {layer["name"]: layer for layer in manifests[0]["kirlian"]["layers"]}
    assert kirlian["aura"]["mode"] == "screen"
    assert kirlian["streamers"]["mode"] == "screen"


def test_out_of_range_field_reports_its_real_range(manifests):
    """A temperature field is Celsius; normalizing it must not hide that."""
    stages = {layer["name"]: layer for layer in manifests[0]["thermograph"]["layers"]}
    temperature = stages["temperature_c"]
    assert temperature["kind"] == "gray"
    assert temperature["range"] is not None, "Celsius range was silently discarded"
    lo, hi = temperature["range"]
    assert lo < hi


def test_materials_layer_carries_its_decode_table(manifests):
    """An index map is useless without the names it indexes."""
    graph = {entry["name"]: entry for entry in manifests[0]["thermograph"]["graph"]}
    assert graph["materials"]["materials"], "material names missing from the manifest"


def test_single_channel_stages_are_not_colourized(manifests):
    """The panel picks the palette; the service ships raw data channels (§5)."""
    manifest, out_dir = manifests[0]["kirlian"], manifests[1]
    aura = next(layer for layer in manifest["layers"] if layer["name"] == "aura")
    assert aura["kind"] == "gray"
    written = io.read_image(out_dir / aura["url"].rsplit("/", 1)[-1])
    # read back as RGB by the loader, but the three channels must be identical
    assert np.allclose(written[..., 0], written[..., 1]), "aura was written as colour"


def test_graph_can_be_omitted(scene, tmp_path):
    """Fewer layers is the sanctioned way to cut work — never flattening."""
    result = render_image(scene, instrument="lumen", seed=1, cache=False)
    manifest = weblayers.build_manifest(
        "/outputs/x.png", result.stages, result.graph, tmp_path, "x", include_graph=False,
    )
    assert manifest["graph"] == []
    assert manifest["layers"], "stages must still be present"
