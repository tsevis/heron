"""End-to-end + graph tests for the classical (--no-ai) pipeline."""

import numpy as np
import pytest

from heron.core import io
from heron.core.graph import RadianceGraph
from heron.physics.thermal import synthesize_thermal
from heron.scene import GraphOptions, build_graph


def _synthetic(h=96, w=96):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([0.15 + 0.1 * xx / w, 0.2 + 0.1 * yy / h, 0.35 - 0.1 * xx / w], -1)
    r = ((xx - w * 0.5) / (w * 0.25)) ** 2 + ((yy - h * 0.5) / (h * 0.35)) ** 2
    fig = (r < 1).astype(np.float32)
    # add texture so saliency has something to bite on
    tex = 0.05 * np.sin(xx * 0.7) * np.cos(yy * 0.6)
    skin = np.stack([0.7, 0.5, 0.42], -1) + tex[..., None]
    img = bg * (1 - fig[..., None]) + skin * fig[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def test_graph_channels_shape_and_range():
    img = _synthetic()
    g = build_graph(img, image_path=None, opts=GraphOptions(cache=False))
    h, w = img.shape[:2]
    assert g.depth.shape == (h, w)
    assert g.normals.shape == (h, w, 3)
    assert g.albedo.shape == (h, w, 3)
    assert g.matte.shape == (h, w)
    for ch in (g.depth, g.matte, g.saliency):
        assert 0.0 <= float(ch.min()) and float(ch.max()) <= 1.0 + 1e-4
    assert np.allclose(np.linalg.norm(g.normals, axis=-1), 1.0, atol=1e-3)


def test_graph_save_load_roundtrip(tmp_path):
    img = _synthetic()
    g = build_graph(img, image_path=None, opts=GraphOptions(cache=False))
    p = tmp_path / "g.npz"
    g.save(p)
    g2 = RadianceGraph.load(p)
    assert np.array_equal(g.depth, g2.depth)
    assert np.array_equal(g.material_ids, g2.material_ids)
    assert g.material_names == g2.material_names


def test_thermal_signal_in_range():
    img = _synthetic()
    g = build_graph(img, image_path=None, opts=GraphOptions(cache=False))
    field = synthesize_thermal(g)
    assert field.signal.shape == img.shape[:2]
    assert 0.0 <= float(field.signal.min()) and float(field.signal.max()) <= 1.0
    assert field.span_c[1] > field.span_c[0]


def test_render_is_deterministic(tmp_path):
    from heron.engine import render_image

    img = _synthetic(128, 128)
    src = tmp_path / "in.png"
    io.write_image(src, img)
    r1 = render_image(src, preset="flir_field", seed=3, cache=False)
    r2 = render_image(src, preset="flir_field", seed=3, cache=False)
    assert r1.image.shape[2] == 3
    assert np.array_equal(r1.image, r2.image)  # same seed -> identical


def test_render_seed_changes_noise(tmp_path):
    from heron.engine import render_image

    img = _synthetic(128, 128)
    src = tmp_path / "in.png"
    io.write_image(src, img)
    a = render_image(src, preset="flir_field", seed=1, cache=False)
    b = render_image(src, preset="flir_field", seed=2, cache=False)
    assert not np.array_equal(a.image, b.image)


def test_no_ai_does_not_require_torch(tmp_path):
    import sys

    torch_was_loaded = "torch" in sys.modules
    from heron.engine import render_image

    img = _synthetic(64, 64)
    src = tmp_path / "in.png"
    io.write_image(src, img)
    render_image(src, use_ai=False, cache=False)
    if not torch_was_loaded:
        assert "torch" not in sys.modules, "classical path must not import torch"
