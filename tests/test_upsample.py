"""Edge-aware graph upsampling — the basis of full-resolution rendering (Phase 4).

Layer A stays at a bounded working resolution and its channels are lifted to the
render resolution instead of re-running the models. That is only legitimate if the
lift preserves silhouettes: the physics keys on the matte and depth edges, so a
soft or bled edge shows up directly as glow leaking off a subject.

The load-bearing assertions here are that upsampling is a no-op at equal size
(so nothing changes on the default path), that it beats bilinear at an edge, and
that label maps are never interpolated into materials that do not exist.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from heron.core.graph import RadianceGraph
from heron.scene.upsample import guided_upsample, upsample_graph


def _scene(h: int, w: int):
    """A hard-edged disc on a gradient — an unambiguous silhouette to preserve."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disc = ((((xx - w * 0.5) / (w * 0.3)) ** 2 + ((yy - h * 0.5) / (h * 0.3)) ** 2) < 1.0)
    source = np.where(disc[..., None], np.float32([0.85, 0.7, 0.6]),
                      np.float32([0.12, 0.14, 0.2])).astype(np.float32)
    source += 0.05 * (xx / w)[..., None]
    return np.clip(source, 0, 1).astype(np.float32), disc.astype(np.float32)


def _graph(h: int, w: int) -> RadianceGraph:
    source, disc = _scene(h, w)
    return RadianceGraph(
        depth=(0.35 + 0.5 * disc).astype(np.float32),
        normals=np.dstack([np.zeros((h, w), np.float32), np.zeros((h, w), np.float32),
                           np.ones((h, w), np.float32)]),
        albedo=source,
        saliency=disc,
        matte=disc,
        material_ids=(disc > 0.5).astype(np.int32),
        material_names=("unknown", "skin"),
        meta={"ai_used": False, "delight_strength": 0.9},
        extras={"face_region": disc.astype(np.float32)},
    )


def test_equal_size_is_a_noop():
    """The default path must be untouched: same size in, same values out."""
    source, _ = _scene(64, 64)
    graph = _graph(64, 64)
    out = upsample_graph(graph, source)
    assert out is graph, "an equal-size upsample should not rebuild the graph"


def test_guided_upsample_is_identity_at_equal_size():
    source, disc = _scene(48, 48)
    assert np.array_equal(guided_upsample(disc, source), disc)


def test_upsampled_graph_matches_the_target_resolution():
    graph = _graph(64, 64)
    source_full, _ = _scene(192, 192)
    out = upsample_graph(graph, source_full)
    assert out.shape == (192, 192)
    for name in ("depth", "saliency", "matte"):
        assert getattr(out, name).shape == (192, 192), name
    assert out.normals.shape == (192, 192, 3)
    assert out.albedo.shape == (192, 192, 3)
    assert out.material_ids.shape == (192, 192)


def test_edge_is_sharper_than_bilinear():
    """The point of guided upsampling: the silhouette comes from the real image.

    Measured as gradient energy along the disc boundary — a bilinear lift spreads
    the transition over the upsampling ratio, a guided one keeps it tight.
    """
    small, disc_small = _scene(64, 64)
    source_full, disc_full = _scene(256, 256)

    guided = guided_upsample(disc_small, source_full)
    bilinear = cv2.resize(disc_small, (256, 256), interpolation=cv2.INTER_LINEAR)

    def edge_energy(field):
        gx = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.sqrt(gx * gx + gy * gy).max())

    assert edge_energy(guided) > edge_energy(bilinear) * 1.2, (
        f"guided {edge_energy(guided):.3f} not meaningfully sharper than "
        f"bilinear {edge_energy(bilinear):.3f}"
    )


def test_matte_stays_in_range_and_keeps_its_coverage():
    """A lift that changes coverage changes what the physics thinks is subject."""
    graph = _graph(64, 64)
    source_full, disc_full = _scene(256, 256)
    out = upsample_graph(graph, source_full)

    assert 0.0 <= float(out.matte.min()) and float(out.matte.max()) <= 1.0
    before = float((graph.matte > 0.5).mean())
    after = float((out.matte > 0.5).mean())
    assert abs(after - before) < 0.03, f"coverage drifted {before:.3f} -> {after:.3f}"


def test_material_ids_are_never_interpolated():
    """An id between skin and metal is not a material — labels need nearest."""
    graph = _graph(64, 64)
    source_full, _ = _scene(256, 256)
    out = upsample_graph(graph, source_full)
    assert set(np.unique(out.material_ids)) <= set(np.unique(graph.material_ids))
    assert out.material_ids.dtype == graph.material_ids.dtype


def test_normals_stay_unit_length():
    """Guided filtering breaks unit length; shading depends on it."""
    graph = _graph(64, 64)
    source_full, _ = _scene(192, 192)
    out = upsample_graph(graph, source_full)
    norms = np.linalg.norm(out.normals, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_spatial_extras_follow_the_graph():
    """Cached Layer A channels (e.g. the SAM 3 face region) must be lifted too."""
    graph = _graph(64, 64)
    source_full, _ = _scene(192, 192)
    out = upsample_graph(graph, source_full)
    assert out.extras["face_region"].shape == (192, 192)


def test_upsampling_is_deterministic():
    graph = _graph(64, 64)
    source_full, _ = _scene(192, 192)
    a = upsample_graph(graph, source_full)
    b = upsample_graph(graph, source_full)
    assert np.array_equal(a.depth, b.depth)
    assert np.array_equal(a.matte, b.matte)


@pytest.mark.parametrize("size", [96, 160, 224])
def test_no_nans_at_various_ratios(size):
    graph = _graph(64, 64)
    source_full, _ = _scene(size, size)
    out = upsample_graph(graph, source_full)
    for name in ("depth", "normals", "albedo", "saliency", "matte"):
        assert np.isfinite(getattr(out, name)).all(), name
