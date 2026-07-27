"""Kirlian-engine tests (CLAUDE.md §1.3): determinism, aura behavior, curl noise."""

import numpy as np

from heron.core.graph import RadianceGraph
from heron.physics import curlnoise
from heron.physics.kirlian import KirlianParams, synthesize_kirlian


def _disc_graph(h=120, w=120, r=32):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    matte = (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < r * r).astype(np.float32)
    return RadianceGraph(
        depth=np.full((h, w), 0.5, np.float32),
        normals=np.zeros((h, w, 3), np.float32),
        albedo=np.full((h, w, 3), 0.4, np.float32),
        saliency=np.zeros((h, w), np.float32),
        matte=matte,
        material_ids=np.zeros((h, w), np.int32),
        material_names=("unknown",),
    ), matte > 0.5


def test_kirlian_intensity_in_range():
    g, _ = _disc_graph()
    f = synthesize_kirlian(g, KirlianParams(), seed=1)
    assert f.intensity.shape == (120, 120)
    assert 0.0 <= float(f.intensity.min()) and float(f.intensity.max()) <= 1.0


def test_kirlian_deterministic():
    g, _ = _disc_graph()
    a = synthesize_kirlian(g, seed=3)
    b = synthesize_kirlian(g, seed=3)
    assert np.array_equal(a.intensity, b.intensity)


def test_kirlian_seed_changes_streamers():
    g, _ = _disc_graph()
    a = synthesize_kirlian(g, seed=1)
    b = synthesize_kirlian(g, seed=2)
    assert not np.array_equal(a.streamers, b.streamers)


def test_aura_glows_outside_not_inside_core():
    g, matte_bin = _disc_graph()
    f = synthesize_kirlian(g, KirlianParams(spark_amount=0.0), seed=1)
    # a ring just outside the disc should be brighter than the deep interior
    from scipy import ndimage
    dist_out = ndimage.distance_transform_edt(~matte_bin)
    ring = (dist_out > 1) & (dist_out < 6)
    core = ndimage.distance_transform_edt(matte_bin) > 20
    assert f.aura[ring].mean() > f.aura[core].mean()


def test_empty_matte_returns_zero():
    g, _ = _disc_graph()
    g.matte[:] = 0.0
    f = synthesize_kirlian(g, seed=1)
    assert float(f.intensity.max()) == 0.0


def test_curl_noise_deterministic():
    a = curlnoise.curl_noise_field((64, 64), seed=5)
    b = curlnoise.curl_noise_field((64, 64), seed=5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
