"""Lumen-chamber tests (CLAUDE.md §1.5): inner-glow emission + subsurface."""

import numpy as np

from heron.core.graph import RadianceGraph
from heron.physics.lumen import LumenParams, synthesize_lumen


def _graph(matte, names=("unknown",), ids=None):
    h, w = matte.shape
    if ids is None:
        ids = np.zeros((h, w), np.int32)
    return RadianceGraph(
        depth=np.full((h, w), 0.5, np.float32),
        normals=np.zeros((h, w, 3), np.float32),
        albedo=np.full((h, w, 3), 0.3, np.float32),
        saliency=matte.astype(np.float32),
        matte=matte.astype(np.float32),
        material_ids=ids.astype(np.int32),
        material_names=tuple(names),
    )


def _disc(h=120, w=120, r=36):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < r * r).astype(np.float32)


def test_lumen_signal_in_range():
    f = synthesize_lumen(_graph(_disc()), LumenParams())
    assert f.signal.shape == (120, 120)
    assert 0.0 <= float(f.signal.min()) and float(f.signal.max()) <= 1.0


def test_empty_matte_is_dark():
    f = synthesize_lumen(_graph(np.zeros((64, 64), np.float32)))
    assert float(f.signal.max()) == 0.0


def test_subsurface_scatters_into_darkness():
    # scattering should lift the signal just outside the body edge above zero
    matte = _disc(120, 120, 30)
    f = synthesize_lumen(_graph(matte), LumenParams())
    from scipy import ndimage
    outside_ring = (ndimage.distance_transform_edt(matte < 0.5) > 2) & \
                   (ndimage.distance_transform_edt(matte < 0.5) < 8)
    assert f.signal[outside_ring].mean() > 0.01


def test_organic_glows_more_than_glass():
    matte = np.ones((16, 40), np.float32)
    ids = np.zeros((16, 40), np.int32)
    ids[:, :20] = 1  # skin (activity 0.7)
    ids[:, 20:] = 2  # glass (activity 0.1)
    g = _graph(matte, names=("unknown", "skin", "glass"), ids=ids)
    f = synthesize_lumen(g, LumenParams(organic_boost=0.8))
    assert f.emitter[:, :20].mean() > f.emitter[:, 20:].mean()


def test_lumen_deterministic():
    a = synthesize_lumen(_graph(_disc()))
    b = synthesize_lumen(_graph(_disc()))
    assert np.array_equal(a.signal, b.signal)
