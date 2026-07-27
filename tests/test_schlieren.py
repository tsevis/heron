"""Schlieren-engine tests (CLAUDE.md §1.4): knife-edge derivative + thermal reuse."""

import numpy as np

from heron.core.graph import RadianceGraph
from heron.physics.schlieren import SchlierenParams, synthesize_schlieren


def _disc_graph(h=120, w=120, r=34):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    matte = (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < r * r).astype(np.float32)
    ids = np.zeros((h, w), np.int32)
    return RadianceGraph(
        depth=np.full((h, w), 0.5, np.float32),
        normals=np.zeros((h, w, 3), np.float32),
        albedo=0.3 + 0.4 * matte[..., None] * np.ones((1, 1, 3), np.float32),
        saliency=matte,
        matte=matte,
        material_ids=ids,
        material_names=("unknown",),
    )


def test_schlieren_signal_in_range():
    f = synthesize_schlieren(_disc_graph(), SchlierenParams(), seed=1)
    assert f.signal.shape == (120, 120)
    assert 0.0 <= float(f.signal.min()) and float(f.signal.max()) <= 1.0


def test_flat_background_is_mid_gray():
    # no plumes; a far-corner flat region should sit near 0.5 (no gradient)
    f = synthesize_schlieren(_disc_graph(), SchlierenParams(plume_amount=0.0), seed=1)
    corner = f.signal[:8, :8]
    assert abs(float(corner.mean()) - 0.5) < 0.05


def test_edge_produces_gradient_response():
    f = synthesize_schlieren(_disc_graph(), SchlierenParams(plume_amount=0.0, gain=8.0), seed=1)
    # somewhere the signed derivative must swing well away from mid-gray
    assert float(f.signal.max()) > 0.65 and float(f.signal.min()) < 0.35


def test_schlieren_deterministic():
    a = synthesize_schlieren(_disc_graph(), seed=3)
    b = synthesize_schlieren(_disc_graph(), seed=3)
    assert np.array_equal(a.signal, b.signal)


def test_reuses_thermal_density():
    # density field should carry structure (not be constant) — it's the thermal field
    f = synthesize_schlieren(_disc_graph(), SchlierenParams(plume_amount=0.0), seed=1)
    assert float(f.density.std()) > 0.02
