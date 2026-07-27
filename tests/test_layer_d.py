"""Layer D 1.6 tests (CLAUDE.md §1.6): Poisson fusion + directional bloom."""

import numpy as np

from heron.art import dirbloom, poisson


def test_poisson_solver_exact_reconstruction():
    rng = np.random.default_rng(0)
    f0 = rng.random((32, 32)).astype(np.float32)
    f0 -= f0.mean()
    lap = -4 * f0 + np.roll(f0, 1, 0) + np.roll(f0, -1, 0) + np.roll(f0, 1, 1) + np.roll(f0, -1, 1)
    rec = poisson.solve_poisson(lap)
    rec -= rec.mean()
    assert np.abs(rec - f0).max() < 1e-4


def test_materiality_zero_is_identity():
    em = np.clip(np.random.default_rng(1).random((40, 40)), 0, 1).astype(np.float32)
    src = np.clip(np.random.default_rng(2).random((40, 40, 3)), 0, 1).astype(np.float32)
    assert np.array_equal(poisson.poisson_detail_fusion(em, src, 0.0), em)


def test_fusion_preserves_low_frequency_and_adds_detail():
    # smooth emission + textured source -> fused keeps the smooth level, gains variance
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    em = (0.3 + 0.4 * xx / 64).astype(np.float32)                  # smooth ramp
    tex = (0.5 + 0.4 * np.sin(xx * 1.5) * np.cos(yy * 1.5))[..., None] * np.ones((1, 1, 3), np.float32)
    fused = poisson.poisson_detail_fusion(em, tex.astype(np.float32), materiality=0.8)
    assert abs(float(fused.mean()) - float(em.mean())) < 0.05       # low-freq preserved
    # high-frequency content increased
    def hf(x):
        import cv2
        return float((x - cv2.GaussianBlur(x, (0, 0), 3)).std())
    assert hf(fused) > hf(em)


def test_directional_bloom_orientation_high_on_stripes():
    stripes = np.tile(np.sin(np.arange(64) * 0.5)[:, None], (1, 64)).astype(np.float32)
    _, coh = dirbloom.structure_orientation(stripes, sigma=3.0)
    assert float(coh.mean()) > 0.8   # strong single orientation


def test_directional_bloom_bounded():
    rng = np.random.default_rng(3)
    s = rng.random((48, 48)).astype(np.float32)
    out = dirbloom.directional_bloom(s, strength=0.6)
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert np.all(out >= s - 1e-5)   # bloom only brightens (screen blend)
