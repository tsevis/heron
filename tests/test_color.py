"""Color-math tests (CLAUDE.md §8): Oklab round-trip, sRGB round-trip."""

import numpy as np

from heron.color import oklab, srgb


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 257, dtype=np.float32).reshape(-1, 1) * np.ones((1, 3), np.float32)
    back = srgb.linear_to_srgb(srgb.srgb_to_linear(x))
    assert np.allclose(back, x, atol=1e-5)


def test_oklab_roundtrip():
    rng = np.random.default_rng(0)
    c = rng.random((512, 3)).astype(np.float32)
    back = oklab.oklab_to_linear_srgb(oklab.linear_srgb_to_oklab(c))
    assert np.abs(back - c).max() < 1e-4


def test_oklab_white_point():
    w = oklab.linear_srgb_to_oklab(np.array([[1.0, 1.0, 1.0]], np.float32))[0]
    assert abs(w[0] - 1.0) < 1e-3
    assert abs(w[1]) < 1e-3 and abs(w[2]) < 1e-3


def test_oklab_lerp_endpoints():
    a = np.array([0.1, 0.2, 0.3], np.float32)
    b = np.array([0.8, 0.7, 0.6], np.float32)
    assert np.allclose(oklab.lerp_oklab(a, b, 0.0), a, atol=1e-4)
    assert np.allclose(oklab.lerp_oklab(a, b, 1.0), b, atol=1e-4)


def test_luminance_monotone_in_gray():
    grays = np.linspace(0, 1, 10, dtype=np.float32).reshape(-1, 1) * np.ones((1, 3), np.float32)
    lum = srgb.luminance(grays)
    assert np.all(np.diff(lum) > 0)
