"""Spectroscope tests (CLAUDE.md §1.5): wavelength color + spectral balance."""

import numpy as np

from heron.color.wavelength import wavelength_to_rgb
from heron.physics.spectral import SpectralParams, synthesize_spectral


def test_wavelength_dominant_channels():
    red = wavelength_to_rgb(650.0)
    green = wavelength_to_rgb(530.0)
    blue = wavelength_to_rgb(465.0)
    assert red[0] > red[1] and red[0] > red[2]
    assert green[1] > green[0] and green[1] > green[2]
    assert blue[2] > blue[0] and blue[2] > blue[1]


def test_wavelength_outside_visible_is_black():
    assert float(wavelength_to_rgb(300.0).sum()) == 0.0
    assert float(wavelength_to_rgb(900.0).sum()) == 0.0


def test_gray_stays_neutral():
    gray = np.full((8, 8, 3), 0.5, np.float32)
    out = synthesize_spectral(gray, SpectralParams(dispersion=0.0, saturation=1.0)).rgb
    ch = out.mean(axis=(0, 1))
    assert np.allclose(ch, ch[0], atol=0.02)  # R≈G≈B


def test_spectral_in_range_and_shape():
    rng = np.random.default_rng(0)
    src = rng.random((16, 24, 3)).astype(np.float32)
    out = synthesize_spectral(src, SpectralParams()).rgb
    assert out.shape == (16, 24, 3)
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_dispersion_changes_output():
    rng = np.random.default_rng(1)
    src = rng.random((16, 40, 3)).astype(np.float32)
    a = synthesize_spectral(src, SpectralParams(dispersion=0.0)).rgb
    b = synthesize_spectral(src, SpectralParams(dispersion=0.08)).rgb
    assert not np.array_equal(a, b)


def test_deterministic():
    rng = np.random.default_rng(2)
    src = rng.random((12, 12, 3)).astype(np.float32)
    a = synthesize_spectral(src, SpectralParams(dispersion=0.05)).rgb
    b = synthesize_spectral(src, SpectralParams(dispersion=0.05)).rgb
    assert np.array_equal(a, b)
