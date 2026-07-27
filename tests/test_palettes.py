"""Palette tests (CLAUDE.md §8): every palette loads and has monotonic-L
(per its declared lightness intent)."""

import numpy as np
import pytest

from heron.color import palettes

ALL = palettes.available_palettes()


def test_canonical_palettes_present():
    expected = {
        "white_hot", "black_hot", "ironbow", "rainbow_hc", "arctic", "lava",
        "fire", "p22_green", "p43_white", "fluoroscope_blue", "turbo",
    }
    assert expected.issubset(set(ALL))


@pytest.mark.parametrize("name", ALL)
def test_palette_loads_and_lut_shape(name):
    lut = palettes.load_palette(name).lut()
    assert lut.shape == (256, 3)
    assert lut.dtype == np.float32
    assert lut.min() >= 0.0 and lut.max() <= 1.0


@pytest.mark.parametrize("name", ALL)
def test_palette_lightness_monotonic(name):
    assert palettes.load_palette(name).is_lightness_monotonic()


def test_apply_shape_and_range():
    field = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    img = palettes.load_palette("ironbow").apply(field)
    assert img.shape == (8, 8, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0
