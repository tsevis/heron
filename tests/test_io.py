"""Image decoding at the sRGB boundary.

Pillow cannot hold 16-bit RGB and clips 16-bit greyscale when converting to
RGBA: a 16-bit grey ramp came back as two values, black and white, and a 16-bit
colour file lost everything below its top eight bits. Decoding is also done
through a lookup table now, which must stay bit-identical to the formula —
§2.4 determinism is measured against renders made before the change.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.color import srgb
from heron.core import io

RAMP16 = np.tile(np.linspace(0, 65535, 1024).astype(np.uint16), (8, 1))


def test_16bit_grey_png_keeps_its_tones(tmp_path):
    path = tmp_path / "grey16.png"
    cv2.imwrite(str(path), RAMP16)

    rgb = io.read_image(path)

    assert len(np.unique(rgb[..., 0])) > 512
    assert abs(float(rgb[0, 512, 0]) - float(srgb.srgb_to_linear(np.float32(512 / 1023)))) < 1e-3


def test_16bit_colour_png_keeps_more_than_8_bits(tmp_path):
    path = tmp_path / "rgb16.png"
    cv2.imwrite(str(path), np.dstack([RAMP16] * 3))

    assert len(np.unique(io.read_image(path)[..., 1])) > 256


def test_rgba_png_keeps_channel_order_and_alpha(tmp_path):
    path = tmp_path / "rgba.png"
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[...] = (255, 128, 0, 64)  # R, G, B, A
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    rgb, alpha = io.read_image_rgba(path)

    assert rgb[0, 0, 0] == 1.0 and rgb[0, 0, 2] == 0.0
    assert abs(float(alpha[0, 0]) - 64 / 255) < 1e-6


def test_8bit_lookup_matches_the_formula_bit_for_bit():
    codes = np.arange(256, dtype=np.uint8)
    expected = srgb.srgb_to_linear(codes.astype(np.float32) / 255.0)

    assert np.array_equal(srgb.decode_uint(codes), expected)


def test_8bit_grey_png_keeps_its_transparency_key(tmp_path):
    from PIL import Image

    path = tmp_path / "grey_trns.png"
    grey = np.full((4, 4), 200, np.uint8)
    grey[0, 0] = 7
    Image.fromarray(grey, "L").save(path, transparency=7)

    _, alpha = io.read_image_rgba(path)

    assert alpha is not None and alpha[0, 0] == 0.0 and alpha[1, 1] == 1.0
