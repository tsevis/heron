"""sRGB <-> linear-light transfer functions (IEC 61966-2-1).

CLAUDE.md §2.3: decode sRGB on input, encode on output; everything between is
linear float [0, 1]. These operate elementwise on numpy arrays of any shape and
always return float32.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_A = 0.055
_THRESH_ENC = 0.0031308
_THRESH_DEC = 0.04045


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Decode gamma-encoded sRGB values in [0,1] to linear light."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    low = x / 12.92
    high = ((x + _A) / (1.0 + _A)) ** 2.4
    return np.where(x <= _THRESH_DEC, low, high).astype(np.float32)


def decode_uint(codes: np.ndarray) -> np.ndarray:
    """Decode uint8/uint16 sRGB code values to linear float32 by table lookup.

    Bit-identical to ``srgb_to_linear(codes / max_code)`` — the table is built
    with that very function — but one gather instead of a ``pow`` per pixel.
    """
    codes = np.asarray(codes)
    if codes.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"decode_uint needs uint8 or uint16 codes, got {codes.dtype}")
    return _decode_table(codes.dtype.str)[codes]


@lru_cache(maxsize=2)
def _decode_table(dtype: str) -> np.ndarray:
    top = np.iinfo(np.dtype(dtype)).max
    table = srgb_to_linear(np.arange(top + 1, dtype=np.float32) / float(top))
    table.setflags(write=False)
    return table


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Encode linear-light values in [0,1] back to gamma sRGB."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    low = x * 12.92
    high = (1.0 + _A) * np.power(x, 1.0 / 2.4) - _A
    return np.where(x <= _THRESH_ENC, low, high).astype(np.float32)


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 relative luminance of a linear-light RGB image (H,W,3)->(H,W)."""
    linear_rgb = np.asarray(linear_rgb, dtype=np.float32)
    r, g, b = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
