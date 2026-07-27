"""Oklab color space (Björn Ottosson, 2020).

CLAUDE.md §2.3 / §9: palette interpolation happens in Oklab, implemented here —
never re-implemented per module, never pulled from a heavyweight library.

Conversions operate on **linear-light** sRGB (not gamma-encoded) with the last
axis of size 3, and preserve input shape. All math is float32.
"""

from __future__ import annotations

import numpy as np

# linear-sRGB -> LMS  (Ottosson)
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)
# nonlinear LMS -> Oklab
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def _apply(mat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix along the last axis of x (...,3)."""
    return x @ mat.T


def linear_srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert linear-sRGB (...,3) to Oklab (...,3) = (L, a, b)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lms = _apply(_M1, rgb)
    lms_ = np.cbrt(lms)  # real cube root, safe for tiny negatives
    return _apply(_M2, lms_).astype(np.float32)


def oklab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert Oklab (...,3) back to linear-sRGB (...,3)."""
    lab = np.asarray(lab, dtype=np.float64)
    lms_ = _apply(_M2_INV, lab)
    lms = lms_ ** 3
    return _apply(_M1_INV, lms).astype(np.float32)


def lerp_oklab(c0: np.ndarray, c1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Interpolate two linear-sRGB colors through Oklab; returns linear-sRGB.

    ``t`` may be a scalar or broadcastable array; 0 -> c0, 1 -> c1.
    """
    a = linear_srgb_to_oklab(c0)
    b = linear_srgb_to_oklab(c1)
    t = np.asarray(t, dtype=np.float32)[..., None] if np.ndim(t) else t
    return oklab_to_linear_srgb(a * (1.0 - t) + b * t)
