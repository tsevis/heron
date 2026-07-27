"""Wavelength -> RGB display color (CLAUDE.md §1.5 spectroscope).

Dan Bruton's classic approximate visible-spectrum-to-RGB mapping, vectorized.
Returns approximate **linear-light** RGB for a wavelength in nanometres, used to
tint the spectroscope's per-band decomposition and its emission-line overlays.
Pure numpy, dependency-free (color math lives in ``heron.color``).
"""

from __future__ import annotations

import numpy as np

from heron.color import srgb


def wavelength_to_rgb(wl: np.ndarray | float) -> np.ndarray:
    """Map wavelength(s) in nm (~380..750) to linear-light RGB (...,3)."""
    wl = np.asarray(wl, dtype=np.float32)
    r = np.zeros_like(wl)
    g = np.zeros_like(wl)
    b = np.zeros_like(wl)

    m = (wl >= 380) & (wl < 440)
    r[m] = -(wl[m] - 440) / 60.0; b[m] = 1.0
    m = (wl >= 440) & (wl < 490)
    g[m] = (wl[m] - 440) / 50.0; b[m] = 1.0
    m = (wl >= 490) & (wl < 510)
    g[m] = 1.0; b[m] = -(wl[m] - 510) / 20.0
    m = (wl >= 510) & (wl < 580)
    r[m] = (wl[m] - 510) / 70.0; g[m] = 1.0
    m = (wl >= 580) & (wl < 645)
    r[m] = 1.0; g[m] = -(wl[m] - 645) / 65.0
    m = (wl >= 645) & (wl <= 750)
    r[m] = 1.0

    # intensity falloff near the limits of vision
    factor = np.zeros_like(wl)
    lo = (wl >= 380) & (wl < 420)
    factor[lo] = 0.3 + 0.7 * (wl[lo] - 380) / 40.0
    mid = (wl >= 420) & (wl < 701)
    factor[mid] = 1.0
    hi = (wl >= 701) & (wl <= 750)
    factor[hi] = 0.3 + 0.7 * (750 - wl[hi]) / 50.0

    rgb_srgb = np.stack([r, g, b], axis=-1) * factor[..., None]
    return srgb.srgb_to_linear(np.clip(rgb_srgb, 0.0, 1.0)).astype(np.float32)
