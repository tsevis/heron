"""Fiber-optic faceplate honeycomb (CLAUDE.md §1.2, §3.3).

Image-intensifier tubes bond the phosphor to a fiber-optic faceplate whose
hexagonal packing leaves a faint honeycomb imprint over the image. Synthesized
here as a hex lattice (three sinusoids 60 deg apart) darkening the cell walls.
Static structure — like FPN, it belongs to the tube, not the frame.
"""

from __future__ import annotations

import numpy as np

_SQRT3_2 = 0.8660254


def hex_faceplate(signal: np.ndarray, cell_px: float = 6.0, strength: float = 0.12) -> np.ndarray:
    """Overlay a subtle hexagonal honeycomb on a [0,1] signal."""
    if strength <= 0.0:
        return np.asarray(signal, dtype=np.float32)
    s = np.asarray(signal, dtype=np.float32)
    h, w = s.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    freq = 2.0 * np.pi / max(cell_px, 2.0)
    # three lattice directions 60 degrees apart -> hexagonal symmetry
    field = (
        np.cos(freq * xx)
        + np.cos(freq * (0.5 * xx + _SQRT3_2 * yy))
        + np.cos(freq * (-0.5 * xx + _SQRT3_2 * yy))
    )
    field = (field + 3.0) / 6.0  # -> [0,1], ~1 at cell centers, low at walls
    walls = np.clip(1.0 - field, 0.0, 1.0) ** 2
    return np.clip(s * (1.0 - strength * walls), 0.0, 1.0).astype(np.float32)
