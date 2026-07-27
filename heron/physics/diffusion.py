"""Edge-stopped heat diffusion (CLAUDE.md §0.4, §3.2).

A few iterations of the heat equation, stopped at material boundaries, melt a
seeded temperature field into the smooth, blobby, isotherm-banded look that
distinguishes real thermograms from gradient maps. This is Perona–Malik
anisotropic diffusion with the diffusivity driven by an external edge map
(albedo gradients), so heat flows *within* a material but not across its border.
"""

from __future__ import annotations

import cv2
import numpy as np


def edge_stop_map(guide: np.ndarray, kappa: float = 0.08) -> np.ndarray:
    """Diffusivity in [0,1] from a guide image: ~1 in flat regions, ~0 at edges."""
    guide = np.asarray(guide, dtype=np.float32)
    gx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return np.exp(-(grad / max(kappa, 1e-4)) ** 2).astype(np.float32)


def anisotropic_diffuse(
    field: np.ndarray,
    diffusivity: np.ndarray,
    iterations: int = 20,
    dt: float = 0.2,
) -> np.ndarray:
    """Iterate edge-stopped heat diffusion on ``field`` guided by ``diffusivity``."""
    T = np.asarray(field, dtype=np.float32).copy()
    g = np.asarray(diffusivity, dtype=np.float32)
    for _ in range(int(iterations)):
        # neighbor differences
        dn = np.roll(T, -1, axis=0) - T
        ds = np.roll(T, 1, axis=0) - T
        de = np.roll(T, -1, axis=1) - T
        dw = np.roll(T, 1, axis=1) - T
        # conductance is the min of the two pixels' diffusivity across each edge
        cn = np.minimum(g, np.roll(g, -1, axis=0))
        cs = np.minimum(g, np.roll(g, 1, axis=0))
        ce = np.minimum(g, np.roll(g, -1, axis=1))
        cw = np.minimum(g, np.roll(g, 1, axis=1))
        T = T + dt * (cn * dn + cs * ds + ce * de + cw * dw)
    return T.astype(np.float32)
