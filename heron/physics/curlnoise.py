"""Curl noise — a divergence-free vector field for organic motion.

Used to give Kirlian streamers/sparks their "electric wind" swirl (CLAUDE.md
§1.3) and, later, Schlieren convection plumes (§1.4). Built from a smooth scalar
potential (value noise); its curl is divergence-free, so the flow swirls without
sources or sinks. Deterministic per seed (§2.4).
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.core.seeds import rng_for


def value_noise(shape: tuple[int, int], seed: int, scale: float = 0.03, tag: str = "vn") -> np.ndarray:
    """Smooth scalar noise in ~[-1,1] at the given resolution."""
    h, w = shape
    lo_h = max(2, int(h * scale))
    lo_w = max(2, int(w * scale))
    rng = rng_for(seed, tag, lo_h, lo_w)
    low = rng.standard_normal((lo_h, lo_w)).astype(np.float32)
    field = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(h, w) * scale * 0.4)
    m = float(np.abs(field).max())
    return (field / m).astype(np.float32) if m > 1e-6 else field


def curl_noise_field(shape: tuple[int, int], seed: int, scale: float = 0.04):
    """Return (vy, vx): a divergence-free flow field, components in ~[-1,1]."""
    psi = value_noise(shape, seed, scale=scale, tag="curl_psi")
    dpsi_dy = cv2.Sobel(psi, cv2.CV_32F, 0, 1, ksize=3)
    dpsi_dx = cv2.Sobel(psi, cv2.CV_32F, 1, 0, ksize=3)
    vy, vx = dpsi_dx, -dpsi_dy  # curl of a 2D potential
    norm = max(float(np.abs(np.stack([vy, vx])).max()), 1e-6)
    return (vy / norm).astype(np.float32), (vx / norm).astype(np.float32)
