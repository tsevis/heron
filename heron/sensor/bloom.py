"""Thermal bloom + halo (CLAUDE.md §0.5, §3.3).

Hot regions bleed light into their surroundings (optics + detector cross-talk).
A spatially-variant, luminance-driven Gaussian glow applied in the signal domain
before palette mapping, plus a wider low-amplitude halo around the hottest areas.
"""

from __future__ import annotations

import cv2
import numpy as np


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def thermal_bloom(
    signal: np.ndarray,
    strength: float = 0.35,
    radius_frac: float = 0.01,
    threshold: float = 0.55,
    halo: float = 0.12,
) -> np.ndarray:
    """Add hot-region bloom to a [0,1] signal (screen-combined, clamped)."""
    s = np.clip(np.asarray(signal, dtype=np.float32), 0.0, 1.0)
    if strength <= 0.0 and halo <= 0.0:
        return s

    hot = _smoothstep(threshold, 1.0, s) * s
    sigma = max(1.0, max(s.shape) * radius_frac)
    glow = cv2.GaussianBlur(hot, (0, 0), sigmaX=sigma)
    wide = cv2.GaussianBlur(hot, (0, 0), sigmaX=sigma * 4.0)

    out = 1.0 - (1.0 - s) * (1.0 - strength * glow)  # screen blend
    out = out + halo * wide
    return np.clip(out, 0.0, 1.0).astype(np.float32)
