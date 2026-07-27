"""Chromatic aberration (CLAUDE.md §1.3, §3.3).

Radially scales the red and blue channels apart, giving corona streamers their
characteristic colored fringing (and lending any instrument a lens-like edge
color-shift). Operates on linear-light RGB.
"""

from __future__ import annotations

import cv2
import numpy as np


def chromatic_aberration(rgb: np.ndarray, strength: float = 0.004) -> np.ndarray:
    """Split R/B channels radially by ``strength`` (fraction of image size)."""
    if strength <= 0.0:
        return np.asarray(rgb, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    def warp(chan, scale):
        map_x = (cx + (xx - cx) * scale).astype(np.float32)
        map_y = (cy + (yy - cy) * scale).astype(np.float32)
        return cv2.remap(chan, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    out = rgb.copy()
    out[..., 0] = warp(rgb[..., 0], 1.0 + strength)   # red expands
    out[..., 2] = warp(rgb[..., 2], 1.0 - strength)   # blue contracts
    return np.clip(out, 0.0, 1.0).astype(np.float32)
