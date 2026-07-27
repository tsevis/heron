"""MSX-style edge fusion (CLAUDE.md §0.5, §3.3).

FLIR's Multi-Spectral Dynamic Imaging composites visible-light edges over the
thermal image — a hugely recognizable part of the "real FLIR screenshot" look,
and trivial for Heron since it *has* the visible source. Edges are extracted
from the source luminance and embossed (darkened) over the palette-mapped image.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.color import srgb


def msx_overlay(
    rgb_linear: np.ndarray,
    source_linear: np.ndarray,
    opacity: float = 0.5,
    edge_sigma: float = 0.6,
) -> np.ndarray:
    """Emboss visible edges of ``source_linear`` onto a palette-mapped image."""
    if opacity <= 0.0:
        return np.asarray(rgb_linear, dtype=np.float32)

    lum = srgb.luminance(np.asarray(source_linear, dtype=np.float32))
    lum = cv2.GaussianBlur(lum, (0, 0), sigmaX=edge_sigma)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx * gx + gy * gy)
    hi = np.percentile(edges, 99.0)
    edges = np.clip(edges / max(hi, 1e-6), 0.0, 1.0)

    factor = (1.0 - opacity * edges)[..., None]
    return np.clip(np.asarray(rgb_linear, dtype=np.float32) * factor, 0.0, 1.0).astype(np.float32)
