"""Optics & detector geometry (CLAUDE.md §0.5, §3.3).

LWIR optics are softer than visible (MTF blur); sensor resolution is coarse
(160x120 .. 640x512) and upsampled with characteristic soft blockiness;
vignetting and the "narcissus" cold-spot round out the lens signature. All in
the [0,1] signal domain.
"""

from __future__ import annotations

import cv2
import numpy as np


def mtf_blur(signal: np.ndarray, sigma_frac: float = 0.0009) -> np.ndarray:
    """Aperture/wavelength MTF softening as a small Gaussian."""
    sigma = max(0.3, max(signal.shape) * sigma_frac)
    return cv2.GaussianBlur(np.asarray(signal, dtype=np.float32), (0, 0), sigmaX=sigma)


def sensor_resolution(signal: np.ndarray, native_rows: int = 0) -> np.ndarray:
    """Render at a coarse detector resolution then upsample (soft blockiness).

    ``native_rows`` of 0 disables the effect; typical values: 120, 240, 512.
    """
    if native_rows <= 0:
        return np.asarray(signal, dtype=np.float32)
    h, w = signal.shape
    nh = int(native_rows)
    nw = max(1, int(round(w * nh / h)))
    small = cv2.resize(signal.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def vignette(signal: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """Darken toward the frame corners."""
    if strength <= 0:
        return np.asarray(signal, dtype=np.float32)
    h, w = signal.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((yy - cy) / (h / 2.0)) ** 2 + ((xx - cx) / (w / 2.0)) ** 2
    mask = 1.0 - strength * np.clip(r2, 0.0, 1.0)
    return (np.asarray(signal, dtype=np.float32) * mask).astype(np.float32)


def narcissus(signal: np.ndarray, strength: float = 0.06) -> np.ndarray:
    """The reflected cold-spot of the detector in the center of the frame."""
    if strength <= 0:
        return np.asarray(signal, dtype=np.float32)
    h, w = signal.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((yy - cy) / (h * 0.35)) ** 2 + ((xx - cx) / (w * 0.35)) ** 2
    dip = strength * np.exp(-r2)
    return np.clip(np.asarray(signal, dtype=np.float32) - dip, 0.0, 1.0).astype(np.float32)
