"""Spectral-residual saliency (Hou & Zhang 2007), pure-numpy.

CLAUDE.md Registry: "Saliency — Not on disk. Phase 0: spectral-residual
(OpenCV) + face-boost". Implemented with numpy FFT so it needs no opencv-contrib
saliency module. Feeds Layer D emphasis and the classical subject matte.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.color import srgb


def spectral_residual(linear_rgb: np.ndarray, work_size: int = 128) -> np.ndarray:
    """Return a saliency map in [0,1] at the input resolution."""
    lum = srgb.luminance(np.asarray(linear_rgb, dtype=np.float32))
    h, w = lum.shape
    scale = work_size / max(h, w)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(lum, (sw, sh), interpolation=cv2.INTER_AREA)

    f = np.fft.fft2(small)
    amplitude = np.abs(f)
    phase = np.angle(f)
    log_amp = np.log(amplitude + 1e-8)

    # spectral residual = log amplitude minus its local average
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    smooth_log = cv2.filter2D(log_amp.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    residual = log_amp - smooth_log

    recon = np.fft.ifft2(np.exp(residual + 1j * phase))
    sal = np.abs(recon) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), sigmaX=max(sw, sh) * 0.02)

    sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(sal.min()), float(sal.max())
    sal = (sal - lo) / max(hi - lo, 1e-8)
    return sal.astype(np.float32)


def center_bias(shape: tuple[int, int], strength: float = 0.4) -> np.ndarray:
    """A soft center-weighting map in [0,1] — photographers frame subjects central."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((yy - cy) / (h * 0.6)) ** 2 + ((xx - cx) / (w * 0.6)) ** 2
    g = np.exp(-r2)
    return (1.0 - strength) + strength * g
