"""Classical de-lighting — the make-or-break step (CLAUDE.md §1, §3.1).

A photograph's brightness encodes *external illumination*; a thermogram encodes
*internal emission*. Before synthesizing emission we must suppress the source
lighting. With no intrinsic-decomposition net on disk (Registry: "Not on disk.
Phase 0 uses classical multi-scale Retinex + bilateral shading split"), this
estimates a smooth, edge-aware shading field and divides it out to recover an
illumination-free albedo.

Everything is linear-light float32.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.color import srgb

_EPS = 1e-4


def estimate_shading(linear_rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Estimate a smooth (H,W) shading field from a linear-RGB image in [0,1].

    Works in log-luminance so multiplicative illumination becomes additive, then
    edge-aware-smooths at a large scale (downscaled bilateral) so shading does
    not bleed across material boundaries.
    """
    lum = np.maximum(srgb.luminance(linear_rgb), _EPS)
    log_lum = np.log(lum)

    h, w = lum.shape
    scale = 256.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(log_lum, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = log_lum

    # normalize into a byte-ish range for bilateral, then restore
    lo, hi = float(small.min()), float(small.max())
    span = max(hi - lo, _EPS)
    norm = (small - lo) / span
    sigma_space = max(small.shape) * 0.15
    smooth = cv2.bilateralFilter(norm.astype(np.float32), d=0, sigmaColor=0.15, sigmaSpace=sigma_space)
    smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=max(small.shape) * 0.08)

    shading_log = smooth * span + lo
    shading = np.exp(shading_log)
    if shading.shape != lum.shape:
        shading = cv2.resize(shading, (w, h), interpolation=cv2.INTER_LINEAR)

    # blend toward flat by `strength` (1 = full de-light, 0 = keep lighting)
    shading = shading * strength + np.mean(shading) * (1.0 - strength)
    return shading.astype(np.float32)


def delight(linear_rgb: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (albedo, shading). Albedo is the de-lit reflectance in [0,1].

    ``strength`` in [0,1] controls how aggressively lighting is removed.
    """
    linear_rgb = np.asarray(linear_rgb, dtype=np.float32)
    shading = estimate_shading(linear_rgb, strength=strength)
    mean_shading = float(np.mean(shading))
    albedo = linear_rgb / (shading[..., None] + _EPS) * mean_shading
    # robust normalize to keep albedo in a sane display range
    hi = np.percentile(albedo, 99.5)
    albedo = np.clip(albedo / max(hi, _EPS), 0.0, 1.0)
    return albedo.astype(np.float32), shading
