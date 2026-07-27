"""Structure-tensor directional bloom (CLAUDE.md §1.6, §3.4).

Ordinary bloom spreads isotropically. Here the local orientation from the
structure tensor steers an **anisotropic** glow, so emission streams *along*
limbs, hair, and brush-forms rather than blooming in circles. Where the image has
no clear orientation (low coherence) it falls back to isotropic bloom.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.color import srgb


def structure_orientation(guide: np.ndarray, sigma: float):
    """Return (along_edge_angle, coherence) from the structure tensor of ``guide``."""
    gx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma)

    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)   # dominant gradient orientation
    along = theta + np.pi / 2.0                       # smear *along* the structure
    tmp = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    lam1 = 0.5 * (jxx + jyy + tmp)
    lam2 = 0.5 * (jxx + jyy - tmp)
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-6)
    return along.astype(np.float32), np.clip(coherence, 0.0, 1.0).astype(np.float32)


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """A normalized 1-D Gaussian streak rotated to ``angle_deg`` in a square kernel."""
    r = max(1, length // 2)
    ks = 2 * r + 1
    base = np.zeros((ks, ks), np.float32)
    xs = np.arange(ks) - r
    base[r, :] = np.exp(-((xs / (r * 0.6 + 1e-6)) ** 2))
    m = cv2.getRotationMatrix2D((r, r), angle_deg, 1.0)
    k = cv2.warpAffine(base, m, (ks, ks), flags=cv2.INTER_LINEAR)
    s = k.sum()
    return (k / s) if s > 1e-6 else k


def directional_bloom(
    signal: np.ndarray,
    guide: np.ndarray | None = None,
    strength: float = 0.4,
    length_frac: float = 0.012,
    threshold: float = 0.55,
    n_angles: int = 8,
) -> np.ndarray:
    """Add orientation-steered anisotropic bloom to a [0,1] signal."""
    if strength <= 0.0:
        return np.asarray(signal, dtype=np.float32)
    s = np.clip(np.asarray(signal, dtype=np.float32), 0.0, 1.0)
    guide = s if guide is None else np.asarray(guide, dtype=np.float32)
    h, w = s.shape

    along, coh = structure_orientation(guide, sigma=max(h, w) * 0.004)
    hot = np.clip((s - threshold) / max(1.0 - threshold, 1e-3), 0.0, 1.0) * s
    length = max(3, int(length_frac * max(h, w)))

    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    for k in range(n_angles):
        ang = np.pi * k / n_angles                    # orientation is mod pi
        bloom_k = cv2.filter2D(hot, -1, _line_kernel(length, np.degrees(ang)))
        wk = coh * (0.5 + 0.5 * np.cos(2.0 * (along - ang)))  # peaks when aligned
        acc += wk * bloom_k
        wsum += wk

    directional = acc / (wsum + 1e-6)
    isotropic = cv2.GaussianBlur(hot, (0, 0), sigmaX=length * 0.4)
    bloom = coh * directional + (1.0 - coh) * isotropic

    out = 1.0 - (1.0 - s) * (1.0 - strength * bloom)   # screen blend
    return np.clip(out, 0.0, 1.0).astype(np.float32)
