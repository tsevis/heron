"""Poisson gradient-domain detail fusion (CLAUDE.md §1.6, §3.4; Pérez 2003).

Recombines the smooth synthesized emission field with the *fine structure* of the
source (fabric weave, hair strands, tulle transparency) in the gradient domain,
so detail is injected seamlessly without the halos a naive add would produce. A
**materiality** slider runs from pure-sensor-smooth (0) to painterly-detailed (1).

The Poisson equation ∇²f = div(g) is solved with a DCT (Neumann boundary
conditions — no periodic wraparound), which is fast and artifact-free on images.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.color import srgb


def solve_poisson(laplacian: np.ndarray) -> np.ndarray:
    """Solve ∇²f = ``laplacian`` via FFT (periodic BCs, matching the roll-based
    forward-gradient / backward-divergence stencil). Result has zero mean."""
    h, w = laplacian.shape
    ll = np.fft.fft2(laplacian.astype(np.float64))
    ky = 2.0 * np.cos(2.0 * np.pi * np.arange(h) / h) - 2.0
    kx = 2.0 * np.cos(2.0 * np.pi * np.arange(w) / w) - 2.0
    denom = ky[:, None] + kx[None, :]
    denom[0, 0] = 1.0
    f_hat = ll / denom
    f_hat[0, 0] = 0.0
    return np.fft.ifft2(f_hat).real.astype(np.float32)


def _divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Divergence matched to forward-difference gradients (backward diff)."""
    gxx = gx - np.roll(gx, 1, axis=1)
    gyy = gy - np.roll(gy, 1, axis=0)
    return (gxx + gyy).astype(np.float32)


def poisson_detail_fusion(
    emission: np.ndarray,
    source_linear: np.ndarray,
    materiality: float = 0.5,
    detail_sigma_frac: float = 0.01,
) -> np.ndarray:
    """Inject source high-frequency detail into ``emission`` (gradient domain).

    ``materiality`` in [0,1] scales the injected detail; 0 returns the emission
    unchanged. Both inputs are single-channel-style [0,1] fields (emission) and a
    linear-RGB source.
    """
    if materiality <= 0.0:
        return np.asarray(emission, dtype=np.float32)

    emission = np.asarray(emission, dtype=np.float32)
    h, w = emission.shape
    lum = srgb.luminance(np.asarray(source_linear, dtype=np.float32))
    sigma = max(1.0, max(h, w) * detail_sigma_frac)
    detail = lum - cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma)  # high-pass fine structure

    # forward-difference gradients: emission structure + injected detail
    def fgrad(f):
        gx = np.roll(f, -1, axis=1) - f
        gy = np.roll(f, -1, axis=0) - f
        return gx.astype(np.float32), gy.astype(np.float32)

    egx, egy = fgrad(emission)
    dgx, dgy = fgrad(detail)
    gx = egx + materiality * dgx
    gy = egy + materiality * dgy

    f = solve_poisson(_divergence(gx, gy))
    f += float(emission.mean()) - float(f.mean())  # anchor DC to the emission level
    return np.clip(f, 0.0, 1.0).astype(np.float32)
