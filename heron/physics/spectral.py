"""Spectroscope physics (CLAUDE.md §1.5, §3.2).

Band-pass decomposition: a smooth spectrum is reconstructed per pixel from linear
RGB (spectral upsampling via three Gaussian channel sensitivities), sampled into
N narrow wavelength bands, each re-rendered in its own spectral hue. Recombining
the bands with a horizontal per-wavelength shift simulates prism/grating
**dispersion** — light fans into a spectrum. Overall brightness is preserved so
the image stays readable.

Works from the source color directly (no Layer A needed), so it runs with
``--no-ai``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heron.color import srgb
from heron.color.wavelength import wavelength_to_rgb

# Gaussian spectral sensitivities of the R/G/B channels (center nm, sigma nm).
_CHANNELS = ((600.0, 55.0), (540.0, 45.0), (460.0, 40.0))


@dataclass(frozen=True)
class SpectralParams:
    n_bands: int = 28
    wl_min: float = 410.0
    wl_max: float = 690.0
    dispersion: float = 0.0    # horizontal spread (fraction of width)
    saturation: float = 1.15


@dataclass
class SpectralField:
    rgb: np.ndarray   # spectral-recombined linear RGB (H,W,3)


def _sensitivity(wl: float) -> np.ndarray:
    """(wR,wG,wB): how strongly wavelength ``wl`` excites each channel."""
    return np.array([np.exp(-((wl - c) ** 2) / (2 * s * s)) for c, s in _CHANNELS], np.float32)


def synthesize_spectral(
    source_linear: np.ndarray,
    params: SpectralParams | None = None,
) -> SpectralField:
    """Decompose into wavelength bands and recombine with dispersion."""
    p = params or SpectralParams()
    src = np.asarray(source_linear, dtype=np.float32)
    h, w, _ = src.shape
    R, G, B = src[..., 0], src[..., 1], src[..., 2]

    wls = np.linspace(p.wl_min, p.wl_max, p.n_bands)
    center = 0.5 * (p.wl_min + p.wl_max)
    span = max(p.wl_max - p.wl_min, 1e-3)

    out = np.zeros((h, w, 3), np.float32)
    white = np.zeros(3, np.float32)  # pipeline response to a neutral input
    for wl in wls:
        s = _sensitivity(float(wl))
        band = R * s[0] + G * s[1] + B * s[2]    # band intensity (H,W)
        wl_rgb = wavelength_to_rgb(float(wl))
        white += float(s.sum()) * wl_rgb
        shift = int(round(p.dispersion * (wl - center) / span * w))
        if shift != 0:
            band = np.roll(band, shift, axis=1)
        out += band[..., None] * wl_rgb

    # white-balance: neutralize the spectrum's green bias so gray -> gray
    out = out / (white[None, None, :] + 1e-6)

    # preserve overall brightness (bands sum brighter than the original)
    src_lum = float(srgb.luminance(src).mean())
    out_lum = float(srgb.luminance(out).mean())
    if out_lum > 1e-6:
        out *= src_lum / out_lum

    if p.saturation != 1.0:
        gray = srgb.luminance(out)[..., None]
        out = np.clip(gray + (out - gray) * p.saturation, 0.0, 1.0)

    return SpectralField(rgb=np.clip(out, 0.0, 1.0).astype(np.float32))
