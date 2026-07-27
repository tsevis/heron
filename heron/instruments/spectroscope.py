"""Spectroscope renderer — band-pass split + grating through the shared stack.

A spectral-color Layer B (physics/spectral.py) plus Layer C furniture unique to a
lab spectroscope: a **diffraction-grating smear** that fans bright highlights into
horizontal spectra, and **emission-line overlays** (sodium doublet, hydrogen
Balmer) as graphic plate elements. Outputs color directly (no 1-D palette), like
the NIR Aerochrome mode.
"""

from __future__ import annotations

import numpy as np

from heron import sensor
from heron.color import srgb
from heron.color.wavelength import wavelength_to_rgb
from heron.core.graph import RadianceGraph
from heron.physics.spectral import SpectralParams, synthesize_spectral

# (wavelength nm, relative brightness) — famous spectral lines
_EMISSION_LINES = [(589.0, 0.7), (656.3, 0.6), (486.1, 0.55), (434.0, 0.45)]


def _spectral_params(cfg: dict) -> SpectralParams:
    b = cfg.get("layerB", {})
    d = SpectralParams()
    return SpectralParams(
        n_bands=int(b.get("n_bands", d.n_bands)),
        wl_min=float(b.get("wl_min", d.wl_min)),
        wl_max=float(b.get("wl_max", d.wl_max)),
        dispersion=float(b.get("dispersion", d.dispersion)),
        saturation=float(b.get("saturation", d.saturation)),
    )


def _grating_smear(source_linear, wl_min, wl_max, spread, strength, threshold, n=22):
    """Fan bright highlights into horizontal spectra (diffraction grating)."""
    lum = srgb.luminance(source_linear)
    hi = np.clip((lum - threshold) / max(1.0 - threshold, 1e-3), 0.0, 1.0) ** 2
    h, w = lum.shape
    center = 0.5 * (wl_min + wl_max)
    span = max(wl_max - wl_min, 1e-3)
    smear = np.zeros((h, w, 3), np.float32)
    for wl in np.linspace(wl_min, wl_max, n):
        shift = int(round(spread * (wl - center) / span * w))
        smear += np.roll(hi, shift, axis=1)[..., None] * wavelength_to_rgb(float(wl))
    return strength * smear / n


def _emission_overlay(rgb, wl_min, wl_max, spread, opacity):
    """Overlay bright vertical lines positioned by wavelength (lab-plate look)."""
    h, w = rgb.shape[:2]
    center = 0.5 * (wl_min + wl_max)
    span = max(wl_max - wl_min, 1e-3)
    out = rgb.copy()
    for wl, bright in _EMISSION_LINES:
        x = int(round(w / 2 + spread * (wl - center) / span * w))
        if 1 <= x < w - 1:
            color = wavelength_to_rgb(wl) * bright * opacity
            out[:, x - 1:x + 2] = np.clip(out[:, x - 1:x + 2] + color, 0.0, 1.0)
    return out


def render_spectral(
    graph: RadianceGraph,
    source_linear: np.ndarray,
    cfg: dict,
    seed: int = 0,
) -> tuple[np.ndarray, dict, dict]:
    """Return (final linear-RGB image, stage maps, meta)."""
    c = cfg.get("layerC", {})
    p = _spectral_params(cfg)

    field = synthesize_spectral(source_linear, p)
    rgb = field.rgb

    grating = float(c.get("grating_strength", 0.35))
    if grating > 0.0:
        rgb = np.clip(rgb + _grating_smear(
            source_linear, p.wl_min, p.wl_max,
            spread=float(c.get("grating_spread", 0.12)),
            strength=grating,
            threshold=float(c.get("grating_threshold", 0.6)),
        ), 0.0, 1.0)

    if bool(c.get("emission_lines", False)):
        rgb = _emission_overlay(rgb, p.wl_min, p.wl_max,
                                spread=float(c.get("line_spread", 0.4)),
                                opacity=float(c.get("line_opacity", 0.5)))

    rgb = sensor.chromatic_aberration(rgb, strength=float(c.get("chromatic", 0.003)))
    # vignette applied per-channel via the single-channel helper on luminance scale
    rgb = rgb * (1.0 - float(c.get("vignette", 0.2)) * _corner_falloff(rgb.shape[:2]))[..., None]
    grain = float(c.get("grain", 0.15))
    if grain > 0.0:
        rgb = sensor.quantum_noise(rgb, dose=float(c.get("grain_dose", 160.0)), strength=grain, seed=seed)
    rgb = sensor.blue_noise_dither(rgb, seed, amplitude_lsb=float(c.get("dither_lsb", 1.0)))
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    stages = {"spectral": field.rgb}
    meta = {"engine": "spectral", "palette": "spectral", "dispersion": p.dispersion, "seed": seed}
    return rgb, stages, meta


def _corner_falloff(shape):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((yy - cy) / (h / 2.0)) ** 2 + ((xx - cx) / (w / 2.0)) ** 2
    return np.clip(r2, 0.0, 1.0)
