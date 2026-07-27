"""Schlieren Bench renderer — knife-edge / color / BOS through the shared stack.

A new Layer B (physics/schlieren.py) that itself *reuses the thermography engine*
for its density field, plus reused Layer C. Three visualization modes:
  * knife  — grayscale signed-derivative (classic knife-edge)
  * color  — rainbow cutoff (color Schlieren) via a spanning palette
  * bos    — background-oriented Schlieren: a reference texture displaced by the
             density gradient, so heat shows as ripples
"""

from __future__ import annotations

import cv2
import numpy as np

from heron import sensor
from heron.color import palettes
from heron.core.graph import RadianceGraph
from heron.physics import curlnoise
from heron.physics.schlieren import SchlierenParams, synthesize_schlieren


def _schlieren_params(cfg: dict) -> SchlierenParams:
    b = cfg.get("layerB", {})
    d = SchlierenParams()
    return SchlierenParams(
        cutoff_angle_deg=float(b.get("cutoff_angle_deg", d.cutoff_angle_deg)),
        gain=float(b.get("gain", d.gain)),
        plume_amount=float(b.get("plume_amount", d.plume_amount)),
        plume_height=float(b.get("plume_height", d.plume_height)),
        plume_scale=float(b.get("plume_scale", d.plume_scale)),
        blur=float(b.get("blur", d.blur)),
    )


def _bos(field, seed: int, gain: float, dot_px: float = 7.0) -> np.ndarray:
    """Background-oriented Schlieren: displace a reference **dot lattice** by the
    density gradient, so heat bends the pattern into visible ripples."""
    h, w = field.signal.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    freq = 2.0 * np.pi / max(dot_px, 3.0)
    bg = 0.5 + 0.5 * (np.cos(freq * xx) * np.cos(freq * yy))  # dot lattice
    map_x = (xx + field.gx * gain).astype(np.float32)
    map_y = (yy + field.gy * gain).astype(np.float32)
    return cv2.remap(bg.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def render_schlieren(
    graph: RadianceGraph,
    source_linear: np.ndarray,
    cfg: dict,
    seed: int = 0,
) -> tuple[np.ndarray, dict, dict]:
    """Return (final linear-RGB image, stage maps, meta)."""
    c = cfg.get("layerC", {})

    field = synthesize_schlieren(graph, _schlieren_params(cfg), seed=seed)

    mode = str(c.get("mode", "knife")).lower()
    if mode == "bos":
        signal = _bos(field, seed, gain=float(c.get("bos_gain", 60.0)))
    else:
        signal = field.signal

    # Layer C: gentle optics + grain
    signal = sensor.mtf_blur(signal, sigma_frac=float(c.get("mtf_sigma_frac", 0.0008)))
    signal = sensor.vignette(signal, strength=float(c.get("vignette", 0.2)))

    palette = palettes.load_palette(str(c.get("palette", "white_hot")))
    rgb = palette.apply(signal)
    grain = float(c.get("grain", 0.25))
    if grain > 0.0:
        rgb = sensor.quantum_noise(rgb, dose=float(c.get("grain_dose", 150.0)), strength=grain, seed=seed)
    rgb = sensor.blue_noise_dither(rgb, seed, amplitude_lsb=float(c.get("dither_lsb", 1.0)))
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    stages = {"knife": field.signal, "density": field.density}
    meta = {"engine": "schlieren", "palette": palette.name, "mode": mode, "seed": seed}
    return rgb, stages, meta
