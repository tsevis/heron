"""NIR Camera renderer — near-infrared reflectance through the shared stack.

Another config-only Instrument: a new reflectance Layer B (physics/nir.py) driven
by the material table's ``nir`` field, plus a light film-style Layer C. Supports
two color paths: a monochrome/tinted phosphor palette (B&W infrared) and an
**Aerochrome** false-color channel remap (NIR->red), the classic Kodak EIR look
where foliage burns red.
"""

from __future__ import annotations

import numpy as np

from heron import sensor
from heron.color import palettes
from heron.core.graph import RadianceGraph
from heron.physics.nir import NIRParams, synthesize_nir


def _nir_params(cfg: dict) -> NIRParams:
    b = cfg.get("layerB", {})
    return NIRParams(
        skin_smooth=float(b.get("skin_smooth", 0.6)),
        contrast=float(b.get("contrast", 1.0)),
    )


def _aerochrome(nir: np.ndarray, source_linear: np.ndarray, saturation: float) -> np.ndarray:
    """False-color EIR: R<-NIR, G<-visible red, B<-visible green (Kodak Aerochrome)."""
    src = np.asarray(source_linear, dtype=np.float32)
    rgb = np.stack([nir, src[..., 0], src[..., 1]], axis=-1)
    if saturation != 1.0:
        gray = rgb.mean(axis=-1, keepdims=True)
        rgb = np.clip(gray + (rgb - gray) * saturation, 0.0, 1.0)
    return rgb.astype(np.float32)


def render_nir(
    graph: RadianceGraph,
    source_linear: np.ndarray,
    cfg: dict,
    seed: int = 0,
) -> tuple[np.ndarray, dict, dict]:
    """Return (final linear-RGB image, stage maps, meta)."""
    c = cfg.get("layerC", {})

    # --- Layer B: NIR reflectance ---
    field = synthesize_nir(graph, source_linear, _nir_params(cfg))
    signal = field.signal

    # --- Layer C: gentle IR-film optics ---
    signal = sensor.thermal_bloom(
        signal,
        strength=float(c.get("bloom_strength", 0.25)),
        threshold=float(c.get("bloom_threshold", 0.7)),
        halo=float(c.get("bloom_halo", 0.08)),
    )
    signal = sensor.mtf_blur(signal, sigma_frac=float(c.get("mtf_sigma_frac", 0.0006)))
    signal = sensor.vignette(signal, strength=float(c.get("vignette", 0.15)))

    # --- color: false-color Aerochrome or a phosphor/mono palette ---
    false_color = str(c.get("false_color", "")).lower()
    if false_color == "aerochrome":
        rgb = _aerochrome(signal, source_linear, float(c.get("saturation", 1.3)))
        palette_name = "aerochrome"
    else:
        palette = palettes.load_palette(str(c.get("palette", "white_hot")))
        rgb = palette.apply(signal)
        palette_name = palette.name

    rgb = sensor.blue_noise_dither(rgb, seed, amplitude_lsb=float(c.get("dither_lsb", 1.2)))
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    stages = {"nir": field.signal, "processed_signal": signal}
    meta = {"engine": "nir", "palette": palette_name, "seed": seed}
    return rgb, stages, meta
