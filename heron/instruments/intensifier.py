"""Intensifier Tube renderer — night-vision through the shared 4-layer stack.

Another config-only Instrument: a new photon-flux Layer B (physics/photon.py)
feeding a phosphor Layer C. Unlike the emission instruments it consumes the
source image's real luminance directly, so it renders well even with ``--no-ai``.
Layer C adds the tube signature: intensifier halos, the hex fiber-optic
faceplate, optional scanlines, heavy vignette, and a phosphor palette.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron import sensor
from heron.color import palettes
from heron.core.graph import RadianceGraph
from heron.physics.photon import PhotonParams, synthesize_photon


def _photon_params(cfg: dict) -> PhotonParams:
    b = cfg.get("layerB", {})
    return PhotonParams(
        gain=float(b.get("gain", 4.0)),
        gamma=float(b.get("gamma", 0.55)),
        flux=float(b.get("flux", 45.0)),
        black_level=float(b.get("black_level", 0.02)),
        bloom_gate=float(b.get("bloom_gate", 0.85)),
    )


def render_photon(
    graph: RadianceGraph,
    source_linear: np.ndarray,
    cfg: dict,
    seed: int = 0,
) -> tuple[np.ndarray, dict, dict]:
    """Return (final linear-RGB image, stage maps, meta)."""
    c = cfg.get("layerC", {})

    # --- Layer B: photon amplification of real scene light ---
    field = synthesize_photon(source_linear, _photon_params(cfg), seed=seed, graph=graph)
    signal = field.signal
    emission = signal.copy()

    # --- Layer C: detector resolution, halos, phosphor furniture ---
    signal = sensor.sensor_resolution(signal, native_rows=int(c.get("sensor_rows", 0)))

    # intensifier bloom + a wide halo grown from bright light sources
    signal = sensor.thermal_bloom(
        signal,
        strength=float(c.get("bloom_strength", 0.5)),
        threshold=float(c.get("bloom_threshold", 0.6)),
        halo=float(c.get("bloom_halo", 0.2)),
    )
    halo = float(c.get("source_halo", 0.35))
    if halo > 0.0 and field.sources.any():
        glow = cv2.GaussianBlur(field.sources, (0, 0), sigmaX=max(signal.shape) * 0.02)
        signal = np.clip(signal + halo * glow, 0.0, 1.0)

    signal = sensor.hex_faceplate(
        signal,
        cell_px=float(c.get("faceplate_cell_px", 6.0)),
        strength=float(c.get("faceplate", 0.12)),
    )
    signal = sensor.scanlines(
        signal,
        period_px=float(c.get("scanline_period", 3.0)),
        strength=float(c.get("scanlines", 0.0)),
        jitter=float(c.get("scanline_jitter", 0.0)),
        seed=seed,
    )
    signal = sensor.vignette(signal, strength=float(c.get("vignette", 0.4)))

    # --- phosphor palette + dither ---
    palette = palettes.load_palette(str(c.get("palette", "p22_green")))
    rgb = palette.apply(signal)
    rgb = sensor.blue_noise_dither(rgb, seed, amplitude_lsb=float(c.get("dither_lsb", 1.0)))
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    stages = {"emission": emission, "processed_signal": signal, "sources": field.sources}
    meta = {"engine": "photon", "palette": palette.name, "seed": seed}
    return rgb, stages, meta
