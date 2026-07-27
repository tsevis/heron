"""Lumen Chamber renderer — bioluminescent inner glow through the shared stack.

Shares the emission->palette family with the Thermograph, which is what made it
cheap to add: a new inner-glow Layer B (physics/lumen.py) feeding a soft Layer C
and a monochromatic high-key palette against darkness.
"""

from __future__ import annotations

import numpy as np

from heron import art, sensor
from heron.color import palettes
from heron.core.graph import RadianceGraph
from heron.physics.lumen import LumenParams, synthesize_lumen


def _lumen_params(cfg: dict) -> LumenParams:
    b = cfg.get("layerB", {})
    d = LumenParams()
    return LumenParams(
        organic_boost=float(b.get("organic_boost", d.organic_boost)),
        thickness_bias=float(b.get("thickness_bias", d.thickness_bias)),
        depth_bias=float(b.get("depth_bias", d.depth_bias)),
        rim=float(b.get("rim", d.rim)),
        scatter_gain=float(b.get("scatter_gain", d.scatter_gain)),
        scatter_scale=float(b.get("scatter_scale", d.scatter_scale)),
    )


def render_lumen(
    graph: RadianceGraph,
    source_linear: np.ndarray,
    cfg: dict,
    seed: int = 0,
) -> tuple[np.ndarray, dict, dict]:
    """Return (final linear-RGB image, stage maps, meta)."""
    c = cfg.get("layerC", {})
    d = cfg.get("layerD", {})

    # --- Layer B: bioluminescent emission ---
    field = synthesize_lumen(graph, _lumen_params(cfg))
    signal = field.signal

    # --- Layer D: emphasis -> directional bloom -> Poisson detail fusion ---
    signal = art.apply_art_direction(signal, graph, source_linear, d)
    emission = signal.copy()

    # --- Layer C: soft glow optics ---
    signal = sensor.thermal_bloom(
        signal,
        strength=float(c.get("bloom_strength", 0.5)),
        threshold=float(c.get("bloom_threshold", 0.3)),
        halo=float(c.get("bloom_halo", 0.25)),
    )
    signal = sensor.vignette(signal, strength=float(c.get("vignette", 0.3)))

    # --- palette (monochromatic high-key) + grain ---
    palette = palettes.load_palette(str(c.get("palette", "p22_green")))
    rgb = palette.apply(signal)
    grain = float(c.get("grain", 0.2))
    if grain > 0.0:
        rgb = sensor.quantum_noise(rgb, dose=float(c.get("grain_dose", 140.0)), strength=grain, seed=seed)
    rgb = sensor.blue_noise_dither(rgb, seed, amplitude_lsb=float(c.get("dither_lsb", 1.0)))
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    stages = {"emission": emission, "emitter": field.emitter}
    meta = {"engine": "lumen", "palette": palette.name, "seed": seed}
    return rgb, stages, meta
