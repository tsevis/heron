"""Kirlian Plate renderer — corona discharge through the shared stack.

The corona intensity field (physics/kirlian.py) is bloomed, mapped through a
phosphor/plate palette, given chromatic streamer fringing, then grained and
vignetted like a film plate. Config-only, like every instrument: a new Layer B
plus reused Layer C.
"""

from __future__ import annotations

import numpy as np

from heron import sensor
from heron.color import palettes
from heron.core.graph import RadianceGraph
from heron.physics.kirlian import KirlianParams, synthesize_kirlian


def _kirlian_params(cfg: dict) -> KirlianParams:
    b = cfg.get("layerB", {})
    d = KirlianParams()
    return KirlianParams(
        aura_falloff=float(b.get("aura_falloff", d.aura_falloff)),
        aura_gain=float(b.get("aura_gain", d.aura_gain)),
        inner_glow=float(b.get("inner_glow", d.inner_glow)),
        n_streamers=int(b.get("n_streamers", d.n_streamers)),
        streamer_length=float(b.get("streamer_length", d.streamer_length)),
        branch_prob=float(b.get("branch_prob", d.branch_prob)),
        jitter=float(b.get("jitter", d.jitter)),
        curl_amp=float(b.get("curl_amp", d.curl_amp)),
        spark_amount=float(b.get("spark_amount", d.spark_amount)),
    )


def render_corona(
    graph: RadianceGraph,
    source_linear: np.ndarray,
    cfg: dict,
    seed: int = 0,
) -> tuple[np.ndarray, dict, dict]:
    """Return (final linear-RGB image, stage maps, meta)."""
    c = cfg.get("layerC", {})

    # --- Layer B: corona synthesis ---
    field = synthesize_kirlian(graph, _kirlian_params(cfg), seed=seed)
    signal = field.intensity

    # --- Layer C: plate optics ---
    signal = sensor.thermal_bloom(
        signal,
        strength=float(c.get("bloom_strength", 0.45)),
        threshold=float(c.get("bloom_threshold", 0.35)),
        halo=float(c.get("bloom_halo", 0.2)),
    )
    signal = sensor.vignette(signal, strength=float(c.get("vignette", 0.35)))

    # --- palette (corona tint) + chromatic fringing + film grain ---
    palette = palettes.load_palette(str(c.get("palette", "fluoroscope_blue")))
    rgb = palette.apply(signal)
    rgb = sensor.chromatic_aberration(rgb, strength=float(c.get("chromatic", 0.004)))

    grain = float(c.get("grain", 0.0))
    if grain > 0.0:
        rgb = sensor.quantum_noise(rgb, dose=float(c.get("grain_dose", 120.0)), strength=grain, seed=seed)
    rgb = sensor.blue_noise_dither(rgb, seed, amplitude_lsb=float(c.get("dither_lsb", 1.0)))
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    stages = {"intensity": field.intensity, "aura": field.aura, "streamers": field.streamers}
    meta = {"engine": "corona", "palette": palette.name, "seed": seed}
    return rgb, stages, meta
