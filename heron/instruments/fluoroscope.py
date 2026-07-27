"""Fluoroscope renderer — the X-ray engine through the shared 4-layer stack.

Proof that an Instrument is *configuration*, not a codebase: this reuses the same
Radiance Graph (Layer A), Layer D emphasis, palette engine, and most of Layer C
as the Thermograph. Only Layer B differs (Beer–Lambert instead of thermal), plus
a phosphor-appropriate Layer C chain (quantum noise + bloom, no microbolometer
FPN/NETD, no MSX).
"""

from __future__ import annotations

import numpy as np

from heron import art, sensor
from heron.color import palettes
from heron.core.graph import RadianceGraph
from heron.materials import load_material_table
from heron.physics.xray import XrayParams, synthesize_xray


def _xray_params(cfg: dict) -> XrayParams:
    b = cfg.get("layerB", {})
    return XrayParams(
        thickness_gain=float(b.get("thickness_gain", 3.5)),
        flesh_floor=float(b.get("flesh_floor", 0.12)),
        contrast=float(b.get("contrast", 0.9)),
        depth_thickness=float(b.get("depth_thickness", 0.4)),
        rim=float(b.get("rim", 0.3)),
        skeleton=float(b.get("skeleton", 0.0)),
        headroom=float(b.get("headroom", 0.9)),
    )


def render_xray(
    graph: RadianceGraph,
    source_linear: np.ndarray,
    cfg: dict,
    seed: int = 0,
) -> tuple[np.ndarray, dict, dict]:
    """Return (final linear-RGB image, stage maps, meta)."""
    c = cfg.get("layerC", {})
    d = cfg.get("layerD", {})

    # --- Layer B: physics (with optional stylized skeleton proxy) ---
    p = _xray_params(cfg)
    # Detected once in Layer A (scene/ai) and cached on the graph — Layer B
    # must never invoke a model itself (CLAUDE.md §2.7).
    bones = graph.extras.get("skeleton_bones") if p.skeleton > 0.0 else None
    field = synthesize_xray(graph, p, load_material_table(), bones=bones)
    signal = field.signal

    # --- Layer D: emphasis -> directional bloom -> Poisson detail fusion ---
    signal = art.apply_art_direction(signal, graph, source_linear, d)
    emission = signal.copy()

    # --- Layer C: phosphor screen & optics ---
    signal = sensor.thermal_bloom(  # phosphor glow (screen blend, reused)
        signal,
        strength=float(c.get("bloom_strength", 0.4)),
        threshold=float(c.get("bloom_threshold", 0.4)),
        halo=float(c.get("bloom_halo", 0.15)),
    )
    signal = sensor.mtf_blur(signal, sigma_frac=float(c.get("mtf_sigma_frac", 0.0012)))
    signal = sensor.quantum_noise(
        signal,
        dose=float(c.get("quantum_dose", 80.0)),
        strength=float(c.get("quantum_strength", 0.5)),
        seed=seed,
    )
    signal = sensor.vignette(signal, strength=float(c.get("vignette", 0.25)))

    # --- palette (phosphor tint via Oklab LUT) ---
    palette = palettes.load_palette(str(c.get("palette", "fluoroscope_blue")))
    rgb = palette.apply(signal)
    rgb = sensor.blue_noise_dither(rgb, seed, amplitude_lsb=float(c.get("dither_lsb", 1.0)))
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    stages = {"emission": emission, "processed_signal": signal, "transmission": field.transmission}
    meta = {"engine": "xray", "palette": palette.name, "seed": seed}
    return rgb, stages, meta
