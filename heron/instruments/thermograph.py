"""Thermograph renderer — wires the 4-layer stack for the thermal engine.

Layer B (physics) is done upstream (a ThermalField); this applies Layer D
emphasis then the Layer C sensor chain in physical order
(optics -> detector -> electronics -> display), maps through the Oklab palette,
and finishes with MSX edge fusion and blue-noise dither.
"""

from __future__ import annotations

import numpy as np

from heron import art, sensor
from heron.color import palettes
from heron.core.graph import RadianceGraph
from heron.materials import load_material_table
from heron.physics.thermal import ThermalParams, synthesize_thermal


def _thermal_params(cfg: dict) -> ThermalParams:
    """Map the instrument's ``layerB`` block onto ThermalParams.

    Every field is wired. An earlier version listed only nine of them, so the
    rest silently kept their dataclass defaults — the YAML could say
    ``transmission: 1.5`` and the render would still use 1.0, and the UI dials
    bound to those keys did nothing at all. A field added to ThermalParams and
    not added here is invisible, so keep this exhaustive.
    """
    b = cfg.get("layerB", {})
    d = ThermalParams()  # defaults, so each key is named exactly once below
    return ThermalParams(
        ambient_c=float(b.get("ambient_c", d.ambient_c)),
        core_c=float(b.get("core_c", d.core_c)),
        exertion=float(b.get("exertion", d.exertion)),
        emissivity_reflection=float(b.get("emissivity_reflection", d.emissivity_reflection)),
        detail=float(b.get("detail", d.detail)),
        shape_cooling=float(b.get("shape_cooling", d.shape_cooling)),
        grazing_drop_c=float(b.get("grazing_drop_c", d.grazing_drop_c)),
        occlusion_warm_c=float(b.get("occlusion_warm_c", d.occlusion_warm_c)),
        transmission=float(b.get("transmission", d.transmission)),
        hair_root_warmth=float(b.get("hair_root_warmth", d.hair_root_warmth)),
        grazing_hair_scale=float(b.get("grazing_hair_scale", d.grazing_hair_scale)),
        hair_detail_c=float(b.get("hair_detail_c", d.hair_detail_c)),
        skin_detail_c=float(b.get("skin_detail_c", d.skin_detail_c)),
        conduction_px=float(b.get("conduction_px", d.conduction_px)),
        subsurface_rim_c=float(b.get("subsurface_rim_c", d.subsurface_rim_c)),
        face_map=float(b.get("face_map", d.face_map)),
        body_structure=float(b.get("body_structure", d.body_structure)),
        extremity_drop_c=float(b.get("extremity_drop_c", d.extremity_drop_c)),
        diffusion_iters=int(b.get("diffusion_iters", 24)),
        span_min_c=b.get("span_min_c"),
        span_max_c=b.get("span_max_c"),
    )


def render_thermal(
    graph: RadianceGraph,
    source_linear: np.ndarray,
    cfg: dict,
    seed: int = 0,
) -> tuple[np.ndarray, dict, dict]:
    """Return (final linear-RGB image, stage maps, meta)."""
    c = cfg.get("layerC", {})
    d = cfg.get("layerD", {})

    # --- Layer B: physics ---
    field = synthesize_thermal(
        graph, _thermal_params(cfg), load_material_table(), source_linear=source_linear
    )
    signal = field.signal

    # --- Layer D: emphasis -> directional bloom -> Poisson detail fusion ---
    signal = art.apply_art_direction(signal, graph, source_linear, d)
    emission = signal.copy()

    # --- Layer C: sensor & optics, in physical order ---
    # optics
    signal = sensor.thermal_bloom(
        signal,
        strength=float(c.get("bloom_strength", 0.35)),
        threshold=float(c.get("bloom_threshold", 0.55)),
        halo=float(c.get("bloom_halo", 0.12)),
    )
    signal = sensor.mtf_blur(signal, sigma_frac=float(c.get("mtf_sigma_frac", 0.0009)))
    signal = sensor.vignette(signal, strength=float(c.get("vignette", 0.15)))
    signal = sensor.narcissus(signal, strength=float(c.get("narcissus", 0.05)))
    # detector
    signal = sensor.sensor_resolution(signal, native_rows=int(c.get("sensor_rows", 0)))
    fpn_field = sensor.make_fpn(
        signal.shape, seed,
        amplitude=float(c.get("fpn_amplitude", 0.03)),
        row_weight=float(c.get("fpn_row_weight", 0.5)),
        col_weight=float(c.get("fpn_col_weight", 0.25)),
    )
    signal = sensor.apply_fpn(signal, fpn_field)
    # electronics (temporal)
    # `frame` varies the temporal noise while leaving the FPN field alone — the
    # difference between a detector's burned-in pattern and its live noise. Video
    # sets it per frame; stills leave it at 0 (CLAUDE.md §2.4).
    signal = sensor.add_netd(signal, float(c.get("netd_mk", 40.0)), field.span_c, seed,
                             frame=int(c.get("frame", 0)))
    # display
    signal = sensor.plateau_equalize(
        signal,
        strength=float(c.get("agc_strength", 0.7)),
        plateau=float(c.get("agc_plateau", 0.02)),
    )

    # --- palette (Oklab LUT) ---
    palette = palettes.load_palette(str(c.get("palette", "ironbow")))
    rgb = palette.apply(signal)

    # --- post: MSX edge fusion + dither ---
    rgb = sensor.msx_overlay(rgb, source_linear, opacity=float(c.get("msx_opacity", 0.0)))
    rgb = sensor.blue_noise_dither(rgb, seed, amplitude_lsb=float(c.get("dither_lsb", 1.0)))

    # camera furniture last: the instrument draws it crisply over a soft image
    if bool(c.get("furniture", False)):
        h, w = field.temperature_c.shape
        rgb = sensor.furniture.draw(
            rgb, field.span_c, str(c.get("palette", "ironbow")),
            spot_c=float(field.temperature_c[h // 2, w // 2]),
            params=sensor.furniture.FurnitureParams(status=str(c.get("status", ""))),
        )
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    stages = {
        "emission": emission,          # (H,W) post-emphasis signal
        "processed_signal": signal,    # (H,W) post-sensor signal
        "temperature_c": field.temperature_c,
    }
    meta = {
        "engine": "thermal",
        "palette": palette.name,
        "span_c": [round(field.span_c[0], 2), round(field.span_c[1], 2)],
        "seed": seed,
    }
    return rgb, stages, meta
