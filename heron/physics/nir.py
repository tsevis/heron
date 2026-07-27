"""Near-infrared (NIR) reflectance physics (CLAUDE.md §1.2, §3.2).

NIR photography is reflectance, not emission or photon amplification — it re-maps
the scene's brightness by *material* using the material table's ``nir`` field:
  * foliage_glow    — chlorophyll reflects NIR strongly -> the Wood effect (near white)
  * skin_translucent— waxy, pale, smoothed skin
  * dye_transparent — dark dyes turn light (many fabrics go transparent)
  * dark            — sky and water absorb/scatter little NIR -> read very dark
  * neutral         — unchanged reflectance

Like every AI-optional layer it degrades: with no material segmentation the whole
image is ``neutral`` (a plausible monochrome-IR grayscale); with SAM 3 materials
the Wood effect and dark skies appear.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from heron.color import srgb
from heron.core.graph import RadianceGraph
from heron.materials import MaterialTable, load_material_table

# nir behavior -> (gamma, gain, lift): reflectance = clip(base**gamma * gain + lift)
_NIR_PARAMS: dict[str, tuple[float, float, float]] = {
    "foliage_glow": (0.7, 1.35, 0.40),
    "skin_translucent": (0.85, 0.95, 0.28),
    "dye_transparent": (0.5, 1.0, 0.12),
    "dark": (1.0, 0.15, 0.0),
    "neutral": (1.0, 1.0, 0.06),
}


@dataclass(frozen=True)
class NIRParams:
    skin_smooth: float = 0.6   # waxy-skin smoothing amount
    contrast: float = 1.0


@dataclass
class NIRField:
    signal: np.ndarray   # NIR reflectance [0,1] (single channel)


def _behavior_param(table: MaterialTable, index: int) -> dict[str, float]:
    return {name: _NIR_PARAMS.get(beh, _NIR_PARAMS["neutral"])[index]
            for name, beh in table.nir_map().items()}


def synthesize_nir(
    graph: RadianceGraph,
    source_linear: np.ndarray,
    params: NIRParams | None = None,
    table: MaterialTable | None = None,
) -> NIRField:
    """Build the NIR reflectance field from the source and the material map."""
    p = params or NIRParams()
    table = table or load_material_table()

    base = srgb.luminance(np.asarray(source_linear, dtype=np.float32))
    gamma = graph.material_map(_behavior_param(table, 0), default=1.0)
    gain = graph.material_map(_behavior_param(table, 1), default=1.0)
    lift = graph.material_map(_behavior_param(table, 2), default=0.06)

    signal = np.clip(np.power(np.clip(base, 0.0, 1.0), gamma) * gain + lift, 0.0, 1.0)

    # waxy skin: smooth the skin regions (translucent, blemish-free)
    if p.skin_smooth > 0.0:
        skin_beh = {name: (1.0 if beh == "skin_translucent" else 0.0)
                    for name, beh in table.nir_map().items()}
        skin_mask = graph.material_map(skin_beh, default=0.0)
        if float(skin_mask.max()) > 0.0:
            smoothed = cv2.bilateralFilter(signal, d=0, sigmaColor=0.15,
                                           sigmaSpace=max(signal.shape) * 0.01)
            w = (skin_mask * p.skin_smooth)[..., None] if signal.ndim == 3 else skin_mask * p.skin_smooth
            signal = signal * (1.0 - w) + smoothed * w

    if p.contrast != 1.0:
        signal = np.clip(0.5 + (signal - 0.5) * p.contrast, 0.0, 1.0)

    return NIRField(signal=signal.astype(np.float32))
