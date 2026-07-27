"""Bioluminescence / chemiluminescence physics (CLAUDE.md §1.5, §3.2).

An inner-glow emission model — the same family as thermography (emission -> Oklab
palette), but the source is *self-luminescence*, not temperature. The subject is
a translucent participating medium: light is generated through its volume
(thicker, deeper regions emit more), escapes brightest at thin translucent edges,
and scatters softly into the surrounding darkness (screen-space subsurface). The
material table's ``kirlian_activity`` doubles as a "which materials luminesce"
proxy — organic surfaces (skin, hair, foliage) glow; metal and glass stay dark.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from heron.core.graph import RadianceGraph
from heron.materials import MaterialTable, load_material_table


@dataclass(frozen=True)
class LumenParams:
    organic_boost: float = 0.6   # how much material activity modulates glow
    thickness_bias: float = 0.55  # volume/thickness contribution to emission
    depth_bias: float = 0.3       # nearer regions glow a little more
    rim: float = 0.35             # translucent bright-edge escape
    scatter_gain: float = 1.0     # subsurface screen-space scattering
    scatter_scale: float = 0.02   # base blur radius (fraction of image)


@dataclass
class LumenField:
    signal: np.ndarray   # emission [0,1]
    emitter: np.ndarray  # pre-scatter emitter density


def synthesize_lumen(
    graph: RadianceGraph,
    params: LumenParams | None = None,
    table: MaterialTable | None = None,
) -> LumenField:
    """Synthesize the bioluminescent inner-glow field from the subject matte."""
    p = params or LumenParams()
    table = table or load_material_table()
    h, w = graph.shape

    matte = np.clip(graph.matte, 0.0, 1.0)
    body = matte > 0.2
    if not body.any():
        z = np.zeros((h, w), np.float32)
        return LumenField(signal=z, emitter=z)

    # thickness of the translucent medium (0 at silhouette, 1 at the core)
    edt = ndimage.distance_transform_edt(body).astype(np.float32)
    thickness = edt / max(float(edt.max()), 1e-6)

    # organic materials luminesce; metal/glass stay dark
    activity = graph.material_map(table.kirlian_map(), default=0.2).astype(np.float32)
    glow_factor = (1.0 - p.organic_boost) + p.organic_boost * activity

    # emitter density: more volume + nearer -> more generated light
    volume = (1.0 - p.thickness_bias) + p.thickness_bias * thickness
    depth = (1.0 - p.depth_bias) + p.depth_bias * graph.depth
    emitter = matte * glow_factor * volume * depth

    # translucent bright edge: light escapes most where the medium is thin
    falloff = max(edt.max() * 0.25, 2.0)
    rim = matte * np.exp(-edt / falloff)
    emitter = emitter + p.rim * rim
    emitter = np.clip(emitter, 0.0, 1.5).astype(np.float32)

    # subsurface screen-space scattering: multi-scale glow into the darkness
    base = max(max(h, w) * p.scatter_scale, 1.0)
    scatter = np.zeros((h, w), np.float32)
    for k in (1.0, 2.5, 6.0):
        scatter += cv2.GaussianBlur(emitter, (0, 0), sigmaX=base * k) / k
    scatter *= 1.0 / (1.0 + 0.5 + 1.0 / 6.0)  # normalize the weighted sum

    signal = np.clip(0.55 * emitter + p.scatter_gain * scatter, 0.0, 1.0)
    return LumenField(signal=signal.astype(np.float32), emitter=emitter)
