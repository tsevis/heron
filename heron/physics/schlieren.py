"""Schlieren physics (CLAUDE.md §1.4, §3.2).

The knife-edge Schlieren image is the **signed directional derivative** of the
refractive-index field (≈ density ≈ temperature) along a chosen cutoff axis,
displayed against a mid-gray field: gradients along the axis brighten on one side
and darken on the other.

Heron's density source is the **thermal field reused from the thermography
engine** — so Schlieren automatically reveals the heat structure of the scene.
On top of it, procedural **convection plumes** rise buoyantly from hot regions,
swirled by the shared curl noise, giving the rising-heat / breath look. Color
Schlieren (rainbow cutoff) and BOS (background displacement) are Layer-C modes;
this engine also returns the raw gradient vectors that BOS needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from heron.core.graph import RadianceGraph
from heron.physics import curlnoise
from heron.physics.thermal import ThermalParams, synthesize_thermal


@dataclass(frozen=True)
class SchlierenParams:
    cutoff_angle_deg: float = 0.0   # knife-edge axis (0 = vertical edge, ∂/∂x)
    gain: float = 7.0               # gradient amplification
    plume_amount: float = 0.5       # convection-plume strength
    plume_height: float = 0.45      # how far plumes rise (fraction of image)
    plume_scale: float = 0.06       # curl-noise scale for plume swirl
    blur: float = 0.0015            # pre-derivative smoothing (fraction of size)


@dataclass
class SchlierenField:
    signal: np.ndarray   # knife-edge image [0,1], mid-gray = no gradient
    gx: np.ndarray       # density gradient x (for BOS)
    gy: np.ndarray       # density gradient y (for BOS)
    density: np.ndarray  # the density field (thermal + plumes)


def _convection_plumes(hot: np.ndarray, p: SchlierenParams, seed: int) -> np.ndarray:
    """Rising buoyant plume field advected upward from hot regions, swirled."""
    h, w = hot.shape
    # smear the hot mask upward with decay -> a rising column above hot areas
    steps = max(1, int(p.plume_height * h))
    col = hot.astype(np.float32).copy()
    acc = np.zeros_like(col)
    decay = 0.985
    shift = max(1, h // 200)
    for _ in range(steps // shift):
        col = np.roll(col, -shift, axis=0) * decay
        col[-shift:, :] = 0.0
        acc += col
    acc /= max(float(acc.max()), 1e-6)

    # horizontal wander + fine swirl from curl noise (plumes are turbulent)
    cy, cx = curlnoise.curl_noise_field((h, w), seed, scale=p.plume_scale)
    swirl = 0.5 + 0.5 * cx
    fine = curlnoise.value_noise((h, w), seed + 7, scale=p.plume_scale * 2.0, tag="plume")
    plume = acc * swirl * (0.6 + 0.4 * fine)
    return plume.astype(np.float32)


def synthesize_schlieren(
    graph: RadianceGraph,
    params: SchlierenParams | None = None,
    seed: int = 0,
) -> SchlierenField:
    """Signed directional derivative of the thermal density field + plumes."""
    p = params or SchlierenParams()

    # density source: the thermography engine's temperature field (reused)
    thermal = synthesize_thermal(graph, ThermalParams())
    density = thermal.signal.astype(np.float32).copy()

    # buoyant convection plumes rising from the hottest regions
    if p.plume_amount > 0.0:
        hot = (thermal.signal > 0.6).astype(np.float32)
        density = density + p.plume_amount * _convection_plumes(hot, p, seed)

    h, w = density.shape
    if p.blur > 0.0:
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=max(1.0, max(h, w) * p.blur))

    gx = cv2.Sobel(density, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(density, cv2.CV_32F, 0, 1, ksize=3)

    theta = np.deg2rad(p.cutoff_angle_deg)
    g_axis = np.cos(theta) * gx + np.sin(theta) * gy
    signal = np.clip(0.5 + p.gain * g_axis, 0.0, 1.0).astype(np.float32)

    return SchlierenField(signal=signal, gx=gx.astype(np.float32), gy=gy.astype(np.float32),
                          density=density)
