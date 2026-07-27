"""Fluoroscope / X-ray physics (CLAUDE.md §1.1, §3.2).

Beer–Lambert attenuation: I = I0 * exp(-mu * d). Thickness ``d`` comes from the
depth map read as a volume — the distance transform of the subject matte gives a
front-to-back proxy (thin at the silhouette, thick through the core), modulated
by depth. ``mu`` comes from each material's X-ray mass class in the table.

Rendered as a luminous, translucent body glowing against darkness: thin edges
and fabric are nearly transparent, dense/thick tissue forms the bright inner
core, with the classic bright-edge falloff the exponential produces. This is
stylized-anatomical, never diagnostic (honesty guardrail §2.8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from heron.core.graph import RadianceGraph
from heron.materials import MaterialTable, load_material_table


@dataclass(frozen=True)
class XrayParams:
    thickness_gain: float = 3.5   # exp attenuation strength (higher = denser core)
    flesh_floor: float = 0.12     # minimum translucent-flesh glow (tissue only)
    contrast: float = 0.9         # gamma on the absorption term
    depth_thickness: float = 0.4  # how much depth modulates thickness
    rim: float = 0.3              # bright silhouette-edge glow
    skeleton: float = 0.0         # bone overlay strength (needs a pose proxy)
    headroom: float = 0.9         # display ceiling for tissue; the rest is bloom's


@dataclass
class XrayField:
    signal: np.ndarray        # windowed [0,1] pre-sensor emission
    transmission: np.ndarray  # Beer-Lambert transmission T(x,y)
    thickness: np.ndarray     # normalized path length


def _boundary_safe_edt(body: np.ndarray, pad_frac: float = 0.12) -> np.ndarray:
    """Distance transform that never treats a frame-cropped edge as open space.

    Plain ``distance_transform_edt`` measures distance to the nearest False
    pixel *within the array*. When the subject silhouette is cut off by the
    frame (shoulders below a tight face crop, near-universal in portraits),
    there is no real background pixel on that side, so the transform reports
    an unbounded-looking distance there — the reported "thickest point" ends
    up on the frame edge itself rather than at the true anatomical core. That
    inflated max then corrupts the tau normalization for every other pixel
    (see heron/physics/xray.py history / CLAUDE.md fluoroscope fix).

    Padding with a false border first caps how far the transform will assume
    the body extends past the frame. The exact pad size barely matters once
    it's large enough to give the transform real slack (~6% of the smaller
    dimension in testing) — it just needs to not be zero. Bodies that don't
    touch the frame are unaffected: a false border added further out than
    the existing real background can only leave the nearest-False distance
    unchanged, never increase it.
    """
    pad = max(int(pad_frac * min(body.shape)), 1)
    padded = np.pad(body, pad, mode="constant", constant_values=False)
    return ndimage.distance_transform_edt(padded)[pad:-pad, pad:-pad].astype(np.float32)


def synthesize_xray(
    graph: RadianceGraph,
    params: XrayParams | None = None,
    table: MaterialTable | None = None,
    bones: np.ndarray | None = None,
) -> XrayField:
    """Run the Beer–Lambert fluoroscope engine over a Radiance Graph.

    ``bones`` is an optional [0,1] stylized skeleton-glow map (see
    physics/skeleton.py) added into the luminous core.
    """
    p = params or XrayParams()
    table = table or load_material_table()

    matte = np.clip(graph.matte, 0.0, 1.0)
    body = matte > 0.3

    # thickness proxy: distance transform of the silhouette (0 edge -> 1 core),
    # modulated by depth so nearer tissue reads slightly thicker.
    edt = _boundary_safe_edt(body)
    tau = edt / max(float(edt.max()), 1e-6)
    tau = tau * (1.0 - p.depth_thickness + p.depth_thickness * graph.depth)

    # Beer–Lambert transmission through the body
    mu = graph.material_map(table.xray_mu_map(), default=0.55)
    transmission = np.exp(-mu * tau * p.thickness_gain).astype(np.float32)
    absorption = 1.0 - transmission  # bright at dense / thick core

    # translucent-flesh glow: always a little, luminous where dense & thick.
    # The floor is *flesh* only — gated by attenuation class, so fabric and hair
    # stay nearly transparent as a real radiograph shows them. Applying it to
    # the whole matte lifted every material off black, which on a big-hair
    # portrait meant half the frame sat on a flat mid-bright pedestal.
    soft_mu, fabric_mu = table.XRAY_MU["soft"], table.XRAY_MU["fabric"]
    tissue = np.clip((mu - fabric_mu) / max(soft_mu - fabric_mu, 1e-6), 0.0, 1.0)
    floor = p.flesh_floor * tissue
    glow = matte * (floor + (1.0 - floor) * np.power(absorption, p.contrast))

    # bright silhouette edge (the exponential's bright-edge falloff)
    rim_glow = matte * np.exp(-tau * 6.0)
    signal = glow + p.rim * rim_glow

    # stylized skeletal core (bones read as the brightest internal structure)
    if bones is not None and p.skeleton > 0.0:
        signal = signal + p.skeleton * np.asarray(bones, dtype=np.float32) * matte

    # Radiographic window/level: map the body's robust signal range onto the
    # display. Dividing by a high percentile alone (what this used to do) only
    # rescales the top and leaves the pedestal, so the output started at ~0.4
    # instead of black — washed out, with the densest tissue pinned against
    # 1.0 where Layer C's screen-blend bloom could only clip it to flat white.
    # Subtracting the low anchor restores the blacks, and ``headroom`` keeps
    # tissue below 1.0 so bloom has somewhere to go.
    ceiling = 1.0
    if body.any():
        lo = float(np.percentile(signal[body], 2.0))
        hi = float(np.percentile(signal[body], 99.5))
        signal = (signal - lo) / max(hi - lo, 1e-6) * p.headroom
        signal = signal * matte  # keep the background black after the shift
        ceiling = p.headroom     # anything past the window's top reads as its max
    signal = np.clip(signal, 0.0, ceiling).astype(np.float32)

    return XrayField(signal=signal, transmission=transmission, thickness=tau.astype(np.float32))
