"""Image-intensifier (night-vision) physics (CLAUDE.md §1.2, §3.2).

Unlike the emission engines (thermal, X-ray), the intensifier tube amplifies the
*real reflected light* of the scene — so it works from the **source luminance**,
not the de-lit albedo. Photon-starved rendering: luminance -> photon flux with
strong Poisson (shot) noise, high gain that lifts shadows into visible-but-grainy
range, and a phosphor response curve. The single-channel signal is later tinted
by a P22-green / P43-white phosphor palette in Layer C.

Poisson noise is the defining artifact and is seeded for determinism (§2.4); for
video it re-rolls per frame while the faceplate/FPN-like structure stays fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heron.color import srgb
from heron.core.graph import RadianceGraph
from heron.core.seeds import rng_for


@dataclass(frozen=True)
class PhotonParams:
    gain: float = 4.0        # light amplification
    gamma: float = 0.55      # phosphor response (<1 lifts shadows)
    flux: float = 45.0       # photon-count scale (lower = grainier)
    black_level: float = 0.02
    bloom_gate: float = 0.85  # scene luminance above this is a "light source"


@dataclass
class PhotonField:
    signal: np.ndarray     # amplified, noisy [0,1] photon signal
    sources: np.ndarray    # bright-source mask (drives intensifier halos)


def synthesize_photon(
    source_linear: np.ndarray,
    params: PhotonParams | None = None,
    seed: int = 0,
    frame: int = 0,
    graph: RadianceGraph | None = None,
) -> PhotonField:
    """Run the intensifier photon engine on the source image's real luminance."""
    p = params or PhotonParams()
    lum = srgb.luminance(np.asarray(source_linear, dtype=np.float32))

    # amplify available light; a soft exposure knee rolls off highlights (real
    # tubes saturate gracefully, they do not hard-clip), then phosphor gamma
    # lifts the shadows into the visible-but-grainy range.
    amplified = np.maximum((lum - p.black_level) * p.gain, 0.0)
    exposed = 1.0 - np.exp(-amplified)          # smooth saturation toward 1
    brightness = np.clip(np.power(exposed, p.gamma), 0.0, 1.0)

    # Poisson shot noise: photon count N = flux * brightness
    rng = rng_for(seed, "photon", frame, *lum.shape)
    counts = rng.poisson(np.clip(p.flux * brightness, 0.0, None)).astype(np.float32)
    signal = np.clip(counts / p.flux, 0.0, 1.0).astype(np.float32)

    sources = (lum >= p.bloom_gate).astype(np.float32)
    return PhotonField(signal=signal, sources=sources)
