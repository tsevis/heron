"""AGC — plateau histogram equalization (CLAUDE.md §0.5, §3.3).

Real thermal cameras flatten scene contrast with plateau equalization (PE): the
histogram is clipped to a plateau before building the transfer CDF, so a few
dominant temperatures cannot wash out the rest. ``strength`` blends from the
linear signal (0) to full PE (1); ``linearity`` blends the PE curve back toward
identity for a softer, less "processed" tonality.
"""

from __future__ import annotations

import numpy as np

_BINS = 1024
_SMOOTH_BINS = 9


def plateau_equalize(
    signal: np.ndarray,
    strength: float = 0.7,
    plateau: float = 0.02,
    max_gain: float = 4.0,
) -> np.ndarray:
    """Plateau-equalize a [0,1] signal. ``plateau`` is the per-bin cap as a
    fraction of total pixels; ``max_gain`` limits local contrast amplification.

    Two safeguards matter, and real AGC implementations have both. A strongly
    bimodal scene (clean subject vs. cold background) concentrates the histogram
    into a few bins, which turns the transfer CDF into a staircase: values inside
    one dense bin get spread across a wide output range, so a few millikelvin of
    fixed-pattern row noise explodes into visible banding. We therefore
    (a) **smooth** the transfer curve and (b) **limit its slope** (gain), which
    is what keeps real plateau-EQ from contouring flat scenes.
    """
    s = np.clip(np.asarray(signal, dtype=np.float32), 0.0, 1.0)
    if strength <= 0.0:
        return s

    hist, edges = np.histogram(s, bins=_BINS, range=(0.0, 1.0))
    total = s.size
    cap = max(1.0, plateau * total)
    clipped = np.minimum(hist.astype(np.float64), cap)
    if clipped.sum() <= 0:
        return s

    # smooth the (clipped) density so the transfer curve has no hard steps
    kernel = np.ones(_SMOOTH_BINS, dtype=np.float64) / _SMOOTH_BINS
    density = np.convolve(clipped, kernel, mode="same")

    # limit local gain: no bin may be amplified more than ``max_gain`` x linear
    mean_density = density.sum() / _BINS
    density = np.minimum(density, max_gain * mean_density)
    if density.sum() <= 0:
        return s

    cdf = np.cumsum(density)
    cdf /= cdf[-1]

    centers = (edges[:-1] + edges[1:]) * 0.5
    mapped = np.interp(s, centers, cdf).astype(np.float32)
    return (s * (1.0 - strength) + mapped * strength).astype(np.float32)
