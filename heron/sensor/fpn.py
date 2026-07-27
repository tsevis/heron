"""Fixed-pattern noise (FPN) + NETD temporal noise (CLAUDE.md §0.5, §3.3, §2.4).

FPN is the dominant noise of staring microbolometer arrays: a *static* per-pixel
gain/offset field with strong row/column-correlated striping from the serial
readout. It is fixed-pattern — generated once per session/sequence from a seed
and reused across frames, never re-rolled (§2.4). NETD noise is the per-frame
temporal Gaussian whose sigma is set by a millikelvin slider.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heron.core.seeds import rng_for


@dataclass(frozen=True)
class FPNField:
    gain: np.ndarray    # (H,W) multiplicative, ~1.0
    offset: np.ndarray  # (H,W) additive, in signal units


def make_fpn(
    shape: tuple[int, int],
    seed: int,
    amplitude: float = 0.03,
    row_weight: float = 0.5,
    col_weight: float = 0.25,
) -> FPNField:
    """Generate a static FPN field. Same (shape, seed, params) -> identical field.

    ``amplitude`` is the offset std in signal units; row/col weights control how
    much of the noise is line-correlated (the readout stripe signature).
    """
    h, w = shape
    rng = rng_for(seed, "fpn", h, w)

    pixel = rng.standard_normal((h, w)).astype(np.float32)
    rows = rng.standard_normal((h, 1)).astype(np.float32)  # per-row (horizontal streaks)
    cols = rng.standard_normal((1, w)).astype(np.float32)  # per-column (vertical stripes)

    pix_weight = max(0.0, 1.0 - row_weight - col_weight)
    offset = amplitude * (
        pix_weight * pixel + row_weight * np.broadcast_to(rows, (h, w)) + col_weight * np.broadcast_to(cols, (h, w))
    )
    gain = (1.0 + 0.5 * amplitude * rng.standard_normal((h, w))).astype(np.float32)
    return FPNField(gain=gain.astype(np.float32), offset=offset.astype(np.float32))


def apply_fpn(signal: np.ndarray, field: FPNField, drift: float = 0.0) -> np.ndarray:
    """Apply gain/offset FPN. ``drift`` (in [-1,1]) scales the offset to emulate
    slow camera-temperature drift without changing the fixed pattern."""
    s = np.asarray(signal, dtype=np.float32)
    return np.clip(s * field.gain + field.offset * (1.0 + drift), 0.0, 1.0).astype(np.float32)


def add_netd(
    signal: np.ndarray,
    netd_mk: float,
    span_c: tuple[float, float],
    seed: int,
    frame: int = 0,
) -> np.ndarray:
    """Add per-frame temporal Gaussian noise sized by NETD (millikelvin).

    Sigma in signal units = (NETD in K) / (window span in K). ``frame`` varies
    the temporal noise per frame while leaving the FPN field untouched.
    """
    span_k = max(span_c[1] - span_c[0], 1e-3)
    sigma = (netd_mk / 1000.0) / span_k
    if sigma <= 0:
        return np.clip(signal, 0.0, 1.0).astype(np.float32)
    rng = rng_for(seed, "netd", frame)
    noise = rng.standard_normal(np.shape(signal)).astype(np.float32) * sigma
    return np.clip(np.asarray(signal, dtype=np.float32) + noise, 0.0, 1.0).astype(np.float32)
