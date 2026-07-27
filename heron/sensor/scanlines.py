"""Analog scanlines & tape degradation (CLAUDE.md §1.2, §3.3).

Horizontal scanline modulation for the CRT/VHS-surveillance look, with optional
per-row horizontal jitter for analog-tape instability. Deterministic per seed.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.core.seeds import rng_for


def scanlines(
    signal: np.ndarray,
    period_px: float = 3.0,
    strength: float = 0.25,
    jitter: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Darken alternating scanlines; optional per-row horizontal jitter."""
    s = np.asarray(signal, dtype=np.float32)
    h, w = s.shape

    if jitter > 0.0:
        rng = rng_for(seed, "scanline_jitter", h)
        shifts = (rng.standard_normal(h) * jitter * w * 0.01).astype(np.float32)
        cols = np.arange(w, dtype=np.float32)[None, :] + shifts[:, None]
        map_x = np.clip(cols, 0, w - 1).astype(np.float32)
        map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
        s = cv2.remap(s, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    if strength > 0.0:
        rows = np.arange(h, dtype=np.float32)
        line = 0.5 + 0.5 * np.cos(2.0 * np.pi * rows / max(period_px, 2.0))
        modulation = (1.0 - strength) + strength * line
        s = s * modulation[:, None]

    return np.clip(s, 0.0, 1.0).astype(np.float32)
