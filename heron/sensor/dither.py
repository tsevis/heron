"""Blue-noise dithering (CLAUDE.md §0.5, §3.3).

Palette output is a smooth 1-D ramp and bands badly at 8-bit. A small amount of
high-frequency (blue-spectrum) noise before quantization breaks the banding
without visible grain. We approximate blue noise by high-pass-filtering white
noise (white minus a blurred copy), which is cheap and spectrally close enough.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.core.seeds import rng_for


def blue_noise_dither(
    rgb_linear: np.ndarray,
    seed: int,
    bit_depth: int = 8,
    amplitude_lsb: float = 1.0,
) -> np.ndarray:
    """Add ~``amplitude_lsb`` LSBs of blue noise to a linear-RGB image."""
    if amplitude_lsb <= 0.0:
        return np.asarray(rgb_linear, dtype=np.float32)
    rgb = np.asarray(rgb_linear, dtype=np.float32)
    h, w = rgb.shape[:2]
    rng = rng_for(seed, "dither", h, w)
    white = rng.standard_normal((h, w)).astype(np.float32)
    blue = white - cv2.GaussianBlur(white, (0, 0), sigmaX=1.0)
    blue /= max(blue.std(), 1e-6)
    amp = amplitude_lsb / float((1 << bit_depth) - 1)
    return np.clip(rgb + amp * blue[..., None], 0.0, 1.0).astype(np.float32)
