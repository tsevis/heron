"""Image IO — the sRGB<->linear boundary of the engine.

CLAUDE.md §2.3: sRGB decode on input, encode on output; internal buffers are
float32 linear-light [0,1]. Loading is via Pillow (broad format support);
writing is via OpenCV so we get real 16-bit PNG/TIFF output.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from heron.color import srgb

# Pillow can open enormous artwork scans; lift the decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None


def read_image(path: str | Path) -> np.ndarray:
    """Load an image and return linear-light RGB float32 (H,W,3) in [0,1].

    Alpha is dropped here; use :func:`read_image_rgba` when the matte matters.
    """
    rgb, _ = read_image_rgba(path)
    return rgb


def read_image_rgba(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an image, returning (linear_rgb (H,W,3), alpha (H,W) or None).

    Alpha is a plain coverage channel — not light — so it is *not* linearized.
    """
    with Image.open(path) as im:
        im = im.convert("RGBA")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    rgb_srgb, alpha = arr[..., :3], arr[..., 3]
    rgb_linear = srgb.srgb_to_linear(rgb_srgb)
    has_alpha = float(alpha.min()) < 1.0
    return rgb_linear, (alpha.astype(np.float32) if has_alpha else None)


def write_image(path: str | Path, linear_rgb: np.ndarray, bit_depth: int = 8) -> Path:
    """Encode a linear-light RGB float32 image to sRGB and write to ``path``.

    ``bit_depth`` is 8 or 16. Output format follows the file extension.
    """
    if bit_depth not in (8, 16):
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")
    linear_rgb = np.asarray(linear_rgb, dtype=np.float32)
    if linear_rgb.ndim != 3 or linear_rgb.shape[2] != 3:
        raise ValueError(f"expected (H,W,3), got {linear_rgb.shape}")

    encoded = srgb.linear_to_srgb(linear_rgb)
    maxval = (1 << bit_depth) - 1
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    quantized = np.clip(encoded * maxval + 0.5, 0, maxval).astype(dtype)

    bgr = cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"failed to write image: {path}")
    return path


def write_gray(path: str | Path, gray: np.ndarray, bit_depth: int = 8) -> Path:
    """Write a single-channel float [0,1] map (e.g. a Radiance Graph channel).

    Written as a raw perceptual/data channel — *not* sRGB-encoded — so it can be
    hand-edited and re-imported without gamma surprises (CLAUDE.md §2.7).
    """
    gray = np.asarray(gray, dtype=np.float32)
    maxval = (1 << bit_depth) - 1
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    quantized = np.clip(gray * maxval + 0.5, 0, maxval).astype(dtype)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), quantized):
        raise IOError(f"failed to write image: {path}")
    return path
