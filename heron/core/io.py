"""Image IO — the sRGB<->linear boundary of the engine.

CLAUDE.md §2.3: sRGB decode on input, encode on output; internal buffers are
float32 linear-light [0,1]. Loading is via OpenCV so 16-bit PNG/TIFF keep
their depth (Pillow clips 16-bit grey and cannot hold 16-bit RGB), with Pillow
as the fallback for anything OpenCV cannot read. Writing is via OpenCV.
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
    codes = _decode_with_opencv(path)
    if codes is None:
        with Image.open(path) as im:
            codes = np.asarray(im.convert("RGBA"))
    rgb_codes, alpha_codes = _split_channels(codes)

    rgb_linear = srgb.decode_uint(rgb_codes)
    if alpha_codes is None:
        return rgb_linear, None
    top = np.iinfo(alpha_codes.dtype).max
    if int(alpha_codes.min()) == top:
        return rgb_linear, None
    return rgb_linear, (alpha_codes.astype(np.float32) / float(top)).astype(np.float32)


def _decode_with_opencv(path: str | Path) -> np.ndarray | None:
    """Decode at full bit depth (Pillow cannot hold 16-bit RGB), RGB(A) order.

    Returns uint8/uint16 codes, or None for anything OpenCV cannot read or
    stores in another sample type; the caller then falls back to Pillow.
    """
    try:
        arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except (cv2.error, OSError, ValueError):
        return None
    if arr is None or arr.dtype not in (np.uint8, np.uint16):
        return None
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    grey = arr[..., 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr
    # 8-bit grey goes to Pillow, which honours a PNG transparency key; OpenCV
    # drops it. 16-bit grey stays here: Pillow clips it to black and white.
    return grey if grey.ndim == 2 and grey.dtype == np.uint16 else None


def _split_channels(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(H,W) grey, (H,W,3) RGB or (H,W,4) RGBA codes -> (RGB codes, alpha or None)."""
    if codes.ndim == 2:
        return np.repeat(codes[..., None], 3, axis=2), None
    if codes.shape[2] == 4:
        return codes[..., :3], codes[..., 3]
    return codes, None


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
