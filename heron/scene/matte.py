"""Classical subject matte (CLAUDE.md §2.6 fallback: manual/threshold masks).

With no matting/segmentation model in play, estimate subject coverage from
saliency, then snap it to the real object region with GrabCut (a color model
fills interiors that saliency — which fires on edges — misses) and fill any
enclosed holes. Result is a soft alpha in [0,1]. AI mode replaces this with
ViTMatte/SAM; the downstream contract (a soft (H,W) matte) is identical.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from heron.color import srgb
from heron.scene import saliency as _saliency


def _grabcut_refine(linear_rgb: np.ndarray, weighted: np.ndarray) -> np.ndarray | None:
    """Snap the saliency estimate to the object region with GrabCut.

    Returns a binary foreground mask, or None if GrabCut cannot seed reliably.
    """
    h, w = weighted.shape
    srgb8 = np.clip(srgb.linear_to_srgb(linear_rgb) * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(srgb8, cv2.COLOR_RGB2BGR)

    hi = float(np.percentile(weighted, 85))
    lo = float(np.percentile(weighted, 40))
    sure_fg = weighted >= hi
    sure_bg = weighted <= lo * 0.5
    # need both foreground and background seeds for a color model
    if sure_fg.sum() < 0.002 * h * w or sure_bg.sum() < 0.05 * h * w:
        return None

    gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc[weighted >= lo] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD
    gc[sure_bg] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)  # GrabCut's GMM k-means uses OpenCV's global RNG (§2.4 determinism)
    try:
        cv2.grabCut(bgr, gc, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    return np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD))


def classical_matte(linear_rgb: np.ndarray, sal: np.ndarray | None = None) -> np.ndarray:
    """Return a soft subject matte (H,W) in [0,1] from a linear-RGB image."""
    linear_rgb = np.asarray(linear_rgb, dtype=np.float32)
    if sal is None:
        sal = _saliency.spectral_residual(linear_rgb)
    h, w = sal.shape
    weighted = sal * _saliency.center_bias((h, w), strength=0.5)
    weighted = (weighted - weighted.min()) / max(weighted.max() - weighted.min(), 1e-8)

    fg = _grabcut_refine(linear_rgb, weighted)
    if fg is None:
        u8 = np.clip(weighted * 255, 0, 255).astype(np.uint8)
        _, core = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = core > 0

    fg = ndimage.binary_fill_holes(fg)
    k = max(3, (min(h, w) // 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    feather = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), sigmaX=max(h, w) * 0.006)
    # keep a little continuous saliency at the edges for soft falloff
    matte = np.clip(0.8 * feather + 0.2 * weighted * feather, 0.0, 1.0)
    return matte.astype(np.float32)
