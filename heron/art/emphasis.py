"""Saliency-driven emphasis (CLAUDE.md §3.4, §1.6).

Makes faces and touching hands the luminous centers of the frame: the saliency
map locally raises apparent emission and tightens palette contrast, in the
[0,1] signal domain (applied before palette mapping).
"""

from __future__ import annotations

import cv2
import numpy as np


def saliency_emphasis(
    signal: np.ndarray,
    saliency: np.ndarray,
    amount: float = 0.25,
    contrast: float = 0.2,
) -> np.ndarray:
    """Locally lift and contrast-tighten the signal where saliency is high."""
    if amount <= 0.0 and contrast <= 0.0:
        return np.asarray(signal, dtype=np.float32)
    s = np.clip(np.asarray(signal, dtype=np.float32), 0.0, 1.0)
    w = np.asarray(saliency, dtype=np.float32)
    w = cv2.GaussianBlur(w, (0, 0), sigmaX=max(s.shape) * 0.01)
    w = (w - w.min()) / max(w.max() - w.min(), 1e-6)

    lifted = s + amount * w * (1.0 - s)          # warm salient regions
    # local contrast tighten around the salient region's mean
    mean = float((s * w).sum() / max(w.sum(), 1e-6))
    tightened = mean + (lifted - mean) * (1.0 + contrast * w)
    return np.clip(tightened, 0.0, 1.0).astype(np.float32)
