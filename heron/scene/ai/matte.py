"""Subject matting via ViTMatte (CLAUDE.md §5).

ViTMatte refines a trimap into a high-quality alpha. We derive the trimap from
the classical saliency matte (eroded core = foreground, dilated boundary =
unknown), then let the model resolve hair/edge detail — replacing the leaky
classical matte that lets background warm patches through. Degrades to None on
any error (caller keeps the classical matte).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from heron.color import srgb
from heron.core.device import get_device
from heron.scene.ai import registry

_HF_MODEL = "hustvl/vitmatte-base-composition-1k"


@lru_cache(maxsize=1)
def _load():
    import torch  # noqa: F401
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    source = registry.resolve("matte") or _HF_MODEL
    processor = VitMatteImageProcessor.from_pretrained(source, local_files_only=True)
    model = VitMatteForImageMatting.from_pretrained(source, local_files_only=True)
    model.to(get_device()).eval()
    return processor, model


def _trimap(matte: np.ndarray) -> np.ndarray:
    """Build a 3-level trimap (0 bg / 128 unknown / 255 fg) from a soft matte."""
    m = (np.asarray(matte, dtype=np.float32) > 0.5).astype(np.uint8) * 255
    h, w = m.shape
    k = max(3, (min(h, w) // 60) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.erode(m, kernel, iterations=1)
    bg = cv2.dilate(m, kernel, iterations=2)
    trimap = np.full_like(m, 128)
    trimap[bg == 0] = 0
    trimap[fg == 255] = 255
    return trimap


def _is_sane(alpha: np.ndarray, seed: np.ndarray) -> bool:
    """Reject a matte that disagrees with its own trimap seed.

    ViTMatte can return noise (e.g. on an unexpected input size), and a garbage
    matte silently poisons every downstream channel — it multiplies almost every
    term in the physics. Verify it is actually foreground where the seed was
    confidently foreground, and background where the seed was confidently
    background, before trusting it.
    """
    fg = seed > 0.9
    bg = seed < 0.1
    if fg.sum() < 64 or bg.sum() < 64:
        return False
    fg_mean, bg_mean = float(alpha[fg].mean()), float(alpha[bg].mean())
    if fg_mean < 0.55 or bg_mean > 0.45 or (fg_mean - bg_mean) < 0.25:
        return False
    # a matte that is mostly high-frequency noise is not a matte
    row_jitter = float(np.diff(alpha.mean(axis=1)).std())
    return row_jitter < 0.15


def estimate_matte(linear_rgb: np.ndarray, fallback: np.ndarray) -> np.ndarray | None:
    """Refine ``fallback`` into a soft alpha with ViTMatte, or None if unusable."""
    try:
        import torch
        from PIL import Image

        processor, model = _load()
        srgb8 = np.clip(srgb.linear_to_srgb(linear_rgb) * 255, 0, 255).astype(np.uint8)
        trimap = _trimap(fallback)
        inputs = processor(
            images=Image.fromarray(srgb8),
            trimaps=Image.fromarray(trimap),
            return_tensors="pt",
        ).to(get_device())
        with torch.no_grad():
            alpha = model(**inputs).alphas.squeeze().detach().float().cpu().numpy()
        h, w = linear_rgb.shape[:2]
        alpha = cv2.resize(np.asarray(alpha, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        if not _is_sane(alpha, np.asarray(fallback, dtype=np.float32)):
            print("[heron.ai.matte] ViTMatte output failed sanity check — keeping the seed mask")
            return None
        return alpha
    except Exception as e:
        print(f"[heron.ai.matte] unavailable, using seed mask: {e}")
        return None
