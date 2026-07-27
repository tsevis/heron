"""Monocular depth via Depth-Anything-V2 (CLAUDE.md §5).

Loaded through transformers from the local model (or the HF cache entry listed in
the registry). Returns depth normalized to [0,1] with larger = nearer, matching
the Radiance Graph convention. Degrades to None (classical proxy) on any error.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from heron.color import srgb
from heron.core.device import get_device
from heron.scene.ai import registry

# Registry (§5): the mozaix Depth-Anything-V2 dir is the ORIGINAL repo (a .pth +
# python package, not HF format), so we load the HF-cached Large-hf checkpoint.
# Both tiers use Large (Fast already gains speed from the engine's work_res
# downscale); the Small-hf variant is NOT cached, and we must not download it
# (§2.1). local_files_only enforces cache-only.
_HF_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


@lru_cache(maxsize=2)
def _load(tier: str):
    import torch  # noqa: F401
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    # registry order: HERON_MODELS -> repo models/ -> legacy paths -> HF cache.
    # None means nothing is installed; local_files_only makes that a clear error
    # rather than a silent download mid-render.
    source = registry.resolve("depth") or _HF_MODEL
    processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(source, local_files_only=True)
    model.to(get_device()).eval()
    return processor, model


def estimate_depth(linear_rgb: np.ndarray, tier: str = "fast") -> np.ndarray | None:
    try:
        import torch
        from PIL import Image

        processor, model = _load(tier)
        h, w = linear_rgb.shape[:2]
        srgb8 = np.clip(srgb.linear_to_srgb(linear_rgb) * 255, 0, 255).astype(np.uint8)
        inputs = processor(images=Image.fromarray(srgb8), return_tensors="pt").to(get_device())
        with torch.no_grad():
            pred = model(**inputs).predicted_depth
        depth = pred.squeeze().detach().float().cpu().numpy()
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = float(np.percentile(depth, 1)), float(np.percentile(depth, 99))
        depth = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return depth.astype(np.float32)  # DA-V2: larger value = nearer
    except Exception as e:
        print(f"[heron.ai.depth] unavailable, using classical proxy: {e}")
        return None
