"""Open-vocabulary segmentation via SAM 3 (CLAUDE.md §5, §3.1).

SAM 3 ("Segment Anything with Concepts") segments all instances of a short text
concept — the F-ViTA-style tagger+SAM pipeline this project's Layer A is built on.
Heron uses it to localize scene materials (water, metal, glass, foliage, …) so
each region gets its true LWIR behavior — the reflective mirror-water/metal case
that gradient-map filters cannot do.

Loaded read-only from the local repo (`mozaix/models/sam3`) with the local
checkpoint (no gated HF download, §2.1). Runs on **CPU**: SAM 3's decoder builds
some tensors on CPU internally, which trips MPS device checks; since this is a
once-per-image, cached Layer-A step, CPU is the robust choice. Degrades to None
so materials fall back to the matte-only assignment.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from heron.color import srgb

def _sam3_dir() -> Path | None:
    """Where SAM 3 is installed, via the registry (env / repo models/ / legacy)."""
    from heron.scene.ai import registry

    resolved = registry.resolve("segment")
    # A bare hub id means "in the HF cache", which this loader cannot use: it
    # imports SAM 3's own python package by path, not through transformers.
    if resolved is None or "/" not in str(resolved) or not Path(resolved).exists():
        return None
    return Path(resolved)

_SCORE_THRESHOLD = 0.4


@lru_cache(maxsize=1)
def _load_processor():
    """Build the SAM 3 image model + processor on CPU (cached for the session)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sam3_dir = _sam3_dir()
    if sam3_dir is None:
        raise FileNotFoundError(
            "SAM 3 is not installed. Materials fall back to a single class. "
            "Run `python scripts/fetch_models.py`, or point HERON_MODELS at an "
            "existing install."
        )
    parent = str(sam3_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)  # read-only import, no install
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=str(sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        device="cpu",
        checkpoint_path=str(sam3_dir / "sam3.pt"),
        load_from_HF=False,
    )
    return Sam3Processor(model, device="cpu", confidence_threshold=_SCORE_THRESHOLD)


def _to_binary_union(masks, scores, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Union SAM 3 instance masks (above threshold) into one (H,W) bool mask."""
    if masks is None:
        return None
    m = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    if m.size == 0:
        return None
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 4:  # (N,1,H,W)
        m = m[:, 0]
    if m.ndim == 2:  # single mask
        m = m[None]
    sc = None
    if scores is not None:
        sc = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)

    binary = m > 0.0 if m.min() < 0.0 else m > 0.5  # logits vs probabilities
    keep = np.ones(len(binary), bool) if sc is None else (sc >= _SCORE_THRESHOLD)
    if not keep.any():
        return None
    union = np.any(binary[keep], axis=0)
    h, w = target_hw
    if union.shape != (h, w):
        union = cv2.resize(union.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return union


def segment_concepts(
    linear_rgb: np.ndarray, prompts: list[str]
) -> dict[str, np.ndarray] | None:
    """Return {prompt: (H,W) bool mask} for each concept SAM 3 finds, or None."""
    try:
        from PIL import Image

        processor = _load_processor()
        h, w = linear_rgb.shape[:2]
        srgb8 = np.clip(srgb.linear_to_srgb(linear_rgb) * 255, 0, 255).astype(np.uint8)
        state = processor.set_image(Image.fromarray(srgb8))

        out: dict[str, np.ndarray] = {}
        for prompt in prompts:
            res = processor.set_text_prompt(prompt=prompt, state=state)
            mask = _to_binary_union(res.get("masks"), res.get("scores"), (h, w))
            if mask is not None and mask.any():
                out[prompt] = mask
        return out
    except Exception as e:
        print(f"[heron.ai.segment] SAM3 unavailable, using matte-only materials: {e}")
        return None
