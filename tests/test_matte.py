"""Classical matte cost.

GrabCut was run at the full working resolution (1280 px) and took ~17 s of a
~21 s classical render. The matte is a smooth, semantic field, so the colour
model is fitted on a reduced copy and the mask is lifted back edge-aware. This
pins that down: a regression to full-resolution GrabCut is invisible in the
output but costs every classical render and every AI render that falls back.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.scene import matte


def _portrait(h: int = 1280, w: int = 1000) -> np.ndarray:
    """A bright disc (the subject) on a darker, textured background."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disc = ((yy - h * 0.45) ** 2 + (xx - w * 0.5) ** 2) < (min(h, w) * 0.3) ** 2
    img = 0.05 + 0.03 * rng.random((h, w, 3), dtype=np.float32)
    img[disc] = (0.6, 0.35, 0.25)
    return img.astype(np.float32)


def test_grabcut_never_runs_at_full_resolution(monkeypatch):
    seen: list[tuple[int, int]] = []
    real = cv2.grabCut

    def spy(img, *args, **kwargs):
        seen.append(img.shape[:2])
        return real(img, *args, **kwargs)

    monkeypatch.setattr(matte.cv2, "grabCut", spy)
    out = matte.classical_matte(_portrait())

    assert seen, "GrabCut did not run on a scene with a clear subject"
    assert max(max(s) for s in seen) <= matte.GRABCUT_LONG_SIDE
    assert out.shape == (1280, 1000)
