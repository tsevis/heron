"""Three-dimensional form from depth (CLAUDE.md §3.1, §3.2).

Depth-Anything resolves real facial relief — nose projecting, eye sockets
recessed, jaw curving away — but the raw depth is normalized over the *whole
scene*, so a subject standing well in front of its background occupies only a
sliver of the [0,1] range. Differentiating that directly yields almost-flat
normals and a figure that reads as a flat plate.

This module re-normalizes depth *within the subject* to recover the relief, then
turns it into the two cues that give a thermogram genuine volume:

* **Grazing-angle emissivity falloff** — real emissivity drops sharply as a
  surface turns away from the sensor, so the rim of a cylinder (a cheek, an arm)
  reads distinctly cooler than its centre. This is the dominant 3-D cue in real
  thermography.
* **Ambient occlusion** — recessed geometry (eye sockets, nostrils, under the
  chin, the ear's folds, the gap between touching bodies) traps heat and shields
  it from convective cooling, so crevices read *warmer*.
"""

from __future__ import annotations

import cv2
import numpy as np


def subject_relief(depth: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """Re-normalize depth inside the subject so its own relief spans [0,1]."""
    body = matte > 0.3
    d = np.asarray(depth, dtype=np.float32)
    if not body.any():
        return d
    lo = float(np.percentile(d[body], 2.0))
    hi = float(np.percentile(d[body], 98.0))
    relief = np.clip((d - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
    return relief.astype(np.float32)


def surface_normals(relief: np.ndarray, strength: float = 60.0) -> np.ndarray:
    """Camera-space unit normals from a relief map (larger value = nearer)."""
    r = cv2.GaussianBlur(np.asarray(relief, dtype=np.float32), (0, 0), sigmaX=1.2)
    dzdx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3) * strength
    dzdy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3) * strength
    n = np.stack([-dzdx, -dzdy, np.ones_like(r)], axis=-1)
    return (n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)).astype(np.float32)


def grazing_falloff(normals: np.ndarray, power: float = 1.6) -> np.ndarray:
    """0 facing the sensor -> 1 at grazing incidence (emissivity roll-off)."""
    cos_theta = np.clip(normals[..., 2], 0.0, 1.0)
    return (1.0 - np.power(cos_theta, power)).astype(np.float32)


def ambient_occlusion(relief: np.ndarray, radius_frac: float = 0.02) -> np.ndarray:
    """Occlusion in [0,1]: high where geometry is recessed relative to its
    neighbourhood (eye sockets, nostrils, under the chin, contact gaps)."""
    r = np.asarray(relief, dtype=np.float32)
    sigma = max(2.0, max(r.shape) * radius_frac)
    local = cv2.GaussianBlur(r, (0, 0), sigmaX=sigma)
    recess = np.clip(local - r, 0.0, None)          # nearer neighbours => recessed
    hi = float(np.percentile(recess, 99.0))
    return np.clip(recess / max(hi, 1e-5), 0.0, 1.0).astype(np.float32)


def form_terms(depth: np.ndarray, matte: np.ndarray):
    """Convenience: return (relief, normals, grazing, occlusion)."""
    relief = subject_relief(depth, matte)
    normals = surface_normals(relief)
    return relief, normals, grazing_falloff(normals), ambient_occlusion(relief)
