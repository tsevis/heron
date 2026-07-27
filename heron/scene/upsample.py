"""Edge-aware upsampling of Radiance Graph channels (Phase 4 §4).

Layer A's models do not scale to 12+ MP, but their output does not need to: depth,
matte, materials and saliency are smooth, semantic fields. So Layer A keeps
running at a bounded working resolution and its channels are lifted to the render
resolution here, guided by the full-resolution source, instead of re-running the
models. That is the cheap 90% of full-resolution rendering.

Bilinear upsampling would soften every silhouette and let the matte bleed across
edges — which is exactly what the physics keys on. This uses the guided filter
(He, Sun & Tang 2013) in its joint-upsampling form: fit the local linear model
``q = a*I + b`` at low resolution, upsample the *coefficients*, then apply them
against the full-resolution guide. Edges therefore come from the real image, not
from interpolation.

Implemented with box filters over ``cv2`` rather than ``cv2.ximgproc`` so it needs
only the base OpenCV the project already depends on. Every operation is
deterministic (CLAUDE.md §2.4).
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from heron.core.graph import RadianceGraph

# Radius as a fraction of the low-res long side: the local linear model needs a
# window large enough to see both sides of an edge at low resolution.
_RADIUS_FRAC = 0.02
_EPS = 1e-4


def _box(img: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * int(radius) + 1
    return cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Luminance guide in [0,1] float32."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def guided_upsample(low: np.ndarray, guide_full: np.ndarray,
                    radius_frac: float = _RADIUS_FRAC, eps: float = _EPS) -> np.ndarray:
    """Lift a single-channel low-res field to the guide's resolution, edge-aware.

    ``low`` is (h,w) float; ``guide_full`` is the full-resolution source, either
    (H,W) or (H,W,3). Returns (H,W) float32.
    """
    low = np.asarray(low, dtype=np.float32)
    guide = _to_gray(guide_full)
    target = (guide.shape[1], guide.shape[0])
    if low.shape[:2] == guide.shape[:2]:
        return low.copy()

    guide_low = cv2.resize(guide, (low.shape[1], low.shape[0]), interpolation=cv2.INTER_AREA)
    radius = max(2, int(round(radius_frac * max(low.shape[:2]))))

    mean_i = _box(guide_low, radius)
    mean_p = _box(low, radius)
    corr_i = _box(guide_low * guide_low, radius)
    corr_ip = _box(guide_low * low, radius)
    var_i = np.maximum(corr_i - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    # Upsampling the coefficients, not the field, is what preserves the edges.
    a_full = cv2.resize(a, target, interpolation=cv2.INTER_LINEAR)
    b_full = cv2.resize(b, target, interpolation=cv2.INTER_LINEAR)
    return (a_full * guide + b_full).astype(np.float32)


def _upsample_vector(low: np.ndarray, guide_full: np.ndarray) -> np.ndarray:
    """Per-channel guided upsample of an (h,w,C) field."""
    channels = [guided_upsample(low[..., c], guide_full) for c in range(low.shape[2])]
    return np.stack(channels, axis=-1).astype(np.float32)


def upsample_graph(graph: RadianceGraph, source_full: np.ndarray) -> RadianceGraph:
    """Return the graph at ``source_full``'s resolution.

    Continuous channels are guided-upsampled. ``material_ids`` is a label map, so
    it is resized nearest-neighbour — interpolating label indices would invent
    materials that are not in the table (an id halfway between skin and metal is
    meaningless). ``extras`` are resized the same way as their kind.
    """
    h, w = source_full.shape[:2]
    if graph.shape == (h, w):
        return graph

    target = (w, h)
    normals = _upsample_vector(np.asarray(graph.normals, np.float32), source_full)
    # Guided filtering breaks unit length; renormalise so shading stays correct.
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = (normals / np.maximum(norm, 1e-6)).astype(np.float32)

    extras = {}
    for name, value in graph.extras.items():
        array = np.asarray(value)
        if array.ndim >= 2 and array.shape[:2] == graph.shape:
            if array.dtype.kind in "iub":
                extras[name] = cv2.resize(array, target, interpolation=cv2.INTER_NEAREST)
            else:
                extras[name] = np.clip(guided_upsample(array.astype(np.float32), source_full), 0.0, 1.0)
        else:
            extras[name] = value      # not a spatial channel (e.g. skeleton bones)

    return replace(
        graph,
        depth=np.clip(guided_upsample(graph.depth, source_full), 0.0, 1.0),
        normals=normals,
        albedo=np.clip(_upsample_vector(np.asarray(graph.albedo, np.float32), source_full), 0.0, 1.0),
        saliency=np.clip(guided_upsample(graph.saliency, source_full), 0.0, 1.0),
        matte=np.clip(guided_upsample(graph.matte, source_full), 0.0, 1.0),
        material_ids=cv2.resize(graph.material_ids, target, interpolation=cv2.INTER_NEAREST),
        meta={**graph.meta, "upsampled_from": list(graph.shape)},
        extras=extras,
    )
