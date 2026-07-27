"""Canonical facial & body heat map (CLAUDE.md §0.4, §3.2, §7).

**This is the feature that makes a thermogram read as a thermogram.** A real face
is not a uniform warm blob: it is hottest at the inner-eye corners (tear ducts),
ears and neck, and markedly *cold* at the nose tip — a signature viewers
recognize even without knowing why. The material table's ``skin.zones`` encodes
those offsets in Celsius; this module warps that canonical layout onto the actual
face and adds a core->extremity perfusion falloff over the body.

Face region comes from SAM 3's ``face`` concept (already part of Layer A — no
extra model or download). The canonical zones are placed in normalized face
coordinates and rotated to the detected face's orientation. With no face found,
only the body-structure term applies, so the engine still works (§2.6).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

# Canonical facial layout in normalized face-ellipse coordinates:
# u = across the face (-1 left .. +1 right), v = down (-1 crown .. +1 chin).
# (u, v, radius) per landmark; several landmarks per zone.
_CANONICAL: dict[str, tuple[tuple[float, float, float], ...]] = {
    "forehead": ((0.0, -0.52, 0.42), (-0.38, -0.45, 0.30), (0.38, -0.45, 0.30)),
    "tearduct": ((-0.20, -0.10, 0.16), (0.20, -0.10, 0.16)),   # hottest points
    "ear": ((-0.92, -0.02, 0.28), (0.92, -0.02, 0.28)),
    "cheek": ((-0.55, 0.18, 0.34), (0.55, 0.18, 0.34)),
    "nose": ((0.0, 0.20, 0.20),),                              # the cold spot
    "lips": ((0.0, 0.54, 0.22),),
    "neck": ((0.0, 1.15, 0.45),),
}


def _mask_axes(mask: np.ndarray):
    """Principal axes of a mask via PCA -> (cx, cy, a_across, a_down, angle_rad).

    PCA is used rather than ``cv2.fitEllipse`` because the latter's axis order and
    angle convention are ambiguous about which axis is the vertical (crown->chin)
    one — which silently mis-places every facial zone. The face's long axis is
    crown->chin, so the major eigenvector defines +v (down); the minor defines u.
    """
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) < 64:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    coords = np.stack([xs - cx, ys - cy]).astype(np.float64)
    cov = np.cov(coords)
    evals, evecs = np.linalg.eigh(cov)          # ascending eigenvalues
    minor_v, major_v = evecs[:, 0], evecs[:, 1]
    a_across = 2.0 * float(np.sqrt(max(evals[0], 1e-6)))
    a_down = 2.0 * float(np.sqrt(max(evals[1], 1e-6)))
    # orient +v downward in image space (faces are near-upright in photographs)
    if major_v[1] < 0:
        major_v = -major_v
    angle = float(np.arctan2(major_v[1], major_v[0]))   # direction of +v
    return cx, cy, a_across, a_down, angle


def face_region(
    linear_rgb: np.ndarray, matte: np.ndarray | None = None, allow_ai: bool = True
) -> np.ndarray | None:
    """Locate the face: SAM 3's ``face`` concept, else the top of the matte.

    ``allow_ai`` must be False on the classical path — importing the SAM 3 wrapper
    pulls in torch, which would break "Layer A is always skippable" (§2.6).
    """
    if allow_ai:
        try:
            from heron.scene.ai import segment as _seg

            masks = _seg.segment_concepts(linear_rgb, ["face"])
            if masks and "face" in masks and masks["face"].any():
                return masks["face"].astype(np.float32)
        except Exception:
            pass

    # classical fallback: the upper part of the subject silhouette
    if matte is not None and (matte > 0.5).any():
        m = (matte > 0.5).astype(np.uint8)
        ys = np.where(m.any(axis=1))[0]
        if len(ys) > 8:
            top, bottom = ys[0], ys[-1]
            cut = int(top + 0.55 * (bottom - top))
            head = np.zeros_like(m)
            head[top:cut] = m[top:cut]
            if head.any():
                return head.astype(np.float32)
    return None


def facial_heat_offsets(
    linear_rgb: np.ndarray,
    zones: dict[str, float],
    matte: np.ndarray | None = None,
    region: np.ndarray | None = None,
    allow_ai: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (offset_c, face_mask): ``zones`` warped onto the detected face.

    ``offset_c`` is an additive temperature field in Celsius. None if no face.
    """
    if region is None:
        region = face_region(linear_rgb, matte, allow_ai=allow_ai)
    if region is None:
        return None
    fit = _mask_axes(region)
    if fit is None:
        return None
    cx, cy, ax, ay, theta = fit

    h, w = linear_rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # project image coords onto the face's local frame: +v along the crown->chin
    # axis (direction ``theta``), +u perpendicular to it.
    dx, dy = xx - cx, yy - cy
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    v = (dx * cos_t + dy * sin_t) / max(ay, 1e-3)      # along the major axis
    u = (-dx * sin_t + dy * cos_t) / max(ax, 1e-3)     # perpendicular

    offsets = np.zeros((h, w), np.float32)
    for zone, points in _CANONICAL.items():
        value = zones.get(zone)
        if value is None:
            continue
        for pu, pv, prad in points:
            d2 = ((u - pu) ** 2 + (v - pv) ** 2) / (2.0 * prad * prad)
            offsets += float(value) * np.exp(-np.clip(d2, 0.0, 60.0))

    # soft face mask, extended below the chin so the neck zone lands on the neck
    face_mask = np.clip(1.2 - (u ** 2 + (v / 1.35) ** 2), 0.0, 1.0)
    face_mask = cv2.GaussianBlur(face_mask.astype(np.float32), (0, 0), sigmaX=max(ax, ay) * 0.15)
    offsets = cv2.GaussianBlur(offsets, (0, 0), sigmaX=max(ax, ay) * 0.08)
    return offsets.astype(np.float32), face_mask.astype(np.float32)


def body_heat_structure(matte: np.ndarray) -> np.ndarray:
    """Core->extremity falloff in [0,1]: 1 deep in the torso, lower at the edges.

    Thin extremities (fingers, arm edges) cool fastest, so the distance transform
    of the silhouette is a good proxy for how well-perfused a region reads.
    """
    body = matte > 0.3
    if not body.any():
        return np.zeros_like(matte, dtype=np.float32)
    edt = ndimage.distance_transform_edt(body).astype(np.float32)
    core = edt / max(float(np.percentile(edt[body], 92)), 1e-6)
    core = np.clip(core, 0.0, 1.0)
    return cv2.GaussianBlur(core, (0, 0), sigmaX=max(matte.shape) * 0.01).astype(np.float32)
