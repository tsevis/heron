"""Kirlian / corona-discharge physics (CLAUDE.md §1.3, §3.2).

The first fully *synthetic* engine — it grows a signal rather than transforming
the scene. Three components, all seeded from the subject matte:

1. **Aura** — a distance-field glow outside the silhouette with exponential
   falloff, modulated by each boundary material's ``kirlian_activity`` (hair and
   skin ignite corona readily; fabric barely).
2. **Streamers** — dielectric-breakdown filaments grown by random walkers
   launched from high-curvature boundary points (fingertips, hair/leaf tips —
   physically correct, since corona ignites where the field curvature is
   highest), pulled outward, swirled by curl noise, and branching stochastically.
3. **Sparks** — high-frequency near-edge noise gated by the curl field (the
   "electric wind").

Everything is deterministic per seed (§2.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from heron.core.graph import RadianceGraph
from heron.core.seeds import rng_for
from heron.materials import MaterialTable, load_material_table
from heron.physics import curlnoise


@dataclass(frozen=True)
class KirlianParams:
    aura_falloff: float = 0.02   # exponential falloff (fraction of image size)
    aura_gain: float = 1.0
    inner_glow: float = 0.15     # faint glow inside the silhouette
    n_streamers: int = 44        # number of launch tips
    streamer_length: float = 0.13
    streamer_step: float = 2.0
    branch_prob: float = 0.05
    max_depth: int = 2
    jitter: float = 0.5
    curl_amp: float = 0.7
    outward_pull: float = 1.1
    inertia: float = 1.3
    spark_amount: float = 0.3
    spark_band: float = 0.035


@dataclass
class KirlianField:
    intensity: np.ndarray   # corona intensity [0,1]
    aura: np.ndarray
    streamers: np.ndarray


def _activity_outside(matte_bin: np.ndarray, activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Return (dist_outside, activity propagated to the exterior, edt indices)."""
    dist, (iy, ix) = ndimage.distance_transform_edt(~matte_bin, return_indices=True)
    activity_out = activity[iy, ix]
    return dist.astype(np.float32), activity_out.astype(np.float32), (iy, ix)


def _launch_tips(matte_bin: np.ndarray, activity: np.ndarray, n: int, seed: int):
    """High-curvature convex boundary points -> (x, y, outward_dir, weight)."""
    contours, _ = cv2.findContours(matte_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.008 * peri, True).reshape(-1, 2)
    if len(approx) < 3:
        return []
    centroid = approx.mean(axis=0)

    tips = []
    m = len(approx)
    for i in range(m):
        v = approx[i].astype(np.float32)
        a = approx[(i - 1) % m].astype(np.float32)
        b = approx[(i + 1) % m].astype(np.float32)
        va, vb = a - v, b - v
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-3 or nb < 1e-3:
            continue
        cosang = float(np.dot(va, vb) / (na * nb))
        sharpness = (cosang + 1.0) / 2.0   # ~1 for a sharp convex tip
        outward = v - centroid
        if np.linalg.norm(outward) < 1e-3:
            continue
        outward = outward / np.linalg.norm(outward)
        # convex-outward check: tip points away from centroid
        yx = (int(np.clip(v[1], 0, activity.shape[0] - 1)), int(np.clip(v[0], 0, activity.shape[1] - 1)))
        act = float(activity[yx])
        weight = sharpness * (0.3 + act)
        tips.append((float(v[0]), float(v[1]), outward, weight))

    tips.sort(key=lambda t: t[3], reverse=True)
    tips = tips[: max(1, n)]
    return tips


def _grow_streamers(tips, dist_out, params: KirlianParams, seed: int) -> np.ndarray:
    """Grow random-walk filaments from the launch tips into a float buffer."""
    h, w = dist_out.shape
    buf = np.zeros((h, w), np.float32)
    if not tips:
        return buf

    gy = cv2.Sobel(dist_out, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(dist_out, cv2.CV_32F, 1, 0, ksize=3)
    cy, cx = curlnoise.curl_noise_field((h, w), seed, scale=0.05)
    step = params.streamer_step
    max_len = params.streamer_length * max(h, w)

    def sample(field, x, y):
        return field[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))]

    rng = rng_for(seed, "streamers", len(tips))
    for ti, (x0, y0, outward, weight) in enumerate(tips):
        stack = [(x0, y0, np.array(outward, np.float32), max_len * (0.6 + 0.6 * weight), 0)]
        path = []
        while stack:
            x, y, d, rem, depth = stack.pop()
            pts = [(x, y)]
            while rem > 0:
                g = np.array([sample(gx, x, y), sample(gy, x, y)], np.float32)
                gn = np.linalg.norm(g)
                g = g / gn if gn > 1e-4 else np.array(outward, np.float32)
                c = np.array([sample(cx, x, y), sample(cy, x, y)], np.float32)
                r = rng.standard_normal(2).astype(np.float32)
                d = d * params.inertia + g * params.outward_pull + c * params.curl_amp + r * params.jitter
                dn = np.linalg.norm(d)
                if dn < 1e-4:
                    break
                d = d / dn
                x, y = x + d[0] * step, y + d[1] * step
                if not (0 <= x < w and 0 <= y < h):
                    break
                pts.append((x, y))
                rem -= step
                if depth < params.max_depth and rng.random() < params.branch_prob:
                    theta = (0.6 + 0.4 * rng.random()) * (1 if rng.random() < 0.5 else -1)
                    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
                    stack.append((x, y, rot @ d, rem * 0.6, depth + 1))
            if len(pts) > 1:
                path.append(np.round(np.array(pts)).astype(np.int32))
        if path:
            tmp = np.zeros((h, w), np.float32)
            cv2.polylines(tmp, path, isClosed=False, color=float(0.7 + 0.3 * weight), thickness=1, lineType=cv2.LINE_AA)
            buf = np.maximum(buf, tmp)
    return buf


def synthesize_kirlian(
    graph: RadianceGraph,
    params: KirlianParams | None = None,
    seed: int = 0,
    table: MaterialTable | None = None,
) -> KirlianField:
    """Synthesize the corona field from the subject matte."""
    p = params or KirlianParams()
    table = table or load_material_table()
    h, w = graph.shape
    matte_bin = graph.matte > 0.4
    if not matte_bin.any():
        z = np.zeros((h, w), np.float32)
        return KirlianField(intensity=z, aura=z, streamers=z)

    activity = graph.material_map(table.kirlian_map(), default=0.2).astype(np.float32)
    dist_out, activity_out, _ = _activity_outside(matte_bin, activity)

    # 1) aura — exponential falloff outside, scaled by boundary material activity
    falloff = max(p.aura_falloff * max(h, w), 1.0)
    aura = np.exp(-dist_out / falloff) * (0.3 + activity_out) * p.aura_gain
    aura[matte_bin] = 0.0
    # faint inner rim + interior glow
    dist_in = ndimage.distance_transform_edt(matte_bin).astype(np.float32)
    inner = np.exp(-dist_in / (falloff * 1.5)) * matte_bin
    interior = p.inner_glow * activity * matte_bin

    # 2) streamers from high-curvature tips
    tips = _launch_tips(matte_bin, activity, p.n_streamers, seed)
    streamers = _grow_streamers(tips, dist_out, p, seed)
    streamers *= np.exp(-dist_out / (falloff * 2.5))  # fade with distance from body
    streamer_glow = cv2.GaussianBlur(streamers, (0, 0), sigmaX=max(h, w) * 0.004)

    # 3) sparks — near-edge high-freq noise gated by the curl magnitude
    sparks = np.zeros((h, w), np.float32)
    if p.spark_amount > 0.0:
        band = np.exp(-dist_out / (p.spark_band * max(h, w))) * (~matte_bin)
        hi = curlnoise.value_noise((h, w), seed, scale=0.25, tag="spark")
        cyf, cxf = curlnoise.curl_noise_field((h, w), seed + 1, scale=0.08)
        gate = np.clip(np.abs(cyf) + np.abs(cxf), 0, 1)
        rng = rng_for(seed, "spark_mask")
        speckle = (rng.random((h, w)) < 0.02).astype(np.float32)
        sparks = p.spark_amount * band * gate * np.clip(hi, 0, 1) * speckle * 6.0

    intensity = np.clip(aura + inner * 0.6 + interior + streamers + 0.6 * streamer_glow + sparks, 0.0, 1.0)
    return KirlianField(intensity=intensity.astype(np.float32), aura=aura.astype(np.float32), streamers=streamers)
