"""Assemble the Radiance Graph (Layer A) — classical by default, AI-enriched
when available and requested (CLAUDE.md §3.1, §2.6).

The classical path uses only numpy/OpenCV; it always produces a sane graph so
every Instrument works with ``--no-ai``. When ``use_ai`` is set, ``scene.ai`` is
imported lazily to overwrite depth/normals/materials/matte with model output.
The result is cached per image under ``.heron_graph/``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from heron.color import srgb
from heron.core import graph as graph_mod
from heron.core.graph import RadianceGraph
from heron.scene import delight as _delight
from heron.scene import matte as _matte
from heron.scene import saliency as _saliency


@dataclass(frozen=True)
class GraphOptions:
    use_ai: bool = False
    delight_strength: float = 0.9
    depth_tier: str = "fast"  # fast | quality
    work_res: int = 1280
    segment_materials: bool = True  # use SAM3 for per-region materials (AI only)
    seed: int = 0
    cache: bool = True

    def signature(self) -> str:
        """Cache key for the expensive channels.

        ``delight_strength`` is deliberately absent: it only affects ``albedo``,
        which is a leaf channel — no other Layer A output is derived from it.
        Keying on it forced a full rebuild (the classical matte alone is ~15 s)
        every time an instrument with a different de-lighting setting ran on the
        same photo, to redo ~0.5 s of work. ``build_graph`` re-derives albedo on
        a cache hit instead.
        """
        mode = f"ai-{self.depth_tier}" if self.use_ai else "classical"
        seg = "+sam3" if (self.use_ai and self.segment_materials) else ""
        return f"{mode}{seg}|res{self.work_res}|v5"


def normals_from_depth(depth: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Guided differentiation of depth into camera-space unit normals (H,W,3)."""
    d = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), sigmaX=1.5)
    dzdx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    nx = -dzdx * strength
    ny = -dzdy * strength
    nz = np.ones_like(d)
    n = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return (n / np.maximum(norm, 1e-6)).astype(np.float32)


def _classical_depth(matte: np.ndarray) -> np.ndarray:
    """A weak pseudo-depth: the subject sits forward of the background.

    Not a measurement — a plausible prior so normals/shape modulation have
    something to work with when no depth model runs.
    """
    depth = 0.35 + 0.5 * matte
    depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), sigmaX=max(matte.shape) * 0.01)
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def _build_classical(linear_rgb: np.ndarray, opts: GraphOptions) -> RadianceGraph:
    albedo, _shading = _delight.delight(linear_rgb, strength=opts.delight_strength)
    sal = _saliency.spectral_residual(linear_rgb)
    matte = _matte.classical_matte(linear_rgb, sal=sal)
    depth = _classical_depth(matte)
    normals = normals_from_depth(depth)
    material_ids = np.zeros(linear_rgb.shape[:2], dtype=np.int32)  # all 'unknown'
    return RadianceGraph(
        depth=depth,
        normals=normals,
        albedo=albedo,
        saliency=sal,
        matte=matte,
        material_ids=material_ids,
        material_names=("unknown",),
        meta={"ai_used": False, "delight_strength": opts.delight_strength},
    )


def _with_delight(graph: RadianceGraph, linear_rgb: np.ndarray,
                  opts: GraphOptions) -> RadianceGraph:
    """Return a graph whose albedo matches ``opts.delight_strength``.

    A cached graph carries the albedo of whichever instrument built it first.
    De-lighting is cheap and depends on nothing else, so a mismatch is fixed by
    re-deriving that one channel rather than discarding the whole graph.
    """
    cached = graph.meta.get("delight_strength")
    if cached is not None and abs(float(cached) - opts.delight_strength) < 1e-6:
        return graph
    albedo, _shading = _delight.delight(linear_rgb, strength=opts.delight_strength)
    return replace(graph, albedo=albedo,
                   meta={**graph.meta, "delight_strength": opts.delight_strength})


def build_graph(
    linear_rgb: np.ndarray,
    image_path: str | Path | None = None,
    opts: GraphOptions | None = None,
) -> RadianceGraph:
    """Build (or load from cache) the Radiance Graph for an image.

    ``image_path`` enables caching; pass ``None`` for in-memory-only builds.
    """
    opts = opts or GraphOptions()

    cache_file: Path | None = None
    if opts.cache and image_path is not None:
        cache_file = graph_mod.cache_path(image_path, opts.signature())
        if cache_file.exists():
            try:
                return _with_delight(RadianceGraph.load(cache_file), linear_rgb, opts)
            except Exception:
                pass  # corrupt cache -> rebuild

    graph = _build_classical(linear_rgb, opts)

    if opts.use_ai:
        try:
            from heron.scene import ai as _ai

            graph = _ai.enrich_graph(graph, linear_rgb, opts)
        except ImportError as e:
            raise RuntimeError(
                "AI mode requested but Layer A dependencies are unavailable. "
                "Install with `pip install -r requirements-ai.txt`, or run with "
                f"--no-ai. (import error: {e})"
            ) from e

    if cache_file is not None:
        try:
            graph.save(cache_file)
        except Exception:
            pass  # caching is best-effort

    return graph
