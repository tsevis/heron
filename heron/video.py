"""Video batch rendering (Phase 4 §6).

Video is not "loop over frames". Rendering each frame independently produces
three artefacts, and all three come from re-estimating things that should be
stable across a shot:

* **Fixed-pattern noise crawling.** FPN is burned into a detector; it does not
  change frame to frame. Re-rolling it per frame is the single most visible
  video artefact. Here the FPN field is a function of the seed alone, so it is
  fixed for the whole clip, while the NETD term takes the frame index and does
  re-roll — which is correct, temporal noise is temporal.
* **Depth/matte flicker.** Layer A re-estimated per frame jitters, and the
  physics keys on it. The graph is therefore built once per shot and warped
  forward by optical flow, so there is no re-estimation to flicker.
* **AGC pumping.** An auto window recomputed per frame makes the whole image
  breathe. The window bounds are smoothed with an EMA whose input is recovered
  from the frame's own temperature field, so hysteresis costs no extra physics.

The engine stays UI-agnostic (CLAUDE.md §2.5) — this module calls into it, and
frontends call this.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from heron.color import srgb
from heron.core.graph import RadianceGraph
from heron.instruments import get_engine, load_instrument
from heron.scene import GraphOptions, build_graph
from heron.scene.upsample import upsample_graph


@dataclass(frozen=True)
class VideoOptions:
    instrument: str = "thermograph"
    preset: str | None = None
    overrides: tuple[str, ...] = ()
    seed: int = 7
    use_ai: bool = False
    render_res: int | None = None
    # Rebuild the scene graph when the frame difference exceeds this (0..1).
    shot_threshold: float = 0.18
    # Force a rebuild at least this often; 0 disables (warp for the whole shot).
    keyframe_interval: int = 0
    # Blend weight for a mid-shot rebuild — a hard swap would pop.
    rebuild_blend: float = 0.5
    # Window-bound smoothing. Higher holds the previous window harder.
    agc_ema: float = 0.85
    max_frames: int | None = None
    fps: float | None = None      # None = inherit from the source


def _to_linear(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return srgb.srgb_to_linear(rgb)


def _to_bgr8(linear_rgb: np.ndarray) -> np.ndarray:
    encoded = np.clip(srgb.linear_to_srgb(linear_rgb), 0.0, 1.0)
    return cv2.cvtColor((encoded * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _fit(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return image
    s = max_side / max(h, w)
    return cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _gray8(linear_gray: np.ndarray) -> np.ndarray:
    """DIS requires 8-bit single channel; the pipeline carries float32 linear."""
    return np.clip(srgb.linear_to_srgb(linear_gray) * 255.0, 0, 255).astype(np.uint8)


def _flow(prev_gray8: np.ndarray, gray8: np.ndarray) -> np.ndarray:
    """Dense optical flow from ``prev_gray8`` to ``gray8``, both uint8.

    DIS in FAST preset: this runs per frame, so a slow flow would dominate the
    budget.
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    return dis.calc(prev_gray8, gray8, None)


def _warp(field: np.ndarray, flow: np.ndarray, nearest: bool = False) -> np.ndarray:
    """Warp a channel forward by ``flow`` (which maps prev -> current)."""
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(field, map_x, map_y, interp, borderMode=cv2.BORDER_REPLICATE)


def warp_graph(graph: RadianceGraph, flow: np.ndarray) -> RadianceGraph:
    """Carry a scene graph forward one frame instead of re-estimating it.

    Re-running Layer A per frame is what makes depth and matte flicker; warping
    a stable graph has no estimator noise to begin with.
    """
    extras = {}
    for name, value in graph.extras.items():
        array = np.asarray(value)
        if array.ndim >= 2 and array.shape[:2] == graph.shape:
            extras[name] = _warp(array.astype(np.float32), flow)
        else:
            extras[name] = value

    return replace(
        graph,
        depth=_warp(graph.depth, flow),
        normals=_warp(graph.normals, flow),
        albedo=_warp(graph.albedo, flow),
        saliency=_warp(graph.saliency, flow),
        matte=_warp(graph.matte, flow),
        material_ids=_warp(graph.material_ids.astype(np.float32), flow, nearest=True).astype(np.int32),
        extras=extras,
    )


def blend_graphs(warped: RadianceGraph, fresh: RadianceGraph, weight: float) -> RadianceGraph:
    """Cross-fade a freshly built graph over the warped one.

    A mid-shot rebuild that swaps outright pops, because the new estimate never
    agrees with the accumulated warp to the pixel.
    """
    w = float(np.clip(weight, 0.0, 1.0))

    def mix(a, b):
        return ((1.0 - w) * a + w * b).astype(np.float32)

    return replace(
        fresh,
        depth=mix(warped.depth, fresh.depth),
        normals=mix(warped.normals, fresh.normals),
        albedo=mix(warped.albedo, fresh.albedo),
        saliency=mix(warped.saliency, fresh.saliency),
        matte=mix(warped.matte, fresh.matte),
    )


def auto_window(temperature_c: np.ndarray, ambient_c: float, core_c: float) -> tuple[float, float]:
    """What the thermal auto-window would have chosen for this field.

    Mirrors ``physics.thermal._window``. Recovering it from the rendered
    temperature stage is what makes AGC hysteresis free: the frame can be
    rendered with a *held* window while still reporting what it wanted.
    """
    tmin = min(ambient_c - 1.0, float(np.percentile(temperature_c, 2.0)))
    tmax = max(core_c + 1.0, float(np.percentile(temperature_c, 98.0)))
    return tmin, max(tmax, tmin + 0.5)


def _shot_changed(prev_small: np.ndarray, small: np.ndarray, threshold: float) -> bool:
    return float(np.abs(small - prev_small).mean()) > threshold


def render_video(
    input_path: str | Path,
    output_path: str | Path,
    opts: VideoOptions | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Render a clip. Returns timing and stability statistics."""
    import time

    opts = opts or VideoOptions()
    cfg_base = load_instrument(opts.instrument, preset=opts.preset,
                               overrides=list(opts.overrides))
    engine = get_engine(cfg_base["engine"])
    layer_a = cfg_base.get("layerA", {})
    work_res = int(layer_a.get("work_res", 1280))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if opts.max_frames:
        total = min(total, opts.max_frames) if total else opts.max_frames

    graph_opts = GraphOptions(
        use_ai=opts.use_ai,
        delight_strength=float(layer_a.get("delight_strength", 0.9)),
        depth_tier=str(layer_a.get("depth_tier", "fast")),
        work_res=work_res,
        segment_materials=bool(layer_a.get("segment_materials", True)),
        seed=opts.seed,
        cache=False,          # every frame is a different image; caching would thrash
    )

    writer = None
    graph_low: RadianceGraph | None = None
    prev_small: np.ndarray | None = None
    prev_gray_low: np.ndarray | None = None
    span_ema: tuple[float, float] | None = None
    rebuilds = 0
    spans: list[tuple[float, float]] = []
    index = 0
    t0 = time.time()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or (opts.max_frames and index >= opts.max_frames):
                break

            source_full = _to_linear(frame_bgr)
            source_low = _fit(source_full, work_res)
            gray_low = _gray8(cv2.cvtColor(source_low, cv2.COLOR_RGB2GRAY))
            small = cv2.resize(gray_low, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

            new_shot = prev_small is None or _shot_changed(prev_small, small, opts.shot_threshold)
            periodic = (opts.keyframe_interval > 0 and index > 0
                        and index % opts.keyframe_interval == 0)

            if new_shot or graph_low is None:
                graph_low = build_graph(source_low, image_path=None, opts=graph_opts)
                rebuilds += 1
                span_ema = None            # a new scene deserves a fresh window
            else:
                flow = _flow(prev_gray_low, gray_low)
                graph_low = warp_graph(graph_low, flow)
                if periodic:
                    fresh = build_graph(source_low, image_path=None, opts=graph_opts)
                    graph_low = blend_graphs(graph_low, fresh, opts.rebuild_blend)
                    rebuilds += 1

            source = _fit(source_full, opts.render_res) if opts.render_res else source_low
            graph = upsample_graph(graph_low, source) if source.shape[:2] != graph_low.shape else graph_low

            cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
            # Fixed FPN (seed constant), per-frame temporal noise (frame index).
            cfg.setdefault("layerC", {})["frame"] = index
            if span_ema is not None:
                cfg.setdefault("layerB", {})["span_min_c"] = span_ema[0]
                cfg.setdefault("layerB", {})["span_max_c"] = span_ema[1]

            image, stages, meta = engine(graph, source, cfg, seed=opts.seed)

            # Recover the window this frame *wanted*, and ease toward it.
            temperature = stages.get("temperature_c")
            if temperature is not None:
                b = cfg_base.get("layerB", {})
                wanted = auto_window(temperature, float(b.get("ambient_c", 19.0)),
                                     float(b.get("core_c", 34.0)))
                if span_ema is None:
                    span_ema = wanted
                else:
                    a = opts.agc_ema
                    span_ema = (a * span_ema[0] + (1 - a) * wanted[0],
                                a * span_ema[1] + (1 - a) * wanted[1])
                spans.append(span_ema)

            out_bgr = _to_bgr8(image)
            if writer is None:
                h, w = out_bgr.shape[:2]
                writer = cv2.VideoWriter(str(output_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(opts.fps or src_fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open video writer for {output_path}")
            writer.write(out_bgr)

            prev_small, prev_gray_low = small, gray_low
            index += 1
            if progress:
                progress(index, total, f"frame {index}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.time() - t0
    # Window drift per frame: the number that shows whether AGC is pumping.
    drift = 0.0
    if len(spans) > 1:
        lows = np.array([s[0] for s in spans], np.float32)
        highs = np.array([s[1] for s in spans], np.float32)
        drift = float(max(np.abs(np.diff(lows)).max(), np.abs(np.diff(highs)).max()))

    return {
        "frames": index,
        "seconds": round(elapsed, 2),
        "fps": round(index / elapsed, 2) if elapsed > 0 else 0.0,
        "graph_rebuilds": rebuilds,
        "max_window_step_c": round(drift, 4),
        "output": str(output_path),
    }
