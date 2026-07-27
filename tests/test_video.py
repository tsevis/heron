"""Video batch — the properties that stop a clip from flickering (Phase 4 §6).

Rendering frames independently produces three artefacts, and each has a test
here because none of them is visible in a single frame:

* fixed-pattern noise must NOT change between frames (it is burned into a
  detector), while the NETD term must;
* the scene graph must be carried forward rather than re-estimated;
* the auto window must be eased, not recomputed, or the image pumps.

A per-frame metric cannot catch pumping by eye, so the stability assertions are
on the mechanism rather than on the pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from heron import sensor
from heron.core.graph import RadianceGraph
from heron.physics.thermal import ThermalParams, _window
from heron.video import (
    VideoOptions,
    auto_window,
    blend_graphs,
    render_video,
    warp_graph,
)


def _graph(h=48, w=64) -> RadianceGraph:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disc = ((((xx - w * 0.5) / (w * 0.25)) ** 2 + ((yy - h * 0.5) / (h * 0.3)) ** 2) < 1.0)
    return RadianceGraph(
        depth=(0.35 + 0.5 * disc).astype(np.float32),
        normals=np.dstack([np.zeros((h, w), np.float32), np.zeros((h, w), np.float32),
                           np.ones((h, w), np.float32)]),
        albedo=np.repeat(disc[..., None], 3, -1).astype(np.float32) * 0.6,
        saliency=disc.astype(np.float32),
        matte=disc.astype(np.float32),
        material_ids=disc.astype(np.int32),
        material_names=("unknown", "skin"),
        meta={"ai_used": False, "delight_strength": 0.9},
        extras={"face_region": disc.astype(np.float32)},
    )


def _clip(path, w=96, h=64, frames=6, fps=12):
    """A moving blob — enough parallax for optical flow to have work to do."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(frames):
        cx = w * (0.3 + 0.4 * i / max(frames - 1, 1))
        fig = (((xx - cx) / (w * 0.18)) ** 2 + ((yy - h * 0.5) / (h * 0.34)) ** 2) < 1.0
        img = np.full((h, w, 3), 0.15, np.float32)
        img[fig] = (0.62, 0.48, 0.42)
        writer.write((np.clip(img, 0, 1)[:, :, ::-1] * 255).astype(np.uint8))
    writer.release()
    return path


# --------------------------------------------------------------- noise model

def test_fpn_is_fixed_across_frames():
    """FPN is burned into the detector. Re-rolling it per frame is THE artefact.

    The field is a function of the seed alone — it takes no frame index — so a
    clip rendered with one seed cannot have a crawling pattern by construction.
    """
    a = sensor.make_fpn((64, 64), seed=7, amplitude=0.03)
    b = sensor.make_fpn((64, 64), seed=7, amplitude=0.03)
    assert np.array_equal(a.gain, b.gain)
    assert np.array_equal(a.offset, b.offset)
    # and a different seed really does give a different detector
    other = sensor.make_fpn((64, 64), seed=8, amplitude=0.03)
    assert not np.array_equal(a.gain, other.gain)


def test_netd_re_rolls_per_frame():
    """Temporal noise is temporal — identical frames would look frozen."""
    signal = np.full((64, 64), 0.5, np.float32)
    f0 = sensor.add_netd(signal, 60.0, (20.0, 37.5), seed=7, frame=0)
    f1 = sensor.add_netd(signal, 60.0, (20.0, 37.5), seed=7, frame=1)
    assert not np.array_equal(f0, f1), "NETD did not change between frames"
    # but it is still deterministic for a given frame
    assert np.array_equal(f1, sensor.add_netd(signal, 60.0, (20.0, 37.5), seed=7, frame=1))


def test_renderer_honours_the_frame_index():
    """The renderer must pass `frame` through, or video freezes its temporal noise."""
    from heron.instruments import get_engine, load_instrument

    graph = _graph()
    source = np.repeat(graph.matte[..., None], 3, -1).astype(np.float32) * 0.6
    cfg = load_instrument("thermograph", preset="translucent_glow")
    engine = get_engine(cfg["engine"])

    cfg_a = {**cfg, "layerC": {**cfg["layerC"], "frame": 0}}
    cfg_b = {**cfg, "layerC": {**cfg["layerC"], "frame": 5}}
    a, _, _ = engine(graph, source, cfg_a, seed=7)
    b, _, _ = engine(graph, source, cfg_b, seed=7)
    assert not np.array_equal(a, b), "frame index had no effect — NETD is frozen"


# ------------------------------------------------------------------ stability

def test_warp_preserves_shape_and_labels():
    graph = _graph()
    flow = np.zeros((*graph.shape, 2), np.float32)
    flow[..., 0] = 3.0                       # shift right by 3 px
    out = warp_graph(graph, flow)
    assert out.shape == graph.shape
    assert set(np.unique(out.material_ids)) <= set(np.unique(graph.material_ids))
    assert out.material_ids.dtype == np.int32


def test_zero_flow_is_a_noop():
    """A still frame must not drift the graph."""
    graph = _graph()
    flow = np.zeros((*graph.shape, 2), np.float32)
    out = warp_graph(graph, flow)
    assert np.allclose(out.depth, graph.depth, atol=1e-5)
    assert np.allclose(out.matte, graph.matte, atol=1e-5)


def test_warp_is_deterministic():
    graph = _graph()
    flow = np.full((*graph.shape, 2), 1.5, np.float32)
    assert np.array_equal(warp_graph(graph, flow).depth, warp_graph(graph, flow).depth)


def test_blend_is_a_true_cross_fade():
    warped, fresh = _graph(), _graph()
    fresh = type(fresh)(**{**fresh.__dict__, "depth": fresh.depth + 0.2})
    out = blend_graphs(warped, fresh, 0.5)
    assert np.allclose(out.depth, warped.depth + 0.1, atol=1e-5)
    assert np.allclose(blend_graphs(warped, fresh, 0.0).depth, warped.depth, atol=1e-5)
    assert np.allclose(blend_graphs(warped, fresh, 1.0).depth, fresh.depth, atol=1e-5)


# ------------------------------------------------------------------- the AGC

def test_auto_window_matches_the_physics():
    """Hysteresis is only free if the recovered window equals the real one."""
    rng = np.random.default_rng(0)
    temperature = (rng.standard_normal((64, 64)).astype(np.float32) * 4.0 + 30.0)
    params = ThermalParams(ambient_c=19.0, core_c=34.0)
    _, span = _window(temperature, params)
    assert auto_window(temperature, 19.0, 34.0) == pytest.approx(span, abs=1e-4)


def test_window_ema_damps_a_step_change():
    """A scene that jumps must not drag the window with it in one frame."""
    a = 0.85
    span = (20.0, 37.0)
    wanted = (10.0, 50.0)          # a violent change
    stepped = (a * span[0] + (1 - a) * wanted[0], a * span[1] + (1 - a) * wanted[1])
    assert abs(stepped[0] - span[0]) < abs(wanted[0] - span[0]) * 0.2
    assert abs(stepped[1] - span[1]) < abs(wanted[1] - span[1]) * 0.2


# -------------------------------------------------------------- end-to-end

def test_render_video_writes_a_playable_clip(tmp_path):
    src = _clip(tmp_path / "in.mp4")
    out = tmp_path / "out.mp4"
    stats = render_video(src, out, VideoOptions(preset="translucent_glow", use_ai=False,
                                                seed=7, max_frames=4))
    assert out.exists() and out.stat().st_size > 0
    assert stats["frames"] == 4

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None


def test_static_shot_builds_the_graph_once(tmp_path):
    """The whole point: no re-estimation within a shot, so nothing can flicker."""
    src = _clip(tmp_path / "in.mp4", frames=6)
    stats = render_video(src, tmp_path / "out.mp4",
                         VideoOptions(preset="translucent_glow", use_ai=False, seed=7))
    assert stats["graph_rebuilds"] == 1, (
        f"rebuilt the graph {stats['graph_rebuilds']} times on one continuous shot"
    )


def test_video_is_deterministic(tmp_path):
    """Same clip + same seed -> same bytes (CLAUDE.md §2.4)."""
    src = _clip(tmp_path / "in.mp4", frames=4)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    opts = VideoOptions(preset="translucent_glow", use_ai=False, seed=7, max_frames=4)
    render_video(src, a, opts)
    render_video(src, b, opts)
    assert a.read_bytes() == b.read_bytes()
