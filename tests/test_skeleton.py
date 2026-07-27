"""Skeleton-drawing tests (CLAUDE.md §1.1) — validate the stylized bone drawing
with synthetic landmarks, independent of MediaPipe pose detection."""

import numpy as np

from heron.physics import skeleton


def _synthetic_figure(h=200, w=150):
    """A simple figure-on-background image — enough for the classical graph
    builder to produce a valid matte/depth, not meant to resemble a real pose."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([0.15 + 0.1 * xx / w, 0.2 + 0.1 * yy / h, 0.35 - 0.1 * xx / w], -1)
    r = ((xx - w * 0.5) / (w * 0.3)) ** 2 + ((yy - h * 0.5) / (h * 0.4)) ** 2
    fig = (r < 1).astype(np.float32)
    tex = 0.05 * np.sin(xx * 0.7) * np.cos(yy * 0.6)
    skin = np.stack([0.7, 0.5, 0.42], -1) + tex[..., None]
    img = bg * (1 - fig[..., None]) + skin * fig[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def _full_body_landmarks(h=400, w=300):
    """A plausible standing pose in pixel coords (33,2) + full visibility."""
    xy = np.zeros((33, 2), np.float32)
    cx = w / 2
    xy[skeleton._NOSE] = (cx, h * 0.10)
    xy[skeleton._L_EAR] = (cx - 20, h * 0.10)
    xy[skeleton._R_EAR] = (cx + 20, h * 0.10)
    xy[skeleton._L_SHO] = (cx - 50, h * 0.25)
    xy[skeleton._R_SHO] = (cx + 50, h * 0.25)
    xy[skeleton._L_ELB] = (cx - 70, h * 0.40)
    xy[skeleton._R_ELB] = (cx + 70, h * 0.40)
    xy[skeleton._L_WRI] = (cx - 80, h * 0.55)
    xy[skeleton._R_WRI] = (cx + 80, h * 0.55)
    xy[skeleton._L_HIP] = (cx - 30, h * 0.55)
    xy[skeleton._R_HIP] = (cx + 30, h * 0.55)
    xy[skeleton._L_KNE] = (cx - 32, h * 0.75)
    xy[skeleton._R_KNE] = (cx + 32, h * 0.75)
    xy[skeleton._L_ANK] = (cx - 34, h * 0.92)
    xy[skeleton._R_ANK] = (cx + 34, h * 0.92)
    vis = np.ones(33, np.float32)
    return xy, vis


def test_draw_skeleton_produces_bones():
    xy, vis = _full_body_landmarks()
    m = skeleton.draw_skeleton(xy, vis, (400, 300))
    assert m is not None
    assert m.shape == (400, 300)
    assert 0.0 <= m.min() and m.max() <= 1.0
    assert (m > 0.1).sum() > 0  # bones drawn


def test_draw_skeleton_spans_torso():
    xy, vis = _full_body_landmarks()
    m = skeleton.draw_skeleton(xy, vis, (400, 300))
    ys = np.where(m > 0.2)[0]
    # bones should span from head region to legs
    assert ys.min() < 400 * 0.3 and ys.max() > 400 * 0.7


def test_draw_skeleton_low_visibility_draws_nothing():
    xy, vis = _full_body_landmarks()
    vis[:] = 0.0  # nothing visible
    m = skeleton.draw_skeleton(xy, vis, (400, 300))
    assert m is None or float(m.max()) == 0.0 or not (m > 0.1).any()


def test_draw_skeleton_upper_body_only_still_draws_partial():
    """Legs cropped out of frame (waist-up shot) — hips/knees/ankles absent, but
    head/torso/arms should still draw instead of an all-or-nothing failure."""
    xy, vis = _full_body_landmarks()
    vis[skeleton._L_HIP] = vis[skeleton._R_HIP] = 0.0
    vis[skeleton._L_KNE] = vis[skeleton._R_KNE] = 0.0
    vis[skeleton._L_ANK] = vis[skeleton._R_ANK] = 0.0
    m = skeleton.draw_skeleton(xy, vis, (400, 300))
    assert m is not None
    assert (m > 0.1).sum() > 0  # skull/clavicles/arms still drawn
    ys = np.where(m > 0.2)[0]
    assert ys.max() < 400 * 0.65  # nothing drawn down at the (invisible) legs


def test_render_xray_uses_cached_skeleton_bones_not_model(monkeypatch):
    """Layer B (the fluoroscope renderer) must read skeleton_bones from the
    graph, never call the pose model itself (CLAUDE.md §2.7) — this was a real
    bug: MediaPipe was being re-run from scratch on every single render/dial
    tweak whenever the skeleton dial was above 0 (true of the default
    cold_blue_nude preset)."""
    from heron.instruments.fluoroscope import render_xray
    from heron.instruments.loader import load_instrument
    from heron.physics import skeleton as skeleton_mod
    from heron.scene import GraphOptions, build_graph

    def _boom(*a, **kw):
        raise AssertionError("Layer B must not call skeleton_proxy() directly")

    monkeypatch.setattr(skeleton_mod, "skeleton_proxy", _boom)

    img = _synthetic_figure()
    graph = build_graph(img, image_path=None, opts=GraphOptions(cache=False))
    graph.extras["skeleton_bones"] = np.full(graph.shape, 0.5, np.float32)

    cfg = load_instrument("fluoroscope", preset="cold_blue_nude")  # skeleton=0.45
    rgb, stages, meta = render_xray(graph, img, cfg, seed=1)
    assert rgb.shape == (*graph.shape, 3)
    assert np.isfinite(rgb).all()


def test_render_xray_without_cached_bones_renders_fine():
    """No skeleton_bones extra (classical path, or AI ran but found no pose) ->
    bones=None, the renderer degrades gracefully rather than crashing."""
    from heron.instruments.fluoroscope import render_xray
    from heron.instruments.loader import load_instrument
    from heron.scene import GraphOptions, build_graph

    img = _synthetic_figure()
    graph = build_graph(img, image_path=None, opts=GraphOptions(cache=False))
    assert "skeleton_bones" not in graph.extras

    cfg = load_instrument("fluoroscope", preset="cold_blue_nude")
    rgb, stages, meta = render_xray(graph, img, cfg, seed=1)
    assert rgb.shape == (*graph.shape, 3)
    assert np.isfinite(rgb).all()


def test_xray_windowing_uses_full_range_and_leaves_bloom_headroom():
    """The fluoroscope used to divide by a high percentile without subtracting
    the low anchor, so output started around 0.4 instead of black (washed out)
    and the densest tissue pinned against 1.0, where Layer C's screen-blend
    bloom could only clip it to flat white."""
    import numpy as np

    from heron.materials import load_material_table
    from heron.physics.xray import XrayParams, synthesize_xray
    from heron.scene import GraphOptions, build_graph

    img = _synthetic_figure(240, 180)
    graph = build_graph(img, image_path=None, opts=GraphOptions(cache=False))
    field = synthesize_xray(graph, XrayParams(headroom=0.9), load_material_table())
    body = graph.matte > 0.3

    assert float(field.signal.min()) < 0.02          # true blacks, no pedestal
    assert float(field.signal[body].max()) <= 0.9 + 1e-4  # headroom respected
    # and the body still spans a usable range rather than sitting flat
    spread = np.percentile(field.signal[body], 90) - np.percentile(field.signal[body], 10)
    assert spread > 0.1


def test_xray_flesh_floor_does_not_lift_transparent_materials():
    """flesh_floor is a *tissue* pedestal; applying it to the whole matte lifted
    hair and fabric off black too, which on a big-hair portrait put half the
    frame on a flat mid-bright plateau."""
    import numpy as np

    from heron.core.graph import RadianceGraph
    from heron.materials import load_material_table
    from heron.physics.xray import XrayParams, synthesize_xray

    h = w = 64
    matte = np.ones((h, w), np.float32)
    ids = np.zeros((h, w), np.int32)
    ids[:, w // 2:] = 1                       # left = skin, right = fabric
    graph = RadianceGraph(
        depth=np.full((h, w), 0.5, np.float32),
        normals=np.zeros((h, w, 3), np.float32),
        albedo=np.zeros((h, w, 3), np.float32),
        saliency=np.zeros((h, w), np.float32),
        matte=matte, material_ids=ids, material_names=("skin", "fabric"),
    )
    field = synthesize_xray(graph, XrayParams(flesh_floor=0.3, rim=0.0),
                            load_material_table())
    skin_mean = float(field.signal[:, :w // 2].mean())
    fabric_mean = float(field.signal[:, w // 2:].mean())
    assert skin_mean > fabric_mean          # tissue is denser/brighter
    assert fabric_mean < 0.35               # fabric stays near-transparent
