"""Radiance Graph cache reuse across instruments.

Instruments disagree about de-lighting: the thermograph wants the source
lighting gone (strength 0.9), night vision and NIR want it kept (0.0) because
they show real illumination. That difference used to be part of the cache key,
so switching between those instruments on the same photo rebuilt the entire
graph — paying ~15 s of classical matting to redo ~0.5 s of de-lighting.

Albedo is a leaf channel: nothing else in Layer A is derived from it. These
tests pin that reasoning down — the expensive channels must survive a change of
de-lighting, and albedo must still be correct for the instrument asking.
"""

from __future__ import annotations

import numpy as np
import pytest

from heron.core import io
from heron.scene import GraphOptions, build_graph

DELIT = GraphOptions(delight_strength=0.9)
LIT = GraphOptions(delight_strength=0.0)


@pytest.fixture
def photo(tmp_path):
    """A lit scene: a bright side and a shaded side, so de-lighting has work."""
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    figure = ((((xx - w * 0.5) / (w * 0.3)) ** 2 + ((yy - h * 0.5) / (h * 0.35)) ** 2) < 1.0)
    lighting = 0.35 + 0.6 * (xx / w)          # strong left-to-right gradient
    base = np.where(figure, 0.62, 0.15).astype(np.float32)
    img = np.clip((base * lighting)[..., None] * np.array([1.0, 0.85, 0.75], np.float32), 0, 1)
    path = tmp_path / "photo.png"
    io.write_image(path, img.astype(np.float32))
    return path


def test_delight_strength_does_not_change_the_cache_key():
    """The key covers the expensive channels only."""
    assert DELIT.signature() == LIT.signature()


def test_expensive_channels_survive_a_delight_change(photo):
    """Switching to a non-de-lit instrument reuses matte/depth/saliency."""
    first = build_graph(io.read_image(photo), image_path=photo, opts=DELIT)
    second = build_graph(io.read_image(photo), image_path=photo, opts=LIT)

    assert np.array_equal(first.matte, second.matte), "matte was rebuilt"
    assert np.array_equal(first.depth, second.depth), "depth was rebuilt"
    assert np.array_equal(first.saliency, second.saliency), "saliency was rebuilt"
    assert np.array_equal(first.normals, second.normals), "normals were rebuilt"


def test_albedo_is_rederived_for_the_requesting_instrument(photo):
    """A cache hit must not hand back the previous instrument's albedo."""
    source = io.read_image(photo)
    delit = build_graph(source, image_path=photo, opts=DELIT)
    lit = build_graph(source, image_path=photo, opts=LIT)

    assert not np.allclose(delit.albedo, lit.albedo), "albedo ignored the de-light setting"
    assert lit.meta["delight_strength"] == 0.0
    assert delit.meta["delight_strength"] == 0.9

    # Strength 0.0 only re-exposes: albedo stays a near-uniform multiple of the
    # source, so the left-to-right lighting gradient survives. Strength 0.9
    # divides it out, and that ratio varies across the frame.
    kept = lit.albedo / np.maximum(source, 1e-6)
    removed = delit.albedo / np.maximum(source, 1e-6)
    assert float(kept.std()) < 0.05, "strength 0.0 altered the lighting it should keep"
    assert float(removed.std()) > 0.5, "strength 0.9 did not remove the lighting"


def test_cached_albedo_matches_an_uncached_build(photo):
    """The cache path and the cold path must agree, whichever order they run."""
    source = io.read_image(photo)
    build_graph(source, image_path=photo, opts=DELIT)      # seed the cache at 0.9
    from_cache = build_graph(source, image_path=photo, opts=LIT)
    cold = build_graph(source, image_path=None, opts=GraphOptions(delight_strength=0.0, cache=False))

    assert np.allclose(from_cache.albedo, cold.albedo, atol=1e-5)


def test_repeat_hit_at_the_same_strength_is_untouched(photo):
    """No needless recompute when the strength already matches."""
    source = io.read_image(photo)
    first = build_graph(source, image_path=photo, opts=DELIT)
    again = build_graph(source, image_path=photo, opts=DELIT)
    assert np.array_equal(first.albedo, again.albedo)


def test_work_res_still_separates_caches():
    """Resolution genuinely changes every channel — it stays in the key."""
    assert GraphOptions(work_res=1280).signature() != GraphOptions(work_res=768).signature()


def test_ai_and_classical_graphs_never_share_a_cache(photo):
    """A --no-ai graph must never masquerade as an AI one."""
    assert GraphOptions(use_ai=False).signature() != GraphOptions(use_ai=True).signature()
