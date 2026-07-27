"""Intensifier-tube tests (CLAUDE.md §1.2): photon determinism, phosphor
furniture, and a render smoke."""

import numpy as np

from heron.physics.photon import PhotonParams, synthesize_photon
from heron.sensor.faceplate import hex_faceplate
from heron.sensor.scanlines import scanlines as apply_scanlines


def _scene(h=96, w=96):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - w * 0.5) / (w * 0.3)) ** 2 + ((yy - h * 0.5) / (h * 0.3)) ** 2))
    return np.repeat((0.1 + 0.8 * g)[..., None], 3, axis=-1).astype(np.float32)


def test_photon_signal_in_range():
    f = synthesize_photon(_scene(), PhotonParams(), seed=1)
    assert f.signal.shape == (96, 96)
    assert 0.0 <= float(f.signal.min()) and float(f.signal.max()) <= 1.0


def test_photon_deterministic_same_seed():
    a = synthesize_photon(_scene(), seed=3, frame=0)
    b = synthesize_photon(_scene(), seed=3, frame=0)
    assert np.array_equal(a.signal, b.signal)


def test_photon_noise_reroll_per_frame():
    a = synthesize_photon(_scene(), seed=3, frame=0)
    b = synthesize_photon(_scene(), seed=3, frame=1)
    assert not np.array_equal(a.signal, b.signal)  # shot noise re-rolls per frame


def test_hex_faceplate_darkens_and_bounded():
    s = np.full((64, 64), 0.7, np.float32)
    out = hex_faceplate(s, cell_px=6.0, strength=0.2)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.mean() < s.mean()          # honeycomb walls darken
    assert out.max() <= s.max() + 1e-6


def test_scanlines_periodic_darkening():
    s = np.ones((60, 40), np.float32)
    out = apply_scanlines(s, period_px=3.0, strength=0.4)
    row_means = out.mean(axis=1)
    assert row_means.min() < 0.95        # some rows darkened
    assert out.min() >= 0.0


def test_render_photon_deterministic(tmp_path):
    from heron.core import io
    from heron.engine import render_image

    src = tmp_path / "in.png"
    io.write_image(src, _scene(128, 128))
    a = render_image(src, instrument="intensifier", preset="gen2_p22", seed=2, cache=False)
    b = render_image(src, instrument="intensifier", preset="gen2_p22", seed=2, cache=False)
    assert a.image.shape[2] == 3
    assert np.array_equal(a.image, b.image)
