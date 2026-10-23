"""In-memory reuse between renders of the same photo.

Every web render re-decoded the source file and re-read the compressed graph
cache from disk: ~0.55 s of a ~1.9 s re-render. Both are now held in memory,
keyed on the file's identity, so the hazards are a stale image after the file
changes, and one render's in-place edit leaking into the next.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from heron import engine
from heron.core import graph as graph_mod
from heron.core import io


def _write(path, level: float) -> None:
    img = np.full((48, 64, 3), 0.1, np.float32)
    img[12:36, 20:44] = level
    io.write_image(path, img)


@pytest.fixture
def photo(tmp_path):
    engine.clear_memo()
    path = tmp_path / "photo.png"
    _write(path, 0.7)
    yield path
    engine.clear_memo()


def _count_calls(monkeypatch, owner, name):
    calls = []
    real = getattr(owner, name)

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)
    return calls


def test_rerender_does_not_decode_the_source_again(photo, monkeypatch):
    reads = _count_calls(monkeypatch, io, "read_image")
    engine.render_image(photo)
    engine.render_image(photo, overrides=["layerC.netd_mk=60"])
    assert len(reads) == 1


def test_rerender_does_not_reload_the_graph_from_disk(photo, monkeypatch):
    loads = _count_calls(monkeypatch, graph_mod.RadianceGraph, "load")
    for _ in range(3):
        engine.render_image(photo)
    assert len(loads) <= 1


def test_a_replaced_source_is_read_again(photo):
    first = engine.render_image(photo).image
    _write(photo, 0.2)
    later = os.stat(photo).st_mtime + 5
    os.utime(photo, (later, later))

    assert not np.array_equal(engine.render_image(photo).image, first)


def test_one_render_cannot_corrupt_the_next(photo):
    engine.render_image(photo)
    held = engine.render_image(photo).graph
    before = held.matte.copy()

    try:
        held.matte[...] = 0.0
    except ValueError:
        pass  # read-only is the intended protection
    held.meta["poison"] = True

    again = engine.render_image(photo).graph
    assert np.array_equal(again.matte, before)
    assert "poison" not in again.meta
