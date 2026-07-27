"""NIR-camera tests (CLAUDE.md §1.2): material-driven reflectance mapping."""

import numpy as np

from heron.core.graph import RadianceGraph
from heron.physics.nir import NIRParams, synthesize_nir


def _graph_with_materials(names, id_map):
    """Minimal RadianceGraph carrying only what the NIR engine reads."""
    h, w = id_map.shape
    return RadianceGraph(
        depth=np.full((h, w), 0.5, np.float32),
        normals=np.zeros((h, w, 3), np.float32),
        albedo=np.full((h, w, 3), 0.5, np.float32),
        saliency=np.zeros((h, w), np.float32),
        matte=np.zeros((h, w), np.float32),
        material_ids=id_map.astype(np.int32),
        material_names=tuple(names),
    )


def _uniform_source(h=8, w=24, value=0.4):
    return np.full((h, w, 3), value, np.float32)


def test_foliage_glows_brighter_than_neutral():
    # left half foliage, right half unknown(neutral), same source brightness
    ids = np.zeros((8, 24), np.int32)
    ids[:, :12] = 1
    g = _graph_with_materials(["unknown", "foliage"], ids)
    out = synthesize_nir(g, _uniform_source(), NIRParams(skin_smooth=0.0)).signal
    assert out[:, :12].mean() > out[:, 12:].mean() + 0.1  # Wood effect


def test_dark_materials_read_dark():
    ids = np.zeros((8, 24), np.int32)
    ids[:, :12] = 1  # water -> nir: dark
    g = _graph_with_materials(["unknown", "water"], ids)
    out = synthesize_nir(g, _uniform_source(value=0.5), NIRParams(skin_smooth=0.0)).signal
    assert out[:, :12].mean() < out[:, 12:].mean()  # water darker than neutral


def test_signal_in_range():
    ids = np.zeros((8, 8), np.int32)
    g = _graph_with_materials(["unknown"], ids)
    out = synthesize_nir(g, _uniform_source(8, 8)).signal
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
