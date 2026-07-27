"""Layer D — Art Direction.

Where Heron stops being a camera and becomes a painter: saliency emphasis
(§3.4), structure-tensor directional bloom (§1.6), and Poisson gradient-domain
detail fusion (§1.6). ``apply_art_direction`` runs the chain from a renderer's
``layerD`` config so every instrument shares it.
"""

from __future__ import annotations

import numpy as np

from heron.art.emphasis import saliency_emphasis
from heron.art.dirbloom import directional_bloom, structure_orientation
from heron.art.poisson import poisson_detail_fusion, solve_poisson
from heron.color import srgb
from heron.core.graph import RadianceGraph

__all__ = [
    "saliency_emphasis",
    "directional_bloom",
    "structure_orientation",
    "poisson_detail_fusion",
    "solve_poisson",
    "apply_art_direction",
]


def apply_art_direction(
    signal: np.ndarray,
    graph: RadianceGraph,
    source_linear: np.ndarray,
    d: dict,
) -> np.ndarray:
    """Emphasis -> directional bloom -> Poisson detail fusion, per ``layerD`` cfg."""
    signal = saliency_emphasis(
        signal, graph.saliency,
        amount=float(d.get("emphasis_amount", 0.2)),
        contrast=float(d.get("emphasis_contrast", 0.15)),
    )
    dir_bloom = float(d.get("dir_bloom", 0.0))
    if dir_bloom > 0.0:
        signal = directional_bloom(
            signal,
            guide=srgb.luminance(source_linear),
            strength=dir_bloom,
            length_frac=float(d.get("dir_bloom_length", 0.012)),
            threshold=float(d.get("dir_bloom_threshold", 0.55)),
        )
    materiality = float(d.get("materiality", 0.0))
    if materiality > 0.0:
        signal = poisson_detail_fusion(
            signal, source_linear,
            materiality=materiality,
            detail_sigma_frac=float(d.get("detail_sigma", 0.01)),
        )
    return signal
