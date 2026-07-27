"""Instruments — YAML configs of the 4-layer stack + the engine registry.

An Instrument's ``engine`` field selects a Layer-B/C/D renderer here. Adding an
Instrument is (mostly) a new YAML plus, where the physics differs, a new engine
entry — never a parallel codebase (CLAUDE.md §1).
"""

from __future__ import annotations

from heron.instruments.loader import (
    available_instruments,
    list_presets,
    load_instrument,
)
from heron.instruments.thermograph import render_thermal
from heron.instruments.fluoroscope import render_xray
from heron.instruments.intensifier import render_photon
from heron.instruments.nir import render_nir
from heron.instruments.kirlian import render_corona
from heron.instruments.schlieren import render_schlieren
from heron.instruments.lumen import render_lumen
from heron.instruments.spectroscope import render_spectral

# engine name -> renderer(graph, source_linear, cfg, seed) -> (image, stages, meta)
ENGINES = {
    "thermal": render_thermal,
    "xray": render_xray,
    "photon": render_photon,
    "nir": render_nir,
    "corona": render_corona,
    "schlieren": render_schlieren,
    "lumen": render_lumen,
    "spectral": render_spectral,
}


def get_engine(name: str):
    if name not in ENGINES:
        raise ValueError(f"unknown engine '{name}'. available: {sorted(ENGINES)}")
    return ENGINES[name]


__all__ = [
    "available_instruments",
    "list_presets",
    "load_instrument",
    "get_engine",
    "ENGINES",
]
