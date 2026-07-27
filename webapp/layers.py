"""Decompose a render into per-stage images + a manifest (Phase 3, task 3.0).

The Photoshop bridge needs an editable layer stack, not a flat PNG. Every
renderer already computes the right intermediates and hands them back in
``RenderResult.stages``; the web layer simply threw them away. This module
writes each one as its own file and describes them in a manifest the panel can
turn into a named layer group.

Nothing here touches ``heron/`` — this is service work (Phase-3 §3.1). Stages
are written as raw data channels (``write_gray``, no sRGB curve) so they can be
hand-edited and re-imported without gamma surprises, exactly like the Radiance
Graph channels of CLAUDE.md §2.7. Single-channel stages are *not* colourized:
the panel decides how to blend them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from heron.core import io as hio
from heron.core.graph import RadianceGraph

# Stages that are additive glow rather than a base signal. Screen keeps the
# layer below readable; normal would hide it.
SCREEN_STAGES = frozenset({"aura", "streamers", "sources", "emitter"})

# Short, honest descriptions so a document is still readable in six months.
STAGE_NOTES: dict[str, str] = {
    "emission": "emitted signal before the sensor stage",
    "processed_signal": "signal after bloom / noise / AGC",
    "temperature_c": "temperature field, degrees Celsius (normalized to 0..1)",
    "transmission": "Beer-Lambert transmission through the body",
    "sources": "detected light sources driving the intensifier",
    "nir": "near-infrared response from the material table",
    "intensity": "corona intensity field",
    "aura": "distance-field aura",
    "streamers": "dielectric-breakdown streamers",
    "knife": "knife-edge signed derivative",
    "density": "density gradient magnitude",
    "spectral": "spectral dispersion, already RGB",
    "emitter": "bioluminescent emitter mask",
}

GRAPH_NOTES: dict[str, str] = {
    "depth": "Layer A depth, larger = nearer",
    "matte": "subject coverage",
    "saliency": "attention field",
    "albedo": "de-lit reflectance (sRGB-encoded RGB)",
    "normals": "surface normals, encoded (n * 0.5 + 0.5)",
    "materials": "material index map; decode id = round(value * (count - 1))",
}


def _entry(name: str, path: Path, url_prefix: str, mode: str, kind: str,
           size: tuple[int, int], note: str, **extra: Any) -> dict:
    return {"name": name, "url": f"{url_prefix}/{path.name}", "mode": mode,
            "kind": kind, "size": [size[0], size[1]], "note": note, **extra}


def _write_channel(path: Path, array: np.ndarray, bit_depth: int) -> dict:
    """Write one stage array, reporting how it was encoded.

    Returns the kind/size/range fields for its manifest entry. Fields that are
    not already in [0,1] (a temperature field is in Celsius) are min-max
    normalized and the original range is reported, so nothing is silently
    rescaled out of existence.
    """
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 3 and array.shape[2] == 3:
        hio.write_image(path, np.clip(array, 0.0, 1.0), bit_depth=bit_depth)
        return {"kind": "rgb", "size": (array.shape[1], array.shape[0]), "range": None}

    if array.ndim != 2:
        raise ValueError(f"stage {path.stem}: expected (H,W) or (H,W,3), got {array.shape}")

    lo, hi = float(np.nanmin(array)), float(np.nanmax(array))
    if lo >= 0.0 and hi <= 1.0:
        normalized, reported = array, None
    else:
        span = hi - lo
        normalized = (array - lo) / span if span > 1e-9 else np.zeros_like(array)
        reported = [round(lo, 4), round(hi, 4)]
    hio.write_gray(path, np.clip(normalized, 0.0, 1.0), bit_depth=bit_depth)
    return {"kind": "gray", "size": (array.shape[1], array.shape[0]), "range": reported}


def stage_layers(stages: dict[str, np.ndarray], out_dir: Path, base: str,
                 url_prefix: str = "/outputs", bit_depth: int = 8) -> list[dict]:
    """Write every renderer stage as its own file; return manifest entries."""
    out: list[dict] = []
    for name, array in stages.items():
        if array is None:
            continue
        path = out_dir / f"{base}_{name}.png"
        written = _write_channel(path, array, bit_depth)
        mode = "screen" if name in SCREEN_STAGES else "normal"
        out.append(_entry(name, path, url_prefix, mode, written["kind"],
                          written["size"], STAGE_NOTES.get(name, ""),
                          range=written["range"]))
    return out


def graph_layers(graph: RadianceGraph, out_dir: Path, base: str,
                 url_prefix: str = "/outputs", bit_depth: int = 8) -> list[dict]:
    """Write the Radiance Graph channels as layers (CLAUDE.md §2.7).

    They are artifacts, not internals — having them in the document *is* the
    hand-editing round-trip that rule promises.
    """
    count = max(len(graph.material_names), 1)
    channels: dict[str, np.ndarray] = {
        "depth": graph.depth,
        "matte": graph.matte,
        "saliency": graph.saliency,
        "albedo": graph.albedo,
        "normals": np.asarray(graph.normals, np.float32) * 0.5 + 0.5,
        "materials": graph.material_ids.astype(np.float32) / max(count - 1, 1),
    }
    out: list[dict] = []
    for name, array in channels.items():
        path = out_dir / f"{base}_graph_{name}.png"
        written = _write_channel(path, array, bit_depth)
        entry = _entry(name, path, url_prefix, "normal", written["kind"],
                       written["size"], GRAPH_NOTES.get(name, ""),
                       range=written["range"])
        if name == "materials":
            entry["materials"] = list(graph.material_names)
        out.append(entry)
    return out


def build_manifest(composite_url: str, stages: dict[str, np.ndarray],
                   graph: RadianceGraph, out_dir: Path, base: str,
                   url_prefix: str = "/outputs", bit_depth: int = 8,
                   include_graph: bool = True) -> dict:
    """Full layered manifest: composite + stage layers + graph channels."""
    return {
        "composite": composite_url,
        "layers": stage_layers(stages, out_dir, base, url_prefix, bit_depth),
        "graph": graph_layers(graph, out_dir, base, url_prefix, bit_depth) if include_graph else [],
    }
