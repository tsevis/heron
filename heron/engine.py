"""Top-level engine orchestration (UI-agnostic; CLAUDE.md §2.5).

Ties Layer A (Radiance Graph) to the Instrument's B/C/D renderer and returns a
``RenderResult``. Frontends (CLI, webapp, UXP bridge) call ``render_image``; the
engine imports nothing above itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from heron.core import io
from heron.core.graph import RadianceGraph
from heron.instruments import get_engine, load_instrument
from heron.scene import GraphOptions, build_graph
from heron.scene.upsample import upsample_graph


@dataclass
class RenderResult:
    image: np.ndarray            # final linear-light RGB (H,W,3) in [0,1]
    graph: RadianceGraph
    stages: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path, bit_depth: int = 8) -> Path:
        return io.write_image(path, self.image, bit_depth=bit_depth)


def _fit_working_res(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the long side <= max_side (Fast tier; full-res is Phase 4).

    Layer A models and the classical CV (GrabCut, diffusion) do not scale to
    12+ MP in this Phase-0 pipeline, so we evaluate at a bounded working res.
    """
    if max_side <= 0:
        return image
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def render_image(
    input_path: str | Path,
    instrument: str = "thermograph",
    preset: str | None = None,
    overrides: list[str] | None = None,
    use_ai: bool = False,
    seed: int = 0,
    cache: bool = True,
    render_res: int | None = None,
    config: dict | None = None,
) -> RenderResult:
    """Render one image with an Instrument, preset, and dotted overrides.

    ``render_res`` caps the long side of the *output* (Phase 4 §4). Layer A always
    runs at ``layerA.work_res`` — its models do not scale and its channels are
    smooth — and the graph is then lifted edge-aware to the render resolution, so
    Layers B/C/D work on real full-resolution pixels. ``None`` keeps the legacy
    behaviour of rendering at the working resolution.
    """
    # A shareable preset arrives fully resolved (Phase 5 §5.3), so it bypasses
    # instrument+preset merging entirely — that is what makes it reproduce
    # bit-exactly on a machine whose instrument YAML has since moved on.
    cfg = config if config is not None else load_instrument(
        instrument, preset=preset, overrides=overrides)

    a = cfg.get("layerA", {})
    work_res = int(a.get("work_res", 1280))
    source_input = io.read_image(input_path)
    # Layer A's view of the scene, always bounded.
    source = _fit_working_res(source_input, work_res)

    opts = GraphOptions(
        use_ai=use_ai,
        delight_strength=float(a.get("delight_strength", 0.9)),
        depth_tier=str(a.get("depth_tier", "fast")),
        work_res=work_res,
        segment_materials=bool(a.get("segment_materials", True)),
        seed=seed,
        cache=cache,
    )
    graph = build_graph(source, image_path=input_path, opts=opts)

    if render_res is not None:
        # Physics/sensor/art at render resolution; understanding stays low-res.
        source = _fit_working_res(source_input, int(render_res))
        graph = upsample_graph(graph, source)

    engine = get_engine(cfg["engine"])
    image, stages, meta = engine(graph, source, cfg, seed=seed)

    meta.update({
        "instrument": instrument,
        "preset": preset,
        "use_ai": use_ai,
        "render_px": [int(source.shape[1]), int(source.shape[0])],
        "graph_px": [int(graph.meta.get("upsampled_from", graph.shape)[1]),
                     int(graph.meta.get("upsampled_from", graph.shape)[0])],
        "source": str(input_path),
        "disclaimer": "Artistic simulation — not real thermography or measurement.",
    })
    return RenderResult(image=image, graph=graph, stages=stages, meta=meta)


def dump_graph_channels(graph: RadianceGraph, out_dir: str | Path) -> list[Path]:
    """Write each Radiance Graph channel as an inspectable image (§2.7)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    written.append(io.write_gray(out_dir / "depth.png", graph.depth))
    written.append(io.write_gray(out_dir / "saliency.png", graph.saliency))
    written.append(io.write_gray(out_dir / "matte.png", graph.matte))
    written.append(io.write_image(out_dir / "albedo.png", graph.albedo))
    normals_vis = (graph.normals * 0.5 + 0.5).astype(np.float32)
    written.append(io.write_gray(out_dir / "normals.png", normals_vis.mean(axis=-1)))
    # material id as a scaled gray so regions are distinguishable
    n = max(len(graph.material_names), 1)
    written.append(io.write_gray(out_dir / "materials.png", graph.material_ids.astype(np.float32) / n))
    return written
