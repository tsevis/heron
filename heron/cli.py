"""Heron command-line interface (CLAUDE.md §0.6).

    python cli.py render input.jpg --instrument thermograph --preset flir_field --out out.png
    python cli.py render input.jpg --no-ai --seed 7 --dump-graph
    python cli.py graph input.jpg --out-dir graph_channels/
    python cli.py palette list
    python cli.py palette preview ironbow --out ironbow.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

app = typer.Typer(add_completion=False, help="Heron — radiometric imaging engine.")
palette_app = typer.Typer(help="Inspect and preview palettes.")
app.add_typer(palette_app, name="palette")


@app.command()
def render(
    input: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source image."),
    instrument: str = typer.Option("thermograph", "--instrument", "-i"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
    out: Path = typer.Option(Path("out.png"), "--out", "-o"),
    ai: bool = typer.Option(False, "--ai/--no-ai", help="Enable Layer A AI models."),
    seed: int = typer.Option(0, "--seed"),
    set_: List[str] = typer.Option(None, "--set", "-s", help="Override, e.g. layerC.netd_mk=40"),
    bit_depth: int = typer.Option(8, "--bit-depth", help="8 or 16."),
    dump_graph: bool = typer.Option(False, "--dump-graph", help="Also write Radiance Graph channels."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore/skip the graph cache."),
    render_res: Optional[int] = typer.Option(
        None, "--render-res",
        help="Long side of the OUTPUT in pixels (e.g. 5120). Scene understanding "
             "stays at layerA.work_res and is lifted edge-aware; physics, sensor "
             "and art direction run at this resolution. Omit for work_res.",
    ),
):
    """Render an image through an Instrument."""
    from heron.engine import dump_graph_channels, render_image

    result = render_image(
        input,
        instrument=instrument,
        preset=preset,
        overrides=list(set_ or []),
        use_ai=ai,
        seed=seed,
        cache=not no_cache,
        render_res=render_res,
    )
    result.save(out, bit_depth=bit_depth)
    span = result.meta.get("span_c")
    span_txt = f", span {span[0]}..{span[1]} C" if span else ""
    px = result.meta.get("render_px")
    graph_px = result.meta.get("graph_px")
    res_txt = ""
    if px and graph_px and px != graph_px:
        res_txt = f", {px[0]}x{px[1]} from a {graph_px[0]}x{graph_px[1]} scene graph"
    elif px:
        res_txt = f", {px[0]}x{px[1]}"
    typer.echo(f"wrote {out}  ({result.meta.get('palette')}{span_txt}{res_txt})")

    if dump_graph:
        d = out.parent / f"{out.stem}_graph"
        dump_graph_channels(result.graph, d)
        typer.echo(f"dumped graph channels -> {d}/")



@app.command("render-video")
def render_video_cmd(
    input: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source video."),
    out: Path = typer.Option(Path("out.mp4"), "--out", "-o"),
    instrument: str = typer.Option("thermograph", "--instrument", "-i"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
    ai: bool = typer.Option(False, "--ai/--no-ai", help="Enable Layer A AI models."),
    seed: int = typer.Option(7, "--seed", help="Fixed for the whole clip: the FPN field."),
    set_: List[str] = typer.Option(None, "--set", "-s", help="Override, e.g. layerC.netd_mk=40"),
    render_res: Optional[int] = typer.Option(None, "--render-res", help="Output long side."),
    keyframe_interval: int = typer.Option(0, "--keyframe-interval",
                                          help="Rebuild the scene graph every N frames (0 = only on shot change)."),
    agc_ema: float = typer.Option(0.85, "--agc-ema", help="Window smoothing; higher holds harder."),
    shot_threshold: float = typer.Option(0.18, "--shot-threshold"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames"),
):
    """Render a video clip, stabilised across frames.

    The scene graph is built once per shot and warped forward by optical flow
    rather than re-estimated, the FPN field is fixed for the clip while the NETD
    term re-rolls per frame, and the auto window is eased rather than recomputed.
    """
    from heron.video import VideoOptions, render_video

    def show(done: int, total: int, stage: str) -> None:
        if total:
            typer.echo(f"\r  {done}/{total} frames", nl=False)
        else:
            typer.echo(f"\r  {done} frames", nl=False)

    stats = render_video(
        input, out,
        VideoOptions(
            instrument=instrument, preset=preset, overrides=tuple(set_ or []),
            seed=seed, use_ai=ai, render_res=render_res,
            keyframe_interval=keyframe_interval, agc_ema=agc_ema,
            shot_threshold=shot_threshold, max_frames=max_frames,
        ),
        progress=show,
    )
    typer.echo("")
    typer.echo(
        f"wrote {stats['output']}  ({stats['frames']} frames, {stats['seconds']}s, "
        f"{stats['fps']} fps, {stats['graph_rebuilds']} graph build(s), "
        f"max window step {stats['max_window_step_c']} C)"
    )



@app.command("preset-export")
def preset_export_cmd(
    out: Path = typer.Argument(..., help="Destination .heron.json file."),
    instrument: str = typer.Option("thermograph", "--instrument", "-i"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
    set_: List[str] = typer.Option(None, "--set", "-s", help="Override baked into the preset."),
    name: Optional[str] = typer.Option(None, "--name"),
    author: str = typer.Option("", "--author"),
    source_image: str = typer.Option("", "--source-image", help="Photo it was tuned on."),
    mode: str = typer.Option("physics-only", "--mode", help="physics-only | neural"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    notes: str = typer.Option("", "--notes"),
):
    """Export a self-contained, shareable preset.

    Every layer value is resolved and the palette ramp is inlined, so the file
    reproduces its render on a machine that has neither this instrument YAML nor
    this palette.
    """
    from heron.presets import export_preset, write_preset

    bundle = export_preset(instrument, preset, overrides=list(set_ or []), name=name,
                           author=author, source_image=source_image, mode=mode,
                           seed=seed, notes=notes)
    write_preset(out, bundle)
    layers = sum(len(v) for v in bundle.layers.values())
    typer.echo(f"wrote {out}  ({bundle.instrument}/{bundle.name}, {layers} resolved values, "
               f"palette {'inlined' if bundle.palette else 'none'})")


@app.command("preset-render")
def preset_render_cmd(
    preset_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    input: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("out.png"), "--out", "-o"),
    ai: bool = typer.Option(False, "--ai/--no-ai"),
    seed: int = typer.Option(0, "--seed"),
    render_res: Optional[int] = typer.Option(None, "--render-res"),
    bit_depth: int = typer.Option(8, "--bit-depth"),
):
    """Render using a shared preset file rather than a built-in preset."""
    from heron.engine import render_image
    from heron.presets import load_preset

    bundle = load_preset(preset_file)
    result = render_image(input, instrument=bundle.instrument, use_ai=ai, seed=seed,
                          render_res=render_res, config=bundle.config())
    result.save(out, bit_depth=bit_depth)
    prov = bundle.provenance or {}
    who = f" by {prov['author']}" if prov.get("author") else ""
    typer.echo(f"wrote {out}  ({bundle.instrument}/{bundle.name}{who}, "
               f"mode {prov.get('mode', 'unknown')})")


@app.command()
def graph(
    input: Path = typer.Argument(..., exists=True, dir_okay=False),
    out_dir: Path = typer.Option(Path("graph_channels"), "--out-dir"),
    ai: bool = typer.Option(False, "--ai/--no-ai"),
    seed: int = typer.Option(0, "--seed"),
):
    """Build and export the Radiance Graph channels for an image."""
    from heron.core import io
    from heron.engine import dump_graph_channels
    from heron.scene import GraphOptions, build_graph

    source = io.read_image(input)
    g = build_graph(source, image_path=input, opts=GraphOptions(use_ai=ai, seed=seed))
    written = dump_graph_channels(g, out_dir)
    typer.echo(f"materials: {g.material_names}")
    typer.echo(f"wrote {len(written)} channels -> {out_dir}/")


@palette_app.command("list")
def palette_list():
    """List available palettes and their lightness intent."""
    from heron.color import palettes

    for name in palettes.available_palettes():
        p = palettes.load_palette(name)
        mono = "ok" if p.is_lightness_monotonic() else "NON-MONOTONIC"
        typer.echo(f"  {name:20s} {p.lightness:11s} L-{mono}")


@palette_app.command("preview")
def palette_preview(
    name: str = typer.Argument(...),
    out: Path = typer.Option(Path("palette.png"), "--out", "-o"),
    width: int = typer.Option(512, "--width"),
    height: int = typer.Option(64, "--height"),
):
    """Write a horizontal strip preview of a palette."""
    from heron.color import palettes
    from heron.core import io

    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    strip = np.broadcast_to(ramp, (height, width))
    img = palettes.load_palette(name).apply(strip)
    io.write_image(out, img)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
