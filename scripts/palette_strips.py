"""Palette sample strips — every camera against every palette (CLAUDE.md §2.10).

One strip per Instrument, showing the same photo rendered through all 16 palette
ramps. The physics graph is built once per photo and the diffusion pipeline is
loaded once for the whole run, so the cost is dominated by the neural pass.

Budget accordingly: a full 113-frame run measured ~58 s/frame (≈110 min) with
RealVisXL fp32 on an M1 Ultra while another server held a second model — the
app's ~25 s/frame figure is for the lighter juggernaut checkpoint with the GPU
to itself. ``--no-neural`` is ~1 s/frame, i.e. seconds for a whole camera.

Individual frames are cached under ``out/strips/tiles/``: re-running skips any
frame already on disk, so an interrupted run resumes instead of starting over.
``tiles/manifest.json`` records whether each camera was rendered physics-only
or neural-finished, and the strip header states it.

    python scripts/palette_strips.py                  # all cameras
    python scripts/palette_strips.py --only thermograph,kirlian
    python scripts/palette_strips.py --draft           # 18 steps @ 832px
    python scripts/palette_strips.py --no-neural       # physics only, fast
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from heron.color import palettes as _palettes
from heron.color import srgb
from heron.engine import render_image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "gallery/source"
OUT = ROOT / "out/strips"
TILES = OUT / "tiles"

# (camera, preset, source photo). Presets are chosen to be palette-driven: the
# NIR aerochrome and spectroscope paths synthesize colour directly and ignore
# the 1-D ramp, so they would render 16 identical frames.
JOBS = [
    ("thermograph", "translucent_glow", "tyla.jpg"),
    ("fluoroscope", "cold_blue_nude", "john.png"),
    ("intensifier", "gen2_p22", "trophy.jpg"),
    ("nir", "wood_effect_grove", "john.png"),
    ("kirlian", "classic_leaf", "kaka.jpg"),
    ("schlieren", "breath_heat_plume", "bowie.jpg"),
    ("lumen", "gfp", "trophy.jpg"),
    ("spectroscope", "prism_split", "coins.jpg"),
]
# instruments whose renderer bypasses the palette LUT entirely
NO_PALETTE = {"spectroscope"}

TILE, PAD, LABEL_H, COLS = 240, 5, 24, 8
NEURAL_STRENGTH, NEURAL_SEED = 0.52, 3


def _font(size: int, bold: bool = False):
    name = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf")
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def _pil_of(linear: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(srgb.linear_to_srgb(linear) * 255, 0, 255).astype(np.uint8))


def _load_pipe(checkpoint: str):
    from neural.controlnet_finisher import ControlParams
    from neural.grid import _load_pipeline

    ckpt = Path("~/AI/ComfyUI/models/checkpoints").expanduser() / checkpoint
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    params = ControlParams(checkpoint=ckpt)
    print(f"loading {checkpoint} (once for the whole run)…")
    return _load_pipeline(params), params


def _neural(pipe, params, render_linear, depth, prompt: str, draft: bool):
    import cv2
    import torch

    from neural.controlnet_finisher import _fit, _pil

    max_side = min(params.max_side, 832) if draft else params.max_side
    image = _fit(_pil(render_linear), max_side)
    if depth.shape != render_linear.shape[:2]:
        depth = cv2.resize(depth, (render_linear.shape[1], render_linear.shape[0]))
    depth_rgb = np.repeat(np.clip(depth, 0, 1)[..., None], 3, axis=-1)
    control = _fit(_pil(srgb.srgb_to_linear(depth_rgb)), max_side).resize(image.size)
    gen = torch.Generator(device="cpu").manual_seed(NEURAL_SEED)
    return pipe(
        prompt=prompt or params.prompt, negative_prompt=params.negative_prompt,
        image=image, control_image=control, strength=NEURAL_STRENGTH,
        controlnet_conditioning_scale=0.75,
        num_inference_steps=18 if draft else 24,
        guidance_scale=params.guidance, generator=gen,
    ).images[0]


def _square(img: Image.Image, size: int) -> Image.Image:
    """Fit the whole frame into a square tile (letterboxed).

    A centre crop would cut the subject out of tall/wide sources — and these
    strips exist to compare palettes over the same composition, so the frame
    matters more than filling the tile.
    """
    fitted = img.copy()
    fitted.thumbnail((size, size), Image.LANCZOS)
    tile = Image.new("RGB", (size, size), (13, 13, 16))
    tile.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return tile


def _manifest() -> dict:
    """Per-camera provenance (physics-only vs neural-finished).

    Without this the strips are indistinguishable: a physics-only camera and a
    neural-finished one look equally plausible, and the neural pass can add
    structure the physics never produced (it invented facial detail under the
    Kirlian corona), so which is which has to be recorded, not remembered.
    """
    import json

    try:
        return json.loads((TILES / "manifest.json").read_text())
    except Exception:
        return {}


def _record(camera: str, mode: str) -> None:
    import json

    m = _manifest()
    m[camera] = mode
    (TILES / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True))


def build_strip(camera: str, preset: str, photo: str, pals: list[str]) -> Path:
    mode = _manifest().get(camera, "neural finish")
    header = f"{camera.upper()}  ·  {preset}  ·  {photo}  ·  {mode}"

    # A single-sample camera (no palette dimension) gets one large frame at its
    # own aspect — squaring it would letterbox a 2:1 photo into dead space.
    if len(pals) == 1:
        src = Image.open(TILES / f"{camera}_{pals[0]}.png").convert("RGB")
        w = 900
        img = src.resize((w, max(1, round(w * src.height / src.width))), Image.LANCZOS)
        sheet = Image.new("RGB", (w + 2 * PAD, 34 + img.height + LABEL_H + PAD), (13, 13, 16))
        d = ImageDraw.Draw(sheet)
        d.text((PAD + 2, 9), header, font=_font(14, bold=True), fill=(235, 235, 240))
        sheet.paste(img, (PAD, 34))
        d.text((PAD + 1, 34 + img.height + 5),
               "no palette — this engine synthesizes colour per wavelength",
               font=_font(11), fill=(150, 150, 158))
        out = OUT / f"strip_{camera}.png"
        sheet.save(out, optimize=True)
        return out

    rows = (len(pals) + COLS - 1) // COLS
    W = max(min(len(pals), COLS) * (TILE + PAD) + PAD, 560)  # keep the header unclipped
    H = 34 + rows * (TILE + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (13, 13, 16))
    d = ImageDraw.Draw(sheet)
    d.text((PAD + 2, 9), header, font=_font(14, bold=True), fill=(235, 235, 240))
    for i, pal in enumerate(pals):
        r, c = divmod(i, COLS)
        x = PAD + c * (TILE + PAD)
        y = 34 + r * (TILE + LABEL_H + PAD)
        tile = TILES / f"{camera}_{pal}.png"
        if not tile.exists():
            continue
        sheet.paste(_square(Image.open(tile).convert("RGB"), TILE), (x, y))
        d.text((x + 1, y + TILE + 5), pal, font=_font(11), fill=(150, 150, 158))
    out = OUT / f"strip_{camera}.png"
    sheet.save(out, optimize=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="RealVisXL_V5.0_fp32.safetensors")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--only", help="comma-separated camera names")
    ap.add_argument("--no-neural", action="store_true", help="physics only (fast preview)")
    args = ap.parse_args()

    TILES.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in JOBS if not args.only or j[0] in args.only.split(",")]
    all_pals = _palettes.available_palettes()

    pipe = params = None
    if not args.no_neural:
        pipe, params = _load_pipe(args.checkpoint)
        from webapp.server import ENGINE_PROMPTS
    else:
        ENGINE_PROMPTS = {}

    total = sum(1 if c in NO_PALETTE else len(all_pals) for c, _, _ in jobs)
    done = 0
    t_start = time.time()
    for camera, preset, photo in jobs:
        _record(camera, "physics only" if args.no_neural else "neural finish")
        pals = ["—"] if camera in NO_PALETTE else all_pals
        for pal in pals:
            done += 1
            tile = TILES / f"{camera}_{pal}.png"
            if tile.exists():
                print(f"[{done}/{total}] {camera}/{pal}: cached")
                continue
            t0 = time.time()
            overrides = [] if camera in NO_PALETTE else [f"layerC.palette={pal}"]
            result = render_image(SRC / photo, instrument=camera, preset=preset,
                                  overrides=overrides, use_ai=True, seed=7)
            img = _pil_of(result.image)
            if pipe is not None:
                engine = str(result.meta.get("engine", "thermal"))
                img = _neural(pipe, params, result.image, result.graph.depth,
                              ENGINE_PROMPTS.get(engine, ""), args.draft)
            img.save(tile)
            eta = (time.time() - t_start) / done * (total - done) / 60
            print(f"[{done}/{total}] {camera}/{pal}: {time.time()-t0:.0f}s  (eta {eta:.0f} min)")
        print("  ->", build_strip(camera, preset, photo, pals))

    print(f"\nall strips in {OUT}  ({(time.time()-t_start)/60:.0f} min)")


if __name__ == "__main__":
    main()
