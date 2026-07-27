"""Build the README's example gallery from gallery/source/ (CLAUDE.md §2.10).

Each source photo is rendered through an instrument chosen to suit it, plus one
hero image (TYLA under dramatic colored stage light — the de-lighting thesis)
gets the full physics + neural hybrid treatment.

    python scripts/render_gallery.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from heron.color import srgb
from heron.core import io
from heron.engine import render_image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "gallery/source"
OUT = ROOT / "gallery/renders"
OUT.mkdir(parents=True, exist_ok=True)

# (source file, instrument, preset, output name)
JOBS = [
    ("tyla.jpg", "thermograph", "translucent_glow", "tyla_thermograph.png"),
    ("tyla.jpg", "lumen", "gfp", "tyla_lumen.png"),
    ("kaka.jpg", "kirlian", "fingertip_storm", "kaka_kirlian.png"),
    ("john.png", "fluoroscope", "cold_blue_nude", "john_fluoroscope.png"),
    ("trophy.jpg", "intensifier", "gen2_p22", "trophy_intensifier.png"),
    ("coins.jpg", "spectroscope", "prism_split", "coins_spectroscope.png"),
    ("john.png", "nir", "aerochrome", "john_nir.png"),
    ("john.png", "schlieren", "breath_heat_plume", "john_schlieren.png"),
]

HERO_SOURCE = "tyla.jpg"
HERO_OUT = OUT / "hero_hybrid.png"


def render_all() -> None:
    for src_name, instrument, preset, out_name in JOBS:
        t0 = time.time()
        photo = SRC / src_name
        result = render_image(photo, instrument=instrument, preset=preset, use_ai=True, seed=7)
        out_path = OUT / out_name
        result.save(out_path)
        print(f"  {out_name:28s} <- {src_name:12s} [{instrument}/{preset}]  {time.time()-t0:.0f}s")


def render_hero() -> None:
    import cv2
    import torch

    from neural.controlnet_finisher import ControlParams, _fit, _pil
    from neural.grid import _load_pipeline

    print("hero: physics…")
    photo = SRC / HERO_SOURCE
    result = render_image(photo, instrument="thermograph", preset="translucent_glow", use_ai=True, seed=7)

    print("hero: neural finish…")
    params = ControlParams()
    pipe = _load_pipeline(params)
    image = _fit(_pil(result.image), params.max_side)
    depth = result.graph.depth
    if depth.shape != result.image.shape[:2]:
        depth = cv2.resize(depth, (result.image.shape[1], result.image.shape[0]))
    depth_rgb = np.repeat(np.clip(depth, 0, 1)[..., None], 3, axis=-1)
    control = _fit(_pil(srgb.srgb_to_linear(depth_rgb)), params.max_side).resize(image.size)
    gen = torch.Generator(device="cpu").manual_seed(3)
    hybrid = pipe(
        prompt=params.prompt, negative_prompt=params.negative_prompt,
        image=image, control_image=control,
        strength=0.52, controlnet_conditioning_scale=0.75,
        num_inference_steps=24, guidance_scale=params.guidance, generator=gen,
    ).images[0]
    hybrid.save(HERO_OUT)
    print(f"  wrote {HERO_OUT}")

    # a before/after strip for the top of the README
    src_img = Image.open(photo).convert("RGB")
    w, h = src_img.size
    s = min(w, h)
    src_sq = src_img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((640, 640))
    hero_sq = hybrid.resize((640, 640))
    strip = Image.new("RGB", (640 * 2 + 8, 640), (10, 10, 12))
    strip.paste(src_sq, (0, 0))
    strip.paste(hero_sq, (648, 0))
    strip.save(OUT / "hero_before_after.png")
    print(f"  wrote {OUT / 'hero_before_after.png'}")


if __name__ == "__main__":
    render_hero()
    render_all()
