"""Phase-0 acceptance sheet (CLAUDE.md §Phase 0 acceptance, §2.10).

Builds the blind-comparison sheet the project defines as its thesis test:
each test photo as a row of  SOURCE | HERON physics | HERON hybrid (neural).
Reference validation is visual: the sheet is meant to be judged by eye against
the criterion "the portrait cases are not instantly identifiable as a filter".

    python scripts/acceptance_sheet.py --out out/ACCEPTANCE_SHEET.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# `neural/` lives at the repo root and is not pip-installed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from heron.color import srgb
from heron.core import io
from heron.engine import render_image

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = [
    ROOT / "tests/sample_images/portrait_john.jpg",
    ROOT / "tests/sample_images/portrait_1.jpg",
    ROOT / "tests/sample_images/portrait_2.jpg",
]

COLUMNS = ["SOURCE", "HERON — PHYSICS", "HERON — HYBRID"]


def _square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size), Image.LANCZOS)


def _pil_of(linear: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(srgb.linear_to_srgb(linear) * 255, 0, 255).astype(np.uint8))


def build(out_path: Path, tile: int, neural_strength: float, neural_seed: int) -> None:
    import torch

    from neural.controlnet_finisher import ControlParams, _fit, _pil
    from neural.grid import _load_pipeline

    print("loading neural pipeline once…")
    params = ControlParams()
    pipe = _load_pipeline(params)

    rows: list[list[Image.Image]] = []
    for i, photo in enumerate(PHOTOS):
        t0 = time.time()
        print(f"[{i + 1}/{len(PHOTOS)}] {photo.name}: physics…")
        result = render_image(photo, instrument="thermograph",
                              preset="translucent_glow", use_ai=True, seed=7)
        physics = result.image

        print(f"   neural finish…")
        image = _fit(_pil(physics), params.max_side)
        depth = result.graph.depth
        import cv2

        if depth.shape != physics.shape[:2]:
            depth = cv2.resize(depth, (physics.shape[1], physics.shape[0]))
        depth_rgb = np.repeat(np.clip(depth, 0, 1)[..., None], 3, axis=-1)
        control = _fit(_pil(srgb.srgb_to_linear(depth_rgb)), params.max_side).resize(image.size)
        gen = torch.Generator(device="cpu").manual_seed(neural_seed)
        hybrid = pipe(
            prompt=params.prompt, negative_prompt=params.negative_prompt,
            image=image, control_image=control,
            strength=neural_strength, controlnet_conditioning_scale=0.75,
            num_inference_steps=24, guidance_scale=params.guidance, generator=gen,
        ).images[0]

        rows.append([
            _square(Image.open(photo).convert("RGB"), tile),
            _square(_pil_of(physics), tile),
            _square(hybrid, tile),
        ])
        print(f"   row done in {time.time() - t0:.0f}s")

    # --- compose ---
    pad, header_h, footer_h = 6, 34, 30
    cols = len(COLUMNS)
    W = cols * (tile + pad) + pad
    H = header_h + len(rows) * (tile + pad) + pad + footer_h
    sheet = Image.new("RGB", (W, H), (12, 12, 14))
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(COLUMNS):
        d.text((pad + c * (tile + pad) + 8, 10), name, fill=(235, 235, 240))
    for r, row in enumerate(rows):
        y = header_h + pad + r * (tile + pad)
        for c, img in enumerate(row):
            sheet.paste(img, (pad + c * (tile + pad), y))
    d.text((pad + 4, H - footer_h + 8),
           "HERON acceptance sheet — outputs are artistic simulation, not measurement.",
           fill=(150, 150, 158))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"wrote {out_path}  ({W}x{H})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "out/ACCEPTANCE_SHEET.png")
    ap.add_argument("--tile", type=int, default=300)
    ap.add_argument("--strength", type=float, default=0.52)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    build(args.out, args.tile, args.strength, args.seed)


if __name__ == "__main__":
    main()
