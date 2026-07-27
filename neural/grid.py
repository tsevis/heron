"""Seed/strength grid for the ControlNet finisher (pick-the-best-draw workflow).

img2img is a draw: the same physics field re-textures differently per seed, and
the denoise strength trades texture realism against identity drift. This runs
the whole grid with ONE pipeline load and composes a labeled contact sheet.

    python -m neural.grid out/PHYS_TRANS.png photo.jpg --out-dir out/grid \
        --seeds 3 7 21 42 --strengths 0.45 0.52
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from heron.color import srgb
from heron.core import io
from neural.controlnet_finisher import ControlParams, _fit, _pil


def _load_pipeline(p: ControlParams):
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

    from heron.core.device import get_device

    dev = get_device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    controlnet = ControlNetModel.from_single_file(str(p.controlnet), torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        str(p.checkpoint), controlnet=controlnet, torch_dtype=dtype, safety_checker=None
    )
    pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _depth_for(source: Path, shape_hw) -> np.ndarray:
    from heron.engine import _fit_working_res
    from heron.scene import GraphOptions, build_graph

    src = _fit_working_res(io.read_image(source), 1280)
    graph = build_graph(src, image_path=source, opts=GraphOptions(use_ai=True))
    depth = graph.depth
    if depth.shape != shape_hw:
        import cv2

        depth = cv2.resize(depth, (shape_hw[1], shape_hw[0]))
    return depth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("render", type=Path)
    ap.add_argument("source", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("out/grid"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 21, 42])
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.45, 0.52])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--control", type=float, default=0.75)
    ap.add_argument("--tile", type=int, default=380)
    args = ap.parse_args()

    import torch

    p = ControlParams()
    render = io.read_image(args.render)
    depth = _depth_for(args.source, render.shape[:2])

    image = _fit(_pil(render), p.max_side)
    depth_rgb = np.repeat(np.clip(depth, 0, 1)[..., None], 3, axis=-1)
    control = _fit(_pil(srgb.srgb_to_linear(depth_rgb)), p.max_side).resize(image.size)

    print(f"loading pipeline once, then {len(args.seeds) * len(args.strengths)} draws...")
    pipe = _load_pipeline(p)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[tuple[str, Image.Image]] = []
    for strength in args.strengths:
        for seed in args.seeds:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            out = pipe(
                prompt=p.prompt, negative_prompt=p.negative_prompt,
                image=image, control_image=control,
                strength=strength, controlnet_conditioning_scale=args.control,
                num_inference_steps=args.steps, guidance_scale=p.guidance, generator=gen,
            ).images[0]
            name = f"s{strength:.2f}_seed{seed}.png"
            out.save(args.out_dir / name)
            tiles.append((f"str {strength:.2f}  seed {seed}", out))
            print(f"  {name}")

    # labeled contact sheet: seeds across, strengths down
    cols, rows = len(args.seeds), len(args.strengths)
    t, label_h, pad = args.tile, 26, 4
    sheet = Image.new("RGB", (cols * (t + pad) + pad, rows * (t + label_h + pad) + pad), (12, 12, 14))
    draw = ImageDraw.Draw(sheet)
    for i, (caption, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (t + pad)
        y = pad + r * (t + label_h + pad)
        sheet.paste(img.resize((t, t)), (x, y))
        draw.text((x + 4, y + t + 5), caption, fill=(220, 220, 225))
    sheet_path = args.out_dir / "GRID.png"
    sheet.save(sheet_path)
    print(f"wrote {sheet_path}")


if __name__ == "__main__":
    main()
