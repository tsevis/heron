"""Driver for the Heron neural finishers (CLAUDE.md §5.1 — outside the engine).

    # ControlNet img2img through a local ComfyUI (needs ComfyUI running)
    python -m neural.run comfy out/SAMPLE3D.png photo.jpg --out out/finish_comfy.png

    # Neural style transfer from a reference painting
    python -m neural.run style out/SAMPLE3D.png reference_painting.png --out out/finish_style.png

    # Training (needs a dataset — see neural/datasets.py)
    python -m neural.run train-cyclegan  --epochs 100
    python -m neural.run train-pix2pixhd --epochs 100
    python -m neural.run datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from heron.core import io


def _graph_depth(source: Path) -> np.ndarray:
    """Reuse Heron's cached Radiance Graph depth as the ControlNet signal."""
    from heron.engine import _fit_working_res
    from heron.scene import GraphOptions, build_graph

    src = _fit_working_res(io.read_image(source), 1280)
    graph = build_graph(src, image_path=source, opts=GraphOptions(use_ai=True))
    return graph.depth


def cmd_comfy(args) -> None:
    from neural import comfy_finisher as cf

    render = io.read_image(args.render)
    depth = _graph_depth(Path(args.source))
    if depth.shape != render.shape[:2]:
        import cv2

        depth = cv2.resize(depth, (render.shape[1], render.shape[0]))
    params = cf.FinishParams(denoise=args.denoise, control_strength=args.control, seed=args.seed)
    out = cf.finish(render, depth, params)
    if out is None:
        raise SystemExit("ComfyUI finisher failed — is ComfyUI running on 127.0.0.1:8188?")
    io.write_image(args.out, out)
    print(f"wrote {args.out}")


def cmd_controlnet(args) -> None:
    from neural import controlnet_finisher as cn

    render = io.read_image(args.render)
    depth = _graph_depth(Path(args.source))
    if depth.shape != render.shape[:2]:
        import cv2

        depth = cv2.resize(depth, (render.shape[1], render.shape[0]))
    out = cn.finish(render, depth, cn.ControlParams(
        strength=args.strength, control_scale=args.control, steps=args.steps, seed=args.seed))
    if out is None:
        raise SystemExit("ControlNet finisher failed")
    io.write_image(args.out, out)
    print(f"wrote {args.out}")


def cmd_style(args) -> None:
    from neural.style_transfer import StyleParams, stylize

    content = io.read_image(args.render)
    style = io.read_image(args.style)
    p = StyleParams(mode=args.mode, steps=args.steps, style_weight=args.style_weight, max_side=args.max_side)
    out = stylize(content, style, p)
    io.write_image(args.out, out)
    print(f"wrote {args.out}")


def cmd_train_cyclegan(args) -> None:
    from torch.utils.data import DataLoader

    from neural.cyclegan import CycleGANConfig, CycleGANTrainer
    from neural.datasets import UnpairedThermalDataset

    ds = UnpairedThermalDataset(args.data, size=args.size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    CycleGANTrainer(CycleGANConfig(epochs=args.epochs)).train(loader)


def cmd_train_pix2pixhd(args) -> None:
    from torch.utils.data import DataLoader

    from neural.datasets import PairedThermalDataset
    from neural.pix2pixhd import Pix2PixHDConfig, Pix2PixHDTrainer

    ds = PairedThermalDataset(args.data, size=args.size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    Pix2PixHDTrainer(Pix2PixHDConfig(epochs=args.epochs)).train(loader)


def cmd_datasets(args) -> None:
    from neural.datasets import describe

    print(describe())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("comfy", help="ControlNet img2img via local ComfyUI")
    c.add_argument("render"); c.add_argument("source")
    c.add_argument("--out", default="out/finish_comfy.png")
    c.add_argument("--denoise", type=float, default=0.45)
    c.add_argument("--control", type=float, default=0.8)
    c.add_argument("--seed", type=int, default=7)
    c.set_defaults(func=cmd_comfy)

    n = sub.add_parser("controlnet", help="SDXL depth-ControlNet img2img (local models)")
    n.add_argument("render"); n.add_argument("source")
    n.add_argument("--out", default="out/finish_controlnet.png")
    n.add_argument("--strength", type=float, default=0.45)
    n.add_argument("--control", type=float, default=0.8)
    n.add_argument("--steps", type=int, default=26)
    n.add_argument("--seed", type=int, default=7)
    n.set_defaults(func=cmd_controlnet)

    s = sub.add_parser("style", help="neural style transfer")
    s.add_argument("render"); s.add_argument("style")
    s.add_argument("--out", default="out/finish_style.png")
    s.add_argument("--mode", choices=["gatys", "adain"], default="gatys")
    s.add_argument("--steps", type=int, default=240)
    s.add_argument("--style-weight", dest="style_weight", type=float, default=1.2e6)
    s.add_argument("--max-side", dest="max_side", type=int, default=768)
    s.set_defaults(func=cmd_style)

    t1 = sub.add_parser("train-cyclegan")
    t1.add_argument("--data", default="data/thermal/unpaired")
    t1.add_argument("--epochs", type=int, default=100)
    t1.add_argument("--batch", type=int, default=1)
    t1.add_argument("--size", type=int, default=256)
    t1.set_defaults(func=cmd_train_cyclegan)

    t2 = sub.add_parser("train-pix2pixhd")
    t2.add_argument("--data", default="data/thermal/paired")
    t2.add_argument("--epochs", type=int, default=100)
    t2.add_argument("--batch", type=int, default=1)
    t2.add_argument("--size", type=int, default=512)
    t2.set_defaults(func=cmd_train_pix2pixhd)

    d = sub.add_parser("datasets"); d.set_defaults(func=cmd_datasets)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
