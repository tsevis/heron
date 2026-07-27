"""Render every Heron technique on one image as a labeled contact sheet.

The visual counterpart to ``tests/test_all_instruments.py``: renders all
Instrument x preset combinations and lays them out in a captioned grid, so the
whole rack can be reviewed at a glance against the references (CLAUDE.md §2.10,
"reference validation is visual").

    python scripts/contact_sheet.py photo.jpg --out out/contact.png
    python scripts/contact_sheet.py photo.jpg --ai --tile 320 --out out/contact_ai.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from heron.color import srgb
from heron.core import io
from heron.engine import render_image
from heron.instruments import available_instruments, list_presets

_LABEL_H = 22
_PAD = 4


def _to_pil(linear_rgb: np.ndarray, size: int) -> Image.Image:
    srgb8 = np.clip(srgb.linear_to_srgb(linear_rgb) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(srgb8).resize((size, size), Image.LANCZOS)


def _tile(img: Image.Image, caption: str, size: int) -> Image.Image:
    cell = Image.new("RGB", (size, size + _LABEL_H), (16, 16, 18))
    cell.paste(img, (0, 0))
    draw = ImageDraw.Draw(cell)
    draw.text((4, size + 5), caption[: size // 6], fill=(210, 210, 215))
    return cell


def build_sheet(
    source: Path, out: Path, use_ai: bool, tile: int, seed: int, columns: int
) -> None:
    jobs = [(i, p) for i in available_instruments() for p in list_presets(i)]
    print(f"rendering {len(jobs)} techniques from {source.name} (ai={use_ai})")

    # the source image itself is the first tile, for reference
    cells = [_tile(_to_pil(io.read_image(source), tile), "SOURCE", tile)]

    for idx, (inst, preset) in enumerate(jobs, 1):
        t0 = time.time()
        try:
            result = render_image(source, instrument=inst, preset=preset, use_ai=use_ai, seed=seed)
            cells.append(_tile(_to_pil(result.image, tile), f"{inst}/{preset}", tile))
            print(f"  [{idx:2d}/{len(jobs)}] {inst}/{preset} ({time.time() - t0:.1f}s)")
        except Exception as e:  # a broken technique must not kill the sheet
            print(f"  [{idx:2d}/{len(jobs)}] {inst}/{preset} FAILED: {e}")
            cells.append(_tile(Image.new("RGB", (tile, tile), (60, 20, 20)), f"{inst}/{preset} FAIL", tile))

    rows = (len(cells) + columns - 1) // columns
    cw, ch = tile + _PAD, tile + _LABEL_H + _PAD
    sheet = Image.new("RGB", (columns * cw + _PAD, rows * ch + _PAD), (10, 10, 12))
    for i, cell in enumerate(cells):
        r, c = divmod(i, columns)
        sheet.paste(cell, (_PAD + c * cw, _PAD + r * ch))

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  ({sheet.size[0]}x{sheet.size[1]}, {len(cells)} tiles)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/contact_sheet.png"))
    ap.add_argument("--ai", action="store_true", help="use the AI Layer A")
    ap.add_argument("--tile", type=int, default=280)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--columns", type=int, default=6)
    args = ap.parse_args()
    build_sheet(args.source, args.out, args.ai, args.tile, args.seed, args.columns)


if __name__ == "__main__":
    main()
