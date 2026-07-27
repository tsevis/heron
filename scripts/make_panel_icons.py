"""Generate the Photoshop panel icons (uxp/icons/).

Drawn with rectangles rather than a font: the panel tab renders these at 23 px,
where any real typeface turns to mush and a font dependency buys nothing. The
mark is Heron's wordmark logic — an H whose right stem carries the ember accent
that the app uses for the final N.

Run: .venv/bin/python scripts/make_panel_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "uxp" / "icons"
EMBER = (255, 122, 26, 255)      # the app's accent
LIGHT = (226, 226, 226, 255)     # Photoshop's panel-tab foreground

# Photoshop asks for the nominal size and a 2x variant of each.
SIZES = (23, 46)


def draw_icon(size: int) -> Image.Image:
    """An H at 4x supersampling, then downsampled for clean edges."""
    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    stem = max(2, round(s * 0.16))          # stroke weight
    inset = round(s * 0.20)                 # side margin
    top, bottom = round(s * 0.16), round(s * 0.84)
    left, right = inset, s - inset - stem

    # Crossbar first, stems over it: painted the other way round, the light bar
    # cuts the ember stem into two floating blocks and stops reading as an H.
    mid = round(top + (bottom - top) * 0.48)      # optical centre, not geometric
    d.rectangle([left, mid - stem // 2, right + stem, mid + stem // 2], fill=LIGHT)
    d.rectangle([left, top, left + stem, bottom], fill=LIGHT)
    d.rectangle([right, top, right + stem, bottom], fill=EMBER)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        path = OUT / f"icon-{size}.png"
        draw_icon(size).save(path)
        print(f"wrote {path.relative_to(OUT.parents[1])} ({size}x{size})")


if __name__ == "__main__":
    main()
