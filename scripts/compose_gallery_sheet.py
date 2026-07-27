"""Compose the README's 8-instrument rack contact sheet from gallery/renders/."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "gallery/renders"
OUT = R / "instrument_rack.png"

CELLS = [
    ("tyla_thermograph.png", "THERMOGRAPH", "emission"),
    ("john_fluoroscope.png", "FLUOROSCOPE", "attenuation"),
    ("trophy_intensifier.png", "INTENSIFIER TUBE", "amplification"),
    ("bowie_nir.png", "NIR CAMERA", "reflectance"),
    ("kaka_kirlian.png", "KIRLIAN PLATE", "corona"),
    ("john_schlieren.png", "SCHLIEREN BENCH", "density gradient"),
    ("trophy_lumen.png", "LUMEN CHAMBER", "bioluminescence"),
    ("coins_spectroscope.png", "SPECTROSCOPE", "spectral"),
]

TILE, PAD, LABEL_H = 300, 6, 46
COLS = 4


def _font(size, bold=False):
    names = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else
             ["/System/Library/Fonts/Supplemental/Arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build():
    rows = (len(CELLS) + COLS - 1) // COLS
    W = COLS * (TILE + PAD) + PAD
    H = rows * (TILE + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (13, 13, 16))
    d = ImageDraw.Draw(sheet)
    f_title = _font(15, bold=True)
    f_sub = _font(11)

    for i, (fname, title, subtitle) in enumerate(CELLS):
        r, c = divmod(i, COLS)
        x = PAD + c * (TILE + PAD)
        y = PAD + r * (TILE + LABEL_H + PAD)
        img = Image.open(R / fname).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((TILE, TILE), Image.LANCZOS)
        sheet.paste(img, (x, y))
        d.text((x + 2, y + TILE + 6), title, font=f_title, fill=(235, 235, 240))
        d.text((x + 2, y + TILE + 26), subtitle, font=f_sub, fill=(140, 140, 148))

    sheet.save(OUT)
    print(f"wrote {OUT}  ({W}x{H})")


if __name__ == "__main__":
    build()
