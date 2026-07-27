"""Extract a Heron palette from a reference image (CLAUDE.md §0.2).

Derives a luminance-ordered dominant-color ramp from an image, smooths it in
Oklab, and enforces monotonically-increasing lightness, then writes it as a
palette JSON.

    python scripts/extract_palettes.py INPUT.png --name "Ember Branch" \
        --out heron/color/palettes/ember.json

    # preview strips for every image in a folder into a scratch dir:
    python scripts/extract_palettes.py --survey references/ --survey-out /tmp/palette_survey
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from heron.color import oklab, srgb
from heron.core import io


def extract_ramp(linear_rgb: np.ndarray, n_bins: int = 24, n_stops: int = 9) -> np.ndarray:
    """Return (n_stops, 3) sRGB [0,1] stops, dark->bright, monotonic in Oklab L."""
    h, w, _ = linear_rgb.shape
    px = linear_rgb.reshape(-1, 3)
    lab = oklab.linear_srgb_to_oklab(px)
    L = lab[:, 0]

    # equal-count luminance bins -> each bin's median color (robust dominant)
    order = np.argsort(L)
    lab_sorted = lab[order]
    edges = np.linspace(0, len(order), n_bins + 1).astype(int)
    ramp_lab = np.stack([
        np.median(lab_sorted[edges[i]:edges[i + 1]], axis=0)
        for i in range(n_bins) if edges[i + 1] > edges[i]
    ])

    # smooth a,b along the ramp (moving average); keep L, then force it monotone
    k = 3
    pad = np.pad(ramp_lab, ((k, k), (0, 0)), mode="edge")
    smooth = np.stack([pad[i:i + 2 * k + 1].mean(axis=0) for i in range(len(ramp_lab))])
    smooth[:, 0] = np.maximum.accumulate(ramp_lab[:, 0])  # monotonic-L

    # resample to n_stops evenly across the ramp
    idx = np.linspace(0, len(smooth) - 1, n_stops).round().astype(int)
    stops_lab = smooth[idx]
    stops_lin = oklab.oklab_to_linear_srgb(stops_lab)
    stops_srgb = np.clip(srgb.linear_to_srgb(np.clip(stops_lin, 0, 1)), 0, 1)
    return stops_srgb


def ramp_to_palette_dict(name: str, stops_srgb: np.ndarray) -> dict:
    n = len(stops_srgb)
    t = np.linspace(0.0, 1.0, n)
    return {
        "name": name,
        "lightness": "ascending",
        "stops": [
            {"t": round(float(t[i]), 4), "rgb": [int(round(c * 255)) for c in stops_srgb[i]]}
            for i in range(n)
        ],
    }


def write_palette(input_path: Path, name: str, out_path: Path, n_stops: int) -> None:
    img = io.read_image(input_path)
    stops = extract_ramp(img, n_stops=n_stops)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ramp_to_palette_dict(name, stops), indent=2) + "\n")
    print(f"wrote {out_path}  ({name}, {n_stops} stops)")


def survey(folder: Path, out_dir: Path, n_stops: int) -> None:
    """Extract + preview a palette for every image in a folder (for labelling)."""
    from heron.color.palettes import Palette

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for i, ip in enumerate(imgs):
        img = io.read_image(ip)
        stops = extract_ramp(img, n_stops=n_stops)
        pd = ramp_to_palette_dict(f"survey_{i}", stops)
        (out_dir / f"survey_{i}.json").write_text(json.dumps(pd, indent=2) + "\n")
        pal = Palette.from_dict(pd)
        strip = np.broadcast_to(np.linspace(0, 1, 512, dtype=np.float32), (64, 512))
        io.write_image(out_dir / f"survey_{i}_strip.png", pal.apply(strip))
        print(f"[{i}] {ip.name} -> survey_{i}_strip.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", type=Path)
    ap.add_argument("--name", default="Extracted")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-stops", type=int, default=9)
    ap.add_argument("--survey", type=Path, help="folder to survey")
    ap.add_argument("--survey-out", type=Path, default=Path("/tmp/heron_palette_survey"))
    args = ap.parse_args()

    if args.survey:
        survey(args.survey, args.survey_out, args.n_stops)
        return
    if not args.input or not args.out:
        ap.error("provide INPUT and --out, or use --survey")
    write_palette(args.input, args.name, args.out, args.n_stops)


if __name__ == "__main__":
    main()
