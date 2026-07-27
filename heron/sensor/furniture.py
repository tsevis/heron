"""On-screen camera furniture (CLAUDE.md §0.5, §3.3).

Real thermal captures are screenshots of an *instrument*: a centre-spot reticle
with its temperature readout, a palette scale bar with span end-points, and a
status line. It is one of the strongest authenticity cues in a genuine FLIR or
Teledyne frame — and it is drawn by the camera, so it sits crisply on top of an
otherwise soft, low-resolution image.

Rendered as a separate removable overlay (§3.3: "all rendered vector, all
removable") so the plain radiometric image is always available underneath.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from heron.color import palettes, srgb


@dataclass(frozen=True)
class FurnitureParams:
    spot: bool = True            # centre reticle + temperature readout
    scale_bar: bool = True       # palette ramp with span end-points
    status: str = ""             # optional status line (e.g. camera name)
    emissivity_note: bool = True # real units readout under the bar
    opacity: float = 1.0


def _put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(
    rgb_linear: np.ndarray,
    span_c: tuple[float, float],
    palette_name: str,
    spot_c: float | None = None,
    params: FurnitureParams | None = None,
) -> np.ndarray:
    """Overlay camera furniture on a linear-light RGB image."""
    p = params or FurnitureParams()
    base = np.asarray(rgb_linear, dtype=np.float32)
    h, w = base.shape[:2]
    s = max(h, w) / 640.0                      # scale UI to the frame
    white = (255, 255, 255)

    # OpenCV's text/AA drawing needs 8-bit; the camera draws its overlay in
    # display space anyway, so compose the furniture in sRGB-8 and convert back.
    out = np.clip(srgb.linear_to_srgb(base) * 255, 0, 255).astype(np.uint8)

    if p.spot:
        cy, cx = h // 2, w // 2
        arm = int(14 * s)
        gap = int(4 * s)
        t = max(1, int(1.5 * s))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(out, (cx + dx * gap, cy + dy * gap),
                     (cx + dx * arm, cy + dy * arm), white, t, cv2.LINE_AA)
        if spot_c is not None:
            _put(out, f"{spot_c:.1f}C", (cx + int(18 * s), cy - int(10 * s)), 0.6 * s, white, t)

    if p.scale_bar:
        bar_w = int(18 * s)
        bar_h = int(h * 0.42)
        x1 = w - int(30 * s)
        x0 = x1 - bar_w
        y0 = int(h * 0.12)
        ramp = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)[:, None]
        ramp = np.repeat(ramp, bar_w, axis=1)
        try:
            bar = palettes.load_palette(palette_name).apply(ramp)
        except Exception:
            bar = np.repeat(ramp[..., None], 3, axis=-1)
        bar8 = np.clip(srgb.linear_to_srgb(bar) * 255, 0, 255).astype(np.uint8)
        out[y0:y0 + bar_h, x0:x1] = bar8
        cv2.rectangle(out, (x0 - 1, y0 - 1), (x1, y0 + bar_h), white, max(1, int(s)), cv2.LINE_AA)
        _put(out, f"{span_c[1]:.1f}", (x0 - int(4 * s), y0 - int(6 * s)), 0.5 * s, white, max(1, int(s)))
        _put(out, f"{span_c[0]:.1f}", (x0 - int(4 * s), y0 + bar_h + int(16 * s)), 0.5 * s, white, max(1, int(s)))

    if p.emissivity_note:
        _put(out, "e=0.98  ARTISTIC SIMULATION", (int(10 * s), h - int(12 * s)),
             0.42 * s, (215, 215, 215), max(1, int(s)))
    if p.status:
        _put(out, p.status, (int(10 * s), int(22 * s)), 0.5 * s, white, max(1, int(s)))

    drawn = srgb.srgb_to_linear(out.astype(np.float32) / 255.0)
    if p.opacity < 1.0:
        drawn = base * (1.0 - p.opacity) + drawn * p.opacity
    return np.clip(drawn, 0.0, 1.0).astype(np.float32)
