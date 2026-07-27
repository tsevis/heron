"""Palette engine — 1-D LUTs interpolated in Oklab (CLAUDE.md §3.3, §2.3).

Palettes are JSON ramps of sRGB stops. They are baked into a dense linear-light
LUT by interpolating **between adjacent stops in Oklab**, giving perceptually
uniform ramps with monotonically-increasing lightness (hotter = brighter). The
palette carries the *look*, never the physics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from heron.color import oklab, srgb

PALETTE_DIR = Path(__file__).parent / "palettes"
_DEFAULT_LUT_SIZE = 256


@dataclass(frozen=True)
class Palette:
    """A named ramp defined by sRGB stops at parameter positions ``t`` in [0,1]."""

    name: str
    stops_t: np.ndarray  # (K,) increasing in [0,1]
    stops_srgb: np.ndarray  # (K,3) in [0,1] gamma sRGB
    lightness: str = "ascending"  # ascending | descending | cyclic

    @classmethod
    def from_dict(cls, data: dict) -> "Palette":
        stops = sorted(data["stops"], key=lambda s: s["t"])
        t = np.array([s["t"] for s in stops], dtype=np.float32)
        rgb = np.array([s["rgb"] for s in stops], dtype=np.float32)
        if rgb.max() > 1.0:  # allow 0-255 authoring
            rgb = rgb / 255.0
        if t[0] != 0.0 or t[-1] != 1.0:
            raise ValueError(f"palette '{data.get('name')}' must span t=0..1")
        return cls(
            name=data.get("name", "unnamed"),
            stops_t=t,
            stops_srgb=rgb,
            lightness=data.get("lightness", "ascending"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Palette":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def lut(self, size: int = _DEFAULT_LUT_SIZE) -> np.ndarray:
        """Bake a dense (size,3) **linear-light** LUT, Oklab-interpolated."""
        stops_lin = srgb.srgb_to_linear(self.stops_srgb)
        stops_lab = oklab.linear_srgb_to_oklab(stops_lin)
        q = np.linspace(0.0, 1.0, size, dtype=np.float32)
        # segment index for each query point
        idx = np.clip(np.searchsorted(self.stops_t, q, side="right") - 1, 0, len(self.stops_t) - 2)
        t0 = self.stops_t[idx]
        t1 = self.stops_t[idx + 1]
        local = np.where(t1 > t0, (q - t0) / np.maximum(t1 - t0, 1e-8), 0.0)[:, None]
        lab = stops_lab[idx] * (1.0 - local) + stops_lab[idx + 1] * local
        lin = oklab.oklab_to_linear_srgb(lab)
        return np.clip(lin, 0.0, 1.0).astype(np.float32)

    def apply(self, scalar: np.ndarray, size: int = _DEFAULT_LUT_SIZE) -> np.ndarray:
        """Map a scalar field in [0,1] (any shape) to linear-light RGB (...,3)."""
        lut = self.lut(size)
        s = np.clip(np.asarray(scalar, dtype=np.float32), 0.0, 1.0)
        pos = s * (size - 1)
        lo = np.floor(pos).astype(np.int32)
        hi = np.minimum(lo + 1, size - 1)
        frac = (pos - lo)[..., None]
        return (lut[lo] * (1.0 - frac) + lut[hi] * frac).astype(np.float32)

    def lightness_profile(self, size: int = _DEFAULT_LUT_SIZE) -> np.ndarray:
        """Oklab L along the ramp — used to check monotonicity (tests §8)."""
        return oklab.linear_srgb_to_oklab(self.lut(size))[:, 0]

    def is_lightness_monotonic(self, tol: float = 1e-3) -> bool:
        """True if Oklab L is monotonic in the palette's declared direction.

        ``cyclic`` palettes (Rainbow, Turbo) are exempt and always pass.
        """
        if self.lightness == "cyclic":
            return True
        L = self.lightness_profile()
        d = np.diff(L)
        if self.lightness == "descending":
            return bool(np.all(d <= tol))
        return bool(np.all(d >= -tol))


# Palettes registered at runtime — the ramp carried inside a shared preset, so
# it resolves on a machine that does not ship that file. Checked before disk so
# an imported preset reproduces its own colours rather than a local namesake.
_RUNTIME_PALETTES: dict[str, Palette] = {}


def register_palette(name: str, data: dict) -> Palette:
    """Make a palette available by name without writing it to the package.

    Used by preset import (Phase 5 §5.3): a shareable preset inlines its ramp so
    it survives without the palette file, and registering it here means every
    renderer resolves it through the normal ``load_palette`` path.
    """
    palette = Palette.from_dict(data)
    _RUNTIME_PALETTES[name] = palette
    # load_palette is memoised, so a name already resolved from disk would keep
    # returning the local ramp and the imported preset would render the wrong
    # colours — silently, since both are valid palettes.
    load_palette.cache_clear()
    return palette


@lru_cache(maxsize=64)
def load_palette(name: str) -> Palette:
    """Load a palette by name — runtime-registered first, then package files."""
    if name in _RUNTIME_PALETTES:
        return _RUNTIME_PALETTES[name]
    path = PALETTE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"unknown palette '{name}' ({path})")
    return Palette.from_json(path)


def palette_source(name: str) -> dict:
    """The palette's authoring data, for inlining into a shareable preset."""
    if name in _RUNTIME_PALETTES:
        p = _RUNTIME_PALETTES[name]
        return {"name": p.name, "lightness": p.lightness,
                "stops": [{"t": float(t), "rgb": [float(c) for c in rgb]}
                          for t, rgb in zip(p.stops_t, p.stops_srgb)]}
    path = PALETTE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"unknown palette '{name}' ({path})")
    return json.loads(path.read_text())


def available_palettes() -> list[str]:
    """Sorted list of palette names shipped in the package."""
    return sorted(p.stem for p in PALETTE_DIR.glob("*.json"))
