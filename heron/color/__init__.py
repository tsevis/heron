"""Color math: sRGB transfer, Oklab, and the palette engine.

Pure-numpy, dependency-free (CLAUDE.md §2.3): all internal processing is
32-bit float, linear light, [0, 1]; palette interpolation happens in Oklab.
Do not pull a heavyweight color library — Oklab lives in ``oklab.py``.
"""

from heron.color import srgb, oklab, palettes

__all__ = ["srgb", "oklab", "palettes"]
