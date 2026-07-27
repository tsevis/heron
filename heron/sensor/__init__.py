"""Layer C — Sensor & Optics (CLAUDE.md §0.5, §3.3).

Deliberately over-engineered: the artifacts carry the authenticity. Each module
is independently toggleable. Palette mapping (Oklab) lives in ``heron.color``.
"""

from heron.sensor.agc import plateau_equalize
from heron.sensor.fpn import FPNField, make_fpn, apply_fpn, add_netd
from heron.sensor.bloom import thermal_bloom
from heron.sensor.optics import mtf_blur, sensor_resolution, vignette, narcissus
from heron.sensor.msx import msx_overlay
from heron.sensor.dither import blue_noise_dither
from heron.sensor.quantum import quantum_noise
from heron.sensor.faceplate import hex_faceplate
from heron.sensor.scanlines import scanlines
from heron.sensor.chromatic import chromatic_aberration
from heron.sensor import furniture

__all__ = [
    "plateau_equalize",
    "FPNField",
    "make_fpn",
    "apply_fpn",
    "add_netd",
    "thermal_bloom",
    "mtf_blur",
    "sensor_resolution",
    "vignette",
    "narcissus",
    "msx_overlay",
    "blue_noise_dither",
    "quantum_noise",
    "hex_faceplate",
    "scanlines",
    "chromatic_aberration",
    "furniture",
]
