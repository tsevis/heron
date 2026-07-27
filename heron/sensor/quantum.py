"""Quantum (photon) noise for X-ray / phosphor imaging (CLAUDE.md §3.2, §3.3).

Fluoroscope images are photon-limited: the visible grain is Poisson shot noise
from a finite X-ray dose, approximated here by signal-dependent Gaussian noise
(std ~ sqrt(signal/dose)). Deterministic per seed (§2.4). This is distinct from
the microbolometer FPN/NETD noise used by the Thermograph.
"""

from __future__ import annotations

import numpy as np

from heron.core.seeds import rng_for


def quantum_noise(signal: np.ndarray, dose: float = 80.0, strength: float = 0.5, seed: int = 0) -> np.ndarray:
    """Add photon-shot noise to a [0,1] signal. Lower ``dose`` = grainier."""
    if strength <= 0.0 or dose <= 0.0:
        return np.clip(signal, 0.0, 1.0).astype(np.float32)
    s = np.clip(np.asarray(signal, dtype=np.float32), 0.0, 1.0)
    std = np.sqrt(s / dose) * strength
    rng = rng_for(seed, "quantum", *s.shape)
    return np.clip(s + rng.standard_normal(s.shape).astype(np.float32) * std, 0.0, 1.0).astype(np.float32)
