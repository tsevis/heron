"""Deterministic seeding — CLAUDE.md §2.4.

Every stochastic module takes an explicit seed; same seed + same params =
identical output. Sub-streams are derived from a master seed by *name*, so the
FPN field, Kirlian streamers, and grain each get an independent, reproducible
generator. FPN is fixed-pattern: derive its field once per session from a
stable tag and reuse it — never re-roll per frame.

We hash tags with SHA-256 (not Python's salted ``hash``) so results do not
depend on ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _tag_entropy(tags: tuple[object, ...]) -> int:
    """Stable 64-bit entropy from a tuple of tags (order-sensitive)."""
    key = "\x1f".join(str(t) for t in tags).encode("utf-8")
    digest = hashlib.sha256(key).digest()[:8]
    return int.from_bytes(digest, "big")


def rng_for(seed: int, *tags: object) -> np.random.Generator:
    """Return a NumPy generator deterministically derived from ``seed`` + tags.

    Calling twice with identical arguments yields identical random streams.
    """
    seq = np.random.SeedSequence([int(seed) & 0xFFFFFFFFFFFFFFFF, _tag_entropy(tags)])
    return np.random.default_rng(seq)


@dataclass(frozen=True)
class SeedBank:
    """A master seed that hands out named, reproducible sub-generators."""

    master: int = 0

    def rng(self, *tags: object) -> np.random.Generator:
        """Generator for a named sub-stream (e.g. ``bank.rng('fpn')``)."""
        return rng_for(self.master, *tags)

    def derive(self, *tags: object) -> int:
        """A reproducible integer sub-seed for passing to other subsystems."""
        return int(_tag_entropy((self.master, *tags)))
