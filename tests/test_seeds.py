"""Determinism tests (CLAUDE.md §2.4, §8): same seed + params = identical
output; FPN is a static fixed pattern."""

import numpy as np

from heron.core.seeds import SeedBank, rng_for
from heron.sensor import fpn


def test_rng_deterministic():
    a = rng_for(42, "grain", 3).standard_normal(1000)
    b = rng_for(42, "grain", 3).standard_normal(1000)
    assert np.array_equal(a, b)


def test_rng_tags_differ():
    a = rng_for(42, "grain").standard_normal(1000)
    b = rng_for(42, "streamers").standard_normal(1000)
    assert not np.array_equal(a, b)


def test_seedbank_named_streams_independent():
    bank = SeedBank(7)
    assert not np.array_equal(bank.rng("a").random(50), bank.rng("b").random(50))
    assert np.array_equal(bank.rng("a").random(50), bank.rng("a").random(50))


def test_fpn_is_fixed_pattern():
    f1 = fpn.make_fpn((32, 48), seed=1, amplitude=0.05)
    f2 = fpn.make_fpn((32, 48), seed=1, amplitude=0.05)
    assert np.array_equal(f1.offset, f2.offset)
    assert np.array_equal(f1.gain, f2.gain)
    f3 = fpn.make_fpn((32, 48), seed=2, amplitude=0.05)
    assert not np.array_equal(f1.offset, f3.offset)


def test_netd_varies_per_frame_not_fpn():
    sig = np.full((16, 16), 0.5, np.float32)
    a = fpn.add_netd(sig, 40.0, (20.0, 30.0), seed=1, frame=0)
    b = fpn.add_netd(sig, 40.0, (20.0, 30.0), seed=1, frame=1)
    assert not np.array_equal(a, b)  # temporal noise re-rolls per frame
    a2 = fpn.add_netd(sig, 40.0, (20.0, 30.0), seed=1, frame=0)
    assert np.array_equal(a, a2)     # but a given frame is reproducible
