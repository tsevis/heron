"""Unit tests for AI-layer helpers that need no models (pure array logic)."""

import numpy as np

from heron.scene.ai import segment


def test_binary_union_probabilities():
    # two instance masks (N,1,H,W) as probabilities, one below threshold
    m = np.zeros((2, 1, 8, 8), np.float32)
    m[0, 0, :4, :4] = 0.9  # strong
    m[1, 0, 4:, 4:] = 0.9  # strong
    scores = np.array([0.9, 0.1], np.float32)  # second below 0.4 threshold
    union = segment._to_binary_union(m, scores, (8, 8))
    assert union is not None
    assert union[:4, :4].all()      # kept instance present
    assert not union[4:, 4:].any()  # low-score instance dropped


def test_binary_union_logits_thresholds_at_zero():
    m = np.full((1, 1, 4, 4), -2.0, np.float32)
    m[0, 0, :2, :2] = 3.0  # positive logits -> foreground
    union = segment._to_binary_union(m, None, (4, 4))
    assert union[:2, :2].all()
    assert not union[2:, 2:].any()


def test_binary_union_resizes_to_target():
    m = np.ones((1, 1, 16, 16), np.float32)
    union = segment._to_binary_union(m, None, (32, 48))
    assert union.shape == (32, 48)


def test_binary_union_empty_returns_none():
    assert segment._to_binary_union(np.zeros((0, 1, 4, 4), np.float32), None, (4, 4)) is None
