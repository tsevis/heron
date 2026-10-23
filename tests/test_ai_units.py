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


def _stub_models(monkeypatch, subject):
    """Replace every model call in enrich_graph with array logic."""
    from heron.physics import skeleton
    from heron.scene.ai import depth, materials
    from heron.scene.ai import matte as ai_matte

    monkeypatch.setattr(depth, "estimate_depth", lambda rgb, tier="fast": np.full(rgb.shape[:2], 0.5, np.float32))
    monkeypatch.setattr(segment, "segment_concepts", lambda rgb, concepts: {})
    monkeypatch.setattr(materials, "subject_mask", lambda rgb, masks=None: subject)
    monkeypatch.setattr(ai_matte, "estimate_matte", lambda rgb, fallback: None)
    monkeypatch.setattr(skeleton, "skeleton_proxy", lambda rgb: None)


def test_ai_graph_skips_the_classical_matte_when_sam_finds_the_subject(monkeypatch):
    """The classical matte (GrabCut) was built and thrown away on every AI render."""
    from heron.scene import GraphOptions, build_graph
    from heron.scene import matte as classical

    calls = []
    monkeypatch.setattr(classical, "classical_matte", lambda *a, **k: calls.append(1) or np.zeros((32, 32), np.float32))
    subject = np.zeros((32, 32), np.float32)
    subject[8:24, 8:24] = 1.0
    _stub_models(monkeypatch, subject)

    g = build_graph(np.full((32, 32, 3), 0.4, np.float32), opts=GraphOptions(use_ai=True, cache=False))

    assert calls == []
    assert g.matte[16, 16] > 0.9 and g.matte[0, 0] < 0.1


def test_ai_graph_falls_back_to_the_classical_matte_without_a_subject(monkeypatch):
    from heron.scene import GraphOptions, build_graph
    from heron.scene import matte as classical

    calls = []
    monkeypatch.setattr(classical, "classical_matte", lambda *a, **k: calls.append(1) or np.ones((32, 32), np.float32))
    _stub_models(monkeypatch, None)

    g = build_graph(np.full((32, 32, 3), 0.4, np.float32), opts=GraphOptions(use_ai=True, cache=False))

    assert calls == [1]
    assert g.matte.min() == 1.0


def test_a_missing_sam3_is_reported_once_not_on_every_call(monkeypatch, capsys):
    """One render asked for SAM 3 three times, each retrying the load and logging."""
    monkeypatch.setattr(segment, "_sam3_dir", lambda: None)
    segment._processor_or_error.cache_clear()
    try:
        img = np.zeros((8, 8, 3), np.float32)
        assert segment.segment_concepts(img, ["person"]) is None
        assert segment.segment_concepts(img, ["face"]) is None
    finally:
        segment._processor_or_error.cache_clear()

    assert capsys.readouterr().out.count("SAM3 unavailable") == 1
