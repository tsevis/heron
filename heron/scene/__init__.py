"""Layer A — Scene Understanding.

Wrappers that produce the Radiance Graph. Every function here has a classical,
model-free path so the whole layer is skippable (CLAUDE.md §2.6): AI enriches,
it is never a hard dependency. AI wrappers live in ``scene.ai`` and are imported
lazily so the classical pipeline needs no torch/onnx.
"""

from heron.scene.graph_builder import build_graph, GraphOptions

__all__ = ["build_graph", "GraphOptions"]
