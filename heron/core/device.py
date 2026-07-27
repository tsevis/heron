"""Device selection — the single choke point for all hardware decisions.

CLAUDE.md §2: Apple Silicon is the target. PyTorch uses the ``mps`` device with
CPU fallback; ONNX Runtime uses the CoreML EP. Never assume CUDA. Every module
that needs a device must route through here rather than calling ``torch`` /
``onnxruntime`` device APIs directly, so the policy lives in one place.

This module imports torch/onnxruntime lazily so the classical (``--no-ai``)
pipeline runs with zero AI dependencies installed.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_device() -> str:
    """Return the preferred torch device string: 'mps', 'cuda', or 'cpu'.

    Honors ``HERON_DEVICE`` for override/testing. Falls back to 'cpu' when torch
    is absent, so importing this module never forces an AI dependency.
    """
    override = os.environ.get("HERON_DEVICE")
    if override:
        return override.strip().lower()

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.backends.mps.is_available():
        # Enable CPU fallback for ops MPS does not implement (idempotent).
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def torch_device():
    """Return a ``torch.device`` for the preferred device. Requires torch."""
    import torch

    return torch.device(get_device())


@lru_cache(maxsize=1)
def onnx_providers() -> list[str]:
    """Return the ONNX Runtime execution-provider preference list.

    CoreML first on Apple Silicon, then CPU. Requires onnxruntime installed.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or ["CPUExecutionProvider"]
