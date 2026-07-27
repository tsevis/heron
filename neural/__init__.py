"""Heron neural finishers — Phase 5 (CLAUDE.md §5.1, §9).

Deliberately OUTSIDE the ``heron`` package: the engine stays deterministic,
offline and UI-agnostic, and never imports any of this. Every module here is an
optional *post-pass* over Heron's physical output, which is exactly the Master
Plan's design — the physics layer becomes the control signal, the network is a
finisher, never the foundation.

Modules
-------
comfy_finisher : ControlNet img2img through a local ComfyUI (no training)
style_transfer : Gatys/AdaIN neural style transfer from reference paintings
cyclegan       : unpaired RGB<->thermal translation (trainable)
pix2pixhd      : paired RGB->thermal translation (trainable)
datasets       : paired/unpaired thermal dataset preparation
"""

__all__ = ["comfy_finisher", "style_transfer", "cyclegan", "pix2pixhd", "datasets"]
