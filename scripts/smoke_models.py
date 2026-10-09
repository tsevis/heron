"""Model registry smoke test (CLAUDE.md §5, §8).

Verifies that every model Heron's Layer A expects is present on disk, reports
device/EP availability, and — when AI deps are installed — runs a tiny 64x64
inference through each loadable model. Models are looked for under
``$HERON_MODELS``, ``<repo>/models`` and every directory in
``$HERON_EXTRA_MODEL_DIRS`` (separated like ``PATH``).

    python scripts/smoke_models.py            # existence + device report
    python scripts/smoke_models.py --infer    # also try tiny inferences (needs AI deps)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from heron.core.device import get_device, onnx_providers
from heron.scene.ai import registry

# Heron role -> location relative to a model root (CLAUDE.md §5).
REGISTRY: dict[str, str] = {
    "depth_pro": "depth-pro/checkpoints",
    "depth_anything_v2": "Depth-Anything-V2",
    "dsine_normals": "DSINE",
    "sam3": "sam3",
    "mlx_sam3": "mlx_sam3",
    "grounding_dino": "groundingdino",
    "vitmatte": "vitmatte-base-composition-1k",
    "realesrgan": "RealESRGAN_x4plus.pth",
}


def model_roots() -> list[Path]:
    """Where to look, in order: $HERON_MODELS, <repo>/models, extra dirs."""
    override = os.environ.get("HERON_MODELS")
    roots = [Path(override).expanduser()] if override else []
    return [*roots, registry.PROJECT_MODELS, *registry.extra_model_dirs()]


def check_existence() -> list[tuple[str, bool, str]]:
    rows = []
    roots = model_roots()
    for role, rel in REGISTRY.items():
        found = next((r / rel for r in roots if (r / rel).exists()), None)
        rows.append((role, found is not None, str(found) if found else rel))
    return rows


def check_ollama() -> tuple[bool, list[str]]:
    try:
        import urllib.request, json

        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.load(r)
        return True, [m["name"] for m in data.get("models", [])]
    except Exception:
        return False, []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--infer", action="store_true", help="attempt tiny inferences (needs AI deps)")
    args = ap.parse_args()

    print(f"device: {get_device()}")
    print(f"onnx providers: {onnx_providers()}")

    print("\nmodel registry:")
    missing = 0
    for role, ok, path in check_existence():
        mark = "OK  " if ok else "MISS"
        if not ok:
            missing += 1
        print(f"  [{mark}] {role:20s} {path}")

    up, models = check_ollama()
    print(f"\nollama: {'up' if up else 'unreachable'}")
    if up:
        vlm = [m for m in models if any(k in m for k in ("qwen2.5vl", "minicpm-v", "moondream", "llava"))]
        print(f"  vlm candidates: {vlm}")

    if args.infer:
        print("\ninference smoke:")
        try:
            import numpy as np  # noqa: F401
            from heron.scene import ai  # noqa: F401

            ai.smoke_infer()
        except Exception as e:
            print(f"  inference smoke unavailable: {e}")

    print(f"\n{'ALL PRESENT' if missing == 0 else f'{missing} MISSING'}")


if __name__ == "__main__":
    main()
