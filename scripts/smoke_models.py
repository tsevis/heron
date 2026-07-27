"""Model registry smoke test (CLAUDE.md §5, §8).

Verifies that every model Heron's Layer A expects is present on disk at its
registered path, reports device/EP availability, and — when AI deps are
installed — runs a tiny 64x64 inference through each loadable model.

    python scripts/smoke_models.py            # existence + device report
    python scripts/smoke_models.py --infer    # also try tiny inferences (needs AI deps)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from heron.core.device import get_device, onnx_providers

# Heron role -> registered local path (CLAUDE.md §5).
REGISTRY: dict[str, str] = {
    "depth_pro": "~/AI/ClaudeCode/mozaix/models/depth-pro/checkpoints",
    "depth_anything_v2": "~/AI/ClaudeCode/mozaix/models/Depth-Anything-V2",
    "dsine_normals": "~/AI/ClaudeCode/mozaix/models/DSINE",
    "sam3": "~/AI/ClaudeCode/mozaix/models/sam3",
    "mlx_sam3": "~/AI/ClaudeCode/mozaix/models/mlx_sam3",
    "grounding_dino": "~/AI/ClaudeCode/mozaix/models/groundingdino",
    "vitmatte": "~/AI/ClaudeCode/mozaix/models/vitmatte-base-composition-1k",
    "realesrgan": "~/AI/ClaudeCode/TsevisEnhancer/models_downloads/RealESRGAN_x4plus.pth",
}


def check_existence() -> list[tuple[str, bool, str]]:
    rows = []
    for role, raw in REGISTRY.items():
        p = Path(raw).expanduser()
        rows.append((role, p.exists(), str(p)))
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
