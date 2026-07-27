"""Where Layer A's optional models live, and whether they are here yet.

The engine used to name absolute paths on the machine it was written on. That
works exactly once — on a fresh clone every lookup fails, the AI path silently
never engages, and the classical fallback quietly takes over. The user sees a
worse result and no explanation.

This resolves each role in a fixed order, so an existing install is always
preferred over a download:

1. ``HERON_MODELS`` — an explicit directory, for anyone who keeps models elsewhere.
2. ``<repo>/models/<name>`` — where ``scripts/fetch_models.py`` puts things.
3. the legacy machine-specific paths — so the original development machine keeps
   working without re-downloading anything it already has.
4. the Hugging Face cache, by model id.

**Nothing here downloads.** Resolution is read-only and the loaders keep
``local_files_only=True``: an offline, deterministic engine must never reach for
the network mid-render (CLAUDE.md §9). Fetching is an explicit, separate step —
``scripts/fetch_models.py`` — and this module only reports what is missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_MODELS = REPO_ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    role: str
    hf_id: str                    # canonical source, used by the fetch script
    approx_mb: int
    purpose: str
    dir_name: str                 # folder name under models/
    legacy: tuple[str, ...] = ()  # absolute paths on the original machine
    required_for_ai: bool = True
    # File that proves the install is usable, not just a half-finished folder.
    marker: str = "config.json"
    # Hub filters. `patterns` allow-lists; `ignore` drops what we never load.
    patterns: tuple[str, ...] = ("*.json", "*.safetensors", "*.bin", "*.txt", "*.model")
    ignore: tuple[str, ...] = ()
    # Requires accepting a licence on the Hub and being logged in.
    gated: bool = False
    note: str = ""


# The roles a fresh clone needs for `--ai`. SAM 3 is included but gated: it is a
# real Hub repo (`facebook/sam3`) and fetchable, but under Meta's SAM License,
# so it needs `huggingface-cli login` and accepting the terms once.
SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        role="depth",
        hf_id="depth-anything/Depth-Anything-V2-Large-hf",
        approx_mb=1340,
        purpose="real depth — drives 3-D form, thickness and shading",
        dir_name="depth-anything-v2-large-hf",
        legacy=("~/AI/ClaudeCode/mozaix/models/Depth-Anything-V2",),
    ),
    ModelSpec(
        role="matte",
        hf_id="hustvl/vitmatte-base-composition-1k",
        approx_mb=390,   # measured: 387 MB on disk after fetch
        purpose="clean subject matte — removes the classical matte's edge leaks",
        dir_name="vitmatte-base-composition-1k",
        legacy=("~/AI/ClaudeCode/mozaix/models/vitmatte-base-composition-1k",),
    ),
    ModelSpec(
        role="segment",
        hf_id="facebook/sam3",
        # The repo ships TWO INCOMPATIBLE SERIALIZATIONS of the same ~860M-param
        # model, not the same weights twice — sam3.pt has 1,465 tensors keyed
        # detector.* / tracker.*, model.safetensors has 1,797 keyed
        # detector_model.* / tracker_model.* / tracker_neck.*, and they share
        # zero keys. Neither is derivable from the other without HF's conversion
        # script. The native .pt is canonical here because the loader imports
        # SAM 3's own python package by path rather than going through
        # transformers, so the safetensors copy is the one to skip.
        approx_mb=3400,
        purpose="per-region materials — skin vs hair vs fabric vs metal",
        dir_name="sam3",
        legacy=("~/AI/ClaudeCode/mozaix/models/sam3",),
        marker="sam3.pt",
        patterns=(),                     # full snapshot: it ships its own python package
        ignore=("model.safetensors", "assets/images/*", "*.gif", "*.mp4"),
        gated=True,
        note="Meta SAM License — accept the terms on the Hub and "
             "`huggingface-cli login` before fetching.",
    ),
)

BY_ROLE = {s.role: s for s in SPECS}


def _hf_cache_has(hf_id: str) -> bool:
    """True if the model is already in the Hugging Face cache."""
    cache = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    folder = cache / "hub" / ("models--" + hf_id.replace("/", "--"))
    if not folder.exists():
        # HF_HOME may already point at the hub dir
        folder = cache / ("models--" + hf_id.replace("/", "--"))
    return folder.exists() and any(folder.rglob("*.safetensors")) or (
        folder.exists() and any(folder.rglob("*.bin")))


def _is_installed(path: Path, spec: ModelSpec) -> bool:
    """A folder counts only if the file we actually load is in it."""
    return (path / spec.marker).exists()


def resolve(role: str) -> str | None:
    """Return a local directory path or an HF id for ``role``, or None.

    A returned HF id means "already in the cache" — the loaders pass
    ``local_files_only=True``, so a miss raises rather than downloading.
    """
    spec = BY_ROLE.get(role)
    if spec is None:
        raise KeyError(f"unknown model role '{role}'. known: {sorted(BY_ROLE)}")

    override = os.environ.get("HERON_MODELS")
    if override:
        candidate = Path(override).expanduser() / spec.dir_name
        if _is_installed(candidate, spec):
            return str(candidate)

    candidate = PROJECT_MODELS / spec.dir_name
    if _is_installed(candidate, spec):
        return str(candidate)

    for legacy in spec.legacy:
        candidate = Path(legacy).expanduser()
        if _is_installed(candidate, spec):
            return str(candidate)

    if _hf_cache_has(spec.hf_id):
        return spec.hf_id
    return None


def missing() -> list[ModelSpec]:
    """Roles a fresh clone still needs before ``--ai`` can work."""
    return [s for s in SPECS if s.required_for_ai and resolve(s.role) is None]


def status() -> list[tuple[ModelSpec, str | None]]:
    """(spec, resolved location or None) for every role — for reporting."""
    return [(s, resolve(s.role)) for s in SPECS]
