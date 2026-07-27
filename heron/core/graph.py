"""The Radiance Graph — Layer A's structured output (CLAUDE.md §3.1, §2.7).

A per-image bundle of scene-understanding channels: depth, normals, de-lit
albedo, saliency, subject matte, and a per-pixel material-id map. It is cached
once per image under ``.heron_graph/`` (NPZ) because Layer A is the slow, AI
part of the pipeline; Layers B/C are fast and recomputed per parameter change.

Every channel is an artifact, not an internal (§2.7): each can be written to an
image, hand-edited, and re-imported. Channels are stored as raw float data, not
sRGB — they are measurements, not pictures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CACHE_DIRNAME = ".heron_graph"
_FORMAT_VERSION = 1


@dataclass
class RadianceGraph:
    """Layer A channels for one image. Arrays are float32 unless noted.

    - ``depth``    (H,W) in [0,1], larger = nearer.
    - ``normals``  (H,W,3) unit vectors, camera space, +Z toward camera.
    - ``albedo``   (H,W,3) linear-light de-lit reflectance in [0,1].
    - ``saliency`` (H,W) in [0,1].
    - ``matte``    (H,W) subject coverage in [0,1].
    - ``material_ids`` (H,W) int32 indexing ``material_names``.
    """

    depth: np.ndarray
    normals: np.ndarray
    albedo: np.ndarray
    saliency: np.ndarray
    matte: np.ndarray
    material_ids: np.ndarray
    material_names: tuple[str, ...]
    meta: dict = field(default_factory=dict)
    # optional named channels produced by Layer A (e.g. the SAM 3 face region)
    # so Layer B never has to re-run a model at render time (§2.7, §3.1)
    extras: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape[:2]

    def material_map(self, name_to_value: dict[str, float], default: float = 0.0) -> np.ndarray:
        """Project the material-id map into a scalar field via a name->value map.

        Used by Layer B to turn materials into emissivity / temperature priors.
        """
        lut = np.full(len(self.material_names), default, dtype=np.float32)
        for i, name in enumerate(self.material_names):
            if name in name_to_value:
                lut[i] = name_to_value[name]
        return lut[self.material_ids]

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra_arrays = {f"x_{k}": np.asarray(v, dtype=np.float32) for k, v in self.extras.items()}
        np.savez_compressed(
            path,
            **extra_arrays,
            depth=self.depth.astype(np.float32),
            normals=self.normals.astype(np.float32),
            albedo=self.albedo.astype(np.float32),
            saliency=self.saliency.astype(np.float32),
            matte=self.matte.astype(np.float32),
            material_ids=self.material_ids.astype(np.int32),
            _sidecar=np.frombuffer(
                json.dumps(
                    {
                        "format": _FORMAT_VERSION,
                        "material_names": list(self.material_names),
                        "meta": self.meta,
                    }
                ).encode("utf-8"),
                dtype=np.uint8,
            ),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RadianceGraph":
        with np.load(path, allow_pickle=False) as data:
            sidecar = json.loads(bytes(data["_sidecar"]).decode("utf-8"))
            extras = {k[2:]: data[k] for k in data.files if k.startswith("x_")}
            return cls(
                extras=extras,
                depth=data["depth"],
                normals=data["normals"],
                albedo=data["albedo"],
                saliency=data["saliency"],
                matte=data["matte"],
                material_ids=data["material_ids"],
                material_names=tuple(sidecar["material_names"]),
                meta=sidecar.get("meta", {}),
            )


def cache_path(image_path: str | Path, signature: str) -> Path:
    """Deterministic cache location for an image's graph under ``.heron_graph/``.

    ``signature`` distinguishes graphs built with different settings (AI tier,
    model set) so a ``--no-ai`` graph never masquerades as an AI one.
    """
    image_path = Path(image_path)
    try:
        stat = image_path.stat()
        content_key = f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        content_key = "0:0"
    digest = hashlib.sha256(f"{image_path.resolve()}|{content_key}|{signature}".encode()).hexdigest()[:16]
    return image_path.parent / CACHE_DIRNAME / f"{image_path.stem}_{digest}.npz"
