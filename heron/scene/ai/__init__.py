"""Layer A — AI enrichment (CLAUDE.md §0.3, §3.1, §5).

Overwrites the classical Radiance Graph channels with model output when AI is
requested. Every component degrades gracefully: if a model or dependency is
missing, that channel keeps its classical value instead of crashing, honoring
"AI enriches; it must never be a hard dependency" (§2.6).

Models are loaded read-only from their registered local paths (§5) via
``heron.core.device`` (MPS/CoreML). Nothing here downloads weights.

STATUS: implemented against the on-disk model formats; requires
``requirements-ai.txt`` installed and a validation pass (`scripts/smoke_models.py
--infer`). Full SAM3 + grounding per-region materials (water/metal masks for
reflective compositing) is the remaining 0.3 piece — see materials.py.
"""

from __future__ import annotations

import numpy as np

from heron.core.graph import RadianceGraph


def enrich_graph(graph: RadianceGraph, linear_rgb: np.ndarray, opts) -> RadianceGraph:
    """Replace classical channels with model output where each model is available.

    Returns a new RadianceGraph (immutable-update style; CLAUDE.md coding-style).
    """
    from heron.scene.ai import depth as _depth
    from heron.scene.ai import matte as _matte
    from heron.scene.ai import materials as _materials

    depth = _depth.estimate_depth(linear_rgb, tier=opts.depth_tier)

    # One SAM 3 pass for everything: its image backbone runs once, then each text
    # concept is a cheap decoder query. The same masks seed the matte AND label
    # the materials, so the figure, its hair and the scene all agree.
    sam_masks = None
    seed_mask = graph.matte
    if getattr(opts, "segment_materials", True):
        from heron.scene.ai import segment as _seg

        sam_masks = _seg.segment_concepts(linear_rgb, _materials.ALL_CONCEPTS)
        subject = _materials.subject_mask(linear_rgb, masks=sam_masks)
        if subject is not None:
            seed_mask = subject

    matte = _matte.estimate_matte(linear_rgb, fallback=seed_mask)
    if matte is None and seed_mask is not graph.matte:
        # ViTMatte was unusable — feather SAM 3's hard mask instead. A slightly
        # soft SAM 3 silhouette is far better than a broken alpha, which would
        # poison every downstream term (it multiplies most of the physics).
        import cv2

        matte = cv2.GaussianBlur(
            seed_mask.astype(np.float32), (0, 0), sigmaX=max(seed_mask.shape) * 0.002
        )
        matte = np.clip(matte, 0.0, 1.0).astype(np.float32)

    # The Ollama VLM is redundant once SAM 3 runs (it only pre-filtered which
    # concepts to segment) and its cold start cost up to a minute per new photo.
    material_ids, material_names = _materials.assign_materials(
        linear_rgb,
        matte if matte is not None else graph.matte,
        use_vlm=False,
        use_sam3=getattr(opts, "segment_materials", True),
        masks=sam_masks,
    )

    # Layer A artifacts for Layer B: physics must never invoke a model itself.
    extras: dict = {}
    if sam_masks:
        face = sam_masks.get("face and skin")
        if face is not None and face.any():
            extras["face_region"] = face.astype(np.float32)

    # Stylized skeleton proxy (Fluoroscope, §1.1) — detected once here, not per
    # render: BlazePose is a full model load+inference, and re-running it on
    # every dial tweak was the same "Layer B invokes a model" bug already fixed
    # once for SAM 3 face segmentation (see physics/thermal.py face_region).
    from heron.physics import skeleton as _skeleton

    bones = _skeleton.skeleton_proxy(linear_rgb)
    if bones is not None:
        extras["skeleton_bones"] = bones.astype(np.float32)

    from heron.scene.graph_builder import normals_from_depth

    new_depth = depth if depth is not None else graph.depth
    ai_channels = [
        name for name, ok in (
            ("depth", depth is not None),
            ("matte", matte is not None),
            ("materials", material_ids is not None),
        ) if ok
    ]
    return RadianceGraph(
        depth=new_depth,
        normals=normals_from_depth(new_depth) if depth is not None else graph.normals,
        albedo=graph.albedo,  # classical de-light (no intrinsic net on disk, §5)
        saliency=graph.saliency,
        matte=matte if matte is not None else graph.matte,
        material_ids=material_ids if material_ids is not None else graph.material_ids,
        material_names=material_names if material_ids is not None else graph.material_names,
        meta={
            **graph.meta,
            "ai_used": bool(ai_channels),
            "ai_channels": ai_channels,
            "depth_tier": opts.depth_tier,
        },
        extras=extras,
    )


def smoke_infer() -> None:
    """Tiny inference through each available model (called by smoke_models.py)."""
    import numpy as np

    from heron.scene.ai import depth as _depth
    from heron.scene.ai import matte as _matte

    img = np.clip(np.random.default_rng(0).random((64, 64, 3)), 0, 1).astype(np.float32)
    d = _depth.estimate_depth(img, tier="fast")
    print(f"  depth:  {'ok ' + str(d.shape) if d is not None else 'unavailable (classical fallback)'}")
    m = _matte.estimate_matte(img, fallback=np.zeros((64, 64), np.float32))
    print(f"  matte:  {'ok ' + str(m.shape) if m is not None else 'unavailable (classical fallback)'}")
