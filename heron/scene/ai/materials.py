"""Material assignment (CLAUDE.md §5, §7, §3.1).

An F-ViTA-style pipeline: an Ollama VLM lists the materials present, SAM 3
localizes each as a mask, and Heron paints a per-pixel material map that the
physics engine turns into emissivity / temperature / reflective behavior. The
subject matte supplies skin/hair on top. Every step degrades gracefully —
no VLM ⇒ person assumption; no SAM 3 ⇒ matte-only.

This is what unlocks the mirror-water/metal case: water and polished metal
get `reflective` LWIR behavior and composite the thermal environment instead
of emitting.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from io import BytesIO

import cv2
import numpy as np

from heron.color import srgb

_OLLAMA = "http://localhost:11434/api/generate"
_VLM_MODEL = "qwen2.5vl:3b"

# Material classes SAM 3 can localize, in background->foreground paint order
# (later entries win overlaps), each with the concept phrase we prompt it with.
_SEGMENTABLE: list[tuple[str, str]] = [
    ("sky", "sky"),
    ("stone", "stone wall"),
    ("wood", "wood"),
    ("foliage", "plants and leaves"),
    ("glass", "glass window"),
    ("water", "water"),
    ("metal_matte", "metal"),
    ("metal_polished", "shiny polished metal"),
    ("fabric", "clothing"),
]

# Concepts that make up a person, painted OVER the scene in this order (later
# wins). Hair must come from SAM 3, not a luminance heuristic: hair is a strong
# insulator and reads far cooler than skin, so mislabeling it as skin is what
# made every head render as one uniform hot mass.
_PERSON_PARTS: list[tuple[str, str]] = [
    ("skin", "face and skin"),
    ("hair", "hair"),
]

# Everything SAM 3 is asked for in a single pass (its backbone runs once per
# image; extra text prompts are cheap decoder passes at ~0.8 s each).
ALL_CONCEPTS: list[str] = (
    ["person"] + [p for _, p in _SEGMENTABLE] + [p for _, p in _PERSON_PARTS]
)


def subject_mask(
    linear_rgb: np.ndarray, masks: dict[str, np.ndarray] | None = None
) -> np.ndarray | None:
    """SAM 3 mask of the main subject, as a soft [0,1] matte seed, or None.

    The union covers the person *and their hair* — hair left outside the matte
    would be treated as background and rendered at ambient temperature, which is
    both wrong and the thing that chopped the hairline flat.
    """
    if masks is None:
        from heron.scene.ai import segment as _seg

        masks = _seg.segment_concepts(linear_rgb, ["person", "face and skin", "hair"])
    if not masks:
        return None
    union = None
    for key in ("person", "face and skin", "hair"):
        m = masks.get(key)
        if m is not None and m.any():
            union = m if union is None else (union | m)
    if union is None or not union.any():
        return None
    return union.astype(np.float32)


def assign_materials(
    linear_rgb: np.ndarray,
    matte: np.ndarray,
    use_vlm: bool = True,
    use_sam3: bool = True,
    masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Return (material_ids, material_names): SAM3 scene materials + subject parts.

    ``masks`` may be a pre-computed {prompt: mask} dict so the caller can run
    SAM 3 once and share it with the matte stage.
    """
    from heron.materials import load_material_table

    table = load_material_table()
    tags = vlm_scene_tags(linear_rgb) if use_vlm else None
    subject_class = (tags or {}).get("subject", "person")
    present = {table.resolve_label(m) for m in (tags or {}).get("materials", [])}

    names: list[str] = ["unknown"]
    ids = np.zeros(matte.shape, dtype=np.int32)

    def _id_for(name: str) -> int:
        if name not in names:
            names.append(name)
        return names.index(name)

    if use_sam3 and masks is None:
        from heron.scene.ai import segment as _seg

        masks = _seg.segment_concepts(linear_rgb, ALL_CONCEPTS)

    # --- scene materials (background -> foreground; later wins overlaps) ---
    if masks:
        for cls, prompt in _SEGMENTABLE:
            if present and cls not in present:
                continue  # the VLM says this material is not in the scene
            m = masks.get(prompt)
            if m is not None and m.any():
                ids[m] = _id_for(cls)

    # --- subject on top: skin, then clothing, then hair painted over it ---
    subject = matte > 0.5
    if subject_class == "person" and subject.any():
        ids[subject] = _id_for("skin")

        # Worn clothing sits INSIDE the person mask, so painting the whole
        # subject as skin erased the shirt SAM 3 had already found — the same
        # layering bug hair had. Fabric goes over skin; hair still goes last so
        # strands falling over a collar keep their material.
        cloth_mask = (masks or {}).get("clothing")
        if cloth_mask is not None and cloth_mask.any():
            ids[cloth_mask] = _id_for("fabric")

        hair_mask = (masks or {}).get("hair")
        if hair_mask is not None and hair_mask.any():
            # trust SAM 3's hair, but keep it near the figure so stray matches
            # elsewhere in the frame do not become "hair"
            near_subject = cv2.dilate(
                subject.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=3
            ).astype(bool)
            hair = hair_mask & near_subject
        else:
            # classical fallback only when SAM 3 gave us nothing
            lum = srgb.luminance(linear_rgb)
            h = matte.shape[0]
            upper = np.arange(h).reshape(-1, 1) < int(h * 0.45)
            hair = subject & upper & (lum < np.percentile(lum[subject], 35))
        if hair.any():
            ids[hair] = _id_for("hair")
    elif subject.any():
        # non-person subject: use its best VLM material, else leave scene labels
        mat = next((m for m in (present or set()) if m != "unknown"), None)
        if mat:
            ids[subject] = _id_for(mat)

    return ids, tuple(names)


def vlm_scene_tags(linear_rgb: np.ndarray, timeout: float = 60.0) -> dict | None:
    """Best-effort Ollama VLM query for subject class + scene materials."""
    try:
        from PIL import Image

        srgb8 = np.clip(srgb.linear_to_srgb(linear_rgb) * 255, 0, 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(srgb8).resize((512, 512)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        prompt = (
            "Look at this photo. Reply ONLY with compact JSON: "
            '{"subject":"person|object|scene","materials":["skin","fabric","water",'
            '"metal","glass","foliage","wood","stone","sky"]} listing materials actually visible.'
        )
        payload = json.dumps({
            "model": _VLM_MODEL,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.0},
        }).encode("utf-8")
        req = urllib.request.Request(_OLLAMA, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.load(r).get("response", "")
        start, end = resp.find("{"), resp.rfind("}")
        if start >= 0 and end > start:
            return json.loads(resp[start:end + 1])
    except Exception as e:
        print(f"[heron.ai.materials] VLM hint unavailable: {e}")
    return None
