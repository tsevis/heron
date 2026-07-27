"""ControlNet img2img finisher via diffusers, driving LOCAL models (§5.1).

Same idea as ``comfy_finisher`` — Heron's physical render is the img2img latent
and its Radiance-Graph depth is the ControlNet signal, so the diffusion model
adds texture without inventing geometry — but it loads the on-disk SDXL and
ControlNet checkpoints directly instead of going through ComfyUI. That removes a
whole moving part (and this machine's ComfyUI install is broken: ``main.py``
imports a ``comfy_aimdo`` module that is not present).

Nothing here is imported by ``heron`` — the engine stays deterministic (§9).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heron.color import srgb
from heron.core.device import get_device

COMFY_MODELS = Path("~/AI/ComfyUI/models").expanduser()
DEFAULT_CHECKPOINT = COMFY_MODELS / "checkpoints" / "juggernautXL_ragnarokBy.safetensors"
DEFAULT_CONTROLNET = COMFY_MODELS / "controlnet" / "diffusers_xl_depth_full.safetensors"


@dataclass(frozen=True)
class ControlParams:
    checkpoint: Path = DEFAULT_CHECKPOINT
    controlnet: Path = DEFAULT_CONTROLNET
    prompt: str = (
        "authentic thermal camera capture, FLIR microbolometer thermogram, ironbow palette, "
        "semi-transparent translucent body, heat glowing through thin clothing, luminous flesh, "
        "glowing body heat, face saturated hot white-yellow, cooler hair reading violet, "
        "deep dark cold background, soft infrared optics, subtle sensor noise, radiometric image"
    )
    negative_prompt: str = (
        "painting, illustration, drawing, ink outlines, hard edges, visible light photograph, "
        "text, watermark, cartoon, flat"
    )
    strength: float = 0.45          # img2img denoise: how far from Heron's render
    control_scale: float = 0.8
    steps: int = 26
    guidance: float = 6.0
    seed: int = 7
    max_side: int = 1024


def _pil(img_linear: np.ndarray):
    from PIL import Image

    arr = np.clip(srgb.linear_to_srgb(np.asarray(img_linear, np.float32)) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _fit(img, max_side: int):
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale) // 8 * 8, int(h * scale) // 8 * 8))
    else:
        img = img.resize((w // 8 * 8, h // 8 * 8))
    return img


def finish(
    render_linear: np.ndarray,
    depth: np.ndarray,
    params: ControlParams | None = None,
) -> np.ndarray | None:
    """Run SDXL + depth-ControlNet img2img over a Heron render."""
    p = params or ControlParams()
    if not Path(p.checkpoint).exists():
        print(f"[heron.neural.controlnet] missing checkpoint {p.checkpoint}")
        return None
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

        dev = get_device()
        dtype = torch.float16 if dev == "cuda" else torch.float32

        controlnet = ControlNetModel.from_single_file(str(p.controlnet), torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            str(p.checkpoint), controlnet=controlnet, torch_dtype=dtype, safety_checker=None
        )
        pipe.to(dev)
        pipe.set_progress_bar_config(disable=False)

        image = _fit(_pil(render_linear), p.max_side)
        depth_rgb = np.repeat(np.clip(np.asarray(depth, np.float32), 0, 1)[..., None], 3, axis=-1)
        control = _fit(_pil(srgb.srgb_to_linear(depth_rgb)), p.max_side).resize(image.size)

        gen = torch.Generator(device="cpu").manual_seed(p.seed)
        out = pipe(
            prompt=p.prompt, negative_prompt=p.negative_prompt,
            image=image, control_image=control,
            strength=p.strength, controlnet_conditioning_scale=p.control_scale,
            num_inference_steps=p.steps, guidance_scale=p.guidance, generator=gen,
        ).images[0]
        return srgb.srgb_to_linear(np.asarray(out, np.float32) / 255.0)
    except Exception as e:
        print(f"[heron.neural.controlnet] failed: {type(e).__name__}: {e}")
        return None
