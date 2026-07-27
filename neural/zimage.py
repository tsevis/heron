"""Z-Image Turbo assembled entirely from on-disk ComfyUI components (§5.1).

Tongyi's Z-Image Turbo is the one modern model whose every heavy part already
sits in ``~/AI/ComfyUI/models``: the 12.3 GB bf16 transformer, its official
Qwen3-4B text encoder (plain bf16, HF-named keys), and the Flux AutoencoderKL it
reuses. Only the *config/tokenizer JSONs* (a few MB of text) are fetched — an
``allow_patterns`` guard makes downloading actual weights impossible.

Turbo means 8-step sampling at guidance 1.0 — on MPS that is comparable to an
SDXL draft, with 2025-class quality. Read-only with respect to ComfyUI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COMFY = Path("~/AI/ComfyUI/models").expanduser()
TRANSFORMER_FILE = COMFY / "diffusion_models" / "z_image_turbo_bf16.safetensors"
TEXT_ENCODER_FILE = COMFY / "text_encoders" / "qwen_3_4b.safetensors"
VAE_FILE = COMFY / "vae" / "ae.safetensors"          # Z-Image reuses the Flux AE

_TEXT_REPO = "Qwen/Qwen3-4B"
_ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
_JSON_ONLY = ["*.json", "*.txt", "tokenizer*", "*.model"]   # never weights


def components_present() -> tuple[bool, str]:
    missing = [p.name for p in (TRANSFORMER_FILE, TEXT_ENCODER_FILE, VAE_FILE) if not p.exists()]
    if missing:
        return False, f"missing on disk: {', '.join(missing)}"
    return True, "transformer + Qwen3-4B text encoder + VAE all on disk"


def _fetch_configs() -> tuple[Path, Path]:
    """Download ONLY json/tokenizer text files for the two repos."""
    from huggingface_hub import snapshot_download

    text_cfg = Path(snapshot_download(_TEXT_REPO, allow_patterns=_JSON_ONLY))
    z_cfg = Path(snapshot_download(_ZIMAGE_REPO, allow_patterns=_JSON_ONLY))
    return text_cfg, z_cfg


def load_pipeline(device: str, dtype):
    """Assemble ZImageImg2ImgPipeline from local weights + tiny configs."""
    import torch
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImageImg2ImgPipeline,
        ZImageTransformer2DModel,
    )
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    ok, why = components_present()
    if not ok:
        raise RuntimeError(f"Z-Image components incomplete: {why}")

    text_cfg_dir, z_cfg_dir = _fetch_configs()

    # --- text encoder: Qwen3-4B config + on-disk bf16 weights ---
    config = AutoConfig.from_pretrained(text_cfg_dir)
    text_encoder = AutoModel.from_config(config)          # bare Qwen3Model
    state = load_file(str(TEXT_ENCODER_FILE))
    # the ComfyUI file uses CausalLM naming ("model.layers…"); a bare model wants
    # them unprefixed
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    missing, unexpected = text_encoder.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if not m.startswith("lm_head")]
    if len(real_missing) > 5:
        raise RuntimeError(f"text encoder mismatch: {len(real_missing)} missing keys, e.g. {real_missing[:3]}")
    text_encoder = text_encoder.to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(text_cfg_dir)

    # --- transformer + VAE from the single files ---
    transformer = ZImageTransformer2DModel.from_single_file(str(TRANSFORMER_FILE), torch_dtype=dtype)
    # ae.safetensors is the 16-latent-channel Flux AE; without an explicit config
    # from_single_file guesses a 4-channel SD VAE and the shapes clash
    vae_cfg = z_cfg_dir / "vae"
    if (vae_cfg / "config.json").exists():
        vae = AutoencoderKL.from_single_file(str(VAE_FILE), config=str(vae_cfg), torch_dtype=dtype)
    else:
        vae = AutoencoderKL.from_single_file(str(VAE_FILE), torch_dtype=dtype)

    sched_cfg = z_cfg_dir / "scheduler" / "scheduler_config.json"
    if sched_cfg.exists():
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(str(sched_cfg.parent))
    else:
        scheduler = FlowMatchEulerDiscreteScheduler()

    pipe = ZImageImg2ImgPipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, transformer=transformer,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def img2img(pipe, image, prompt: str, strength: float, seed: int,
            steps: int = 9, on_step=None) -> np.ndarray:
    """Run the turbo img2img pass; returns linear-light RGB float32."""
    import torch

    from heron.color import srgb

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    kwargs = dict(
        prompt=prompt, image=image, strength=float(strength),
        num_inference_steps=steps, guidance_scale=1.0,   # turbo: CFG off
        generator=gen,
    )
    if on_step is not None:
        kwargs["callback_on_step_end"] = on_step
    out = pipe(**kwargs).images[0]
    return srgb.srgb_to_linear(np.asarray(out, np.float32) / 255.0)
