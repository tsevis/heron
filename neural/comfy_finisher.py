"""ControlNet img2img finisher through a local ComfyUI (CLAUDE.md §5.1).

Heron's physical output becomes the *control signal*: the Radiance Graph's depth
drives a depth-ControlNet while the rendered thermogram is the img2img latent, so
the diffusion model adds photographic/painterly texture without inventing new
geometry. Strictly a post-pass — the engine remains deterministic and this file
is never imported by ``heron``.

Talks to ComfyUI over its HTTP API (default 127.0.0.1:8188); no cloud calls.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heron.color import srgb

COMFY_HOST = "127.0.0.1:8988"  # the user's ComfyUI lives here — read-only API use only
COMFY_ROOT = Path("~/AI/ComfyUI").expanduser()


@dataclass(frozen=True)
class FinishParams:
    checkpoint: str = "juggernautXL_ragnarokBy.safetensors"
    controlnet: str = "diffusers_xl_depth_full.safetensors"
    positive: str = (
        "thermal infrared photograph, thermographic camera image, glowing body heat, "
        "luminous skin, deep dark background, high detail, cinematic"
    )
    negative: str = "text, watermark, blurry, lowres, cartoon, flat, washed out"
    denoise: float = 0.45      # how far from Heron's render the model may travel
    control_strength: float = 0.8
    steps: int = 26
    cfg: float = 6.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    seed: int = 7


def is_running(host: str = COMFY_HOST, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}/system_stats", timeout=timeout):
            return True
    except Exception:
        return False


def _post(host: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://{host}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _save_input(img_linear: np.ndarray, name: str) -> Path:
    """Write an sRGB PNG into ComfyUI's input folder so nodes can load it."""
    from PIL import Image

    out_dir = COMFY_ROOT / "input"
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.clip(srgb.linear_to_srgb(np.asarray(img_linear, np.float32)) * 255, 0, 255).astype(np.uint8)
    path = out_dir / name
    Image.fromarray(arr).save(path)
    return path


def _graph(render_name: str, depth_name: str, p: FinishParams) -> dict:
    """Build the ComfyUI API graph: img2img + depth ControlNet."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": p.checkpoint}},
        "2": {"class_type": "LoadImage", "inputs": {"image": render_name}},
        "3": {"class_type": "LoadImage", "inputs": {"image": depth_name}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": p.positive, "clip": ["1", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": p.negative, "clip": ["1", 1]}},
        "7": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": p.controlnet}},
        "8": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["5", 0], "negative": ["6", 0], "control_net": ["7", 0],
                "image": ["3", 0], "strength": p.control_strength,
                "start_percent": 0.0, "end_percent": 0.85,
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["4", 0],
                "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
                "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": p.denoise,
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "heron_finish"}},
    }


def finish(
    render_linear: np.ndarray,
    depth: np.ndarray,
    params: FinishParams | None = None,
    host: str = COMFY_HOST,
    timeout_s: float = 600.0,
) -> np.ndarray | None:
    """Run the finisher; returns a linear-light RGB image, or None on failure."""
    p = params or FinishParams()
    if not is_running(host):
        print(f"[heron.neural.comfy] ComfyUI not reachable at {host} — start it first")
        return None

    tag = uuid.uuid4().hex[:8]
    render_name = f"heron_render_{tag}.png"
    depth_name = f"heron_depth_{tag}.png"
    _save_input(render_linear, render_name)
    depth_rgb = np.repeat(np.clip(np.asarray(depth, np.float32), 0, 1)[..., None], 3, axis=-1)
    _save_input(srgb.srgb_to_linear(depth_rgb), depth_name)  # depth is data, keep it linear-through

    client_id = uuid.uuid4().hex
    try:
        resp = _post(host, "/prompt", {"prompt": _graph(render_name, depth_name, p), "client_id": client_id})
    except Exception as e:
        print(f"[heron.neural.comfy] prompt rejected: {e}")
        return None
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        print(f"[heron.neural.comfy] no prompt_id in response: {resp}")
        return None

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}/history/{prompt_id}", timeout=15) as r:
                hist = json.load(r)
        except Exception:
            hist = {}
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            for node in outputs.values():
                for meta in node.get("images", []):
                    return _fetch_image(host, meta)
            print("[heron.neural.comfy] job finished but produced no image")
            return None
        time.sleep(2.0)
    print("[heron.neural.comfy] timed out waiting for ComfyUI")
    return None


def _fetch_image(host: str, meta: dict) -> np.ndarray:
    from io import BytesIO

    from PIL import Image

    q = urllib.parse.urlencode({k: meta[k] for k in ("filename", "subfolder", "type") if k in meta})
    with urllib.request.urlopen(f"http://{host}/view?{q}", timeout=60) as r:
        img = Image.open(BytesIO(r.read())).convert("RGB")
    return srgb.srgb_to_linear(np.asarray(img, np.float32) / 255.0)
