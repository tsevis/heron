"""Heron web app — Phase 2 demo surface (CLAUDE.md §2.1–2.3).

FastAPI wrapper around the engine: upload a photo, turn physical dials, render
the physics field, optionally finish with the neural pass. Renders run as
background jobs that report real progress (the neural stage streams actual
diffusion steps), polled by the UI. The engine stays UI-agnostic (§2.5).

Run:
    .venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8077
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from heron.color import palettes as _palettes
from heron.core import io as hio
from heron.engine import render_image
from heron.instruments import available_instruments, list_presets
from webapp import layers as weblayers

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="Heron — Optical Bench")

SETTINGS_FILE = ROOT / "settings.json"
DIALS_FILE = ROOT / "dials.json"


def _load_dials() -> dict:
    """The per-instrument dial map — one source for every frontend.

    Served through ``/api/meta`` so the web UI and the Photoshop panel read the
    same definitions over HTTP instead of each keeping a copy that drifts.
    Fails loudly: a frontend with no dials is broken, not degraded.
    """
    import json

    try:
        return json.loads(DIALS_FILE.read_text())
    except Exception as e:
        raise RuntimeError(f"cannot read the dial map at {DIALS_FILE}: {e}") from e


DIALS = _load_dials()


def _load_settings() -> dict:
    try:
        import json

        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    import json

    SETTINGS_FILE.write_text(json.dumps(data, indent=2))

COMFY_MODELS_DIR = Path("~/AI/ComfyUI/models").expanduser()   # read-only


def neural_models() -> list[dict]:
    """Scan the local checkpoints and report honestly what can run offline.

    Single-file SDXL/SD15 checkpoints are self-contained. Flux-2 / Qwen-Image
    exist here only as bare transformers whose text encoders are ComfyUI-format
    single files — diffusers cannot assemble those offline, so they are listed
    as gated (enabling them means downloading HF-format components).
    """
    out: list[dict] = []
    ckpt_dir = COMFY_MODELS_DIR / "checkpoints"
    if ckpt_dir.exists():
        for f in sorted(ckpt_dir.glob("*.safetensors")):
            family = "sdxl" if "xl" in f.name.lower() else "sd15"
            note = "depth-ControlNet img2img" if family == "sdxl" else                    "plain img2img (no depth control — SD 1.5)"
            out.append({"id": f.name, "label": f.stem, "family": family,
                        "ready": True, "note": note})
    dm = COMFY_MODELS_DIR / "diffusion_models"
    try:
        from neural.zimage import components_present

        z_ok, z_why = components_present()
    except Exception as e:
        z_ok, z_why = False, str(e)
    if z_ok:
        out.append({"id": "zimage-turbo", "label": "Z-Image Turbo (on-disk)", "family": "zimage",
                    "ready": True,
                    "note": "assembled from your ComfyUI files — 9-step turbo, ~70 s/frame; keep strength ≤ 0.35 or it will replace the person"})
    if any(dm.glob("*lux*2*lein*")) or any(dm.glob("*Flux2*")):
        out.append({"id": "flux2-klein", "label": "Flux-2-Klein 9B", "family": "flux",
                    "ready": False,
                    "note": "cannot run from this system: its Mistral-3 text encoder is not on "
                            "disk anywhere (a large download you have declined)"})
    if any(dm.glob("qwen_image*")):
        out.append({"id": "qwen-image", "label": "Qwen-Image 20B", "family": "qwen",
                    "ready": False,
                    "note": "all weights are on disk but the 20B fp8 transformer must upcast to "
                            "~40 GB and runs minutes-per-frame on MPS — say the word if you want it anyway"})
    return out


@app.get("/api/models")
def api_models():
    return {"models": neural_models(),
            "default": "juggernautXL_ragnarokBy.safetensors"}


# img2img prompt per Layer-B engine — the finisher must speak each instrument's
# visual language, not thermal-camera language everywhere.
ENGINE_PROMPTS: dict[str, str] = {
    "thermal": "",   # empty -> ControlParams default (thermal-tuned)
    "xray": ("authentic x-ray fluoroscopy radiograph, translucent glowing body, "
             "cobalt phosphor screen, soft radiographic grain, deep dark background"),
    "photon": ("authentic night vision image intensifier capture, phosphor green glow, "
               "photon shot noise grain, soft tube optics, dark night scene"),
    "nir": ("infrared film photograph, wood effect glowing foliage, waxy translucent skin, "
            "dark sky, fine analog grain"),
    "corona": ("kirlian electrophotography plate, corona discharge streamers, luminous aura, "
               "dark photographic plate, film grain"),
    "schlieren": ("schlieren optics laboratory photograph, refractive density gradients, "
                  "smooth mid-gray field, subtle grain"),
    "lumen": ("bioluminescent organism photograph, translucent luminous body glowing in "
              "darkness, soft subsurface scatter"),
    "spectral": ("prism spectral dispersion photograph, chromatic rainbow separation, "
                 "laboratory plate, subtle grain"),
}

_neural_pipe = None                 # hot pipeline (exactly one model at a time)
_neural_key: str | None = None      # which checkpoint the hot pipeline serves
_neural_family: str = "sdxl"
_render_lock = threading.Lock()     # one render at a time
_jobs: dict[str, dict] = {}         # job_id -> {state, progress, stage, ...}


class RenderRequest(BaseModel):
    image_id: str
    instrument: str = "thermograph"
    preset: str | None = "translucent_glow"
    overrides: dict[str, float | int | str] = Field(default_factory=dict)
    seed: int = 7
    use_ai: bool = True
    neural: bool = False
    neural_strength: float = 0.52   # the user-picked default draw
    neural_seed: int = 3
    neural_draft: bool = False      # 832px / 18 steps — ~2.5x faster preview
    neural_model: str = "juggernautXL_ragnarokBy.safetensors"
    render_res: int | None = None   # long side of the output; None = work_res (Phase 4)
    layers: bool = False            # also write per-stage PNGs + manifest (Phase 3)
    graph_layers: bool = True       # include the Radiance Graph channels when layers=True


def _load_model(job: dict, model_id: str):
    """Load (or reuse) the requested model; evict the previous one first —
    a second fp32 SDXL in memory would double ~10 GB of RAM."""
    global _neural_pipe, _neural_key, _neural_family
    import gc

    import torch

    entry = next((m for m in neural_models() if m["id"] == model_id), None)
    if entry is None:
        raise RuntimeError(f"unknown neural model '{model_id}'")
    if not entry["ready"]:
        raise RuntimeError(f"{entry['label']} is not runnable offline: {entry['note']}")

    if _neural_pipe is not None and _neural_key == model_id:
        return entry["family"]

    if _neural_pipe is not None:
        job["stage"] = "unloading previous model"
        _neural_pipe = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    job["stage"] = f"loading {entry['label']} (~30–60 s first use)"
    ckpt = COMFY_MODELS_DIR / "checkpoints" / model_id
    from heron.core.device import get_device

    dev = get_device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    if entry["family"] == "zimage":
        from neural.zimage import load_pipeline as _load_z

        _neural_pipe = _load_z(dev, dtype)
    elif entry["family"] == "sdxl":
        from neural.controlnet_finisher import ControlParams
        from neural.grid import _load_pipeline

        _neural_pipe = _load_pipeline(ControlParams(checkpoint=ckpt))
    else:  # sd15 — plain img2img, no depth ControlNet on disk for 1.5
        from diffusers import StableDiffusionImg2ImgPipeline

        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            str(ckpt), torch_dtype=dtype, safety_checker=None)
        pipe.to(dev)
        pipe.set_progress_bar_config(disable=True)
        _neural_pipe = pipe
    _neural_key = model_id
    _neural_family = entry["family"]
    return entry["family"]


def _neural_pass(job: dict, render_linear: np.ndarray, depth: np.ndarray,
                 strength: float, seed: int, draft: bool = False,
                 prompt: str = "", model_id: str = "") -> np.ndarray:
    """img2img finish over the physics render; reports per-step progress."""
    import cv2
    import torch

    from heron.color import srgb
    from neural.controlnet_finisher import ControlParams, _fit, _pil

    p = ControlParams()
    family = _load_model(job, model_id or "juggernautXL_ragnarokBy.safetensors")

    if depth.shape != render_linear.shape[:2]:
        depth = cv2.resize(depth, (render_linear.shape[1], render_linear.shape[0]))
    native = 768 if family == "sd15" else p.max_side   # SD1.5 was trained small
    max_side = min(native, 832) if draft else native
    image = _fit(_pil(render_linear), max_side)

    if family == "zimage":
        from neural import zimage as _z

        steps = 6 if draft else 9
        est = max(1, int(steps * float(strength)))

        def _zstep(pipe, step, timestep, kw):
            job["progress"] = 0.40 + 0.58 * min(1.0, (step + 1) / est)
            job["stage"] = f"neural finish — step {step + 1}/{est}"
            return kw

        return _z.img2img(_neural_pipe, image, prompt or p.prompt,
                          strength, seed, steps=steps, on_step=_zstep)

    steps = 18 if draft else 24
    est_steps = max(1, int(steps * float(strength)))

    def _on_step(pipe, step, timestep, kw):
        job["progress"] = 0.40 + 0.58 * min(1.0, (step + 1) / est_steps)
        job["stage"] = f"neural finish — step {step + 1}/{est_steps}"
        return kw

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    kwargs = dict(
        prompt=prompt or p.prompt, negative_prompt=p.negative_prompt,
        image=image, strength=float(strength),
        num_inference_steps=steps, guidance_scale=p.guidance, generator=gen,
        callback_on_step_end=_on_step,
    )
    if family == "sdxl":
        depth_rgb = np.repeat(np.clip(depth, 0, 1)[..., None], 3, axis=-1)
        kwargs["control_image"] = _fit(_pil(srgb.srgb_to_linear(depth_rgb)), max_side).resize(image.size)
        kwargs["controlnet_conditioning_scale"] = 0.75
    out = _neural_pipe(**kwargs).images[0]
    return srgb.srgb_to_linear(np.asarray(out, np.float32) / 255.0)


def _run_job(job_id: str, req: RenderRequest) -> None:
    job = _jobs[job_id]
    try:
        with _render_lock:
            job["state"] = "running"
            t0 = time.time()
            job["stage"] = "scene graph + physics"
            job["progress"] = 0.05
            overrides = [f"{k}={v}" for k, v in req.overrides.items()]
            result = render_image(
                UPLOADS / req.image_id, instrument=req.instrument,
                preset=req.preset or None, overrides=overrides,
                use_ai=req.use_ai, seed=req.seed,
                render_res=req.render_res,
            )
            physics_s = time.time() - t0
            job["progress"] = 0.40 if req.neural else 0.95

            image = result.image
            neural_s = 0.0
            if req.neural:
                t1 = time.time()
                engine = str(result.meta.get("engine", "thermal"))
                image = _neural_pass(job, image, result.graph.depth,
                                     req.neural_strength, req.neural_seed,
                                     draft=req.neural_draft,
                                     prompt=ENGINE_PROMPTS.get(engine, ""),
                                     model_id=req.neural_model)
                neural_s = time.time() - t1

            job["stage"] = "writing output"
            base = uuid.uuid4().hex[:10]
            out_name = f"{base}.png"
            hio.write_image(OUTPUTS / out_name, image)

            manifest = None
            if req.layers:
                job["stage"] = "writing layers"
                # A diffusion output cannot be decomposed back into emission /
                # noise / palette, so folding it into the physics group would
                # destroy the stack (Phase-3 §6). The group's composite is
                # therefore always the PHYSICS render, and the neural pass is
                # published separately as its own top layer.
                composite_url = f"/outputs/{out_name}"
                if req.neural:
                    physics_name = f"{base}_physics.png"
                    hio.write_image(OUTPUTS / physics_name, result.image)
                    composite_url = f"/outputs/{physics_name}"

                manifest = weblayers.build_manifest(
                    composite_url, result.stages, result.graph,
                    OUTPUTS, base, include_graph=req.graph_layers,
                )
                if req.neural:
                    # The finisher runs at max_side 1024 (832 draft), so on a
                    # full-resolution render this layer is much smaller than the
                    # group it sits above and will be upscaled on import. Report
                    # its true size so the client can say so rather than quietly
                    # stretching it.
                    manifest["finish"] = {
                        "name": "neural finish",
                        "url": f"/outputs/{out_name}",
                        "mode": "normal",
                        "model": req.neural_model,
                        "strength": req.neural_strength,
                        "size": [int(image.shape[1]), int(image.shape[0])],
                        "composite_size": [int(result.image.shape[1]),
                                           int(result.image.shape[0])],
                        "note": "diffusion finish — cannot be decomposed; keep it "
                                "above the physics group, never merged into it",
                    }

            saved_to = None
            cfg = _load_settings()
            if cfg.get("save_dir"):
                try:
                    save_dir = Path(cfg["save_dir"]).expanduser()
                    save_dir.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    nice = f"heron_{stamp}_{req.instrument}_{req.preset or 'custom'}.png"
                    hio.write_image(save_dir / nice, image,
                                    bit_depth=int(cfg.get("bit_depth", 8)))
                    saved_to = str(save_dir / nice)
                except Exception as e:
                    job["save_warning"] = f"could not save copy: {e}"

            job.update({
                "saved_to": saved_to,
                "state": "done", "progress": 1.0, "stage": "done",
                "url": f"/outputs/{out_name}",
                "timing": {"physics_s": round(physics_s, 1), "neural_s": round(neural_s, 1)},
                "meta": {k: v for k, v in result.meta.items() if k != "source"},
            })
            if manifest is not None:
                job["manifest"] = manifest
    except Exception as e:
        job.update({"state": "error", "stage": "error", "error": str(e)[:500]})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "img.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        raise HTTPException(400, "unsupported image type")
    image_id = uuid.uuid4().hex[:12] + suffix
    (UPLOADS / image_id).write_bytes(await file.read())
    return {"image_id": image_id, "url": f"/uploads/{image_id}"}


@app.post("/api/upload-raw")
async def upload_raw(request: Request):
    """Accept image bytes as the request body, no multipart envelope.

    The Photoshop panel exports the document to a PNG and POSTs it directly:
    UXP's FormData support is unreliable, and a raw body needs nothing but an
    ArrayBuffer. The magic bytes are checked rather than a filename, because
    there is no filename to trust here.
    """
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty body")
    signatures = {
        b"\x89PNG\r\n\x1a\n": ".png",
        b"\xff\xd8\xff": ".jpg",
        b"II*\x00": ".tif",
        b"MM\x00*": ".tif",
    }
    suffix = next((s for magic, s in signatures.items() if data.startswith(magic)), None)
    if suffix is None:
        raise HTTPException(400, "body is not a PNG, JPEG or TIFF")

    image_id = uuid.uuid4().hex[:12] + suffix
    (UPLOADS / image_id).write_bytes(data)
    return {"image_id": image_id, "url": f"/uploads/{image_id}", "bytes": len(data)}


@app.get("/api/meta")
def meta():
    from heron.instruments import load_instrument

    def _presets_with_desc(inst: str) -> dict[str, str]:
        out = {}
        for preset in list_presets(inst):
            try:
                out[preset] = str(load_instrument(inst, preset=preset).get("description", ""))
            except Exception:
                out[preset] = ""
        return out

    return {
        "instruments": {i: _presets_with_desc(i) for i in available_instruments()},
        "palettes": _palettes.available_palettes(),
        "dials": DIALS,
        "disclaimer": "Artistic simulation — not real thermography or measurement.",
    }


def _start_job(req: RenderRequest) -> dict:
    if not (UPLOADS / req.image_id).exists():
        raise HTTPException(404, f"unknown image_id {req.image_id}")
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {"state": "queued", "progress": 0.0, "stage": "queued"}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/render")
def render(req: RenderRequest):
    return _start_job(req)


@app.post("/api/render-layers")
def render_layers(req: RenderRequest):
    """Same job, but the finished job also carries a layer manifest.

    The Photoshop bridge (Phase 3) re-imports these as a named, editable group;
    a flattened result is a failure mode there, not a shortcut.
    """
    return _start_job(req.model_copy(update={"layers": True}))


class Settings(BaseModel):
    save_dir: str = ""
    bit_depth: int = 8


def _volatile_warning(save_dir: str) -> str:
    """Warn when the save folder sits somewhere the OS will empty.

    Not hypothetical: the save folder was set to /tmp/heron_saves and macOS
    purged it, silently destroying every "saved" render. A warning rather than
    a rejection — a deliberate scratch folder is a legitimate choice.
    """
    if not save_dir:
        return ""
    try:
        p = Path(save_dir).expanduser().resolve()
    except Exception:
        return ""
    volatile = (Path("/tmp").resolve(), Path("/private/tmp").resolve(), Path("/var/tmp").resolve())
    if any(p == v or v in p.parents for v in volatile):
        return (f"{p} is a temporary folder — macOS empties it periodically, so "
                "saved renders will disappear. Somewhere under ~/Pictures is safer.")
    return ""


@app.get("/api/settings")
def get_settings():
    cfg = _load_settings()
    save_dir = cfg.get("save_dir", "")
    return {"save_dir": save_dir, "bit_depth": int(cfg.get("bit_depth", 8)),
            "save_dir_warning": _volatile_warning(save_dir)}


@app.post("/api/settings")
def set_settings(cfg: Settings):
    if cfg.bit_depth not in (8, 16):
        raise HTTPException(400, "bit_depth must be 8 or 16")
    if cfg.save_dir:
        try:
            d = Path(cfg.save_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(400, f"cannot use save folder: {e}")
    _save_settings({"save_dir": cfg.save_dir, "bit_depth": cfg.bit_depth})
    return get_settings()


@app.post("/api/browse-folder")
def browse_folder():
    """Open the native macOS folder chooser and return the picked path.

    Possible only because Heron runs locally: the server process can summon a
    real system dialog, which a browser page never could.
    """
    import subprocess

    script = ('POSIX path of (choose folder with prompt '
              '"Choose the Heron save folder" default location (path to pictures folder))')
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"cancelled": True}
    if r.returncode != 0:
        return {"cancelled": True}
    return {"path": r.stdout.strip()}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
