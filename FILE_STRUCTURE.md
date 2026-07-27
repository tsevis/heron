# File structure

How Heron is laid out and, more usefully, **why** — which direction dependencies
run, and where a given kind of change belongs.

## The one rule that explains the layout

```
uxp/  webapp/  cli.py        frontends — may call the engine
        │
        ▼
     heron/                  the engine — calls nothing above it
        │
        ▼
   numpy / opencv / (optional torch in scene/ai only)

     neural/                 optional post-passes — heron/ NEVER imports this
```

`heron/` is UI-agnostic and offline. It does not import FastAPI, UXP, or any
diffusion runtime. Two tests enforce this (`tests/test_presets.py`), because it
is the property that lets the same engine serve a CLI, a web app and a Photoshop
panel without any of them knowing about each other.

## Top level

| Path | What it is |
|---|---|
| `heron/` | the engine package — the 4-layer stack, 94 files |
| `webapp/` | FastAPI service + static web UI |
| `uxp/` | Photoshop panel (4 source files, no build step) |
| `neural/` | optional finishers — SDXL/Z-Image img2img, GAN architectures |
| `scripts/` | gallery, palette strips, contact sheets, model smoke tests |
| `tests/` | 328 tests |
| `gallery/` | everything the README embeds |
| `cli.py`, `heron.sh`, `Heron.command` | entry points |
| `MODELS.md` | what the optional models buy, and how they are found |
| `models/` | fetched Layer A models (gitignored, created on demand) |

## `heron/` — the engine

The stack is fixed; every instrument is a *configuration* of it, never a
separate codebase.

```
heron/
├── core/         io, device, seeds, graph.py (the RadianceGraph + its cache)
├── color/        oklab, srgb, wavelength, palettes/ (16 JSON ramps)
├── scene/        LAYER A — scene understanding
│   ├── graph_builder.py   classical path (delight, saliency, GrabCut matte)
│   ├── upsample.py        guided-filter lift to render resolution (Phase 4)
│   └── ai/                optional models (depth, ViTMatte, SAM 3), and
│                          registry.py — where they live, what is missing
├── physics/      LAYER B — thermal, xray, photon, nir, kirlian, schlieren,
│                          lumen, spectral (+ diffusion, curlnoise, facemap)
├── sensor/       LAYER C — bloom, fpn, netd, agc, msx, optics, quantum,
│                          scanlines, faceplate, chromatic, dither, furniture
├── art/          LAYER D — emphasis, dirbloom, poisson detail fusion
├── instruments/  one renderer + one YAML per instrument, and the loader
├── materials/    table.yaml — THE material table (curated, never generated)
├── engine.py     render_image(): ties Layer A to an instrument's B/C/D
├── video.py      clip rendering with temporal stabilisation (Phase 4)
├── presets.py    shareable self-contained presets (Phase 5)
└── cli.py        typer commands
```

### Where a change belongs

| You want to… | Edit |
|---|---|
| add an instrument | a YAML in `instruments/` + a renderer; register the engine |
| change how a camera looks | its YAML, or a new preset in it — **not** Python |
| add a physical effect | `physics/` (Layer B) or `sensor/` (Layer C) |
| change what a dial is called or its range | `webapp/dials.json` — one source for every frontend |
| add a palette | a JSON ramp in `color/palettes/` |
| change scene understanding | `scene/` — and remember Layer A must stay skippable (`--no-ai`) |

## `webapp/`

| File | Role |
|---|---|
| `server.py` | FastAPI: upload, render jobs with real progress, settings |
| `layers.py` | decomposes a render into per-stage PNGs + a manifest |
| `dials.json` | **the** per-instrument dial map, served via `/api/meta` |
| `static/index.html` | the Optical Bench UI |

`dials.json` exists so the web UI and the Photoshop panel read one definition
and cannot drift. `tests/test_dials.py` guards it — including that every dialled
physics key is actually read by its renderer, which caught six dead dials.

## `uxp/` — the Photoshop panel

| File | Role |
|---|---|
| `manifest.json` | UXP manifest v5 |
| `index.html` | markup + styles, UXP's built-in `sp-*` widgets |
| `main.js` | UI, service discovery, request assembly |
| `bridge.js` | everything touching the Photoshop DOM |

No bundler and no `node_modules` — deliberate. See `uxp/README.md` for install
and for the UXP quirks that cost real time.

## `neural/` — strictly optional

Post-passes over the physics output. `heron/` never imports this. Contains the
working SDXL/Z-Image img2img finishers, and CycleGAN/pix2pixHD architectures
that are **built but untrained** (no data on disk).

## Ignored, and why

`CLAUDE.md` and `documents/` are gitignored on purpose: they name absolute
machine paths and other private projects. `CLAUDE.md` was purged from git
history, not merely deleted. Also ignored: `.venv/`, model weights,
`.heron_graph/` caches, `webapp/{uploads,outputs,settings.json}`, `out/`.
