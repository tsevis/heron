# Models

**Heron runs with no models at all.** Every instrument works on the classical
path (`--no-ai`), which uses only numpy and OpenCV. Nothing below is required to
get a render.

What the models buy is Layer A — *scene understanding*. The classical fallback
estimates a subject matte with GrabCut and derives a weak pseudo-depth from it.
That is enough for a plausible image, but real depth and a clean matte are the
difference between a figure with three-dimensional form and a silhouette with a
gradient in it.

## First run

```bash
python scripts/fetch_models.py
```

It reports what is present, what is missing, and roughly how large the download
is, then asks. `./heron.sh` runs the same check on startup and offers once.

```bash
python scripts/fetch_models.py --check   # report only; exit 1 if incomplete
python scripts/fetch_models.py --yes     # no prompt
```

The Hub rate-limits unauthenticated downloads. If a fetch is slow or throttled,
`huggingface-cli login` (or exporting `HF_TOKEN`) fixes it — and it is required
anyway for SAM 3, which is gated.

| Role | Model | Size | What it gives you |
|---|---|---|---|
| depth | `depth-anything/Depth-Anything-V2-Large-hf` | ~1.3 GB | real depth — 3-D form, thickness, grazing-angle falloff |
| matte | `hustvl/vitmatte-base-composition-1k` | ~390 MB | clean subject matte; removes the classical matte's edge leaks |
| segment | `facebook/sam3` **(gated)** | ~3.4 GB | per-region materials — skin vs hair vs fabric vs metal |

**SAM 3 is gated.** Accept the SAM License at
<https://huggingface.co/facebook/sam3>, run `huggingface-cli login`, then fetch.
The script says so if it hits a 403 rather than leaving you with a bare error.

The repo ships the same weights twice (`model.safetensors` and `sam3.pt`, 3.2 GB
each) and Heron loads `sam3.pt`, so the other is skipped — hence ~3.4 GB rather
than 6.5 GB. It also ships its own python package, which Heron imports by path,
so that download is a full snapshot rather than a weights-only filter.

## Where they are looked for

In this order, so an existing install always wins over a download:

1. `$HERON_MODELS/<name>` — a shared directory outside the repo
2. `<repo>/models/<name>` — where `fetch_models.py` installs
3. legacy machine-specific paths — so the original development machine keeps
   working without re-downloading anything
4. the Hugging Face cache, by model id

`models/` is gitignored. Weights are never committed.

## The engine never downloads

Model loaders pass `local_files_only=True`, and fetching is a separate explicit
script. An engine that quietly reaches for the network mid-render is neither
offline nor deterministic, and both properties are load-bearing here: the same
seed and parameters must give the same pixels, on a machine with no connection.

If a model is missing, `--ai` falls back to the classical path for that channel.

## What each one changes if you skip it

- **No depth** — form falls back to a pseudo-depth derived from the matte. The
  figure reads flatter; grazing-angle roll-off and occlusion warmth have little
  to work with.
- **No matte** — GrabCut instead. Usable, but it leaks at hair edges and
  bokeh backgrounds, and the physics keys on that boundary.
- **No SAM 3** — materials collapse to one "unknown" class. The render still
  works, but everything material-driven goes inert: hair root-to-tip conduction,
  fabric transmission, the Wood effect in NIR, corona activity in Kirlian.

## Not fetched automatically

**Neural finishers** (`neural/`) are optional post-passes and are separate again.
They read checkpoints from a local ComfyUI install read-only; the app lists what
it finds and explains what it cannot run. Nothing is downloaded on your behalf.
