"""Thermography — the flagship physics engine (CLAUDE.md §0.4, §3.2).

Builds a temperature field T(x,y) from *what things are* (materials + subject
matte), not from how they are lit, then:
  * melts it with edge-stopped heat diffusion (blobby isotherms),
  * composites a blurred thermal-environment *reflection* into low-emissivity
    materials (water, polished metal) — the reason water and metal mirror the
    sitter instead of glowing, and the one thing gradient-map filters cannot do,
  * windows the apparent temperature to a Kelvin span into a [0,1] signal.

The signal is temperature-linear (as a radiometrically-corrected FLIR display
shows), fed downstream to Layer C (AGC, noise, palette). Honesty guardrail
(§2.8): this is artistic simulation, not measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import ndimage

import cv2
import numpy as np

from heron.color import srgb
from heron.core.graph import RadianceGraph
from heron.materials import MaterialTable, load_material_table
from heron.physics import diffusion, facemap, shading

_KELVIN = 273.15


@dataclass(frozen=True)
class ThermalParams:
    ambient_c: float = 22.0
    core_c: float = 34.0
    exertion: float = 0.0            # 0..1, raises subject temperature
    emissivity_reflection: float = 1.0  # strength of reflective compositing
    detail: float = 0.0              # visible-luminance term: OFF (reflectance != temperature)
    shape_cooling: float = 0.12      # legacy flat normal term (kept for configs)
    grazing_drop_c: float = 5.0      # emissivity roll-off at grazing angles (3-D form)
    occlusion_warm_c: float = 2.2    # crevices trap heat (eye sockets, under chin)
    transmission: float = 1.0        # LWIR see-through of fabric/hair (§7)
    hair_root_warmth: float = 0.45   # roots conduct scalp heat; tips cool to hair temp
    grazing_hair_scale: float = 0.35 # hair edges are strands, not a hard silhouette
    hair_detail_c: float = 2.0       # geometric structure in hair (deg C)
    skin_detail_c: float = 0.5       # gentle geometric texture over skin
    conduction_px: float = 3.0       # heals label-boundary steps (continuous field)
    subsurface_rim_c: float = 1.8    # translucent flesh: thin edges glow (blue-nude register)
    face_map: float = 1.0            # canonical facial heat map strength (§0.4)
    body_structure: float = 1.0      # core->extremity perfusion falloff
    extremity_drop_c: float = 3.5    # how much cooler thin extremities read
    diffusion_iters: int = 6
    span_min_c: float | None = None  # None -> auto window from field
    span_max_c: float | None = None


@dataclass
class ThermalField:
    temperature_c: np.ndarray  # apparent temperature (H,W) in Celsius
    signal: np.ndarray         # windowed [0,1] pre-sensor signal
    emissivity: np.ndarray     # (H,W) LWIR emissivity map
    span_c: tuple[float, float]


def _geometric_detail(relief: np.ndarray, sigmas=(2.0, 5.0), weights=(1.0, 0.6)) -> np.ndarray:
    """Surface curvature from the DEPTH relief, normalized.

    Detail must come from geometry, never from albedo. A thermal camera cannot
    see a reflectance edge: a dark eyebrow is not colder than pale skin. High-
    passing visible luminance injects a bright/dark rim at every edge, which
    reads as pen-and-ink illustration and is exactly the gradient-map failure
    CLAUDE.md §1 forbids. Curvature is physically meaningful instead — concave
    creases trap heat, convex ridges shed it.
    """
    out = np.zeros_like(relief, dtype=np.float32)
    for sigma, w in zip(sigmas, weights):
        curv = cv2.GaussianBlur(relief, (0, 0), sigmaX=sigma) - relief
        out += w * curv / max(float(curv.std()), 1e-6)
    return out.astype(np.float32)


def _seed_temperature(
    graph: RadianceGraph,
    table: MaterialTable,
    p: ThermalParams,
    source_linear: np.ndarray | None = None,
) -> np.ndarray:
    """Seed the raw temperature field before diffusion.

    The subject is not a uniform warm blob: a core->extremity perfusion falloff
    shapes the body, and the canonical facial heat map (hot tear ducts/ears/neck,
    cold nose) is warped onto the detected face (§0.4). Those two terms are what
    separate a thermogram from a gradient map.
    """
    matte = graph.matte
    material_T = graph.material_map(table.temp_mean_map(), default=p.ambient_c)
    core = p.core_c + p.exertion * 4.0

    # Inside the subject each material keeps ITS OWN temperature — hair is a good
    # insulator and reads far cooler than skin, clothing cooler still. Pulling the
    # whole silhouette to a single core temperature (the previous behaviour) is
    # what flattened every figure into one uniform warm blob.
    unknown_id = graph.material_names.index("unknown") if "unknown" in graph.material_names else -1
    labelled = graph.material_ids != unknown_id
    subject_T = np.where(labelled, material_T, core)
    T = (1.0 - matte) * material_T + matte * subject_T

    # --- hair is a gradient, not a flat mass ---
    # Hair is heated by conduction from the scalp and loses that heat along its
    # length, so roots read close to skin temperature and tips fall toward
    # ambient. Painting the whole hair region at one temperature produced a flat
    # black cut-out; this restores the roots-warm/tips-cool falloff that makes
    # hair read as hair.
    if "hair" in graph.material_names and p.hair_root_warmth > 0.0:
        hair_id = graph.material_names.index("hair")
        hair = graph.material_ids == hair_id
        if hair.any():
            skin_like = matte > 0.5
            if "hair" in graph.material_names:
                skin_like = skin_like & ~hair
            if skin_like.any():
                dist = ndimage.distance_transform_edt(~skin_like).astype(np.float32)
                scale = max(float(np.percentile(dist[hair], 75)), 2.0)
                t = np.clip(dist / scale, 0.0, 1.0)           # 0 at scalp -> 1 at tips
                t = cv2.GaussianBlur(t, (0, 0), sigmaX=max(matte.shape) * 0.003)
                skin_T = table.get("skin").temp_mean_c
                hair_T = table.get("hair").temp_mean_c
                warm_root = skin_T - p.hair_root_warmth * (skin_T - hair_T)
                T = np.where(hair, warm_root * (1.0 - t) + hair_T * t, T)

    # body structure: thin extremities are cooler than the well-perfused core
    if p.body_structure > 0.0:
        perfusion = facemap.body_heat_structure(matte)
        T = T - p.body_structure * p.extremity_drop_c * (1.0 - perfusion) * matte

    # canonical facial heat map warped onto the real face
    if p.face_map > 0.0 and source_linear is not None:
        skin = table.get("skin")
        # face region is a Layer-A artifact carried by the graph; the physics
        # layer must stay model-free (re-running SAM 3 here cost ~10 s per
        # render before this was cached)
        cached_face = getattr(graph, "extras", {}).get("face_region")
        found = facemap.facial_heat_offsets(
            source_linear, skin.zones, matte=matte,
            region=cached_face, allow_ai=False,
        )
        if found is not None:
            offsets, face_mask = found
            T = T + p.face_map * offsets * np.maximum(matte, face_mask)

    # inverted-luminance micro-detail inside the subject (V2T grayscale trick)
    inv = 1.0 - srgb.luminance(graph.albedo)
    inv_centered = inv - float(np.mean(inv))
    T = T + p.detail * (core - p.ambient_c) * inv_centered * matte

    # --- strand detail ---
    # Real thermal hair is not a smooth mass: individual strands and waves are
    # fully resolved, with hot scalp showing between cooler strands. The smooth
    # root->tip gradient above would erase that, so the source photograph's own
    # high-frequency structure is injected back, at several scales (fine strands,
    # then waves), strongest inside hair and lighter over skin.
    if p.hair_detail_c > 0.0 or p.skin_detail_c > 0.0:
        strands = _geometric_detail(shading.subject_relief(graph.depth, matte))
        weight = np.zeros_like(matte)
        if "hair" in graph.material_names:
            hid = graph.material_names.index("hair")
            weight = np.where(graph.material_ids == hid, p.hair_detail_c, 0.0).astype(np.float32)
        weight = np.where(weight > 0.0, weight, p.skin_detail_c * matte)
        T = T + strands * weight

    # --- three-dimensional form from depth relief (§3.2) ---
    # The raw depth is scene-normalized, so the subject's own relief must be
    # recovered before differentiating; otherwise the figure reads as a flat
    # plate. Grazing-angle emissivity roll-off models the volume, and occlusion
    # warms the crevices.
    _relief, _n, grazing, occlusion = shading.form_terms(graph.depth, matte)
    # hair is a mass of strands, not a smooth surface, so the grazing roll-off
    # must not carve a hard black rim around the head
    graze_w = np.ones_like(matte)
    if "hair" in graph.material_names:
        hid = graph.material_names.index("hair")
        graze_w = np.where(graph.material_ids == hid, p.grazing_hair_scale, 1.0).astype(np.float32)
    T = T - p.grazing_drop_c * grazing * matte * graze_w
    T = T + p.occlusion_warm_c * occlusion * matte

    # --- LWIR transmission: the body glowing THROUGH semi-transparent layers ---
    # Fabric and hair are partially transparent to thermal radiation, so the warm
    # body behind them shows through (the semi-transparent tutu of the ballerina
    # reference). Observed temperature is a mix of the surface's own temperature
    # and the body heat behind it, weighted by the material's transmissivity.
    if p.transmission > 0.0:
        tau = np.clip(graph.material_map(table.transmissivity_map(), default=0.0)
                      * p.transmission, 0.0, 0.92)
        # What shows through must carry the BODY'S FORM, not a featureless glow:
        # the warm mass behind a shirt is the torso, brightest along its thick
        # core and falling off toward the silhouette (the tutu reads transparent
        # exactly because the legs' form glows through it).
        body = matte > 0.3
        if body.any():
            edt_body = ndimage.distance_transform_edt(body).astype(np.float32)
            thickness = np.clip(edt_body / max(float(np.percentile(edt_body[body], 95)), 1e-3), 0.0, 1.0)
        else:
            thickness = np.zeros_like(matte)
        behind = p.ambient_c + (core - p.ambient_c) * np.clip(0.25 + 0.9 * thickness, 0.0, 1.1)
        behind = cv2.GaussianBlur(behind.astype(np.float32), (0, 0), sigmaX=max(matte.shape) * 0.008)
        # a layer is optically thicker at grazing incidence -> less shows through
        tau = tau * (1.0 - 0.5 * grazing)
        T = (1.0 - tau) * T + tau * behind

    # --- translucent flesh: subsurface rim ---
    # In the translucent register light escapes where the medium is THIN, so the
    # silhouette edge carries a soft luminous lift instead of dying to black.
    if p.subsurface_rim_c > 0.0:
        body = matte > 0.3
        if body.any():
            edt_body = ndimage.distance_transform_edt(body).astype(np.float32)
            rim_scale = max(float(np.percentile(edt_body[body], 95)) * 0.10, 2.0)
            rim = np.exp(-edt_body / rim_scale) * matte
            T = T + p.subsurface_rim_c * rim

    # --- conduction across label boundaries ---
    # Temperature is a continuous field: heat conducts across every interface, so
    # a step in T is unphysical. Material labels produce exactly such steps (the
    # hairline seam), which read as a composite rather than a capture. A short
    # conduction pass over the WHOLE frame heals them without touching the
    # large-scale structure. Only emissivity may jump, never temperature.
    if p.conduction_px > 0.0:
        sigma = max(1.0, p.conduction_px * max(T.shape) / 640.0)
        T = cv2.GaussianBlur(T.astype(np.float32), (0, 0), sigmaX=sigma)

    return T.astype(np.float32)


def _apparent_temperature(T: np.ndarray, refl_weight: np.ndarray) -> np.ndarray:
    """Composite a blurred thermal-environment reflection into reflective areas."""
    if float(refl_weight.max()) <= 1e-4:
        return T
    sigma = max(T.shape) * 0.03
    T_env = cv2.GaussianBlur(T, (0, 0), sigmaX=sigma)
    return ((1.0 - refl_weight) * T + refl_weight * T_env).astype(np.float32)


def _window(T: np.ndarray, p: ThermalParams) -> tuple[np.ndarray, tuple[float, float]]:
    """Window apparent temperature to [0,1].

    The auto window is anchored to the physical ambient->core range and then
    widened by the data's tails. Anchoring matters when a clean matte makes the
    subject fill the frame: pure percentiles would otherwise collapse the window
    into the narrow skin range and blow the subject out.
    """
    core = p.core_c + p.exertion * 4.0
    if p.span_min_c is not None:
        tmin = p.span_min_c
    else:
        tmin = min(p.ambient_c - 1.0, float(np.percentile(T, 2.0)))
    if p.span_max_c is not None:
        tmax = p.span_max_c
    else:
        tmax = max(core + 1.0, float(np.percentile(T, 98.0)))
    tmax = max(tmax, tmin + 0.5)
    signal = np.clip((T - tmin) / (tmax - tmin), 0.0, 1.0)
    return signal.astype(np.float32), (tmin, tmax)


def synthesize_thermal(
    graph: RadianceGraph,
    params: ThermalParams | None = None,
    table: MaterialTable | None = None,
    source_linear: np.ndarray | None = None,
) -> ThermalField:
    """Run the thermal physics engine over a Radiance Graph.

    ``source_linear`` enables the canonical facial heat map (face detection needs
    the original image); without it the body structure still applies.
    """
    p = params or ThermalParams()
    table = table or load_material_table()

    T0 = _seed_temperature(graph, table, p, source_linear)

    # edge-stopped diffusion, guided by albedo structure (material boundaries)
    # Edge-stopping used to key on the albedo, which pinned heat at every
    # VISIBLE edge — the camera cannot see those. Key it on the temperature
    # field itself so only genuine thermal interfaces resist diffusion.
    g = diffusion.edge_stop_map(T0 / max(float(np.ptp(T0)), 1e-3), kappa=0.35)
    T = diffusion.anisotropic_diffuse(T0, g, iterations=p.diffusion_iters, dt=0.2)

    emissivity = graph.material_map(table.emissivity_map(), default=0.92)
    refl_weight = np.clip(
        graph.material_map(table.reflective_map(), default=0.0) * p.emissivity_reflection,
        0.0,
        1.0,
    )
    T_app = _apparent_temperature(T, refl_weight)

    signal, span = _window(T_app, p)
    return ThermalField(temperature_c=T_app, signal=signal, emissivity=emissivity, span_c=span)
