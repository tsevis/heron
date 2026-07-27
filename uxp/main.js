/* Heron Photoshop panel — Phase 3.1 scaffold.
 *
 * The panel is a frontend and nothing else: it talks HTTP to the local Heron
 * service and never imports from `heron/`. Every control it shows is described
 * by the service (`/api/meta`), including the per-instrument dial map, so this
 * panel and the web app cannot drift apart — there is only one definition.
 */

const bridge = require("./bridge.js");

const DEFAULT_PORT = 8077;

const state = {
  base: `http://127.0.0.1:${DEFAULT_PORT}`,
  meta: null,        // /api/meta payload
  dials: {},         // instrument -> dial definition (from the service)
  instrument: "thermograph",
  openGroups: {},    // instrument -> Set of expanded group names
  retryTimer: null,  // background reconnect while the service is down
};

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- status */

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

/* ------------------------------------------------------------ service io */

async function getJSON(path, opts) {
  const res = await fetch(state.base + path, opts);
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

/** Retry connecting until the service appears, so starting ./heron.sh is enough.
 *  Slow on purpose (4 s): this runs forever while offline and must not busy-poll. */
function scheduleRetry() {
  if (state.retryTimer) clearTimeout(state.retryTimer);
  state.retryTimer = setTimeout(() => connect(true), 4000);
}

/**
 * Probe the service. Offline must be obvious and actionable, never a spinner.
 * @param quiet  true for background retries, which must not stomp on the
 *               visible message with "Looking for…" every few seconds.
 */
async function connect(quiet) {
  if (state.retryTimer) { clearTimeout(state.retryTimer); state.retryTimer = null; }
  const port = parseInt($("port").value, 10) || DEFAULT_PORT;
  state.base = `http://127.0.0.1:${port}`;
  if (!quiet) {
    setStatus("Looking for the Heron service…", "busy");
    $("controls").classList.add("hidden");
  }

  try {
    state.meta = await getJSON("/api/meta");
  } catch (e) {
    // Two very different failures look identical from here: the service is not
    // running, or UXP refused the request (the manifest's network permission).
    // Reporting the raw error is what tells them apart — a generic "not
    // running" message sent us looking in the wrong place once already.
    const detail = `${e && e.name ? e.name : "Error"}: ${e && e.message ? e.message : e}`;
    setStatus(
      `Could not reach ${state.base}/api/meta — ${detail}\n\n` +
      `If the Heron service IS running (open ${state.base} in a browser to ` +
      `check), UXP blocked the request and this is a manifest network ` +
      `permission problem. Otherwise start it with ./heron.sh — this panel ` +
      `keeps checking and will connect on its own.`,
      "bad",
    );
    console.error("[Heron] fetch failed", state.base, e);
    $("controls").classList.add("hidden");
    scheduleRetry();
    return;
  }

  state.dials = state.meta.dials || {};
  if (!Object.keys(state.dials).length) {
    setStatus("The service answered but sent no dial map — is it an older build?", "bad");
    return;
  }

  $("disclaimer").textContent = state.meta.disclaimer || "";
  buildInstruments();
  $("controls").classList.remove("hidden");
  setStatus(`Connected on port ${port}.`, "ok");
}

/* ------------------------------------------------------------ dropdowns */

function fillMenu(menuId, items, selected) {
  const menu = $(menuId);
  menu.innerHTML = "";
  for (const item of items) {
    const el = document.createElement("sp-menu-item");
    el.textContent = item;
    if (item === selected) el.setAttribute("selected", "");
    menu.appendChild(el);
  }
}

function buildInstruments() {
  const names = Object.keys(state.dials).filter((i) => state.meta.instruments[i]);
  if (!names.includes(state.instrument)) state.instrument = names[0];
  fillMenu("instrument-menu", names, state.instrument);
  applyInstrument();
}

function applyInstrument() {
  const cfg = state.dials[state.instrument];
  const presets = state.meta.instruments[state.instrument] || {};
  const names = Object.keys(presets).sort();
  const preset = names.includes(cfg.defPreset) ? cfg.defPreset : names[0] || "";
  fillMenu("preset-menu", names, preset);
  showPresetDesc(preset);

  // Spectroscope and NIR-aerochrome synthesize colour directly and bypass the
  // 1-D palette, so offering a palette there would be a lie.
  if (cfg.palette === null) {
    fillMenu("palette-menu", ["—"], "—");
    $("palette").setAttribute("disabled", "");
  } else {
    $("palette").removeAttribute("disabled");
    fillMenu("palette-menu", state.meta.palettes || [], cfg.palette);
  }

  buildDials(cfg);
  showResolutionNote();
  showAiNote();
  showNeuralNote();
  showScopeNote();
  showFullResNote();
}

function showPresetDesc(preset) {
  const presets = state.meta.instruments[state.instrument] || {};
  $("preset-desc").textContent = presets[preset] || "";
}

/* ---------------------------------------------------------------- dials */

/** Which groups are expanded for this camera. First group opens by default so
 *  the panel shows what dials exist without filling a small screen. */
function openSet() {
  if (!state.openGroups[state.instrument]) {
    const first = Object.keys(state.dials[state.instrument].groups)[0];
    state.openGroups[state.instrument] = new Set(first ? [first] : []);
  }
  return state.openGroups[state.instrument];
}

function buildDials(cfg) {
  const host = $("dials");
  host.innerHTML = "";
  const open = openSet();

  for (const [groupName, dials] of Object.entries(cfg.groups)) {
    const group = document.createElement("div");
    group.className = "group";

    const head = document.createElement("div");
    head.className = "ghead";
    const chev = document.createElement("span");
    chev.className = "chev";
    const title = document.createElement("span");
    title.textContent = groupName;
    const count = document.createElement("span");
    count.className = "gcount";
    count.textContent = String(dials.length);
    head.appendChild(chev);
    head.appendChild(title);
    head.appendChild(count);

    const body = document.createElement("div");
    body.className = "gbody";

    const paint = () => {
      const isOpen = open.has(groupName);
      chev.textContent = isOpen ? "▼" : "▶";
      body.classList.toggle("hidden", !isOpen);
    };
    head.addEventListener("click", () => {
      if (open.has(groupName)) open.delete(groupName);
      else open.add(groupName);
      paint();
    });

    for (const d of dials) {
      body.appendChild(makeDial(d));
    }
    paint();

    group.appendChild(head);
    group.appendChild(body);
    host.appendChild(group);
  }
}

function makeDial(d) {
  const row = document.createElement("div");
  row.className = "dial";

  const line = document.createElement("div");
  line.className = "dline";
  const name = document.createElement("span");
  name.className = "dname";
  name.textContent = d.label;
  name.setAttribute("title", d.key);      // the real engine key, on hover
  const val = document.createElement("span");
  val.className = "dval";
  val.textContent = String(d.def);
  line.appendChild(name);
  line.appendChild(val);

  const slider = document.createElement("sp-slider");
  slider.setAttribute("min", d.min);
  slider.setAttribute("max", d.max);
  slider.setAttribute("step", d.step);
  slider.setAttribute("value", d.def);
  slider.dataset.key = d.key;
  slider.dataset.def = String(d.def);

  const sync = () => {
    val.textContent = slider.value;
    // Highlight dials the user has moved off the preset, so it is obvious what
    // is no longer "the preset" when a render looks unexpected.
    row.classList.toggle("moved", parseFloat(slider.value) !== parseFloat(d.def));
  };
  slider.addEventListener("input", sync);
  slider.addEventListener("change", sync);

  row.appendChild(line);
  row.appendChild(slider);
  return row;
}

/* Scene understanding always runs at 1280 and is lifted edge-aware, so the cost
 * of full resolution is the physics/sensor/art pass: ~1.4 s per megapixel
 * measured on this machine. Quote that rather than leave it a surprise. */
function showFullResNote() {
  const size = bridge.documentSize();
  if (!size) {
    $("fullres-note").textContent = "No document open.";
    return;
  }
  const long = Math.max(size.width, size.height);
  if (!$("fullres").checked) {
    $("fullres-note").textContent =
      `Off — renders at 1280 px and Photoshop scales it to ${size.width}×${size.height}. ` +
      `Fast, good for finding the look.`;
    return;
  }
  const mp = (size.width * size.height) / 1e6;
  $("fullres-note").textContent =
    `On — physics runs at ${long} px on the long side (${mp.toFixed(1)} MP, ` +
    `roughly ${Math.round(mp * 1.4)} s). Scene understanding still runs at 1280 ` +
    `and is lifted edge-aware; that part is not full resolution.`;
}

/** Only dials moved off their default are sent, so presets stay meaningful. */
function collectOverrides() {
  const out = {};
  for (const slider of $("dials").querySelectorAll("sp-slider")) {
    const value = parseFloat(slider.value);
    if (Number.isFinite(value) && value !== parseFloat(slider.dataset.def)) {
      out[slider.dataset.key] = value;
    }
  }
  const palette = $("palette").value;
  if (palette && palette !== "—") out["layerC.palette"] = palette;
  return out;
}

function resetDials() {
  for (const row of $("dials").querySelectorAll(".dial")) {
    const slider = row.querySelector("sp-slider");
    if (!slider) continue;
    slider.value = slider.dataset.def;
    const val = row.querySelector(".dval");
    if (val) val.textContent = slider.dataset.def;
    row.classList.remove("moved");
  }
}

/* ----------------------------------------------------------- honest note */

function showAiNote() {
  $("ai-note").textContent = $("useai").checked
    ? "Real depth, matte and materials. First render of a photo is slow (models load, ~40–60 s); after that the scene graph is cached."
    : "Classical fallback — no models. Fast, but coarser matte and a weak pseudo-depth.";
}

/* The finisher is not equally trustworthy across instruments. On Kirlian it
 * invented facial structure the physics never produced, so any camera whose
 * dial map does not vouch for it says so plainly instead of just being slower. */
function showNeuralNote() {
  const cfg = state.dials[state.instrument] || {};
  const on = $("neural").checked;
  $("neural-dials").classList.toggle("hidden", !on);

  if (!cfg.neural) {
    $("neural-note").textContent =
      "Not tuned for this camera — it tends to add geometry rather than texture " +
      "here, so physics-only is the trustworthy path. ~25–60 s per frame.";
    return;
  }
  if (!on) {
    $("neural-note").textContent =
      "Off. Adds sensor texture at ~25–60 s per frame, as a separate top layer.";
    return;
  }
  // The finisher is capped at ~1024 px. Against a full-resolution group that
  // layer gets upscaled several times over, and fine sensor texture — its whole
  // reason to exist — is exactly what does not survive that.
  const size = bridge.documentSize();
  const long = size ? Math.max(size.width, size.height) : 0;
  const clash = $("fullres").checked && long > 1400;
  $("neural-note").textContent = clash
    ? `⚠ Caps at ~1024 px, so against a ${long} px render it arrives soft and ` +
      `gets upscaled ~${(long / 1024).toFixed(1)}×. Its value is fine texture, ` +
      `which does not survive that. Use one or the other, not both.`
    : "Adds sensor texture (hair strands, skin grain). ~25–60 s per frame " +
      "depending on checkpoint. Arrives as its own top layer, above the group.";
}

/* What actually gets sent. Photoshop's PNG-as-copy export flattens the VISIBLE
 * layers, so hidden ones — including the stage layers of a previous Heron group
 * — contribute nothing. Saying so beats letting the button imply it. */
function showScopeNote() {
  $("scope-note").textContent = $("useselection").checked
    ? "Reads the flattened visible layers inside the selection (whole canvas if nothing is selected). Hidden layers are excluded."
    : "Reads the flattened visible layers of the whole canvas. Hidden layers are excluded.";
}

function showResolutionNote() {
  $("resnote").textContent =
    "Scene understanding (depth, matte, materials) always runs at 1280 px and is " +
    "lifted edge-aware. Physics, sensor and art direction run at whatever the " +
    "resolution checkbox selects.";
}

/* ------------------------------------------------------------- wiring up */

$("instrument").addEventListener("change", (e) => {
  state.instrument = e.target.value;
  applyInstrument();
});
$("preset").addEventListener("change", (e) => showPresetDesc(e.target.value));
// Not `connect` directly: the click Event would arrive as `quiet`, and an Event
// is truthy, so an explicit press would take the silent background path.
$("reconnect").addEventListener("click", () => connect(false));
$("reset").addEventListener("click", resetDials);
$("selftest").addEventListener("click", async () => {
  setStatus("verifying layer placement…", "busy");
  try {
    const r = await bridge.verifyPlacement();
    setStatus(r.message, r.ok ? "ok" : "bad");
  } catch (e) {
    setStatus(`Verification failed — ${e && e.message ? e.message : e}`, "bad");
    console.error("[Heron] verifyPlacement", e);
  }
});
$("dice").addEventListener("click", () => {
  $("seed").value = String(Math.floor(Math.random() * 10000));
});
$("useai").addEventListener("change", showAiNote);
$("useselection").addEventListener("change", () => { showScopeNote(); showFullResNote(); });
$("fullres").addEventListener("change", () => { showFullResNote(); showNeuralNote(); });
$("neural").addEventListener("change", showNeuralNote);
$("nstr").addEventListener("input", () => { $("nstr-val").textContent = $("nstr").value; });
$("nstr").addEventListener("change", () => { $("nstr-val").textContent = $("nstr").value; });

/* ------------------------------------------------------------- rendering */

let rendering = false;

/* The long side to render at. For a selection this is the selection's own
 * native size, which is why a selection render is sharper than a full-canvas one
 * at the same setting: fewer pixels to cover, none of them invented. */
function renderRes(doc) {
  if (!$("fullres").checked) return null;
  return doc.bounds
    ? Math.max(doc.bounds.right - doc.bounds.left, doc.bounds.bottom - doc.bounds.top)
    : Math.max(doc.width, doc.height);
}

async function pollJob(jobId) {
  // ~600 ms, matching the web UI. Never blocks: the panel stays interactive.
  for (;;) {
    const job = await getJSON(`/api/job/${jobId}`);
    if (job.state === "error") throw new Error(job.error || "render failed");
    if (job.state === "done") return job;
    const pct = Math.round((job.progress || 0) * 100);
    setStatus(`${job.stage || job.state}… ${pct}%`, "busy");
    await new Promise((r) => setTimeout(r, 600));
  }
}

async function render() {
  if (rendering) return;
  rendering = true;
  $("render").setAttribute("disabled", "");
  const t0 = Date.now();

  try {
    setStatus("exporting the document…", "busy");
    const doc = await bridge.exportDocument(!!$("useselection").checked);
    const scope = doc.bounds
      ? `selection ${doc.bounds.right - doc.bounds.left}×${doc.bounds.bottom - doc.bounds.top}`
      : `visible layers, ${doc.width}×${doc.height}`;

    setStatus(`uploading ${scope}, ${Math.round(doc.bytes.byteLength / 1024)} KB…`, "busy");
    const up = await getJSON("/api/upload-raw", { method: "POST", body: doc.bytes });

    setStatus("starting the render…", "busy");
    const preset = $("preset").value;
    const { job_id } = await getJSON("/api/render-layers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: up.image_id,
        instrument: state.instrument,
        preset,
        overrides: collectOverrides(),
        seed: parseInt($("seed").value, 10) || 7,   // explicit, never defaulted silently
        render_res: renderRes(doc),
        use_ai: !!$("useai").checked,
        neural: !!$("neural").checked,          // §6: off by default, own top layer
        neural_strength: parseFloat($("nstr").value) || 0.52,
        neural_seed: parseInt($("seed").value, 10) || 7,
      }),
    });

    const job = await pollJob(job_id);
    if (!job.manifest) throw new Error("the service returned no layer manifest");

    const result = await bridge.importLayers(
      state.base, job.manifest, `${state.instrument}/${preset}`,
      (text) => setStatus(text + "…", "busy"), true, doc.bounds,
    );

    let finishNote = "";
    if (job.manifest.finish && job.manifest.finish.size) {
      const [fw, fh] = job.manifest.finish.size;
      const [cw, ch] = job.manifest.finish.composite_size || [fw, fh];
      finishNote = Math.max(cw, ch) > Math.max(fw, fh) * 1.2
        ? ` The neural layer rendered at ${fw}×${fh} against a ${cw}×${ch} composite, so it is upscaled and softer.`
        : ` Neural layer ${fw}×${fh}.`;
      // State where it actually landed. Inside the group would mean the physics
      // stack is no longer separable — the one thing the finish must never do.
      if (result.finishInsideGroup === true) {
        finishNote += " ⚠ It landed INSIDE the group — drag it out; the physics" +
                      " stack should stay separable.";
      } else if (result.finishInsideGroup === null) {
        finishNote += " (Could not confirm it sits above the group — check the" +
                      " layer panel.)";
      } else {
        finishNote += " It sits above the group, not inside it.";
      }
    }

    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    const where = doc.bounds
      ? (result.masked
          ? "fitted to the selection and masked to its shape"
          : "fitted to the selection bounds (shape mask failed — see console)")
      : ($("useselection").checked ? "whole canvas — nothing was selected" : "whole canvas");
    setStatus(
      `“${result.groupName}” — ${result.count} layers in ${secs}s ` +
      `(physics ${job.timing.physics_s}s), ${where}. Composite is on top; the ` +
      `stages below it are hidden — unhide any of them to edit.` + finishNote,
      "ok",
    );
  } catch (e) {
    const detail = `${e && e.name ? e.name : "Error"}: ${e && e.message ? e.message : e}`;
    setStatus(`Render failed — ${detail}`, "bad");
    console.error("[Heron] render failed", e);
  } finally {
    rendering = false;
    $("render").removeAttribute("disabled");
  }
}

$("render").addEventListener("click", render);

connect();
