/* Photoshop bridge — export pixels out, bring named layers back (task 3.2).
 *
 * Everything that touches the Photoshop DOM lives here, so main.js stays UI.
 * The product is an editable layer GROUP: a flattened result is a failure mode,
 * not a shortcut. The composite goes on top visible; each physics stage sits
 * below it, hidden, for the user to unhide and edit.
 *
 * Photoshop mutations must run inside executeAsModal or they throw.
 */

const photoshop = require("photoshop");
const uxp = require("uxp");

const { app, core, action, constants } = photoshop;
const localFs = uxp.storage.localFileSystem;
const formats = uxp.storage.formats;

/* Manifest blend-mode names -> Photoshop's own enum. Anything unrecognised
 * stays normal rather than guessing, so a new stage name cannot break a run. */
const BLEND_MODES = {
  normal: "normal",
  screen: "screen",
  overlay: "overlay",
  multiply: "multiply",
  linearDodge: "linearDodge",
};

function blendMode(name) {
  return BLEND_MODES[name] || "normal";
}

/** Active document pixel size, or null when nothing is open. Read-only, so it
 *  needs no modal scope and is safe to call while building the UI. */
function documentSize() {
  if (!app.documents.length) return null;
  const doc = app.activeDocument;
  return { width: doc.width, height: doc.height };
}

/** Pixel bounds of the active selection, or null when nothing is selected. */
async function selectionBounds() {
  const res = await action.batchPlay(
    [{
      _obj: "get",
      _target: [
        { _property: "selection" },
        { _ref: "document", _enum: "ordinal", _value: "targetEnum" },
      ],
    }],
    { synchronousExecution: true },
  );
  const sel = res && res[0] && res[0].selection;
  if (!sel || sel.top === undefined) return null;
  const px = (v) => Math.round(v && v._value !== undefined ? v._value : v);
  const bounds = {
    left: px(sel.left), top: px(sel.top),
    right: px(sel.right), bottom: px(sel.bottom),
  };
  if (bounds.right - bounds.left < 1 || bounds.bottom - bounds.top < 1) return null;
  return bounds;
}

/**
 * Export the active document (or just the selection) as a PNG.
 *
 * Rendering only the selection is what makes this better than the web app for
 * retouching: a duplicate is cropped to the marquee so the service never sees
 * the rest of the canvas, and the returned layers are translated back onto
 * those exact bounds at import time.
 */
async function exportDocument(useSelection) {
  if (!app.documents.length) throw new Error("No document is open in Photoshop.");
  const doc = app.activeDocument;
  const folder = await localFs.getTemporaryFolder();
  const file = await folder.createFile("heron_source.png", { overwrite: true });

  let bounds = null;
  if (useSelection) {
    try {
      bounds = await selectionBounds();
    } catch (e) {
      // A selection we cannot read is not worth failing a render over.
      console.warn("[Heron] could not read the selection", e);
    }
  }

  await core.executeAsModal(
    async () => {
      if (bounds) {
        // Keep the real selection shape before anything disturbs it, then crop
        // a throwaway duplicate. The user's document is never modified.
        await saveSelectionChannel();
        const dup = await doc.duplicate();
        try {
          await dup.crop(bounds);
          await dup.flatten();
          await dup.saveAs.png(file, { compression: 6 }, true);
        } finally {
          await dup.closeWithoutSaving();
        }
      } else {
        await doc.saveAs.png(file, { compression: 6 }, true);   // true = as a copy
      }
    },
    { commandName: "Heron: export document" },
  );

  const bytes = await file.read({ format: formats.binary });
  return { bytes, bounds, width: doc.width, height: doc.height, title: doc.title };
}

/** Fetch one returned layer image into a temp file and hand back a token. */
async function stageToToken(base, entry, index) {
  const res = await fetch(base + entry.url);
  if (!res.ok) throw new Error(`${entry.name}: HTTP ${res.status}`);
  const data = await res.arrayBuffer();

  const folder = await localFs.getTemporaryFolder();
  // Index-prefixed so the temp folder stays readable while debugging, and so
  // two stages with the same name across runs cannot collide.
  const safe = entry.name.replace(/[^a-z0-9_-]/gi, "_");
  const file = await folder.createFile(`heron_${index}_${safe}.png`, { overwrite: true });
  await file.write(data, { format: formats.binary });
  return localFs.createSessionToken(file);
}

/** Bounds values arrive either as plain numbers or as {_value} unit objects. */
function pixels(value) {
  if (typeof value === "number") return value;
  return value && value._value !== undefined ? value._value : 0;
}

/**
 * Make a placed layer occupy exactly `target`, in both size and position.
 *
 * Two things conspire here. The service renders at Layer A's working resolution
 * (<=1280 px long side), so the returned pixels are almost never the size of the
 * region they belong to; and `placeEvent` scales what it places to fit the
 * CANVAS and centres it. Translating alone therefore parks a canvas-sized layer
 * at the target's origin — which is what made selection renders land as a huge
 * offset rectangle. Scale first, then position.
 */
async function fitToBounds(layer, target) {
  const wantW = target.right - target.left;
  const wantH = target.bottom - target.top;
  if (wantW < 1 || wantH < 1) return;

  const b = layer.bounds;
  const haveW = pixels(b.right) - pixels(b.left);
  const haveH = pixels(b.bottom) - pixels(b.top);
  if (haveW > 0 && haveH > 0) {
    const sx = (wantW / haveW) * 100;
    const sy = (wantH / haveH) * 100;
    if (Math.abs(sx - 100) > 0.1 || Math.abs(sy - 100) > 0.1) {
      // Anchor at the top-left so the follow-up translate is a small correction
      // rather than half the size delta. If this build names the enum member
      // differently, scaling about the default anchor is still correct — the
      // translate below fixes the position either way.
      const topLeft = constants && constants.AnchorPosition
        ? constants.AnchorPosition.TOPLEFT
        : undefined;
      try {
        await layer.scale(sx, sy, topLeft);
      } catch (e) {
        console.warn("[Heron] scale with anchor failed, retrying without", e);
        await layer.scale(sx, sy);
      }
    }
  }

  // Re-read: scaling about the top-left still moves bounds by a rounding pixel.
  const after = layer.bounds;
  const dx = target.left - pixels(after.left);
  const dy = target.top - pixels(after.top);
  if (Math.abs(dx) >= 1 || Math.abs(dy) >= 1) {
    await layer.translate(dx, dy);
  }
}

const SELECTION_CHANNEL = "heron_selection";

/** Stash the live selection in an alpha channel so it survives the render.
 *
 *  Cropping can only ever produce the selection's bounding box, so a lasso or
 *  a feathered marquee would come back as a hard rectangle. Keeping the real
 *  shape lets the finished group be masked with it.
 */
async function saveSelectionChannel() {
  await action.batchPlay(
    [{
      _obj: "duplicate",
      _target: [{ _ref: "channel", _property: "selection" }],
      name: SELECTION_CHANNEL,
      _options: { dialogOptions: "dontDisplay" },
    }],
    { synchronousExecution: true },
  );
}

/** Mask `group` with the stashed selection, then drop the temp channel. */
async function maskGroupWithSelection(doc, group) {
  doc.activeLayers = [group];
  await action.batchPlay(
    [
      {
        _obj: "set",
        _target: [{ _ref: "channel", _property: "selection" }],
        to: { _ref: "channel", _name: SELECTION_CHANNEL },
        _options: { dialogOptions: "dontDisplay" },
      },
      {
        _obj: "make",
        new: { _class: "channel" },
        at: { _ref: "channel", _enum: "channel", _value: "mask" },
        using: { _enum: "userMaskOptions", _value: "revealSelection" },
        _options: { dialogOptions: "dontDisplay" },
      },
      {
        _obj: "delete",
        _target: [{ _ref: "channel", _name: SELECTION_CHANNEL }],
        _options: { dialogOptions: "dontDisplay" },
      },
    ],
    { synchronousExecution: true },
  );
}

/** Place a PNG as a new layer in the active document; returns the layer. */
async function placeAsLayer(token, name) {
  const before = new Set(app.activeDocument.layers.map((l) => l.id));
  await action.batchPlay(
    [{
      _obj: "placeEvent",
      null: { _path: token, _kind: "local" },
      freeTransformCenterState: { _enum: "quadCenterState", _value: "QCSAverage" },
      _options: { dialogOptions: "dontDisplay" },
    }],
    { synchronousExecution: true },
  );

  // placeEvent leaves the new layer active, but resolve it by diffing ids so a
  // surprise (an extra layer, a different active target) cannot go unnoticed.
  const placed = app.activeDocument.layers.find((l) => !before.has(l.id));
  if (!placed) throw new Error(`placing '${name}' produced no new layer`);
  placed.name = name;
  return placed;
}

/**
 * Import a render manifest as one named, editable group.
 *
 * @param base      service origin, e.g. http://127.0.0.1:8077
 * @param manifest  {composite, layers[], graph[]} from /api/render-layers
 * @param label     what to call the group (instrument/preset)
 * @param onStage   progress callback(text)
 * @param includeGraph  also import depth/matte/etc.
 * @param bounds    selection bounds to align onto, or null for the whole canvas
 */
async function importLayers(base, manifest, label, onStage, includeGraph, bounds) {
  const stages = manifest.layers || [];
  const graph = includeGraph ? (manifest.graph || []) : [];

  // Fetch everything BEFORE entering the modal scope: network waits inside
  // executeAsModal keep Photoshop's UI hostage for the whole download.
  const jobs = [];
  let index = 0;
  jobs.push({
    name: "composite",
    mode: "normal",
    visible: true,
    token: await stageToToken(base, { name: "composite", url: manifest.composite }, index++),
  });
  for (const entry of stages) {
    onStage(`fetching ${entry.name}`);
    jobs.push({
      name: entry.name,
      mode: blendMode(entry.mode),
      visible: false,
      token: await stageToToken(base, entry, index++),
    });
  }
  for (const entry of graph) {
    onStage(`fetching graph/${entry.name}`);
    jobs.push({
      name: `graph · ${entry.name}`,
      mode: "normal",
      visible: false,
      token: await stageToToken(base, entry, index++),
    });
  }

  // Fetched last so it is the final thing placed, landing on top of the group.
  let finishToken = null;
  if (manifest.finish) {
    onStage("fetching the neural finish");
    finishToken = await stageToToken(base, manifest.finish, index++);
  }

  onStage("building the layer group");
  let groupName = "";
  let masked = false;
  let finishInsideGroup = false;   // null = could not be determined
  await core.executeAsModal(
    async () => {
      const doc = app.activeDocument;
      // Whole-canvas renders need fitting too: the service returns working-res
      // pixels, so "no selection" still means "resize to the document".
      const target = bounds || { left: 0, top: 0, right: doc.width, bottom: doc.height };
      const placed = [];
      // Place in reverse so the composite ends up on top of the stack.
      for (const job of jobs.slice().reverse()) {
        const layer = await placeAsLayer(job.token, job.name);
        await fitToBounds(layer, target);
        try {
          layer.blendMode = job.mode;
        } catch (e) {
          // A blend mode Photoshop rejects must not abort the whole import.
          console.warn(`[Heron] blend mode '${job.mode}' rejected for ${job.name}`, e);
        }
        layer.visible = job.visible;
        placed.push(layer);
      }
      groupName = `Heron — ${label}`;
      const group = await doc.createLayerGroup({ name: groupName, fromLayers: placed });

      // A lasso or feathered selection cannot survive a rectangular crop, so
      // the real shape is restored here as a mask on the whole group. Failing
      // to mask must not lose the render, hence the catch.
      if (bounds) {
        try {
          await maskGroupWithSelection(doc, group);
          masked = true;
        } catch (e) {
          console.warn("[Heron] could not mask the group with the selection", e);
        }
      }

      // The neural finish goes ABOVE the group, never inside it: it cannot be
      // decomposed back into stages, so merging it would destroy the stack.
      //
      // Where placeEvent puts a layer when a group is selected is not something
      // to take on trust, so this does not rely on it: the layer is moved above
      // the group explicitly, and the result is reported back rather than
      // assumed. `placedInsideGroup` is what the panel surfaces.
      if (finishToken) {
        const finish = await placeAsLayer(finishToken, "neural finish");
        await fitToBounds(finish, target);
        try {
          // Guarded like AnchorPosition was: if this build names the member
          // differently, fall through to whatever placeEvent already did rather
          // than throwing away a finished render.
          const above = constants && constants.ElementPlacement
            ? constants.ElementPlacement.PLACEBEFORE
            : undefined;
          if (above !== undefined) await finish.move(group, above);
        } catch (e) {
          console.warn("[Heron] could not move the neural finish above the group", e);
        }
        // Verify rather than hope: a layer inside the group would mean the
        // physics stack is no longer separable, which is the whole point.
        try {
          const parent = finish.parent;
          finishInsideGroup = !!(parent && parent.id === group.id);
        } catch (e) {
          finishInsideGroup = null;      // unknown — say so, do not claim success
        }
      }
    },
    { commandName: "Heron: import layers" },
  );

  return {
    groupName,
    count: jobs.length + (finishToken ? 1 : 0),
    finish: !!finishToken,
    finishInsideGroup,
    masked,
  };
}

/**
 * Exercise the layer-placement calls the neural finish depends on, and clean up.
 *
 * Three calls in `importLayers` could only be confirmed by running a full
 * diffusion pass and inspecting the result by hand: placing a layer after a
 * group exists, moving it above that group, and reading back its parent. This
 * runs exactly those against the plugin's own icon file — no service, no
 * diffusion, about a second — and deletes everything it made.
 *
 * Returns {ok, message}. It never leaves layers behind, even on failure.
 */
async function verifyPlacement() {
  if (!app.documents.length) {
    return { ok: false, message: "Open a document first — placement needs a canvas." };
  }

  // The panel's own icon is a real PNG that is guaranteed to be present.
  const pluginFolder = await localFs.getPluginFolder();
  const icon = await pluginFolder.getEntry("icons/icon-46.png");
  const token = localFs.createSessionToken(icon);

  const made = [];
  let verdict = { ok: false, message: "verification did not run" };

  await core.executeAsModal(
    async () => {
      const doc = app.activeDocument;
      try {
        const member = await placeAsLayer(token, "heron selftest member");
        made.push(member);
        const group = await doc.createLayerGroup({
          name: "heron selftest group", fromLayers: [member],
        });
        made.push(group);

        const finish = await placeAsLayer(token, "heron selftest finish");
        made.push(finish);

        const above = constants && constants.ElementPlacement
          ? constants.ElementPlacement.PLACEBEFORE
          : undefined;
        const moved = above !== undefined;
        if (moved) await finish.move(group, above);

        const parent = finish.parent;
        const inside = !!(parent && parent.id === group.id);

        verdict = inside
          ? { ok: false,
              message: "FAIL — the finish layer landed inside the group. The " +
                       "physics stack would not stay separable." }
          : { ok: true,
              message: "PASS — placed a layer, grouped it, moved a second layer " +
                       "above the group, and confirmed it is a sibling" +
                       (moved ? "." : " (ElementPlacement was unavailable; " +
                                       "placeEvent's own ordering held).") };
      } catch (e) {
        verdict = { ok: false, message: `FAIL — ${e && e.message ? e.message : e}` };
      } finally {
        // Never leave test layers in the user's document.
        for (const layer of made.reverse()) {
          try { await layer.delete(); } catch (e) { /* already gone with its group */ }
        }
      }
    },
    { commandName: "Heron: verify layer placement" },
  );

  return verdict;
}

module.exports = { exportDocument, importLayers, documentSize, verifyPlacement };
