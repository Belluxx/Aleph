/* Local dashboard; intentionally no framework or build step. */
"use strict";

const $ = (id) => document.getElementById(id);
const empty = () => ({type: "FeatureCollection", features: []});
const state = {config: null, captures: [], selected: null, map: null, ready: null,
  selection: 0, photos: [], photoCount: 0, photoGroups: new Map(), photoIndex: 0, job: null,
  composing: false, drawing: false, drag: null, draftBounds: null, submitting: false,
  pendingPlan: null, plan: null, loadingCapture: false,
  previousView: null, rotating: null, terrainFocus: null, refreshing: false, lastRefresh: 0, popup: null};
const form = $("capture-form");
const maxDrawLatitude = 85.0511287; // Remain inside Mercator bounds after rounding to seven decimals.
const names = {streetview: "STREET VIEW", satellite: "SATELLITE", osm: "OSM + TERRAIN"};
const resolutionNames = ["Extra low", "Very low", "Low", "Medium", "High", "Very high", "Extra high"];
const layerGroups = {satellite: ["satellite"], buildings: ["buildings", "building-outlines"],
  roads: ["road-casing", "roads"], water: ["land", "water", "water-lines"], photos: ["stops"]};
const contextRoadFilter = ["all", ["==", ["geometry-type"], "LineString"],
  ["match", ["get", "class"], ["motorway", "trunk", "primary", "secondary", "tertiary", "minor", "service", "track", "path", "pedestrian"], true, false]];
const contextRoadWeight = ["match", ["get", "class"], ["motorway", "trunk"], 1,
  ["primary", "secondary"], 0.8, ["tertiary", "minor"], 0.62, 0.42];
const contextLayers = [
  {id: "context-landuse", type: "fill", "source-layer": "landuse",
    filter: ["match", ["get", "class"], ["residential", "commercial", "industrial", "hospital", "school", "cemetery"], true, false],
    paint: {"fill-color": ["match", ["get", "class"], "residential", "--map-residential", ["commercial", "industrial"], "--map-commercial", ["hospital", "school"], "--map-public", "--map-cemetery"], "fill-opacity": 0.7}},
  {id: "context-landcover", type: "fill", "source-layer": "landcover",
    filter: ["match", ["get", "class"], ["wood", "grass", "farmland", "scrub"], true, false],
    paint: {"fill-color": ["match", ["get", "class"], "wood", "--map-wood", ["grass", "scrub"], "--map-grass", "--map-farmland"], "fill-opacity": 0.72}},
  {id: "context-parks", type: "fill", "source-layer": "park",
    paint: {"fill-color": "--map-park", "fill-opacity": 0.78}},
  {id: "context-water", type: "fill", "source-layer": "water",
    paint: {"fill-color": "--map-water"}},
  {id: "context-waterways", type: "line", "source-layer": "waterway",
    paint: {"line-color": "--map-waterway", "line-width": ["interpolate", ["linear"], ["zoom"], 8, 0.5, 18, 3]}},
  {id: "context-road-casing", type: "line", "source-layer": "transportation", minzoom: 5,
    filter: contextRoadFilter,
    layout: {"line-cap": "round", "line-join": "round"},
    paint: {"line-color": "--map-road-casing", "line-width": ["interpolate", ["exponential", 1.35], ["zoom"], 5, ["*", 0.7, contextRoadWeight], 20, ["*", 19, contextRoadWeight]]}},
  {id: "context-roads", type: "line", "source-layer": "transportation", minzoom: 5,
    filter: contextRoadFilter,
    layout: {"line-cap": "round", "line-join": "round"},
    paint: {"line-color": ["match", ["get", "class"], ["motorway", "trunk"], "--map-motorway", ["primary", "secondary"], "--map-primary-road", ["track", "path", "pedestrian"], "--map-path", "--map-road"], "line-width": ["interpolate", ["exponential", 1.35], ["zoom"], 5, ["*", 0.3, contextRoadWeight], 20, ["*", 15, contextRoadWeight]]}},
  {id: "context-buildings", type: "fill", "source-layer": "building", minzoom: 13,
    paint: {"fill-color": "--map-building", "fill-outline-color": "--map-building-outline"}},
  {id: "context-boundaries", type: "line", "source-layer": "boundary", minzoom: 3,
    filter: ["all", ["<=", ["get", "admin_level"], 4], ["!=", ["get", "maritime"], 1]],
    paint: {"line-color": "--map-boundary", "line-dasharray": [3, 2], "line-opacity": 0.65, "line-width": ["interpolate", ["linear"], ["zoom"], 3, 0.7, 10, 1.6]}},
  {id: "context-road-labels", type: "symbol", "source-layer": "transportation_name", minzoom: 12,
    layout: {"symbol-placement": "line", "text-field": ["coalesce", ["get", "name_en"], ["get", "name"]], "text-font": ["Noto Sans Regular"], "text-size": ["interpolate", ["linear"], ["zoom"], 12, 10, 18, 13]},
    paint: {"text-color": "--map-road-label", "text-halo-color": "--map-label-halo", "text-halo-width": 1}},
  {id: "context-place-labels", type: "symbol", "source-layer": "place",
    filter: ["match", ["get", "class"], ["country", "state", "city", "town", "village"], true, false],
    layout: {"text-field": ["coalesce", ["get", "name_en"], ["get", "name"]], "text-font": ["Noto Sans Regular"], "text-size": ["match", ["get", "class"], "country", 15, "state", 13, "city", 14, "town", 12, 11]},
    paint: {"text-color": "--map-place-label", "text-halo-color": "--map-label-halo", "text-halo-width": 1.2}},
];
const contextLayerIds = contextLayers.map((layer) => layer.id);
const mapPaints = new Map();

// Resolve CSS palette names, including colors inside MapLibre expressions.
function mapColor(value, palette) {
  if (Array.isArray(value)) return value.map((item) => mapColor(item, palette));
  return typeof value === "string" && value.startsWith("--") ? palette.getPropertyValue(value).trim() : value;
}

function themedLayer(layer) {
  const palette = getComputedStyle(document.documentElement);
  const colors = Object.entries(layer.paint || {}).filter(([name]) => name.endsWith("-color"));
  mapPaints.set(layer.id, colors);
  return {...layer, paint: {...layer.paint,
    ...Object.fromEntries(colors.map(([name, value]) => [name, mapColor(value, palette)]))}};
}

function syncMapTheme() {
  const palette = getComputedStyle(document.documentElement);
  for (const [id, colors] of mapPaints) {
    if (!state.map.getLayer(id)) continue;
    for (const [name, value] of colors) state.map.setPaintProperty(id, name, mapColor(value, palette));
  }
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function notice(message) {
  $("notice").querySelector("span").textContent = message;
  $("notice").hidden = false;
}

async function api(path, data) {
  const response = await fetch(path, data === undefined ? undefined : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Aleph-Token": state.config.token},
    body: JSON.stringify(data),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status}).`);
  return value;
}

function runURL(identity, suffix = "") {
  return `/api/captures/${encodeURIComponent(identity)}${suffix}`;
}

function fileURL(identity, filename) {
  return runURL(identity, `/files/${filename.split("/").map(encodeURIComponent).join("/")}`);
}

function date(value, options = {}) {
  return new Date(value).toLocaleString(undefined, {month: "short", day: "numeric", year: "numeric", ...options});
}

function center(area) {
  return [(area[0] + area[2]) / 2, (area[1] + area[3]) / 2];
}

function mapBounds(area) {
  return [[area[1], area[0]], [area[3], area[2]]];
}

function rectangle(area) {
  const [south, west, north, east] = area;
  return {type: "Feature", properties: {}, geometry: {type: "Polygon", coordinates: [
    [[west, south], [east, south], [east, north], [west, north], [west, south]],
  ]}};
}

function renderLibrary() {
  const term = $("search").value.toLowerCase().trim();
  $("capture-count").textContent = state.captures.length;
  const jobPanel = $("job");
  const visibleJob = state.job && !jobPanel.hidden;
  let jobCard = null;
  const fragment = document.createDocumentFragment();
  for (const run of state.captures) {
    const currentJob = visibleJob && state.job.run_id === run.id;
    const status = state.job?.run_id === run.id ? state.job.state : run.state;
    const coordinates = center(run.bounds).map((v) => v.toFixed(5)).join(", ");
    if (term && !currentJob && !`${run.id} ${date(run.started_at)} ${coordinates} ${status}`.toLowerCase().includes(term)) continue;
    const card = node("article", undefined, `capture-card${state.selected?.id === run.id ? " active" : ""}`);
    card.dataset.state = status;
    const button = node("button", undefined, "capture-select");
    button.title = `${run.id}\n${status}`;
    button.setAttribute("aria-pressed", String(state.selected?.id === run.id));
    const heading = node("div", undefined, "card-heading");
    heading.append(node("strong", date(run.started_at)));
    const sources = node("div", undefined, "card-sources");
    Object.keys(run.layers).forEach((key) => sources.append(node("span", names[key], "card-source")));
    button.append(heading, node("div", coordinates, "card-coordinates"), sources);
    button.addEventListener("click", () => selectCapture(run.id).catch((error) => notice(error.message)));
    const available = {
      resume: status !== "complete",
      export: Object.values(run.layers).some((layer) => layer.done),
    };
    const actions = $("capture-actions").content.firstElementChild.cloneNode(true);
    const more = actions.querySelector(".card-more");
    more.hidden = !available.export;
    actions.hidden = (currentJob && state.job.active) || !Object.values(available).some(Boolean);
    for (const action of actions.querySelectorAll("button")) {
      action.hidden = !available[action.dataset.action];
      action.disabled = Boolean(state.job?.active);
      if (action.dataset.action === "resume" && run.state === "planned") {
        action.title = "Review capture plan";
        action.setAttribute("aria-label", action.title);
        action.querySelector("span").textContent = "Review plan";
      }
      action.addEventListener("click", () => {
        more.open = false;
        if (action.dataset.action === "resume" && run.state === "planned") {
          return reviewPlan(run.id).catch((error) => notice(error.message));
        }
        return runAction(run.id, action.dataset.action);
      });
    }
    const footer = node("div", undefined, "card-footer");
    if (!currentJob) footer.append(node("span", status.charAt(0).toUpperCase() + status.slice(1), "card-status"));
    footer.append(actions);
    card.append(button, footer);
    fragment.append(card);
    if (currentJob) jobCard = card;
  }
  if (visibleJob && !jobCard) {
    jobCard = node("article", undefined, "capture-card");
    jobCard.dataset.state = state.job.state;
    const heading = node("div", undefined, "card-heading pending-heading");
    heading.append(node("strong", "New capture"));
    jobCard.append(heading, node("div", undefined, "card-footer"));
    fragment.prepend(jobCard);
  }
  if (!fragment.childNodes.length) fragment.append(node("p", state.captures.length ? "No matching captures." : "No captures yet.", "muted empty-list"));
  $("captures").replaceChildren(fragment);
  (jobCard?.querySelector(".card-footer") || $("captures")).prepend(jobPanel);
  $("view-job").hidden = !state.job?.run_id || state.captures.some((run) => run.id === state.job.run_id);
}

async function refreshLibrary(initial = false) {
  if (state.refreshing) return;
  state.refreshing = true;
  try {
    const data = await api("/api/captures");
    const changed = data.captures.length !== state.captures.length || data.captures.some((run, index) => {
      const previous = state.captures[index];
      return run.id !== previous.id || run.state !== previous.state || run.exports_saved !== previous.exports_saved;
    });
    state.captures = data.captures;
    state.lastRefresh = Date.now();
    // Keep the active progress bar attached so polling does not restart its animation.
    if (!state.job?.active || changed) renderLibrary();
    if (initial && data.errors.length) notice(`${data.errors.length} capture folder(s) could not be read: ${data.errors[0].id}: ${data.errors[0].error}`);
    if (state.composing) return;
    if (initial && state.captures.length) await selectCapture(state.captures[0].id);
    else if (state.selected) {
      const updated = state.captures.find((run) => run.id === state.selected.id);
      if (updated && updated.revision !== state.selected.revision) await selectCapture(updated.id, false);
    }
  } finally {
    state.refreshing = false;
  }
}

function initializeMap() {
  let map;
  try {
    if (!window.maplibregl) throw new Error("Map library unavailable.");
    map = new maplibregl.Map({container: "map", center: [12.492, 41.890], zoom: 14,
      maxZoom: 22, maxPitch: 75, renderWorldCopies: false,
      dragRotate: false,
      style: {version: 8, glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf", sources: {},
        layers: [themedLayer({id: "background", type: "background", paint: {"background-color": "--map-background"}})]},
      attributionControl: true,
    });
  } catch (error) {
    $("draw-area").disabled = $("move-map").disabled = $("use-view").disabled = true;
    notice("This browser cannot create a WebGL map. Enable WebGL to select an area for a new capture.");
    return Promise.resolve();
  }
  state.map = map;
  map.on("idle", () => {
    if (!state.loadingCapture) $("map-loading").hidden = true;
  });
  initializeMiddleRotation(map);
  initializeAreaDrawing(map);
  map.addControl(new maplibregl.NavigationControl({visualizePitch: true}), "top-right");
  map.addControl(new maplibregl.ScaleControl({maxWidth: 100}), "bottom-right");
  map.on("error", (event) => {
    if (event.error?.status === 404) return;
    const message = event.error?.message || "Map layer could not be loaded.";
    notice(`${message} You can disable the affected layer and continue.`);
  });
  map.on("mousemove", (event) => {
    $("coordinates").textContent = `${event.lngLat.lat.toFixed(6)}, ${event.lngLat.lng.toFixed(6)} (zoom ${map.getZoom().toFixed(1)})`;
  });
  return new Promise((resolve) => map.on("load", () => {
    map.addSource("context", {type: "vector", url: "https://tiles.openfreemap.org/planet",
      attribution: '<a href="https://openfreemap.org/" target="_blank" rel="noopener">OpenFreeMap</a> © <a href="https://openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a> Data from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>'});
    for (const layer of contextLayers) map.addLayer(themedLayer({...layer, source: "context"}));
    map.addSource("selection", {type: "geojson", data: empty()});
    map.addLayer(themedLayer({id: "selection-fill", type: "fill", source: "selection", paint: {"fill-color": "--accent", "fill-opacity": 0.15}}));
    map.addLayer(themedLayer({id: "selection-line", type: "line", source: "selection", paint: {"line-color": "--accent", "line-width": 2}}));
    map.addSource("selection-corners", {type: "geojson", data: empty()});
    map.addLayer(themedLayer({id: "selection-corners", type: "circle", source: "selection-corners", paint: {"circle-radius": 4, "circle-color": "--surface", "circle-stroke-color": "--accent", "circle-stroke-width": 2}}));
    // Also refresh the background if the system theme changed while the map loaded.
    syncMapTheme();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncMapTheme);
    resolve();
  }));
}

function initializeMiddleRotation(map) {
  const canvas = map.getCanvas();
  const finish = (event) => {
    const rotation = state.rotating;
    if (!rotation || event.pointerId !== rotation.pointerId) return;
    state.rotating = null;
    canvas.style.cursor = "";
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  };
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 1 || !state.selected || state.composing) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    state.rotating = {pointerId: event.pointerId, x: event.clientX, y: event.clientY,
      bearing: map.getBearing(), pitch: map.getPitch(), pivot: map.getCenter(), active: false};
    canvas.style.cursor = "grabbing";
  });
  canvas.addEventListener("pointermove", (event) => {
    const rotation = state.rotating;
    if (!rotation || event.pointerId !== rotation.pointerId) return;
    event.preventDefault();
    const dx = event.clientX - rotation.x;
    const dy = event.clientY - rotation.y;
    if (!rotation.active && Math.hypot(dx, dy) < 3) return;
    if (!rotation.active) {
      rotation.active = true;
      map.stop();
    }
    map.jumpTo({center: rotation.pivot, bearing: rotation.bearing + dx * 0.8,
      pitch: Math.max(map.getMinPitch(), Math.min(map.getMaxPitch(), rotation.pitch - dy * 0.5))});
  });
  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", finish);
  canvas.addEventListener("lostpointercapture", finish);
  canvas.addEventListener("auxclick", (event) => {
    if (event.button === 1) event.preventDefault();
  });
}

function setSelection(values) {
  if (!state.map?.getSource("selection")) return;
  if (!values) {
    state.map.getSource("selection").setData(empty());
    state.map.getSource("selection-corners").setData(empty());
    return;
  }
  const shape = rectangle(normalizedBounds(values));
  state.map.getSource("selection").setData(shape);
  state.map.getSource("selection-corners").setData({type: "FeatureCollection", features:
    shape.geometry.coordinates[0].slice(0, 4).map((coordinates) => ({type: "Feature", properties: {}, geometry: {type: "Point", coordinates}}))});
}

function clearCaptureLayers() {
  const map = state.map;
  const persistentLayers = new Set(["background", ...contextLayerIds, "selection-fill", "selection-line", "selection-corners"]);
  state.terrainFocus?.();
  map.stop();
  map.setTerrain(null);
  for (const layer of [...map.getStyle().layers].reverse()) {
    if (!persistentLayers.has(layer.id)) {
      map.removeLayer(layer.id);
      mapPaints.delete(layer.id);
    }
  }
  for (const id of Object.keys(map.getStyle().sources)) {
    if (!["context", "selection", "selection-corners"].includes(id)) map.removeSource(id);
  }
  state.popup?.remove();
}

function showLayers(ids, visible) {
  if (!state.map) return;
  for (const id of ids) {
    if (state.map.getLayer(id)) state.map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
  }
}

function addLayer(layer) {
  state.map.addLayer(themedLayer(layer), "selection-fill");
}

function applyLayers() {
  if (!state.map?.getLayer("selection-fill")) return;
  for (const [group, layers] of Object.entries(layerGroups)) {
    const shown = !state.composing && !$("layer-" + group).disabled && $("layer-" + group).checked;
    showLayers(layers, shown);
  }
  const terrain = !state.composing && $("terrain").checked && !$("terrain").disabled;
  if (state.map.getLayer("buildings-3d")) {
    const buildings = !state.composing && $("layer-buildings").checked;
    showLayers(["buildings-3d"], terrain && buildings);
    showLayers(["buildings"], !terrain && buildings);
  }
  showLayers(["capture-bounds"], !state.composing);
}

function applyTerrain(animate = true) {
  if (!state.map) return;
  state.terrainFocus?.();
  const enabled = !state.composing && $("terrain").checked && !$("terrain").disabled;
  $("terrain-scale").hidden = !enabled;
  const exaggeration = Number($("exaggeration").value);
  $("height-value").value = `${exaggeration}×`;
  // MapLibre 5.6 can freeze when terrain changes during a camera transition.
  state.map.stop();
  state.map.setTerrain(enabled ? {source: "terrain", exaggeration} : null);
  showLayers(["terrain-shade"], enabled);
  if (enabled) focusTerrain(animate);
  else if (animate) state.map.easeTo({pitch: 0, duration: 650});
  applyLayers();
}

function focusTerrain(animate) {
  const map = state.map;
  const cancel = () => {
    map.off("render", focus);
    map.off("movestart", cancel);
    state.terrainFocus = null;
  };
  const focus = () => {
    if (!map.isSourceLoaded("terrain")) return;
    cancel();
    // A tilt started before the DEM loads can leave MapLibre 5.6 focused at sea level.
    // Query after rendering so the terrain tiles (including height exaggeration) are ready.
    const elevation = map.queryTerrainElevation(map.getCenter());
    if (Number.isFinite(elevation)) map.jumpTo({elevation});
    if (animate) map.easeTo({pitch: 60, duration: 650});
  };
  state.terrainFocus = cancel;
  map.on("render", focus);
  map.on("movestart", cancel);
}

function updateHeader(run) {
  $("capture-title").textContent = date(run.started_at, {hour: "2-digit", minute: "2-digit"});
  const [lat, lon] = center(run.bounds);
  $("capture-subtitle").textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)} (${run.state}, ${Object.values(run.layers).reduce((sum, layer) => sum + layer.done, 0).toLocaleString()} saved items)`;
  $("capture-subtitle").hidden = false;
  $("view-job").hidden = !state.job?.run_id || state.job.run_id === run.id;
}

async function selectCapture(identity, fit = true) {
  if (state.composing) return;
  const ticket = ++state.selection;
  state.loadingCapture = true;
  if (fit || state.selected?.id !== identity) $("map-loading").hidden = false;
  try {
    const run = await api(runURL(identity));
    await state.ready;
    if (ticket !== state.selection) return;
    const previous = state.selected;
    const map = state.map;
    const updating = previous?.id === identity && map?.getSource("capture-bounds");
    state.selected = run;
    updateHeader(run);
    renderLibrary();
    $("welcome").hidden = true;
    if (!updating) {
      closePhotos();
      state.photos = [];
      state.photoCount = 0;
      state.photoGroups.clear();
      if (map) clearCaptureLayers();
    }
    if (!map) return;
    const bounds = [run.bounds[1], run.bounds[0], run.bounds[3], run.bounds[2]];
    const template = (kind) => runURL(identity, `/tiles/${kind}/{z}/{x}/{y}.png`);
    $("layer-satellite").disabled = !run.layers.satellite?.done;
    ["buildings", "roads", "water"].forEach((key) => $("layer-" + key).disabled = !run.osm);
    $("layer-photos").disabled = !run.layers.streetview?.done;
    $("terrain").disabled = !run.terrain;
    $("terrain").title = !state.config.terrain ? "Terrain viewing requires Pillow with libtiff support." : !run.terrain ? "No saved terrain is available in this capture." : "Tilt the map and show terrain heights.";
    const newTerrain = run.terrain && !map.getSource("terrain");
    if (!updating || newTerrain) $("terrain").checked = run.terrain;
    if (run.layers.satellite && !map.getSource("satellite")) {
      // HTTP expiry refreshes incomplete regions; completed tiles stay cached.
      map.addSource("satellite", {type: "raster", tiles: [template("satellite")], tileSize: 256,
        bounds, minzoom: 0, maxzoom: run.layers.satellite.zoom, attribution: "Satellite imagery © Google"});
      addLayer({id: "satellite", type: "raster", source: "satellite", paint: {"raster-fade-duration": 0}});
    }
    if (newTerrain) {
      const terrainSource = {type: "raster-dem", tiles: [template("terrain")], tileSize: 512,
        encoding: "terrarium", bounds, minzoom: 0, maxzoom: run.layers.osm.zoom,
        attribution: '<a href="https://registry.opendata.aws/terrain-tiles/" target="_blank" rel="noopener">Terrain Tiles</a>'};
      map.addSource("terrain", terrainSource);
      map.addSource("terrain-hillshade", {...terrainSource});
      addLayer({id: "terrain-shade", type: "hillshade", source: "terrain-hillshade", layout: {visibility: "none"},
        paint: {"hillshade-shadow-color": "--map-hillshade-shadow", "hillshade-highlight-color": "--map-hillshade-highlight", "hillshade-exaggeration": 0.3}});
    }
    if (!updating) {
      map.addSource("capture-bounds", {type: "geojson", data: rectangle(run.bounds)});
      addLayer({id: "capture-bounds", type: "line", source: "capture-bounds", paint: {"line-color": "--accent", "line-width": 1.5, "line-dasharray": [5, 3]}});
    }
    setSelection(null);
    // Load the bounded DEM from above, then focus at its elevation before tilting.
    if (fit && run.terrain) fitCapture(false);
    if (!updating || newTerrain || fit) applyTerrain(fit && run.terrain);
    if (fit && !run.terrain) fitCapture();
    if (run.osm && !map.getSource("osm")) loadOSM(run);
    if (run.layers.streetview && (!map.getSource("photos")
        || state.photoCount < run.layers.streetview.done)) {
      try { await loadPhotos(run, ticket); }
      catch (error) { if (ticket === state.selection) notice(error.message); }
    }
    if (ticket !== state.selection) return;
    applyLayers();
  } finally {
    if (ticket === state.selection) {
      state.loadingCapture = false;
      // URL sources keep loading after addSource returns. Hide only after rendering settles.
      if (state.map) state.map.triggerRepaint();
      else $("map-loading").hidden = true;
    }
  }
}

function loadOSM(run) {
  state.map.addSource("osm", {type: "geojson", data: runURL(run.id, "/osm.geojson"),
    attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>'});
  const building = ["has", "building"], road = ["has", "road"];
  const layers = [
    {id: "land", type: "fill", filter: ["has", "land"], paint: {"fill-color": "--map-land", "fill-opacity": 0.27}},
    {id: "water", type: "fill", filter: ["has", "water"], paint: {"fill-color": "--map-capture-water", "fill-opacity": 0.6}},
    {id: "water-lines", type: "line", filter: ["has", "waterway"], paint: {"line-color": "--map-capture-waterway", "line-width": 2}},
    {id: "road-casing", type: "line", filter: road, paint: {"line-color": "--map-capture-road-casing", "line-opacity": 0.8, "line-width": ["interpolate", ["linear"], ["zoom"], 12, 1.5, 19, 8]}},
    {id: "roads", type: "line", filter: road, paint: {"line-color": "--map-capture-road", "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.7, 19, 5]}},
    {id: "buildings", type: "fill", filter: building, paint: {"fill-color": "--map-capture-building", "fill-opacity": 0.5}},
    {id: "building-outlines", type: "line", filter: building, paint: {"line-color": "--map-capture-building-outline", "line-width": 1, "line-opacity": 0.85}},
    {id: "buildings-3d", type: "fill-extrusion", filter: building, layout: {visibility: "none"}, paint: {"fill-extrusion-color": "--map-capture-building", "fill-extrusion-height": ["get", "_height"], "fill-extrusion-base": ["get", "_base"], "fill-extrusion-opacity": 0.88}},
  ];
  for (const layer of layers) addLayer({...layer, source: "osm"});
  // OSM can become available after photo dots during a live capture.
  if (state.map.getLayer("stops")) state.map.moveLayer("stops", "selection-fill");
}

async function loadPhotos(run, ticket) {
  const data = await api(runURL(run.id, `/photos?after=${state.photoCount}`));
  if (ticket !== state.selection) return;
  const added = [];
  let currentChanged = false;
  for (const photo of data.features) {
    const key = photo.properties.pano_id;
    if (!state.photoGroups.has(key)) {
      const index = state.photos.length;
      state.photoGroups.set(key, index);
      state.photos.push({coordinates: photo.geometry.coordinates, photos: []});
      added.push({type: "Feature", id: index, geometry: photo.geometry, properties: {index}});
    }
    const index = state.photoGroups.get(key);
    state.photos[index].photos.push(photo.properties);
    if (index === state.photoIndex) currentChanged = true;
  }
  state.photoCount = data.count;
  $("layer-photos").disabled = !state.photos.length;
  const source = state.map.getSource("photos");
  if (source) {
    if (added.length) source.updateData({add: added});
  } else {
    state.map.addSource("photos", {type: "geojson", data: {type: "FeatureCollection", features: added}});
    const selected = ["boolean", ["feature-state", "selected"], false];
    addLayer({id: "stops", type: "circle", source: "photos", paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"],
        10, ["case", selected, 5, 2.5], 18, ["case", selected, 9, 5]],
      "circle-color": ["case", selected, "--surface", "--accent"],
      "circle-stroke-color": ["case", selected, "--accent", "--surface"],
      "circle-stroke-width": ["case", selected, 3, 1.5],
    }});
  }
  if (!$("photo-drawer").hidden) {
    if (currentChanged) showPhoto(state.photoIndex);
    else updatePhotoNavigation();
  }
}

function updatePhotoNavigation() {
  $("photo-position").textContent = `${state.photoIndex + 1} / ${state.photos.length}`;
  $("previous-photo").disabled = $("next-photo").disabled = state.photos.length < 2;
}

function setPhotoMarker(index, selected) {
  if (state.map?.getSource("photos")) state.map.setFeatureState({source: "photos", id: index}, {selected});
}

function closePhotos() {
  const drawer = $("photo-drawer");
  if (!drawer.hidden) setPhotoMarker(state.photoIndex, false);
  drawer.hidden = true;
}

function showPhoto(index, move = false) {
  if (!state.photos.length) return;
  index = (index + state.photos.length) % state.photos.length;
  const previous = $("photo-drawer").hidden ? null : state.photoIndex;
  if (previous !== null && previous !== index) setPhotoMarker(previous, false);
  state.photoIndex = index;
  setPhotoMarker(index, true);
  const stop = state.photos[index];
  const first = stop.photos[0];
  $("photo-title").textContent = first.path_name || "Street View stop";
  $("photo-meta").textContent = `${stop.coordinates[1].toFixed(6)}, ${stop.coordinates[0].toFixed(6)}`;
  updatePhotoNavigation();
  const figures = stop.photos.map((photo) => {
    const figure = node("figure", undefined, "photo-figure");
    const image = node("img");
    const sphere = photo.projection === "equirectangular";
    image.alt = `${sphere ? "Full sphere panorama" : photo.side + " view"}${photo.path_name ? " of " + photo.path_name : ""}`;
    image.src = fileURL(state.selected.id, photo.filename);
    image.addEventListener("error", () => notice(`The saved photo ${photo.filename} could not be loaded.`), {once: true});
    const caption = node("figcaption");
    const imageryDate = photo.imagery_date;
    const description = sphere ? `Full sphere · ${photo.width} × ${photo.height}` : `${photo.side === "left" ? "Left" : "Right"} view, ${Math.round(photo.heading)}°`;
    caption.append(node("span", `${description}${imageryDate ? " (" + imageryDate + ")" : ""}`));
    const address = photo.streetview_url;
    if (address && /^https:\/\/(www\.)?google\.com\//.test(address)) {
      const link = node("a", "Open in Google ↗");
      link.href = address;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      caption.append(link);
    }
    figure.append(image, caption);
    return figure;
  });
  $("photo-images").replaceChildren(...figures);
  $("photo-drawer").hidden = false;
  if (move) state.map?.easeTo({center: stop.coordinates, duration: 400});
}

function fitCapture(animate = true) {
  if (!state.selected || !state.map) return;
  const narrow = window.innerWidth < 1000;
  state.map.fitBounds(mapBounds(state.selected.bounds), {
    padding: {top: 50, bottom: 60, left: narrow ? 40 : 265, right: 50}, maxZoom: 19, duration: animate ? 600 : 0,
    pitch: 0,
  });
}

function selectArea(values) {
  const area = normalizedBounds(values.map((value) => Number(value.toFixed(7))));
  const error = boundsError(area);
  if (error) {
    $("draw-instruction").textContent = error;
    return false;
  }
  state.draftBounds = area;
  renderDraft();
  return true;
}

function areaMetrics(area) {
  const [south, west, north, east] = area.map((v) => v * Math.PI / 180);
  const radius = 6371008.8;
  return {
    width: 2 * radius * Math.asin(Math.min(1, Math.cos((south + north) / 2) * Math.sin((east - west) / 2))),
    height: radius * (north - south),
    area: radius * radius * (east - west) * (Math.sin(north) - Math.sin(south)),
  };
}

function distanceLabel(meters) {
  return meters < 1000 ? `${Math.round(meters).toLocaleString()} m` : `${(meters / 1000).toLocaleString(undefined, {maximumFractionDigits: 2})} km`;
}

function areaLabel(squareMeters) {
  return squareMeters < 1000000 ? `${Math.round(squareMeters).toLocaleString()} m²`
    : `${(squareMeters / 1000000).toLocaleString(undefined, {maximumFractionDigits: 2})} km²`;
}

function normalizedBounds(values) {
  return [Math.min(values[0], values[2]), Math.min(values[1], values[3]), Math.max(values[0], values[2]), Math.max(values[1], values[3])];
}

function boundsError(area) {
  if (area.some((v) => !Number.isFinite(v)) || area[0] < -85.0511287798066 || area[2] > 85.0511287798066 || area[1] < -180 || area[3] > 180) return "Use latitude −85.0511287…85.0511287 and longitude −180…180.";
  if (area[3] - area[1] > 180) return "Choose a smaller area. Areas cannot cross the date line.";
  const size = areaMetrics(area);
  if (size.width < 1 || size.height < 1) return "Choose an area at least one meter wide and high.";
  return "";
}

function satellitePatchCount(area, zoom) {
  const [south, west, north, east] = area;
  const size = 256 * 2 ** zoom;
  const snap = (value) => Math.abs(value - Math.round(value)) < 1e-7 ? Math.round(value) : value;
  const x = (longitude) => (longitude + 180) / 360 * size;
  const y = (latitude) => Math.max(0, Math.min(size,
    (1 - Math.asinh(Math.tan(latitude * Math.PI / 180)) / Math.PI) / 2 * size));
  const left = Math.floor(snap(x(west)));
  const top = Math.floor(snap(y(north)));
  const right = Math.ceil(snap(x(east)));
  const bottom = Math.ceil(snap(y(south)));
  const columns = Math.ceil(right / 256) - Math.floor(left / 256);
  const rows = Math.ceil(bottom / 256) - Math.floor(top / 256);
  return columns * rows;
}

function patchCountLabel(count) {
  const amount = count >= 1000 ? `${Math.round(count / 1000)}k` : count.toLocaleString();
  return `${amount} ${count === 1 ? "photo" : "photos"} needed`;
}

function renderDraft() {
  const area = state.draftBounds;
  $("form-error").hidden = true;
  const valid = Boolean(state.draftBounds);
  $("area-summary").classList.toggle("has-area", valid);
  $("area-status").textContent = valid ? "Selected ✓" : "Required";
  $("redraw-area").hidden = !valid || !state.map;
  if (valid) {
    const size = areaMetrics(area);
    $("area-size").textContent = areaLabel(size.area);
    $("area-dimensions").textContent = `${distanceLabel(size.width)} wide × ${distanceLabel(size.height)} high`;
  } else {
    $("area-size").textContent = state.map ? "No area selected" : "Map unavailable";
    $("area-dimensions").textContent = "";
  }
  $("area-dimensions").hidden = !valid;
  setSelection(state.draftBounds);
  renderResolution();
  renderCapturePanel();
}

function renderCapturePanel() {
  const reviewing = Boolean(state.plan);
  const planning = Boolean(state.pendingPlan);
  const busy = state.submitting || planning;
  form.hidden = reviewing;
  $("plan-recap").hidden = !reviewing;
  $("capture-panel").setAttribute("aria-labelledby", reviewing ? "plan-title" : "new-capture-title");
  form.querySelector(".capture-panel-body").inert = busy;
  if (state.composing) $("capture-map-tools").hidden = reviewing || busy || !state.map;
  $("close-capture").disabled = busy;
  for (const button of $("plan-recap").querySelectorAll("button")) button.disabled = state.submitting;
  $("confirm-capture").disabled = state.submitting || Boolean(state.job?.active);
  $("confirm-capture").textContent = state.submitting ? "Starting…" : "Start capture";
  $("planning-progress").hidden = !planning;
  $("start-capture").hidden = planning;
  if (planning) {
    const job = state.job;
    $("planning-phase").textContent = job.phase;
    renderProgress("planning-meter", "planning-detail", job);
    $("stop-planning").disabled = !job.active || job.state === "stopping";
  }
  const sources = form.querySelectorAll('input[name="include"]:checked').length;
  const sphere = form.elements.full_sphere.value === "true";
  form.elements.fov.closest("label").hidden = sphere;
  $("sphere-resolution").hidden = !sphere;
  for (const section of form.querySelectorAll("[data-source]")) {
    section.hidden = !form.querySelector(`input[name="include"][value="${section.dataset.source}"]`).checked;
  }
  $("capture-settings").hidden = !["satellite", "streetview"].some((source) => form.querySelector(`input[name="include"][value="${source}"]`).checked);
  $("sources-error").hidden = sources > 0;
  $("start-capture").disabled = !state.config || !state.draftBounds || !sources || busy || Boolean(state.job?.active);
  $("start-capture-label").textContent = state.submitting ? "Planning…" : "Plan capture";
  const message = planning ? "" : state.job?.active ? "A capture is already running." : !state.draftBounds ? (state.map ? "Select an area to continue." : "A working map is needed to select an area.") : "Review photo counts and estimated time before starting downloads.";
  $("capture-readiness").textContent = message;
  $("capture-readiness").hidden = !message;
}

function renderResolution() {
  const input = form.elements.satellite_zoom;
  const label = resolutionNames[input.valueAsNumber - Number(input.min)];
  $("resolution-value").value = label;
  input.setAttribute("aria-valuetext", label);
  const patches = $("satellite-patches");
  patches.hidden = !state.draftBounds;
  patches.value = state.draftBounds ? patchCountLabel(satellitePatchCount(state.draftBounds, input.valueAsNumber)) : "";
}

const mapHandlers = ["dragPan", "boxZoom", "doubleClickZoom", "touchZoomRotate", "touchPitch", "keyboard", "scrollZoom"];

async function openCapture(plan = null) {
  if (state.job?.active || state.composing) return;
  state.composing = true;
  state.plan = null;
  state.pendingPlan = null;
  ++state.selection; // Ignore pending library layer requests while selecting a new area.
  state.loadingCapture = false;
  state.previousView = state.map ? {center: state.map.getCenter(), zoom: state.map.getZoom(), pitch: state.map.getPitch(), bearing: state.map.getBearing(),
    handlers: Object.fromEntries(mapHandlers.map((key) => [key, state.map[key].isEnabled()]))} : null;
  form.reset();
  for (const [key, value] of Object.entries(plan?.options || state.config.defaults)) {
    if (key === "include") {
      for (const input of form.elements.include) input.checked = value.includes(input.value);
    } else form.elements[key].value = value;
  }
  state.draftBounds = plan ? [...plan.bounds] : null;
  form.querySelectorAll("details").forEach((details) => details.open = false);
  $("capture-panel").hidden = false;
  document.querySelector(".app").classList.add("composing");
  $("map-loading").hidden = true;
  closePhotos();
  state.popup?.remove();
  renderDraft();
  $("new-capture-title").focus();
  await state.ready;
  if (!state.composing || !state.map) return;
  state.map.resize();
  state.map.stop();
  applyTerrain(false);
  state.map.jumpTo({pitch: 0, bearing: 0});
  showLayers(contextLayerIds, true);
  setSelection(state.draftBounds);
  $("capture-map-tools").hidden = false;
  state.map.keyboard.disableRotation();
  if (plan) state.map.fitBounds(mapBounds(plan.bounds), {padding: 60, maxZoom: 19});
  setDrawMode(!plan);
}

function closeCapture(restore = true) {
  if (state.submitting || state.pendingPlan) return;
  cancelAreaDrag();
  state.composing = false;
  state.plan = null;
  state.pendingPlan = null;
  state.drawing = false;
  $("capture-panel").hidden = true;
  $("capture-map-tools").hidden = true;
  document.querySelector(".app").classList.remove("composing");
  if (state.map) {
    state.map.getCanvas().style.cursor = "";
    state.map.keyboard.enableRotation();
    state.map.touchZoomRotate.enableRotation();
    for (const [key, enabled] of Object.entries(state.previousView?.handlers || {})) state.map[key][enabled ? "enable" : "disable"]();
    state.map.resize();
    setSelection(null);
    showLayers(contextLayerIds, $("layer-context").checked);
    if (restore && state.previousView) state.map.jumpTo(state.previousView);
    if (restore) applyTerrain(false);
    else applyLayers();
  }
  $("welcome").hidden = Boolean(state.selected || state.job?.active);
  if (restore && state.selected) selectCapture(state.selected.id, false).catch((error) => notice(error.message));
  $("new-capture").focus();
}

function setDrawMode(drawing) {
  cancelAreaDrag();
  state.drawing = drawing && Boolean(state.map);
  $("draw-area").setAttribute("aria-pressed", String(state.drawing));
  $("move-map").setAttribute("aria-pressed", String(!state.drawing));
  $("draw-instruction").textContent = state.drawing ? "Drag to draw. Scroll to zoom." : "Drag corners to resize. Drag the map to move.";
  $("coordinates").textContent = "";
  if (!state.map) return;
  for (const key of ["dragPan", "doubleClickZoom", "touchZoomRotate"]) state.map[key][state.drawing ? "disable" : "enable"]();
  state.map.touchZoomRotate.disableRotation();
  state.map.touchPitch.disable();
  state.map.boxZoom.disable();
  state.map.getCanvas().style.cursor = state.drawing ? "crosshair" : "grab";
}

function cancelAreaDrag() {
  const drag = state.drag;
  if (!drag) return;
  state.drag = null;
  const canvas = state.map.getCanvas();
  if (canvas.hasPointerCapture(drag.pointerId)) canvas.releasePointerCapture(drag.pointerId);
  for (const [key, enabled] of Object.entries(drag.handlers)) state.map[key][enabled ? "enable" : "disable"]();
  canvas.style.cursor = state.drawing ? "crosshair" : "grab";
  $("draw-measure").hidden = true;
  setSelection(state.draftBounds);
}

function initializeAreaDrawing(map) {
  const canvas = map.getCanvas();
  const editable = () => state.composing && !state.submitting && !state.pendingPlan && !state.plan;
  const point = (event) => {
    const rect = canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(rect.width, event.clientX - rect.left)), Math.max(0, Math.min(rect.height, event.clientY - rect.top))];
  };
  const cornerAt = (position, pointerType) => {
    if (!state.draftBounds) return null;
    const corners = rectangle(state.draftBounds).geometry.coordinates[0].slice(0, 4);
    let nearest = null;
    let distance = pointerType === "touch" ? 22 : 12;
    corners.forEach((coordinates, index) => {
      const projected = map.project(coordinates);
      const delta = Math.hypot(projected.x - position[0], projected.y - position[1]);
      if (delta < distance) {
        distance = delta;
        nearest = {anchor: corners[(index + 2) % 4], position: projected,
          cursor: index % 2 ? "nwse-resize" : "nesw-resize"};
      }
    });
    return nearest;
  };
  const endpoint = (event) => point(event).map((value, index) => value + state.drag.offset[index]);
  const bounds = (end) => {
    const [lng, lat] = state.drag.anchor;
    const finish = map.unproject(end);
    return normalizedBounds([lat, lng, finish.lat, finish.lng].map((value, index) => {
      const limit = index % 2 ? 180 : maxDrawLatitude;
      return Math.max(-limit, Math.min(limit, value));
    }));
  };
  canvas.addEventListener("pointerdown", (event) => {
    if (!editable() || state.drag || event.button !== 0 || !event.isPrimary) return;
    const start = point(event);
    const corner = cornerAt(start, event.pointerType);
    if (!corner && !state.drawing) return;
    event.preventDefault();
    map.stop();
    canvas.focus({preventScroll: true});
    const origin = map.unproject(start);
    state.drag = {pointerId: event.pointerId, start, resizing: Boolean(corner),
      anchor: corner ? corner.anchor : [origin.lng, origin.lat],
      offset: corner ? [corner.position.x - start[0], corner.position.y - start[1]] : [0, 0],
      handlers: Object.fromEntries(mapHandlers.map((key) => [key, map[key].isEnabled()]))};
    for (const key of mapHandlers) map[key].disable();
    canvas.style.cursor = corner ? corner.cursor : "crosshair";
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.drag) {
      if (state.composing) canvas.style.cursor = (editable() && cornerAt(point(event), event.pointerType)?.cursor) || (state.drawing ? "crosshair" : "grab");
      return;
    }
    if (event.pointerId !== state.drag.pointerId) return;
    event.preventDefault();
    const end = endpoint(event);
    const area = bounds(end);
    setSelection(area);
    const size = areaMetrics(area);
    const measure = $("draw-measure");
    measure.textContent = `${distanceLabel(size.width)} × ${distanceLabel(size.height)}`;
    measure.hidden = false;
    measure.style.left = `${Math.max(8, Math.min(canvas.clientWidth - measure.offsetWidth - 8, end[0] + 15))}px`;
    measure.style.top = `${Math.max(8, Math.min(canvas.clientHeight - measure.offsetHeight - 8, end[1] + 15))}px`;
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!state.drag || event.pointerId !== state.drag.pointerId) return;
    event.preventDefault();
    const end = endpoint(event);
    const area = bounds(end);
    const [x, y] = state.drag.start;
    const tooSmall = !state.drag.resizing && (Math.abs(end[0] - x) < 6 || Math.abs(end[1] - y) < 6);
    cancelAreaDrag();
    const error = boundsError(area);
    if (tooSmall || error) {
      $("draw-instruction").textContent = tooSmall ? "Drag a larger rectangle to select an area." : error;
      return;
    }
    if (selectArea(area)) setDrawMode(false);
  });
  canvas.addEventListener("pointerleave", () => {
    if (state.composing && !state.drag) canvas.style.cursor = state.drawing ? "crosshair" : "grab";
  });
  for (const type of ["pointercancel", "lostpointercapture"]) canvas.addEventListener(type, (event) => {
    if (event.pointerId === state.drag?.pointerId) cancelAreaDrag();
  });
  window.addEventListener("blur", cancelAreaDrag);
  window.addEventListener("resize", cancelAreaDrag);
}

async function planCapture(event) {
  event.preventDefault();
  if (state.submitting || state.pendingPlan || state.plan || state.job?.active || !state.composing) return;
  renderDraft();
  if (!state.draftBounds || !form.querySelector('input[name="include"]:checked')) return;
  const invalid = [...form.elements].find((input) => input.willValidate && !input.validity.valid);
  if (invalid) {
    const details = invalid.closest("details");
    if (details) details.open = true;
    invalid.reportValidity();
    return;
  }
  $("form-error").hidden = true;
  const data = new FormData(form);
  const options = {include: data.getAll("include"), depth: data.get("depth"),
    full_sphere: data.get("full_sphere") === "true",
    streetview_format: data.get("streetview_format"), satellite_format: data.get("satellite_format")};
  for (const key of ["step", "fov", "delay", "satellite_zoom", "terrain_zoom", "sphere_zoom"]) options[key] = Number(data.get(key));
  const bounds = [...state.draftBounds];
  state.submitting = true;
  setDrawMode(false);
  renderCapturePanel();
  try {
    const job = await api("/api/plans", {bounds, options});
    state.pendingPlan = job.plan_id;
    renderJob(job);
  } catch (error) {
    $("form-error").textContent = error.message;
    $("form-error").hidden = false;
  } finally {
    state.submitting = false;
    renderCapturePanel();
  }
}

function durationLabel(seconds) {
  if (seconds < 60) return `${Math.ceil(seconds)} sec`;
  const minutes = Math.ceil(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const remainder = minutes % 60;
  return `${Math.floor(minutes / 60)} hr${remainder ? ` ${remainder} min` : ""}`;
}

async function reviewPlan(identity) {
  const plan = await api(runURL(identity, "/plan"));
  if (state.job?.active || state.composing) return;
  await openCapture(plan);
  showPlan(plan);
}

function showPlan(plan) {
  if (!state.composing) return;
  state.plan = plan;
  const estimate = plan.estimate;
  $("plan-duration").textContent = estimate.seconds === null ? "Download time depends on sphere resolution" : `About ${durationLabel(estimate.seconds)}`;
  const counts = $("plan-counts");
  counts.replaceChildren();
  const row = (label, value) => {
    const entry = node("div");
    entry.append(node("dt", label), node("dd", value));
    counts.append(entry);
  };
  row("Selected area", areaLabel(areaMetrics(plan.bounds).area));
  const sources = plan.options.include;
  if (sources.includes("streetview")) {
    row(plan.options.full_sphere ? "Street View spheres" : "Street View photos", estimate.streetview_photos.toLocaleString());
    row("Street View stops", estimate.streetview_stops.toLocaleString());
  }
  if (sources.includes("satellite")) row("Satellite tiles", estimate.satellite_tiles.toLocaleString());
  if (sources.includes("osm")) {
    row("OSM map extracts", estimate.osm_maps.toLocaleString());
    row("Terrain tiles", estimate.terrain_tiles.toLocaleString());
  }
  $("plan-error").hidden = true;
  setDrawMode(false);
  renderCapturePanel();
  $("plan-title").focus();
}

function editPlan() {
  if (state.submitting || state.pendingPlan) return;
  state.plan = null;
  renderDraft();
  $("new-capture-title").focus();
}

async function confirmCapture() {
  if (!state.plan || state.submitting || state.job?.active) return;
  const plan = state.plan;
  state.submitting = true;
  $("plan-error").hidden = true;
  renderCapturePanel();
  try {
    const job = await api(plan.id ? runURL(plan.id, "/resume") : "/api/captures",
      plan.id ? {} : {plan_id: plan.plan_id});
    renderJob(job);
    state.submitting = false;
    closeCapture(false);
    await refreshLibrary();
    if (job.run_id) await selectCapture(job.run_id);
  } catch (error) {
    if (state.plan) {
      $("plan-error").textContent = error.message;
      $("plan-error").hidden = false;
    } else notice(error.message);
  } finally {
    state.submitting = false;
    renderCapturePanel();
  }
}

function renderProgress(meterId, detailId, job) {
  const meter = $(meterId);
  const measured = job.total > 0;
  const done = job.done || 0;
  const percent = measured ? `${Math.floor(Math.max(0, Math.min(100, done / job.total * 100)))}%` : "—%";
  let detail = measured ? `${done.toLocaleString()} / ${job.total.toLocaleString()}` : "Working…";
  if (job.unit === "bytes") {
    const mb = (value) => (value / 1_000_000).toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
    detail = `${mb(done)} MB / ${measured ? mb(job.total) : "—"} MB`;
  }
  if (measured) {
    meter.max = job.total;
    meter.value = done;
  } else meter.removeAttribute("value");
  $(`${meterId}-percent`).textContent = percent;
  $(detailId).textContent = detail;
  meter.setAttribute("aria-valuetext", measured ? `${percent}, ${detail}` : detail);
}

function renderJob(job) {
  const previous = state.job;
  state.job = job;
  const panel = $("job");
  const wasHidden = panel.hidden;
  panel.hidden = job.kind === "plan" || job.state === "idle" || job.state === "complete";
  // Only announce changed phase text, not every polling response.
  const phase = job.phase;
  if ($("job-phase").textContent !== phase) $("job-phase").textContent = phase;
  $("job-phase").title = phase;
  $("job-meter").hidden = !job.active;
  renderProgress("job-progress", "job-detail", job);
  $("stop-job").hidden = !job.active;
  $("stop-job").disabled = job.state === "stopping";
  const stopLabel = job.state === "stopping" ? "Stopping capture…" : "Stop capture";
  $("stop-job").setAttribute("aria-label", stopLabel);
  $("stop-job").title = stopLabel;
  $("view-job").hidden = !job.run_id || state.captures.some((run) => run.id === job.run_id);
  $("job-error").hidden = !job.error;
  $("job-error").title = job.error || "";
  $("new-capture").disabled = $("welcome-new").disabled = job.active;
  renderCapturePanel();
  if (wasHidden !== panel.hidden || previous?.run_id !== job.run_id
      || previous?.state !== job.state || previous?.active !== job.active) renderLibrary();
}

async function poll() {
  try {
    const previous = state.job;
    const job = await api("/api/job");
    renderJob(job);
    const finished = previous?.active && !job.active;
    const created = job.run_id && job.run_id !== previous?.run_id;
    if ((job.kind !== "plan" && (job.active || created || finished)) || Date.now() - state.lastRefresh > 15000) await refreshLibrary();
    if (job.run_id && (created || (finished && !state.selected))) await selectCapture(job.run_id);
    if (state.pendingPlan && job.plan_id === state.pendingPlan && !job.active) {
      try {
        if (job.state !== "planned") throw new Error(job.error || "Planning stopped. Adjust your settings or plan again.");
        showPlan(await api(`/api/plans/${encodeURIComponent(state.pendingPlan)}`));
      } catch (error) {
        $("form-error").textContent = error.message;
        $("form-error").hidden = false;
      } finally {
        state.pendingPlan = null;
        renderCapturePanel();
      }
    }
  } catch (error) {
    notice(`Dashboard connection lost: ${error.message} Keep the terminal server running, then reload this page.`);
  } finally {
    window.setTimeout(poll, 1000);
  }
}

async function runAction(identity, action) {
  if (state.job?.active) return;
  try {
    renderJob(await api(runURL(identity, "/" + action), {}));
    if (state.selected?.id !== identity) await selectCapture(identity);
  } catch (error) { notice(error.message); }
}

function bindEvents() {
  $("new-capture").addEventListener("click", () => openCapture());
  $("welcome-new").addEventListener("click", () => openCapture());
  $("close-capture").addEventListener("click", () => closeCapture());
  $("capture-form").addEventListener("submit", planCapture);
  $("confirm-capture").addEventListener("click", confirmCapture);
  $("close-plan").addEventListener("click", () => closeCapture());
  $("edit-plan").addEventListener("click", editPlan);
  $("satellite-resolution").addEventListener("input", renderResolution);
  form.addEventListener("change", renderCapturePanel);
  $("search").addEventListener("input", renderLibrary);
  $("refresh").addEventListener("click", () => refreshLibrary(true).catch((error) => notice(error.message)));
  $("close-notice").addEventListener("click", () => $("notice").hidden = true);
  $("close-photo").addEventListener("click", closePhotos);
  $("previous-photo").addEventListener("click", () => showPhoto(state.photoIndex - 1, true));
  $("next-photo").addEventListener("click", () => showPhoto(state.photoIndex + 1, true));
  $("terrain").addEventListener("change", () => applyTerrain());
  $("exaggeration").addEventListener("input", () => applyTerrain(false));
  Object.keys(layerGroups).forEach((key) => $("layer-" + key).addEventListener("change", applyLayers));
  $("layer-context").addEventListener("change", async () => {
    await state.ready;
    showLayers(contextLayerIds, $("layer-context").checked);
  });
  $("draw-area").addEventListener("click", () => setDrawMode(true));
  $("redraw-area").addEventListener("click", () => setDrawMode(true));
  $("move-map").addEventListener("click", () => setDrawMode(false));
  $("use-view").addEventListener("click", () => {
    if (!state.map) return;
    const bounds = state.map.getBounds();
    const area = [Math.max(-maxDrawLatitude, bounds.getSouth()), Math.max(-180, bounds.getWest()), Math.min(maxDrawLatitude, bounds.getNorth()), Math.min(180, bounds.getEast())];
    const error = boundsError(area);
    if (error) { $("draw-instruction").textContent = "Zoom in before using the visible map."; return; }
    if (selectArea(area)) setDrawMode(false);
  });
  for (const id of ["stop-job", "stop-planning"]) $(id).addEventListener("click", async () => {
    try { renderJob(await api("/api/job/stop", {})); } catch (error) { notice(error.message); }
  });
  $("view-job").addEventListener("click", () => {
    if (state.job?.run_id) selectCapture(state.job.run_id).catch((error) => notice(error.message));
  });
  $("job-error").addEventListener("click", () => notice(state.job.error));
  document.addEventListener("click", (event) => {
    for (const menu of document.querySelectorAll(".card-more[open]")) {
      if (!menu.contains(event.target)) menu.open = false;
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (state.plan) { editPlan(); return; }
      const menu = document.querySelector(".card-more[open]");
      if (menu) {
        menu.open = false;
        menu.querySelector("summary").focus();
        return;
      }
      if (state.drag) cancelAreaDrag();
      else if (state.composing && state.drawing && state.draftBounds) setDrawMode(false);
      else if (state.composing) closeCapture();
      else closePhotos();
    }
  });
}

async function main() {
  bindEvents();
  state.config = await api("/api/config");
  $("root-path").textContent = state.config.root;
  $("root-path").title = state.config.root;
  state.ready = initializeMap();
  if (state.map) {
    await state.ready;
    state.map.on("click", (event) => {
      if (state.composing) return;
      if (state.map.getLayer("stops")) {
        const points = state.map.queryRenderedFeatures([[event.point.x - 6, event.point.y - 6], [event.point.x + 6, event.point.y + 6]], {layers: ["stops"]});
        if (points.length) {showPhoto(Number(points[0].properties.index)); return;}
      }
      const layers = ["buildings", "buildings-3d"].filter((id) => state.map.getLayer(id));
      if (!layers.length) return;
      const buildings = state.map.queryRenderedFeatures(event.point, {layers});
      if (!buildings.length) return;
      const p = buildings[0].properties;
      const content = node("div");
      content.append(node("strong", p.name || "Building"),
        node("p", `Type: ${p.building}`),
        node("p", `Height: ${Number(p._height).toFixed(1)} m${p._estimated ? " (estimated)" : ""}`));
      state.popup?.remove();
      state.popup = new maplibregl.Popup().setLngLat(event.lngLat).setDOMContent(content).addTo(state.map);
    });
    state.map.on("mousemove", (event) => {
      if (state.composing || state.rotating) return;
      const layers = ["stops", "buildings", "buildings-3d"].filter((id) => state.map.getLayer(id));
      state.map.getCanvas().style.cursor = layers.length && state.map.queryRenderedFeatures(event.point, {layers}).length ? "pointer" : "";
    });
    if (window.innerWidth < 680) document.querySelector(".layer-panel").open = false;
  }
  await refreshLibrary(true);
  poll();
}

main().catch((error) => notice(`Dashboard could not start: ${error.message}`));
