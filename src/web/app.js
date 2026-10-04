"use strict";

const BASEMAP = "https://tiles.openfreemap.org/styles/positron";
const EMPTY = {type: "FeatureCollection", features: []};
const NAMES = {streetview: "Street View", satellite: "Satellite", osm: "OSM map", terrain: "Terrain", mesh: "3D mesh"};
const STATES = {planned: "Planned", running: "Capturing", interrupted: "Interrupted", stopped: "Stopped",
                failed: "Failed", complete: "Complete", unreadable: "Unreadable"};
const NUMBERS = ["step", "fov", "sphere_zoom", "satellite_zoom", "terrain_zoom", "mesh_level", "delay",
                 "streetview_workers", "satellite_workers", "terrain_workers", "mesh_workers"];
const EARTH = 6371008.8;
// Level names shown under slider knobs.
const SCALES = {
  satellite_zoom: {12: "City", 13: "Town", 14: "District", 15: "Blocks", 16: "Buildings", 17: "Roofs", 18: "Cars",
                   19: "Road markings", 20: "Details", 21: "Finest"},
  terrain_zoom: {8: "Region", 9: "Province", 10: "Metro area", 11: "City", 12: "Town", 13: "District", 14: "Blocks"},
  mesh_level: {14: "Hills", 15: "Districts", 16: "Blocks", 17: "Buildings", 18: "Roofs", 19: "Windows", 20: "Cars",
               21: "Details", 22: "Finest"},
  sphere_zoom: Object.fromEntries(["Thumbnail", "Low", "Medium", "High", "Very high", "Maximum"]
    .map((name, z) => [z, `${name} · ${(512 * 2 ** z).toLocaleString("en-US")} px`])),
};
const ICONS = {
  streetview: '<path d="M3 8h4l2-3h6l2 3h4v11H3z"/><circle cx="12" cy="13" r="3.5"/>',
  satellite: '<path d="M9.5 9.5h5v7h-5zM2 10h5v5H2zM17 10h5v5h-5zM4.5 10v5M19.5 10v5M7 12.5h2.5M14.5 12.5H17M12 9.5v-3M9.5 5c1.5 1.3 3.5 1.3 5 0"/>',
  osm: '<path d="M3 6l6-2 6 2 6-2v14l-6 2-6-2-6 2zM9 4v14M15 6v14"/>',
  terrain: '<path d="M2 20 9 8l4 6 3-4 6 10z"/>',
  mesh: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9zM4 7.5l8 4.5 8-4.5M12 12v9"/>',
  map: '<path d="M12 4l9 5-9 5-9-5zM3 14l9 5 9-5"/>',
};
// Progress phases that a capture stage row already shows.
const DOWNLOADS = new Set(["Street View", "Satellite", "Downloading terrain tiles", "3D mesh"]);

// Helpers

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);
const number = value => Math.round(value).toLocaleString("en-US");
const plural = (count, one, many = `${one}s`) => `${number(count)} ${count === 1 ? one : many}`;
const icon = name => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const absolute = path => location.origin + path;
const capturePath = (id, path = "") => `/api/captures/${encodeURIComponent(id)}${path}`;

const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(`aleph:${key}`)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`aleph:${key}`, JSON.stringify(value)); } catch { /* Private windows may refuse. */ }
  },
};

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(data?.error || `${response.status} ${response.statusText}`);
  return data;
}

function duration(seconds) {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  const minutes = Math.round(seconds / 60);
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

const meters = m => m >= 1000 ? `${(m / 1000).toFixed(m >= 10000 ? 0 : 1)} km` : `${Math.round(m)} m`;
const resolution = m => `${m < 1 ? m.toFixed(2) : m.toFixed(1)} m/px`;
const megabytes = bytes => bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB`
  : `${Math.max(1, Math.round(bytes / 1e3))} KB`;

function distance([lat1, lon1], [lat2, lon2]) {
  const r = Math.PI / 180;
  const a = Math.sin((lat2 - lat1) * r / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin((lon2 - lon1) * r / 2) ** 2;
  return 2 * EARTH * Math.asin(Math.sqrt(a));
}

function size([s, w, n, e]) {
  const middle = (s + n) / 2;
  return [distance([middle, w], [middle, e]), distance([s, w], [n, w])];
}

const dimensions = area => size(area).map(meters).join(" × ");
const ring = ([s, w, n, e]) => [[w, s], [e, s], [e, n], [w, n], [w, s]];
const polygon = (area, properties = {}) => ({type: "Feature", properties, geometry: {type: "Polygon", coordinates: [ring(area)]}});
const point = (coordinates, properties = {}) => ({type: "Feature", properties, geometry: {type: "Point", coordinates}});
const collection = features => ({type: "FeatureCollection", features});
const lngLatBounds = ([s, w, n, e]) => [[w, s], [e, n]];

function when(iso) {
  const date = new Date(iso);
  return `${date.toLocaleDateString(undefined, {day: "numeric", month: "short", year: "numeric"})} · ${
    date.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit"})}`;
}

function destination([lon, lat], heading, length) {
  const r = Math.PI / 180, d = length / EARTH, h = heading * r, phi = lat * r;
  const phi2 = Math.asin(Math.sin(phi) * Math.cos(d) + Math.cos(phi) * Math.sin(d) * Math.cos(h));
  const lambda = lon * r + Math.atan2(Math.sin(h) * Math.sin(d) * Math.cos(phi), Math.cos(d) - Math.sin(phi) * Math.sin(phi2));
  return [lambda / r, phi2 / r];
}

// State

const state = {
  config: null,
  captures: [],
  job: null,
  view: null,
  area: store.get("area", null),
  drawing: false,
  drag: null,
  preview: null,
  planError: null,
  captureId: null,
  capture: null,
  photos: [],
  photoCount: 0,
  photoLoading: false,
  photoAgain: false,
  live: {},
  selected: null,
  search: [],
  toggles: {photos: true, route: true, satellite: true, terrain: true, buildings: true, roads: true, water: true,
            outline: true, basemap: true, ...store.get("toggles", {})},
};

let map;
let basemapLayers = [];
const themed = new Map();  // Layer id → paint with CSS variable names, resolved per color scheme.

// Map styling

// Replace "--name" strings, including inside expressions, with the current CSS value.
const resolve = paint => JSON.parse(JSON.stringify(paint), (key, value) =>
  typeof value === "string" && value.startsWith("--") ? css(value) : value);

function tint(layer) {
  const source = layer["source-layer"] || "";
  if (layer.type === "background") return {"background-color": "--paper"};
  if (layer.type === "fill") {
    if (source === "water") return {"fill-color": "--map-water"};
    if (source === "building") return {"fill-color": "--map-building", "fill-outline-color": "--map-building-line"};
    return {"fill-color": "--map-land"};
  }
  if (layer.type === "line") {
    if (source === "waterway") return {"line-color": "--map-water"};
    if (source === "boundary") return {"line-color": "--map-boundary"};
    if (layer.id.includes("rail")) return {"line-color": "--map-rail"};
    return {"line-color": layer.id.includes("casing") ? "--map-casing" : "--map-road"};
  }
  if (layer.type === "symbol") return {"text-color": "--map-label", "text-halo-color": "--paper"};
  return {};
}

async function basemap() {
  try {
    const style = await (await fetch(BASEMAP)).json();
    // Shields and icons would bring color back; keep the map to paper and ink.
    style.layers = style.layers.filter(layer => !(layer.type === "symbol" && layer.layout?.["icon-image"]));
    for (const layer of style.layers) {
      themed.set(layer.id, tint(layer));
      layer.paint = {...layer.paint, ...resolve(tint(layer))};
    }
    basemapLayers = style.layers.map(layer => layer.id);
    return style;
  } catch {
    // Offline: saved captures still display on plain paper.
    themed.set("background", {"background-color": "--paper"});
    basemapLayers = ["background"];
    return {version: 8, sources: {}, layers: [{id: "background", type: "background", paint: resolve({"background-color": "--paper"})}]};
  }
}

const is = (key, value) => ["==", ["get", key], value];
const hover = (on, off) => ["case", ["boolean", ["feature-state", "hover"], false], on, off];
const grow = (low, high) => ["interpolate", ["linear"], ["zoom"], 14, low, 19, high];

const OVERLAYS = [
  {id: "osm-land", source: "osm", type: "fill", filter: is("kind", "land"), paint: {"fill-color": "--ink", "fill-opacity": 0.05}},
  {id: "osm-water", source: "osm", type: "fill", filter: is("kind", "water"), paint: {"fill-color": "--ink", "fill-opacity": 0.14}},
  {id: "osm-waterways", source: "osm", type: "line", filter: is("kind", "waterway"),
   paint: {"line-color": "--ink", "line-opacity": 0.4, "line-width": 1}},
  {id: "osm-buildings", source: "osm", type: "fill", filter: is("kind", "building"), paint: {"fill-color": "--ink", "fill-opacity": 0.14}},
  {id: "osm-building-lines", source: "osm", type: "line", filter: is("kind", "building"),
   paint: {"line-color": "--ink", "line-opacity": 0.7, "line-width": 0.75}},
  {id: "osm-paths", source: "osm", type: "line", filter: ["all", is("kind", "road"), is("road", "path")],
   paint: {"line-color": "--ink", "line-opacity": 0.6, "line-width": 1, "line-dasharray": [2, 1.5]}},
  {id: "osm-roads", source: "osm", type: "line", filter: ["all", is("kind", "road"), ["!=", ["get", "road"], "path"]],
   layout: {"line-cap": "round", "line-join": "round"},
   paint: {"line-color": "--ink", "line-opacity": 0.75, "line-width": ["interpolate", ["linear"], ["zoom"],
     13, ["match", ["get", "road"], "major", 1.2, 0.6], 18, ["match", ["get", "road"], "major", 4, 2]]}},
  {id: "bounds", source: "bounds", type: "line", paint: {"line-color": "--ink", "line-width": 1, "line-dasharray": [4, 3]}},
  {id: "plan-grid", source: "plan", type: "line", filter: is("kind", "satellite"),
   paint: {"line-color": "--ink", "line-opacity": 0.3, "line-width": 0.5}},
  {id: "plan-terrain", source: "plan", type: "line", filter: is("kind", "terrain"),
   paint: {"line-color": "--muted", "line-width": 1, "line-dasharray": [2, 2]}},
  {id: "plan-paths", source: "plan", type: "line", filter: is("kind", "path"), layout: {"line-cap": "round", "line-join": "round"},
   paint: {"line-color": "--muted", "line-opacity": 0.6, "line-width": 2}},
  {id: "route-paths", source: "plan", type: "line", filter: is("kind", "path"), layout: {"line-cap": "round", "line-join": "round"},
   paint: {"line-color": "--muted", "line-opacity": 0.35, "line-width": 1.5}},
  {id: "plan-gaps", source: "plan", type: "line", filter: is("kind", "gap"),
   paint: {"line-color": "--accent", "line-width": 2, "line-dasharray": [1, 1.5]}},
  {id: "plan-stops", source: "plan", type: "circle", filter: is("kind", "stop"),
   paint: {"circle-radius": grow(1.5, 4), "circle-color": "--accent"}},
  {id: "route-stops", source: "plan", type: "circle", filter: is("kind", "stop"),
   paint: {"circle-radius": grow(1.5, 4.5), "circle-color": "--paper", "circle-stroke-color": "--muted", "circle-stroke-width": 1}},
  {id: "photos", source: "photos", type: "circle",
   paint: {"circle-radius": grow(2, 5.5), "circle-color": "--ink", "circle-stroke-color": "--paper", "circle-stroke-width": 1.5}},
  {id: "selected-view", source: "selected", type: "line", filter: ["==", ["geometry-type"], "LineString"],
   layout: {"line-cap": "round"}, paint: {"line-color": "--accent", "line-width": 2.5}},
  {id: "selected", source: "selected", type: "circle", filter: ["==", ["geometry-type"], "Point"],
   paint: {"circle-radius": 6, "circle-color": "--accent", "circle-stroke-color": "--paper", "circle-stroke-width": 2}},
  {id: "outlines-fill", source: "outlines", type: "fill", paint: {"fill-color": "--accent", "fill-opacity": hover(0.08, 0)}},
  {id: "outlines", source: "outlines", type: "line", paint: {"line-color": hover("--accent", "--ink"), "line-width": hover(2, 1)}},
  {id: "area-fill", source: "area", type: "fill", paint: {"fill-color": "--accent", "fill-opacity": 0.07}},
  {id: "area-line", source: "area", type: "line", paint: {"line-color": "--accent", "line-width": 1.5}},
  {id: "corners", source: "corners", type: "symbol",
   layout: {"icon-image": "corner", "icon-allow-overlap": true, "icon-ignore-placement": true}},
];

// Result layers each toggle shows.
const GROUPS = {
  photos: ["photos", "selected", "selected-view"],
  route: ["route-paths", "route-stops"],
  satellite: ["satellite"],
  terrain: ["hillshade"],
  buildings: ["osm-buildings", "osm-building-lines"],
  roads: ["osm-roads", "osm-paths"],
  water: ["osm-land", "osm-water", "osm-waterways"],
  outline: ["bounds"],
};

function addLayer(layer, before) {
  themed.set(layer.id, layer.paint || {});
  map.addLayer({...layer, paint: resolve(layer.paint || {})}, before);
}

function cornerImage() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 20;
  const context = canvas.getContext("2d");
  context.fillStyle = css("--accent");
  context.fillRect(0, 0, 20, 20);
  context.fillStyle = css("--paper");
  context.fillRect(3, 3, 14, 14);
  return context.getImageData(0, 0, 20, 20);
}

function applyTheme() {
  for (const [id, paint] of themed) {
    if (!map.getLayer(id)) continue;
    for (const [key, value] of Object.entries(resolve(paint))) map.setPaintProperty(id, key, value);
  }
  map.updateImage("corner", cornerImage());
}

function visibleLayers() {
  const on = new Set();
  const add = (...ids) => ids.forEach(id => on.add(id));
  if (state.view === "library") add("outlines", "outlines-fill");
  if (state.view === "new" || state.view === "preview") add("area-fill", "area-line");
  if (state.view === "new" && state.area && !state.drawing) add("corners");
  if (state.view === "preview" && state.preview) add("plan-grid", "plan-terrain", "plan-paths", "plan-gaps", "plan-stops");
  if (state.view === "capture") {
    for (const [group, ids] of Object.entries(GROUPS)) if (state.toggles[group]) add(...ids);
  }
  return on;
}

function syncMap() {
  if (!map?.getLayer("corners")) return;
  const on = visibleLayers();
  for (const id of [...OVERLAYS.map(layer => layer.id), "satellite", "hillshade"]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on.has(id) ? "visible" : "none");
  }
  const base = state.view !== "capture" || state.toggles.basemap;
  for (const id of basemapLayers) {
    if (id !== "background" && map.getLayer(id)) map.setLayoutProperty(id, "visibility", base ? "visible" : "none");
  }
}

function setData(source, data) {
  map.getSource(source)?.setData(data);
}

// Rasters are added once a capture has saved tiles; their URLs carry the saved count.
function setRaster(mode, info) {
  const tiles = [absolute(capturePath(state.capture.id, `/${mode}/{z}/{x}/{y}?v=${info.done}`))];
  const source = map.getSource(mode);
  if (source) return source.setTiles(tiles);
  const [s, w, n, e] = state.capture.bounds;
  const spec = {tiles, bounds: [w, s, e, n], minzoom: Math.max(0, info.zoom - state.config.overviews), maxzoom: info.zoom};
  if (mode === "satellite") {
    map.addSource(mode, {type: "raster", tileSize: 256, ...spec});
    addLayer({id: "satellite", type: "raster", source: mode, paint: {"raster-fade-duration": 0}}, "osm-land");
  } else {
    map.addSource(mode, {type: "raster-dem", encoding: "terrarium", tileSize: 512, ...spec});
    addLayer({id: "hillshade", type: "hillshade", source: mode, paint: {
      "hillshade-shadow-color": "--hill-shadow", "hillshade-highlight-color": "--hill-light",
      "hillshade-accent-color": "--hill-shadow", "hillshade-exaggeration": 0.55}}, "osm-land");
  }
  syncMap();
}

function clearCaptureLayers() {
  for (const [layer, source] of [["satellite", "satellite"], ["hillshade", "terrain"]]) {
    if (map.getLayer(layer)) map.removeLayer(layer);
    if (map.getSource(source)) map.removeSource(source);
    themed.delete(layer);
  }
  for (const source of ["osm", "photos", "selected", "bounds", "plan"]) setData(source, EMPTY);
  Object.assign(state, {photos: [], photoCount: 0, photoAgain: false, live: {}, selected: null});
  $("#viewer").hidden = true;
}

// Area drawing

function setArea(area) {
  state.area = area;
  setData("area", area ? collection([polygon(area)]) : EMPTY);
  setData("corners", area ? collection(ring(area).slice(0, 4).map(c => point(c))) : EMPTY);
  renderArea();
}

function areaFrom(a, b) {
  const clampLat = value => Math.max(-85.05, Math.min(85.05, value));
  const clampLon = value => Math.max(-180, Math.min(180, value));
  return [clampLat(Math.min(a.lat, b.lat)), clampLon(Math.min(a.lng, b.lng)),
          clampLat(Math.max(a.lat, b.lat)), clampLon(Math.max(a.lng, b.lng))];
}

function cornerAt(pixel) {
  if (!state.area) return null;
  const corners = ring(state.area).slice(0, 4);
  const index = corners.findIndex(corner => {
    const p = map.project(corner);
    return Math.abs(p.x - pixel.x) <= 9 && Math.abs(p.y - pixel.y) <= 9;
  });
  if (index < 0) return null;
  const [lng, lat] = corners[(index + 2) % 4];
  return {index, anchor: {lng, lat}};
}

function setDrawing(on) {
  state.drawing = on;
  $("#hint").hidden = !on;
  map.getCanvas().style.cursor = on ? "crosshair" : "";
  renderArea();
  syncMap();
}

function startDrag(event) {
  if (state.view !== "new" || event.originalEvent.button !== 0) return;
  const corner = state.drawing ? null : cornerAt(event.point);
  if (!state.drawing && !corner) return;
  event.preventDefault();  // Keeps the map from panning.
  state.drag = {anchor: corner ? corner.anchor : event.lngLat, previous: state.area};
}

function moveDrag(event) {
  if (state.drag) return setArea(areaFrom(state.drag.anchor, event.lngLat));
  if (state.view === "new" && !state.drawing) {
    const corner = cornerAt(event.point);
    map.getCanvas().style.cursor = corner ? (corner.index % 2 ? "nwse-resize" : "nesw-resize") : "";
  }
}

function endDrag() {
  if (!state.drag) return;
  const {previous} = state.drag;
  state.drag = null;
  const [width, height] = state.area ? size(state.area) : [0, 0];
  if (width < 5 || height < 5) {
    setArea(previous);  // A click, not a drag.
    return;
  }
  store.set("area", state.area);
  setDrawing(false);
  estimate();
}

// Settings

function fillForm(options) {
  for (const input of $("#settings").elements) {
    const name = input.name;
    if (!name) continue;
    if (name === "include") input.checked = options.include.includes(input.value);
    else if (name === "camera") input.checked = (input.value === "sphere") === options.full_sphere;
    else if (input.type === "radio") input.checked = String(options[name]) === input.value;
    else if (name in options) input.value = options[name];
  }
  updateForm();
}

function readForm() {
  const data = new FormData($("#settings"));
  const options = {include: data.getAll("include"), full_sphere: data.get("camera") === "sphere", depth: data.get("depth"),
                   streetview_format: data.get("streetview_format"), satellite_format: data.get("satellite_format")};
  for (const name of NUMBERS) options[name] = Number(data.get(name));
  return options;
}

function updateForm() {
  const options = readForm();
  for (const fieldset of $$(".source")) fieldset.classList.toggle("on", options.include.includes(fieldset.dataset.source));
  for (const field of $$("[data-camera]")) field.hidden = (field.dataset.camera === "sphere") !== options.full_sphere;
  for (const input of $$(".range input")) {
    const range = input.parentElement, min = Number(input.min), max = Number(input.max);
    range.style.setProperty("--at", (input.value - min) / (max - min));
    range.style.setProperty("--steps", max - min);
    $(".knob", range).textContent = input.value;
    $(".tag", range).textContent = SCALES[input.name][input.value];
  }
  $("#preview-button").disabled = !state.area || !options.include.length;
  store.set("options", options);
}

let estimateTimer = null;
let estimateCount = 0;

function estimate() {
  clearTimeout(estimateTimer);
  estimateTimer = setTimeout(async () => {
    const count = ++estimateCount;
    const {include, satellite_zoom, terrain_zoom, satellite_workers, terrain_workers, delay} = readForm();
    const outputs = {satellite: $("[data-source=satellite] .estimate"), osm: $("[data-source=osm] .estimate")};
    const details = {satellite: $("[data-detail=satellite]"), terrain: $("[data-detail=terrain]")};
    let result = {};
    if (state.area && (include.includes("satellite") || include.includes("osm"))) {
      try {
        result = await api("/api/estimate", {bounds: state.area, options:
          {include, satellite_zoom, terrain_zoom, satellite_workers, terrain_workers, delay}});
      } catch {
        // Invalid values are reported by Preview.
      }
    }
    if (count !== estimateCount) return;
    const {satellite, terrain} = result;
    outputs.satellite.textContent = satellite ? `${plural(satellite.tiles, "tile")} · ~${duration(satellite.seconds)}` : "";
    details.satellite.textContent = satellite
      ? `${resolution(satellite.meters_per_pixel)} · ${number(satellite.width)} × ${number(satellite.height)} px` : "";
    outputs.osm.textContent = terrain ? `${plural(terrain.tiles, "terrain tile")}` : "";
    details.terrain.textContent = terrain ? `${resolution(terrain.meters_per_pixel)} elevation` : "";
  }, 120);
}

function renderArea() {
  const area = state.area;
  $("#redraw").textContent = state.drawing ? "Cancel" : area ? "Redraw" : "Draw area";
  $("#area").innerHTML = area ? `<strong>${esc(dimensions(area))}</strong>`
    : `<span class="muted">${state.drawing ? "Drag on the map to draw it." : "No area yet."}</span>`;
  $("#preview-button").disabled = !area || !readForm().include.length;
}

// Progress

function renderProgress(container, job) {
  const bar = $(".bar", container);
  const known = job.total != null && job.total > 0;
  bar.classList.toggle("indeterminate", !known);
  $("i", bar).style.width = known ? `${Math.min(100, job.done / job.total * 100)}%` : "";
  let amount = "";
  if (job.unit === "bytes") amount = `${(job.done / 1e6).toFixed(1)}${job.total ? ` / ${(job.total / 1e6).toFixed(1)}` : ""} MB`;
  else if (known) amount = `${number(job.done)} / ${number(job.total)}`;
  else if (job.done) amount = number(job.done);
  $(".progress-text", container).innerHTML =
    `<span>${esc(job.stopping ? "Stopping…" : job.phase)}</span><span>${esc(amount)}</span>`;
}

const running = (kind, id) => {
  const job = state.job;
  return job?.state === "running" && job.kind === kind && (id === undefined || job.capture === id);
};

// Library

async function loadCaptures() {
  try {
    state.captures = await api("/api/captures");
  } catch (error) {
    $("#captures").innerHTML = `<li class="empty error">${esc(error.message)}</li>`;
    return;
  }
  setData("outlines", collection(state.captures.filter(c => c.bounds).map(c => polygon(c.bounds, {id: c.id}))));
  renderLibrary();
}

function renderLibrary() {
  const list = $("#captures");
  $("#capture-count").textContent = state.captures.length || "";
  if (!state.captures.length) {
    list.innerHTML = `<li class="empty">No captures here yet.</li>`;
    return;
  }
  list.innerHTML = state.captures.map(c => {
    if (!c.bounds) {
      return `<li><div class="capture"><span class="when">${esc(c.id)}</span><span class="state failed">${STATES[c.state]}</span>
        <span class="detail">${esc(c.error)}</span></div></li>`;
    }
    const live = running("capture", c.id);
    const bar = live ? `<span class="bar"><i style="width:${libraryProgress()}%"></i></span>` : "";
    const status = live ? "running" : c.state;
    return `<li><button type="button" class="capture" data-id="${esc(c.id)}">
      <span class="when">${esc(when(c.started_at))}</span><span class="state ${status}">${STATES[status]}</span>
      <span class="detail">${esc(dimensions(c.bounds))} · ${esc(c.options.include.map(s => NAMES[s]).join(", "))}</span>${bar}
    </button></li>`;
  }).join("");
}

function libraryProgress() {
  const stages = Object.values(state.job.stages);
  return (stages.reduce((sum, stage) => sum + stage.done / Math.max(1, stage.total), 0) / stages.length * 100).toFixed(1);
}

function hoverCapture(id) {
  for (const button of $$(".capture[data-id]")) button.classList.toggle("hover", button.dataset.id === id);
  if (state.hovered === id) return;
  if (state.hovered) map.setFeatureState({source: "outlines", id: state.hovered}, {hover: false});
  if (id) map.setFeatureState({source: "outlines", id}, {hover: true});
  state.hovered = id;
}

// Preview

const stageRows = {
  streetview: s => [plural(s.photos, s.sphere ? "sphere" : "photo"), !s.meters ? "No mapped roads at this depth"
    : !s.stops ? "No panoramas on the selected roads"
    : `${plural(s.stops, "stop")} along ${meters(s.meters)} of road` + (s.gaps ? ` · ${plural(s.gaps, "spacing gap")}` : "")],
  satellite: s => [plural(s.tiles, "tile"), `${number(s.width)} × ${number(s.height)} px · ${resolution(s.meters_per_pixel)}`],
  osm: () => ["1 extract", "Clipped from a Geofabrik regional file"],
  terrain: s => [plural(s.tiles, "tile"), `Zoom ${s.zoom} · ${resolution(s.meters_per_pixel)}`],
  mesh: s => [plural(s.nodes, "node"), `Detail level ${s.level}`],
};

function renderPreview() {
  const job = state.job;
  const planning = running("plan");
  $("#planning").hidden = !planning;
  if (planning) renderProgress($("#planning .progress"), job);
  $("#cancel-plan").hidden = !planning;
  $("#cancel-plan").disabled = !!job?.stopping;
  $("#plan").hidden = !state.preview;
  $("#start").hidden = !state.preview;
  $("#plan-error").hidden = !state.planError;
  $("#plan-error").textContent = state.planError || "";
  if (!state.preview) return;
  $("#plan-stages").innerHTML = Object.entries(state.preview.stages).map(([mode, info]) => {
    const [value, detail] = stageRows[mode](info);
    return `<li>${icon(mode)}<strong>${NAMES[mode]}</strong><span class="value">${esc(value)}</span>
      <span class="detail">${esc(detail)}</span></li>`;
  }).join("");
  $("#plan-time").textContent = `~ ${duration(state.preview.seconds)}`;
}

async function previewPlan(event) {
  event.preventDefault();
  const error = $("#settings-error");
  error.hidden = true;
  if (!state.area) return;
  Object.assign(state, {preview: null, planError: null});
  setData("plan", EMPTY);
  try {
    state.job = await api("/api/plan", {bounds: state.area, options: readForm()});
  } catch (failure) {
    error.textContent = failure.message;
    error.hidden = false;
    return;
  }
  location.hash = "#/preview";
  poll();
}

function showPlan() {
  setData("plan", state.preview.layers);
  map.fitBounds(lngLatBounds(state.preview.bounds), {padding: 80, maxZoom: 18, duration: 600});
  syncMap();
}

// Capture

async function openCapture(id) {
  let detail;
  try {
    detail = await api(capturePath(id));
  } catch (error) {
    if (state.captureId !== id) return;
    $("#capture-title").textContent = "Capture";
    $("#capture-error").textContent = error.message;
    $("#capture-error").hidden = false;
    return;
  }
  if (state.view !== "capture" || state.captureId !== id) return;
  const first = state.capture?.id !== id;
  state.capture = detail;
  if (first) {
    setData("bounds", collection([polygon(detail.bounds)]));
    setData("plan", absolute(capturePath(id, "/plan")));
    map.fitBounds(lngLatBounds(detail.bounds), {padding: 60, maxZoom: 18, duration: 600});
  }
  refresh(detail.stages, true);
  renderCapture();
  syncMap();
}

// Bring map data up to the saved counts; imagery reloads at most every few seconds while it grows.
function refresh(stages, force = false) {
  const id = state.capture.id, live = state.live;
  if (stages.streetview && stages.streetview.done !== live.streetview) {
    live.streetview = stages.streetview.done;
    loadPhotos();
  }
  for (const mode of ["satellite", "terrain"]) {
    const info = stages[mode];
    if (!info?.done || info.done === live[mode]) continue;
    if (!force && info.done < info.total && Date.now() - (live[`${mode}At`] || 0) < 2500) continue;
    live[mode] = info.done;
    live[`${mode}At`] = Date.now();
    setRaster(mode, {...info, zoom: info.zoom ?? state.capture.stages[mode].zoom});
  }
  if (stages.osm?.done && !live.osm) {
    live.osm = true;
    setData("osm", absolute(capturePath(id, "/osm")));
  }
}

async function loadPhotos() {
  if (state.photoLoading) {
    state.photoAgain = true;
    return;
  }
  const id = state.capture.id;
  state.photoLoading = true;
  try {
    const data = await api(capturePath(id, `/photos?after=${state.photoCount}`));
    if (state.capture?.id !== id) return;
    state.photoCount = data.count;
    state.photos.push(...data.features);
    setData("photos", collection(state.photos));
    renderLayers();
  } catch {
    // The next progress update retries.
  } finally {
    state.photoLoading = false;
    if (state.photoAgain && state.capture?.id === id) {
      state.photoAgain = false;
      loadPhotos();
    }
  }
}

function renderCapture() {
  const c = state.capture;
  if (!c) return;
  const job = state.job;
  const live = running("capture", c.id);
  const status = live ? "running" : c.state;
  $("#capture-title").textContent = when(c.started_at);
  $("#capture-meta").textContent = `${STATES[status]} · ${dimensions(c.bounds)}`;
  $("#progress").hidden = !live;
  if (live) {
    const downloadPhase = DOWNLOADS.has(job.phase) && !job.stopping;
    $("#phase").hidden = downloadPhase;
    renderProgress($("#phase"), job);
    const downloading = Object.values(job.stages).some(stage => stage.done < stage.total);
    $("#progress .label").textContent = downloading ? `Capturing · about ${duration(job.seconds)} left` : "Building files";
    // Only the stage downloading now gets a bar, unless the phase line below already shows one.
    const active = downloadPhase && Object.keys(job.stages).find(mode => job.stages[mode].done < job.stages[mode].total);
    $("#progress-stages").innerHTML = Object.entries(job.stages).map(([mode, s]) => {
      const finished = s.done >= s.total;
      const bar = mode === active ? `<span class="bar"><i style="width:${(s.done / Math.max(1, s.total) * 100).toFixed(1)}%"></i></span>` : "";
      return `<li class="${finished ? "done" : ""}">${icon(mode)}<span>${NAMES[mode]}</span>
        <span class="value">${finished ? "Done" : `${number(s.done)} / ${number(s.total)}`}</span>${bar}</li>`;
    }).join("");
  }
  // Stopping on request is not an error worth showing.
  const error = !live && c.error && c.state !== "complete" && c.error !== "Interrupted";
  $("#capture-error").hidden = !error;
  $("#capture-error").textContent = error ? c.error : "";
  $("#stop").hidden = !live;
  $("#stop").disabled = !!job?.stopping;
  $("#stop").textContent = job?.stopping ? "Stopping…" : "Stop";
  $("#resume").hidden = live || c.state === "complete";
  $("#resume").disabled = running("plan") || running("capture");
  $("#folder").textContent = c.folder;
  $("#files").innerHTML = Object.entries(c.files).map(([name, bytes]) =>
    `<li><span>${esc(name)}</span><span class="value">${megabytes(bytes)}</span></li>`).join("")
    || `<li class="muted">No merged files yet.</li>`;
  renderLayers();
}

function renderLayers() {
  const c = state.capture;
  if (!c) return;
  const s = c.stages;
  const toggle = (key, name, value = "", disabled = false) => `<label class="toggle">
    <input type="checkbox" data-toggle="${key}" ${state.toggles[key] ? "checked" : ""} ${disabled ? "disabled" : ""}>
    <span>${name}</span><span class="value">${esc(value)}</span></label>`;
  const groups = [];
  if (s.streetview) {
    const skipped = s.streetview.skipped ? ` · ${number(s.streetview.skipped)} skipped` : "";
    groups.push(["streetview", toggle("photos", "Photos", `${number(state.photos.length)}${skipped}`)
      + toggle("route", "Planned route")]);
  }
  if (s.satellite) {
    groups.push(["satellite", toggle("satellite", "Imagery", `zoom ${s.satellite.zoom} · ${tileCount("satellite")}`)]);
  }
  if (s.osm) {
    const saved = state.live.osm || s.osm.done;
    const note = saved ? "" : "not saved";
    groups.push(["osm", toggle("buildings", "Buildings", note, !saved) + toggle("roads", "Roads and paths", note, !saved)
      + toggle("water", "Water and land", note, !saved)]);
  }
  if (s.terrain) groups.push(["terrain", toggle("terrain", "Hillshade", `zoom ${s.terrain.zoom} · ${tileCount("terrain")}`)]);
  if (s.mesh) {
    const glb = c.files["mesh.glb"];
    groups.push(["mesh", `<p class="toggle"><span>mesh.glb</span><span class="value">${
      glb ? megabytes(glb) : `${number(s.mesh.done)} / ${plural(s.mesh.total, "node")}`}</span></p>`]);
  }
  groups.push(["map", toggle("outline", "Capture outline") + toggle("basemap", "Basemap")]);
  $("#layers").innerHTML = groups.map(([mode, rows]) =>
    `<li class="group"><p>${icon(mode)}${NAMES[mode] ?? "Map"}</p>${rows}</li>`).join("");
}

function tileCount(mode) {
  const stage = state.capture.stages[mode];
  const done = state.live[mode] ?? stage.done;
  return done >= stage.total ? plural(stage.total, "tile") : `${number(done)} / ${plural(stage.total, "tile")}`;
}

// Photo viewer

function showPhoto(index) {
  const photo = state.photos[index];
  if (!photo) return;
  state.selected = index;
  const p = photo.properties;
  const image = $("#viewer-image");
  image.src = absolute(capturePath(state.capture.id, `/files/${p.filename.split("/").map(encodeURIComponent).join("/")}`));
  image.classList.toggle("sphere", p.heading == null);
  const facts = [`#${p.sequence}`, p.side ?? "360°", p.heading != null ? `${Math.round(p.heading)}°` : null, p.imagery_date];
  $("#viewer-text").innerHTML = `<strong>${esc(p.path_name || "Unnamed road")}</strong><span>${esc(facts.filter(Boolean).join(" · "))}</span>`;
  $("#viewer-link").href = p.streetview_url;
  $("#viewer-prev").disabled = index === 0;
  $("#viewer-next").disabled = index === state.photos.length - 1;
  $("#viewer").hidden = false;
  const at = photo.geometry.coordinates;
  const features = [point(at)];
  if (p.heading != null) features.unshift({type: "Feature", properties: {},
    geometry: {type: "LineString", coordinates: [at, destination(at, p.heading, 18)]}});
  setData("selected", collection(features));
  if (!map.getBounds().contains(at)) map.easeTo({center: at});
}

function closeViewer() {
  $("#viewer").hidden = true;
  state.selected = null;
  setData("selected", EMPTY);
}

// Jobs

let pollTimer = null;

async function poll() {
  clearTimeout(pollTimer);
  let job;
  try {
    job = await api("/api/job");
  } catch {
    pollTimer = setTimeout(poll, 3000);
    return;
  }
  const previous = state.job;
  state.job = job;
  const finished = job && job.state !== "running" && previous?.id === job.id && previous.state === "running";
  if (job?.kind === "plan" && finished) {
    if (job.state === "done") {
      try {
        state.preview = await api("/api/preview");
        if (state.view === "preview") showPlan();
      } catch (error) {
        state.planError = error.message;
      }
    } else if (job.state === "failed") {
      state.planError = job.error;
    } else if (state.view === "preview") {
      location.hash = "#/new";
    }
  }
  if (state.view === "preview") renderPreview();
  if (job?.kind === "capture" && state.capture?.id === job.capture && state.view === "capture") {
    if (finished) {
      await openCapture(job.capture);
    } else {
      refresh(job.stages);
      renderCapture();
    }
  }
  if (state.view === "library") {
    const bar = job?.kind === "capture" && $(`.capture[data-id="${CSS.escape(job.capture)}"] .bar i`);
    if (finished || (running("capture") && !bar)) loadCaptures();
    else if (bar) bar.style.width = `${libraryProgress()}%`;
  }
  if (job?.state === "running") pollTimer = setTimeout(poll, 700);
}

// Views

function show(view) {
  for (const section of $$(".view")) section.hidden = section.id !== view;
  if (view !== "new" && state.drawing) setDrawing(false);
  if (view !== "capture" && state.captureId) {
    clearCaptureLayers();
    state.capture = state.captureId = null;
  }
  state.view = view;
  if (map.getCanvas().style.cursor !== "crosshair") map.getCanvas().style.cursor = "";
  syncMap();
}

function route() {
  const [, view, id] = (location.hash.slice(1) || "/").split("/");
  if (view === "new") {
    show("new");
    renderArea();
    estimate();
    if (!state.area) setDrawing(true);
  } else if (view === "preview") {
    if (!state.preview && !running("plan")) {
      location.replace("#/new");
      return;
    }
    show("preview");
    renderPreview();
    if (state.preview) showPlan();
  } else if (view === "capture" && id) {
    const captureId = decodeURIComponent(id);
    if (state.captureId !== captureId) {
      show("capture");
      clearCaptureLayers();
      state.captureId = captureId;
      state.capture = null;
      $("#capture-title").textContent = "";
      $("#capture-meta").textContent = "";
      $("#layers").innerHTML = "";
      $("#files").innerHTML = "";
      $("#capture-error").hidden = true;
      $("#progress").hidden = true;
    }
    openCapture(captureId);
  } else {
    show("library");
    loadCaptures();
  }
}

// Events

function bind() {
  $("#settings").addEventListener("input", () => {
    updateForm();
    estimate();
  });
  $("#settings").addEventListener("submit", previewPlan);
  $("#redraw").addEventListener("click", () => setDrawing(!state.drawing));

  $("#preview-back").addEventListener("click", async () => {
    if (running("plan")) await api("/api/stop", {}).catch(() => null);
    location.hash = "#/new";
  });
  $("#cancel-plan").addEventListener("click", async () => {
    state.job = await api("/api/stop", {}).catch(() => state.job);
    renderPreview();
  });
  $("#start").addEventListener("click", async () => {
    $("#start").disabled = true;
    try {
      state.job = await api("/api/capture", {plan: state.preview.plan});
      state.preview = null;
      location.hash = `#/capture/${encodeURIComponent(state.job.capture)}`;
      poll();
    } catch (error) {
      state.planError = error.message;
      renderPreview();
    } finally {
      $("#start").disabled = false;
    }
  });

  $("#captures").addEventListener("click", event => {
    const button = event.target.closest(".capture[data-id]");
    if (button) location.hash = `#/capture/${encodeURIComponent(button.dataset.id)}`;
  });
  $("#captures").addEventListener("mouseover", event => hoverCapture(event.target.closest(".capture[data-id]")?.dataset.id ?? null));
  $("#captures").addEventListener("mouseleave", () => hoverCapture(null));

  $("#layers").addEventListener("change", event => {
    const key = event.target.dataset.toggle;
    if (!key) return;
    state.toggles[key] = event.target.checked;
    store.set("toggles", state.toggles);
    if (key === "photos" && !event.target.checked) closeViewer();
    syncMap();
  });
  $("#stop").addEventListener("click", async () => {
    state.job = await api("/api/stop", {}).catch(() => state.job);
    renderCapture();
  });
  $("#resume").addEventListener("click", async () => {
    try {
      state.job = await api(capturePath(state.capture.id, "/resume"), {});
      renderCapture();
      poll();
    } catch (error) {
      $("#capture-error").textContent = error.message;
      $("#capture-error").hidden = false;
    }
  });
  $("#copy-folder").addEventListener("click", async event => {
    try {
      await navigator.clipboard.writeText(state.capture.folder);
      event.target.textContent = "Copied";
      setTimeout(() => { event.target.textContent = "Copy path"; }, 1500);
    } catch { /* Clipboard access can be refused. */ }
  });

  $("#viewer-prev").addEventListener("click", () => showPhoto(state.selected - 1));
  $("#viewer-next").addEventListener("click", () => showPhoto(state.selected + 1));
  $("#viewer-close").addEventListener("click", closeViewer);

  const search = $("#search"), results = $("#search-results");
  search.addEventListener("submit", async event => {
    event.preventDefault();
    const query = search.q.value.trim();
    if (!query) return;
    results.hidden = false;
    results.innerHTML = `<li class="note">Searching…</li>`;
    try {
      state.search = await api(`/api/search?q=${encodeURIComponent(query)}`);
      results.innerHTML = state.search.map((place, i) => `<li><button type="button" data-index="${i}">${esc(place.name)}
        <span class="detail">${esc(place.label.split(", ").slice(1).join(", "))}</span></button></li>`).join("")
        || `<li class="note">No places found.</li>`;
    } catch (error) {
      results.innerHTML = `<li class="note">${esc(error.message)}</li>`;
    }
  });
  results.addEventListener("click", event => {
    const place = state.search[event.target.closest("[data-index]")?.dataset.index];
    if (!place) return;
    results.hidden = true;
    map.flyTo({center: [place.lon, place.lat], zoom: 16});
  });
  search.q.addEventListener("input", () => { if (!search.q.value) results.hidden = true; });

  document.addEventListener("keydown", event => {
    if (event.target.matches("input, select, textarea") && event.key !== "Escape") return;
    if (event.key === "Escape") {
      if (state.drawing && !state.drag) setDrawing(false);
      else if (!results.hidden) results.hidden = true;
      else if (!$("#viewer").hidden) closeViewer();
    } else if (!$("#viewer").hidden && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      showPhoto(state.selected + (event.key === "ArrowLeft" ? -1 : 1));
    }
  });
  window.addEventListener("hashchange", route);
  window.addEventListener("mouseup", endDrag);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
}

function bindMap() {
  map.on("mousedown", startDrag);
  map.on("mousemove", moveDrag);
  map.on("mouseup", endDrag);
  map.on("click", "photos", event => {
    const sequence = event.features[0].properties.sequence;
    showPhoto(state.photos.findIndex(photo => photo.properties.sequence === sequence));
  });
  map.on("click", "outlines-fill", event => {
    location.hash = `#/capture/${encodeURIComponent(event.features[0].properties.id)}`;
  });
  map.on("mousemove", "outlines-fill", event => hoverCapture(event.features[0].properties.id));
  map.on("mouseleave", "outlines-fill", () => hoverCapture(null));
  for (const layer of ["photos", "outlines-fill"]) {
    map.on("mouseenter", layer, () => { if (!state.drawing) map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", layer, () => { if (!state.drawing) map.getCanvas().style.cursor = ""; });
  }
  map.on("moveend", () => store.set("view", {center: map.getCenter().toArray(), zoom: map.getZoom()}));
  map.on("error", event => {
    // Tiles not saved yet are expected while a capture runs.
    if (event.sourceId === "satellite" || event.sourceId === "terrain") return;
    console.warn(event.error);
  });
}

async function init() {
  state.config = await api("/api/config");
  $("#root").textContent = state.config.root;
  $("#root").title = state.config.root;
  for (const slot of $$("[data-icon]")) slot.outerHTML = icon(slot.dataset.icon);
  fillForm({...state.config.defaults, ...store.get("options", {})});

  const view = store.get("view", {center: [12.49, 41.89], zoom: 2});
  map = new maplibregl.Map({container: "map", style: await basemap(), center: view.center, zoom: view.zoom,
                            attributionControl: {compact: true}, dragRotate: false, pitchWithRotate: false});
  map.touchZoomRotate.disableRotation();
  map.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
  map.addControl(new maplibregl.ScaleControl({unit: "metric"}), "bottom-left");
  await map.once("load");

  map.addImage("corner", cornerImage(), {pixelRatio: 2});
  for (const source of ["osm", "bounds", "plan", "photos", "selected", "area", "corners"]) {
    map.addSource(source, {type: "geojson", data: EMPTY});
  }
  map.addSource("outlines", {type: "geojson", data: EMPTY, promoteId: "id"});
  for (const layer of OVERLAYS) addLayer(layer);
  bindMap();
  bind();
  setArea(state.area);

  state.job = await api("/api/job").catch(() => null);
  if (state.job?.kind === "plan" && state.job.state === "done") state.preview = await api("/api/preview").catch(() => null);
  if (running("plan") && !location.hash.startsWith("#/preview")) location.hash = "#/preview";
  if (running("capture") && !location.hash.startsWith("#/capture")) location.hash = `#/capture/${encodeURIComponent(state.job.capture)}`;
  route();
  if (state.job?.state === "running") poll();
  if (!store.get("view", null) && state.view === "library") {
    await loadCaptures();
    const bounds = state.captures.filter(c => c.bounds).map(c => c.bounds);
    if (bounds.length) {
      map.fitBounds(lngLatBounds([Math.min(...bounds.map(b => b[0])), Math.min(...bounds.map(b => b[1])),
                                  Math.max(...bounds.map(b => b[2])), Math.max(...bounds.map(b => b[3]))]),
                    {padding: 80, maxZoom: 16, duration: 0});
    }
  }
}

init().catch(error => {
  document.body.insertAdjacentHTML("afterbegin", `<p class="error fatal">Aleph could not start: ${esc(error.message)}</p>`);
});
