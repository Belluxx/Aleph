"""Local dashboard: draw an area, preview its plan, capture it, and explore the results."""

import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import capture, layers, places
from .common import CachedClient, Client, RequestError, contained, now
from .geo import bounds, collection, coordinate, feature, grid

WEB = Path(__file__).resolve().parent / "web"
PHOTO_KEYS = ("sequence", "filename", "heading", "side", "imagery_date", "pano_id", "path_name", "streetview_url")
OUTPUTS = ("satellite.tif", "satellite.png", "terrain.tif", "map.osm", "mesh.glb")


class Stopped(KeyboardInterrupt):
    """Raised from a job's requests; capture.download then saves the run as stopped."""


class Busy(Exception):
    pass


class Job:
    """One planning or capture task at a time, polled by the browser."""

    count = 0

    def __init__(self, kind, work, folder=None, run=None):
        Job.count += 1
        self.id, self.kind, self.folder, self.run = Job.count, kind, folder, run
        self.state, self.error, self.preview = "running", None, None
        self.phase, self.done, self.total, self.unit = "Starting", 0, None, None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.work, args=(work,), daemon=True)
        self.thread.start()

    def check(self):
        if self.stop.is_set():
            raise Stopped

    def progress(self, phase, done=0, total=None, unit=None):
        # Captures only stop between requests, so their partial exports can still finish.
        if self.kind == "plan":
            self.check()
        self.phase, self.done, self.total, self.unit = phase, done, total, unit

    def work(self, work):
        try:
            work(self)
            self.state = "done"
        except KeyboardInterrupt:
            self.state = "stopped"
        except Exception as error:
            self.state, self.error = "failed", str(error) or type(error).__name__
            print(f"alephgeo dashboard: {self.error}", file=sys.stderr)


def imagery(area, stage, tile_size):
    g = stage["grid"]
    return dict(tiles=g["rows"] * g["columns"], zoom=g["zoom"], width=g["width"], height=g["height"],
                meters_per_pixel=capture.resolution(area, g["zoom"], tile_size))


def grid_lines(g, kind):
    """Tile boundaries, or just the outline of large grids."""
    x0, y0, x1, y1 = g["x0"], g["y0"], g["x0"] + g["columns"], g["y0"] + g["rows"]

    def point(x, y):
        lat, lon = coordinate(x * 256, y * 256, g["zoom"])
        return [lon, lat]

    xs = range(x0, x1 + 1) if g["columns"] <= 64 else (x0, x1)
    ys = range(y0, y1 + 1) if g["rows"] <= 64 else (y0, y1)
    lines = [[point(x, y0), point(x, y1)] for x in xs] + [[point(x0, y), point(x1, y)] for y in ys]
    return feature("MultiLineString", lines, dict(kind=kind))


def plan_layers(run):
    """Planned street paths, stops, and spacing gaps, and the imagery tile grids."""
    features = []
    for stage in run["stages"]:
        if stage["mode"] == "streetview":
            features += [feature("LineString", [[p[1], p[0]] for p in part["points"]], dict(kind="path"))
                         for part in stage["paths"]]
            features += [feature("LineString", [[gap["start"][1], gap["start"][0]], [gap["end"][1], gap["end"][0]]],
                                 dict(kind="gap")) for gap in stage["coverage"]["gaps"]]
            features += [feature("Point", [stop["lon"], stop["lat"]], dict(kind="stop")) for stop in stage["samples"]]
        elif "grid" in stage:
            features.append(grid_lines(stage["grid"], stage["mode"]))
    return collection(features)


def preview(run):
    area, stages = run["bounds"], {}
    for stage in run["stages"]:
        mode = stage["mode"]
        if mode == "streetview":
            stages[mode] = dict(stops=len(stage["samples"]), photos=capture.total(stage), sphere=stage["full_sphere"],
                                panoramas=stage["coverage"]["available"], gaps=len(stage["coverage"]["gaps"]),
                                meters=stage["length"])
        elif mode in ("satellite", "terrain"):
            stages[mode] = imagery(area, stage, 512 if mode == "terrain" else 256)
        elif mode == "mesh":
            stages[mode] = dict(nodes=len(stage["nodes"]), level=stage["level"])
        else:
            stages[mode] = {}
    return dict(bounds=area, options=run["options"], seconds=capture.estimate(run)["seconds"], stages=stages,
                layers=plan_layers(run))


def summary(identity, run, live):
    stages = {}
    for stage in run["stages"]:
        results = stage["results"]
        count = len(results)
        stages[stage["mode"]] = info = dict(done=count, total=capture.total(stage))
        if stage["mode"] in ("streetview", "mesh"):
            info["skipped"] = sum(result.get("status") == "skipped" for result in results[:count])
        if "grid" in stage:
            info.update(zoom=stage["grid"]["zoom"], minzoom=layers.lowest(stage))
    state = "interrupted" if run["state"] == "running" and not live else run["state"]
    return dict(id=identity, bounds=run["bounds"], started_at=run["started_at"], finished_at=run.get("finished_at"),
                state=state, error=run.get("error"), options=run["options"], stages=stages)


def area_and_options(body):
    try:
        area = bounds([float(value) for value in body["bounds"]])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Draw a valid rectangle first. {error}".strip()) from error
    return area, capture.settings(body.get("options"))


class Dashboard:
    def __init__(self, root, cache_dir, refresh):
        self.root = Path(root).expanduser().resolve()
        self.cache_dir, self.refresh = cache_dir, refresh
        self.lock = threading.RLock()
        self.job = None
        self.runs = {}
        self.osm = {}
        self.tiles = layers.Tiles()

    # Jobs

    def start(self, kind, work, folder=None, run=None):
        with self.lock:
            if self.job and self.job.state == "running":
                raise Busy("Another task is running. Stop it first.")
            self.job = Job(kind, work, folder, run)
            return self.job_state()

    def client(self, job, delay):
        return Client(delay, cancel=job.check, cache_dir=self.cache_dir, refresh=self.refresh)

    def plan(self, body):
        area, options = area_and_options(body)

        def work(job):
            job.run = capture.plan(self.client(job, options["delay"]), area, options, job.progress)
            job.preview = preview(job.run)

        return self.start("plan", work)

    def capture(self, body):
        with self.lock:
            job = self.job
            if not job or job.kind != "plan" or job.state != "done" or job.id != body.get("plan"):
                raise ValueError("This preview is out of date. Preview the capture again.")
            run = job.run
            run["started_at"] = now()
            folder = capture.create_folder(run, self.root)
            capture.save_preview(run, folder)
            return self.download(folder, run)

    def resume(self, identity):
        folder = self.folder(identity)
        with self.lock:
            if self.job and self.job.state == "running":
                raise Busy("Another task is running. Stop it first.")
            return self.download(folder, capture.load(folder))

    def download(self, folder, run):
        def work(job):
            capture.download(run, folder, self.client(job, run["options"]["delay"]), job.progress)

        return self.start("capture", work, folder, run)

    def stop(self):
        if self.job and self.job.state == "running":
            self.job.stop.set()
        return self.job_state()

    def job_state(self):
        job = self.job
        if job is None:
            return None
        state = dict(id=job.id, kind=job.kind, state=job.state, phase=job.phase, done=job.done, total=job.total,
                     unit=job.unit, error=job.error, stopping=job.stop.is_set())
        if job.kind == "capture":
            state["capture"] = job.folder.name
            state["seconds"] = capture.estimate(job.run)["seconds"]
            state["stages"] = {stage["mode"]: dict(done=len(stage["results"]), total=capture.total(stage))
                               for stage in job.run["stages"]}
        return state

    # Captures

    def folder(self, identity):
        if "/" in identity or identity.startswith("."):
            raise FileNotFoundError("Capture not found.")
        folder = contained(self.root, identity)
        if not (folder / "manifest.json").is_file():
            raise FileNotFoundError("Capture not found.")
        return folder

    def load(self, identity):
        """The live run while it downloads, otherwise its manifest, reloaded when it changes."""
        folder = self.folder(identity)
        job = self.job
        if job and job.kind == "capture" and job.folder == folder and job.state == "running":
            return folder, job.run, True
        stat = (folder / "manifest.json").stat()
        signature = stat.st_mtime_ns, stat.st_size
        cached = self.runs.get(identity)
        if not cached or cached[0] != signature:
            cached = self.runs[identity] = signature, capture.load(folder, check_files=False)
        return folder, cached[1], False

    def stage(self, identity, mode):
        folder, run, live = self.load(identity)
        stage = next((stage for stage in run["stages"] if stage["mode"] == mode), None)
        if stage is None:
            raise FileNotFoundError(f"This capture has no {mode} data.")
        return folder, stage, live

    def captures(self):
        items = []
        for folder in self.root.iterdir() if self.root.is_dir() else ():
            if folder.is_dir() and (folder / "manifest.json").is_file():
                try:
                    _, run, live = self.load(folder.name)
                    items.append(summary(folder.name, run, live))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    items.append(dict(id=folder.name, state="unreadable", error=str(error)))
        return sorted(items, key=lambda item: item.get("started_at", ""), reverse=True)

    def detail(self, identity):
        folder, run, live = self.load(identity)
        files = {name: (folder / name).stat().st_size for name in OUTPUTS if (folder / name).is_file()}
        return dict(summary(identity, run, live), folder=str(folder), files=files)

    def photos(self, identity, after):
        try:
            _, stage, _ = self.stage(identity, "streetview")
        except FileNotFoundError:
            return dict(count=0, features=[])
        results = stage["results"]
        count = len(results)
        return dict(count=count, features=[
            feature("Point", [photo["lon"], photo["lat"]], {key: photo.get(key) for key in PHOTO_KEYS})
            for photo in results[max(0, after):count] if photo["status"] == "saved"])

    def osm_layer(self, identity):
        folder = self.folder(identity)
        path = folder / "map.osm"
        if not path.is_file():
            raise FileNotFoundError("This capture has no saved OSM map yet.")
        stat = path.stat()
        signature = stat.st_mtime_ns, stat.st_size
        cached = self.osm.get(identity)
        if not cached or cached[0] != signature:
            data = json.dumps(layers.osm(path), separators=(",", ":")).encode()
            cached = self.osm[identity] = signature, data
        return cached[1], "application/geo+json", 0

    def tile(self, identity, mode, z, x, y):
        folder, stage, _ = self.stage(identity, mode)
        z = int(z)
        tile = self.tiles.get(folder, stage, z, int(x), int(y.split(".")[0]))
        if tile is None:
            raise FileNotFoundError("No saved tile here.")
        # Saved patches never change. Built tiles are not cached, so rendering changes show up at once.
        return *tile, 86400 if mode == "satellite" and z == stage["grid"]["zoom"] else 0

    def file(self, identity, path):
        target = contained(self.folder(identity), "/".join(path))
        if not target.is_file():
            raise FileNotFoundError("File not found.")
        return target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream", 86400

    # Routes

    def handle(self, method, parts, query, body):
        match method, parts:
            case "GET", []:
                return (WEB / "index.html").read_bytes(), "text/html; charset=utf-8", 0
            case "GET", [("app.js" | "style.css") as name]:
                return (WEB / name).read_bytes(), mimetypes.guess_type(name)[0], 0
            case "GET", ["api", "config"]:
                return dict(root=str(self.root), defaults=capture.DEFAULT_OPTIONS)
            case "GET", ["api", "job"]:
                return self.job_state()
            case "GET", ["api", "preview"]:
                job = self.job
                if not job or job.kind != "plan" or job.preview is None:
                    raise FileNotFoundError("No preview is ready.")
                return dict(job.preview, plan=job.id)
            case "GET", ["api", "search"]:
                client = CachedClient(cache_dir=self.cache_dir, refresh=self.refresh)
                return places.geocode(client, query=query.get("q", [""])[0], limit=6)
            case "GET", ["api", "captures"]:
                return self.captures()
            case "GET", ["api", "captures", identity]:
                return self.detail(identity)
            case "GET", ["api", "captures", identity, "plan"]:
                _, run, _ = self.load(identity)
                return plan_layers(run)
            case "GET", ["api", "captures", identity, "photos"]:
                return self.photos(identity, int(query.get("after", ["0"])[0]))
            case "GET", ["api", "captures", identity, "osm"]:
                return self.osm_layer(identity)
            case "GET", ["api", "captures", identity, "satellite", z, x, y]:
                return self.tile(identity, "satellite", z, x, y)
            case "GET", ["api", "captures", identity, "hillshade", z, x, y]:
                return self.tile(identity, "terrain", z, x, y)
            case "GET", ["api", "captures", identity, "files", *path] if path:
                return self.file(identity, path)
            case "POST", ["api", "estimate"]:
                area, options = area_and_options(body)
                result = {}
                for mode in ("satellite", "terrain"):
                    if mode in options["include"]:
                        stage = dict(mode=mode, grid=grid(area, options[f"{mode}_zoom"]), results=[])
                        result[mode] = dict(imagery(area, stage, 512 if mode == "terrain" else 256),
                                            seconds=capture.estimate(dict(stages=[stage], options=options))["seconds"])
                result["planning"] = capture.planning(area, options)
                return result
            case "POST", ["api", "plan"]:
                return self.plan(body)
            case "POST", ["api", "capture"]:
                return self.capture(body)
            case "POST", ["api", "stop"]:
                return self.stop()
            case "POST", ["api", "captures", identity, "resume"]:
                return self.resume(identity)
        raise FileNotFoundError("Not found.")


class Handler(BaseHTTPRequestHandler):
    server_version = "Aleph"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def send(self, status, data, content_type, max_age=0):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", f"max-age={max_age}" if max_age else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def dispatch(self, method):
        app = self.server.app
        # Refuse other sites reaching this server through DNS rebinding.
        if self.headers.get("Host") not in self.server.hosts:
            return self.send(403, b"Forbidden", "text/plain")
        address = urlsplit(self.path)
        parts = [unquote(part) for part in address.path.split("/") if part]
        try:
            body = {}
            if method == "POST":
                # JSON bodies need a CORS preflight, which this server never grants.
                if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                    raise ValueError("Send JSON.")
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("Send a JSON object.")
            result = app.handle(method, parts, parse_qs(address.query), body)
            if isinstance(result, tuple):
                return self.send(200, *result)
            self.send(200, json.dumps(result, separators=(",", ":")).encode(), "application/json")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            status = (404 if isinstance(error, FileNotFoundError) else 409 if isinstance(error, Busy)
                      else 502 if isinstance(error, RequestError) else 400 if isinstance(error, ValueError) else 500)
            if status == 500:
                print(f"alephgeo dashboard: {method} {address.path}: {error!r}", file=sys.stderr)
            message = str(error) or type(error).__name__
            try:
                self.send(status, json.dumps(dict(error=message)).encode(), "application/json")
            except OSError:
                pass


def serve(root, port, *, cache_dir, refresh=False, open_browser=True):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.app = app = Dashboard(root, cache_dir, refresh)
    port = server.server_address[1]
    server.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    address = f"http://127.0.0.1:{port}/"
    print(f"Aleph dashboard at {address}\nCaptures in {app.root}\nPress Ctrl-C to quit.", file=sys.stderr)
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        job = app.job
        if job and job.state == "running":
            print("\nStopping the running task. Press Ctrl-C again to quit now.", file=sys.stderr)
            job.stop.set()
            try:
                job.thread.join()
            except KeyboardInterrupt:
                return 130
    finally:
        server.server_close()
    return 0
