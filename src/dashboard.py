"""A local, single-user dashboard. Capture folders remain the source of truth."""

import hashlib
import json
import mimetypes
import secrets
import shutil
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from PIL import features

from . import capture
from .common import Client, contained, now
from .dashboard_tiles import Tiles
from .geo import bounds

ASSETS = Path(__file__).resolve().parent / "assets"


def plan_preview(run):
    return dict(bounds=run["bounds"], options=run["options"], estimate=capture.estimate(run))


def log_network_request(method, address, attempt, error=None):
    target = urlsplit(address)
    location = target.netloc + target.path
    if target.query:
        location += f"?{target.query}"
    retry = f" (retry {attempt})" if attempt else ""
    if error is None:
        message = f"[network] {method} {location}{retry}"
    else:
        code = getattr(error, "code", None)
        detail = (
            f"HTTP {code}: {getattr(error, 'reason', error)}"
            if code is not None
            else str(error) or type(error).__name__
        )
        message = f"[network] ERROR {method} {location}{retry} → {detail}"
    print(message, file=sys.stderr, flush=True)


class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}
        self.live = {}
        self.lock = threading.Lock()

    def follow(self, identity, run=None):
        with self.lock:
            if run is None:
                self.live.pop(identity, None)
            else:
                self.live[identity] = run
            self.cache.pop(identity, None)

    def folder(self, identity):
        if "/" in identity or identity.startswith("."):
            raise FileNotFoundError("Capture not found.")
        folder = contained(self.root, identity)
        if not folder.is_dir():
            raise FileNotFoundError("Capture not found.")
        return folder

    def get(self, identity):
        folder = self.folder(identity)
        with self.lock:
            live = self.live.get(identity)
            if live is not None:
                counts = tuple(len(stage["results"]) for stage in live["stages"])
                signature = ("live", counts, live["state"], live.get("exports_saved"), live.get("error"))
            else:
                stat = contained(folder, "manifest.json").stat()
                signature = (stat.st_mtime_ns, stat.st_size)
            entry = self.cache.get(identity)
            if entry and entry[0] == signature:
                return folder, entry[1], entry[2]
            # Results are appended only after their files are saved, and never mutated.
            # Copy their lists on demand, not on every download or tile request.
            run = (dict(live, stages=[dict(stage, results=stage["results"][:count])
                                     for stage, count in zip(live["stages"], counts)])
                   if live is not None else capture.load(folder, check_files=False))
            revision = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
            self.cache[identity] = signature, run, revision
            return folder, run, revision

    def summary(self, identity):
        folder, run, revision = self.get(identity)
        layers = {}
        for stage in run["stages"]:
            mode = stage["mode"]
            layers[mode] = dict(done=len(stage["results"]), total=capture.total(stage))
            if "grid" in stage:
                layers[mode]["zoom"] = stage["grid"]["zoom"]
        terrain_path = contained(folder, "terrain.tif")
        return dict(
            id=identity, bounds=run["bounds"], started_at=run["started_at"],
            state=run["state"], error=run.get("error"), revision=revision,
            exports_saved=run.get("exports_saved", False), options=run["options"], layers=layers,
            osm=contained(folder, "map.osm").is_file(),
            terrain=terrain_path.is_file() and features.check("libtiff"),
        )

    def list(self):
        runs, errors = [], []
        for folder in self.root.iterdir():
            if not folder.is_dir() or not (folder / "manifest.json").is_file():
                continue
            try:
                runs.append(self.summary(folder.name))
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                errors.append(dict(id=folder.name, error=str(error)))
        runs.sort(key=lambda run: run["started_at"], reverse=True)
        return dict(captures=runs, errors=errors)

    def photos(self, identity, after=0):
        if after < 0:
            raise ValueError("Photo offset must not be negative.")
        _, run, _ = self.get(identity)
        count = sum(len(stage["results"]) for stage in run["stages"] if stage["mode"] == "streetview")
        return dict(capture.photos(run, after), count=count)

    def file(self, identity, name):
        folder, run, _ = self.get(identity)
        allowed = {"map.osm", "terrain.tif", "satellite.png", "manifest.json", "README.txt",
                   "streetview/photos.geojson", "streetview/paths.geojson", "streetview/plan.svg",
                   "satellite/patches.geojson", "satellite/plan.svg"}
        allowed.update(item["filename"] for stage in run["stages"] for item in stage["results"]
                       if "filename" in item)
        if name not in allowed:
            raise FileNotFoundError("File not found.")
        return contained(folder, name)


class BusyError(Exception):
    pass


class Jobs:
    def __init__(self, catalog):
        self.catalog = catalog
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.interrupted = False
        self.plan = None
        self.plan_id = None
        self.status = dict(active=False, state="idle", phase="Ready", run_id=None)

    def snapshot(self):
        with self.lock:
            return dict(self.status)

    def preview(self, identity):
        with self.lock:
            if identity != self.plan_id or self.plan is None:
                raise ValueError("This plan is no longer available. Plan the capture again.")
            run = self.plan
        return dict(plan_preview(run), plan_id=identity)

    def update(self, **values):
        with self.lock:
            self.status.update(values)

    def cancel(self):
        # Raise once: the capture engine then saves checkpoints and exports.
        if self.stop_event.is_set() and not self.interrupted:
            self.interrupted = True
            self.update(state="stopping", phase="Stopping planning" if self.status.get("kind") == "plan" else "Saving progress")
            raise KeyboardInterrupt()

    def progress(self, phase, done=0, total=None):
        self.cancel()
        self.update(phase=phase, done=done, total=total)

    def start(self, *, area=None, options=None, identity=None, plan_only=False, export_only=False,
              plan_id=None):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise BusyError("A capture is already running. Stop it before starting another.")
            run = None
            if plan_id is not None:
                if plan_id != self.plan_id or self.plan is None:
                    raise ValueError("This plan is no longer available. Plan the capture again.")
                run, self.plan = self.plan, None
            elif plan_only:
                self.plan = None
                self.plan_id = secrets.token_urlsafe(16)
            elif identity is None:
                raise ValueError("Plan a capture before starting it.")
            self.stop_event.clear()
            self.interrupted = False
            self.status = dict(active=True, state="running", done=0, total=None, run_id=identity, error=None,
                               phase="Planning capture" if plan_only else "Starting",
                               kind="plan" if plan_only else "capture",
                               plan_id=self.plan_id if plan_only else None)
            self.thread = threading.Thread(
                target=self._work, args=(area, options, identity, plan_only, export_only, run),
                name="aleph-capture", daemon=False,
            )
            self.thread.start()
            return dict(self.status)

    def _work(self, area, options, identity, plan_only, export_only, run=None):
        folder = None
        try:
            if identity:
                folder = self.catalog.folder(identity)
                run = capture.load(folder)
            options = run["options"] if run is not None else options
            client = Client(options["delay"], cancel=self.cancel, request_log=log_network_request)
            if plan_only:
                run = capture.plan(client, area, options, self.progress)
                self.cancel()
                with self.lock:
                    self.plan = run
                self.update(state="planned", phase="Plan ready")
                return
            if folder is None:
                run.update(started_at=now(), state="running")
                folder = capture.create_folder(run, self.catalog.root)
                self.update(run_id=folder.name)
            self.catalog.follow(folder.name, run)
            capture.save_preview(run, folder)
            if export_only:
                capture.export(run, folder, self.progress)
            else:
                capture.download(run, folder, client, self.progress)
            self.update(state="complete", phase="Exports rebuilt" if export_only else "Finished")
        except KeyboardInterrupt:
            self.update(state="stopped", phase="Stopped")
        except Exception as error:
            self.update(state="failed", phase="Planning failed" if plan_only else "Capture failed", error=str(error))
        finally:
            if folder is not None:
                self.catalog.follow(folder.name)
            self.update(active=False)

    def stop(self):
        with self.lock:
            if self.status["active"]:
                self.stop_event.set()
                self.status.update(state="stopping", phase="Stopping after the current request")
            return dict(self.status)

    def close(self):
        self.stop()
        if self.thread:
            self.thread.join()


class Server(ThreadingHTTPServer):
    # Finish file responses before the temporary tile cache is removed.
    daemon_threads = False

    def __init__(self, root, port, cache):
        self.catalog = Catalog(root)
        self.jobs = Jobs(self.catalog)
        self.tiles = Tiles(cache)
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), Handler)
        port = self.server_address[1]
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.origins = {f"http://{host}" for host in self.hosts}


class Handler(BaseHTTPRequestHandler):
    server_version = "Aleph"

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *_):
        pass

    def headers_for(self, status, content_type, size, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https://tiles.openfreemap.org; connect-src 'self' https://tiles.openfreemap.org; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()

    def json(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.headers_for(status, "application/json; charset=utf-8", len(data))
        if self.command != "HEAD":
            self.wfile.write(data)

    def file(self, path, cache="no-cache"):
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(0)
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.headers_for(200, mime, size, cache)
            if self.command != "HEAD":
                shutil.copyfileobj(stream, self.wfile)

    def dispatch(self):
        if self.headers.get("Host") not in self.server.hosts:
            return self.json(dict(error="Use the dashboard's local URL."), 403)
        path = unquote(urlsplit(self.path).path)
        parts = path.strip("/").split("/")
        try:
            if self.command == "POST":
                if (not secrets.compare_digest(self.headers.get("X-Aleph-Token", ""), self.server.token)
                        or self.headers.get("Origin", "") not in self.server.origins | {""}):
                    return self.json(dict(error="Reload the dashboard before continuing."), 403)
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16384 or self.headers.get_content_type() != "application/json":
                    return self.json(dict(error="Expected a small JSON request."), 400)
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")
                if path == "/api/plans":
                    area = data.get("bounds")
                    if (not isinstance(area, list) or len(area) != 4
                            or any(type(n) not in (int, float) for n in area)):
                        raise ValueError("Enter four numeric latitude/longitude bounds.")
                    return self.json(self.server.jobs.start(
                        area=bounds(area), options=capture.settings(data.get("options")),
                        plan_only=True), 202)
                if path == "/api/captures":
                    if not isinstance(data.get("plan_id"), str):
                        raise ValueError("Plan a capture before starting it.")
                    return self.json(self.server.jobs.start(plan_id=data["plan_id"]), 202)
                if path == "/api/job/stop":
                    return self.json(self.server.jobs.stop())
                if len(parts) == 4 and parts[:2] == ["api", "captures"] and parts[3] in ("resume", "export"):
                    self.server.catalog.get(parts[2])
                    return self.json(self.server.jobs.start(identity=parts[2], export_only=parts[3] == "export"), 202)
                raise FileNotFoundError("Unknown action.")
            if path == "/api/config":
                return self.json(dict(token=self.server.token, root=str(self.server.catalog.root),
                                      defaults=capture.DEFAULT_OPTIONS, terrain=features.check("libtiff")))
            if path == "/api/captures":
                return self.json(self.server.catalog.list())
            if path == "/api/job":
                return self.json(self.server.jobs.snapshot())
            if len(parts) == 3 and parts[:2] == ["api", "plans"]:
                return self.json(self.server.jobs.preview(parts[2]))
            if len(parts) >= 3 and parts[:2] == ["api", "captures"]:
                identity = parts[2]
                if len(parts) == 3:
                    return self.json(self.server.catalog.summary(identity))
                if len(parts) == 4 and parts[3] == "plan":
                    _, run, _ = self.server.catalog.get(identity)
                    return self.json(dict(plan_preview(run), id=identity))
                if len(parts) == 4 and parts[3] == "photos":
                    query = parse_qs(urlsplit(self.path).query)
                    return self.json(self.server.catalog.photos(identity, int(query.get("after", ["0"])[0])))
                if len(parts) > 4 and parts[3] == "files":
                    return self.file(self.server.catalog.file(identity, "/".join(parts[4:])))
                if len(parts) == 8 and parts[3] == "tiles" and parts[4] in ("satellite", "terrain"):
                    folder, run, _ = self.server.catalog.get(identity)
                    if not parts[7].endswith(".png"):
                        raise FileNotFoundError("Unknown tile format.")
                    tile, complete = self.server.tiles.get(folder, run, parts[4],
                                                           int(parts[5]), int(parts[6]), int(parts[7][:-4]))
                    # MapLibre refreshes expiring tiles while retaining their textures.
                    return self.file(tile, "max-age=31536000" if complete else "max-age=1")
            if path.startswith("/api/"):
                raise FileNotFoundError("Unknown endpoint.")
            return self.file(contained(ASSETS, "index.html" if path == "/" else path.lstrip("/")))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except FileNotFoundError as error:
            self.json(dict(error=str(error)), 404)
        except BusyError as error:
            self.json(dict(error=str(error)), 409)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.json(dict(error=str(error)), 400)
        except Exception as error:
            self.json(dict(error=str(error)), 500)

    do_GET = dispatch
    do_HEAD = dispatch
    do_POST = dispatch


def serve(root, port=8100, *, open_browser=True):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aleph-dashboard-") as cache:
        server = Server(root, port, Path(cache))
        address = f"http://127.0.0.1:{server.server_address[1]}"
        print(f"Dashboard: {address}\nCaptures: {root}\nPress Ctrl+C to stop.", flush=True)
        if open_browser:
            try:
                webbrowser.open(address)
            except webbrowser.Error:
                pass
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            print("Stopping dashboard; saving capture progress…", file=sys.stderr)
        finally:
            server.jobs.close()
            server.server_close()
