"""Small shared pieces: paced HTTP, progress, and atomic writes."""

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

OVERPASS_SERVERS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
OVERPASS = OVERPASS_SERVERS[0]
APP_AGENT = "Terrain downloader"
BROWSER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15"


class MissingImagery(Exception):
    pass


class RequestError(Exception):
    """An operation failed with a code and optional details."""

    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def url(base, **params):
    return base + "?" + urlencode(params)


def contained(parent, name):
    """Resolve a relative filename without allowing it to escape its directory."""
    if (not isinstance(name, str) or not name or "\\" in name or "\0" in name
            or Path(name).is_absolute() or ".." in Path(name).parts):
        raise FileNotFoundError("File not found.")
    parent = Path(parent).resolve()
    path = (parent / name).resolve()
    if not path.is_relative_to(parent) or path == parent:
        raise FileNotFoundError("File not found.")
    return path


class Client:
    def __init__(self, delay=0, cancel=None, request_log=None):
        self.delay = delay
        self.cancel = cancel
        self.request_log = request_log
        self.finished = float("-inf")

    def check_cancel(self):
        if self.cancel:
            self.cancel()

    def get(self, address, *, missing_ok=False, data=None, timeout=60, user_agent=BROWSER_AGENT):
        for attempt in range(4):
            backoff = 2 ** (attempt - 1) if attempt else 0
            until = time.monotonic() + max(backoff, self.delay - (time.monotonic() - self.finished))
            self.check_cancel()
            while (remaining := until - time.monotonic()) > 0:
                time.sleep(min(remaining, 0.2) if self.cancel else remaining)
                self.check_cancel()
            if self.request_log:
                self.request_log("POST" if data is not None else "GET", address, attempt)
            try:
                with urlopen(
                    Request(address, data=data, headers={"User-Agent": user_agent}), timeout=timeout
                ) as response:
                    result = response.read()
                self.check_cancel()
                return result
            except Exception as error:
                if self.request_log:
                    self.request_log("POST" if data is not None else "GET", address, attempt, error)
                if isinstance(error, HTTPError):
                    error.close()
                    if missing_ok and error.code == 404:
                        raise MissingImagery("The requested image is not available.") from error
                if attempt < 3:
                    continue
                if isinstance(error, HTTPError):
                    raise OSError(
                        f"{urlsplit(address).hostname}: HTTP {error.code}. Resume to retry."
                    ) from error
                raise
            finally:
                self.finished = time.monotonic()

    def overpass(self, query):
        # POST avoids long query URLs; allow the query budget plus server queue time.
        # Overpass rejects browser UAs.
        data = urlencode({"data": query}).encode()
        for endpoint in OVERPASS_SERVERS:
            try:
                return self.get(endpoint, data=data, timeout=240, user_agent=APP_AGENT)
            except OSError as error:
                cause = error.__cause__
                if isinstance(cause, HTTPError) and cause.code < 500 and cause.code != 429:
                    raise
                failure = error
        raise failure


class CachedClient(Client):
    """Cache successful quick-query responses for one day, including OSM queries."""

    def __init__(self, directory, *, refresh=False):
        super().__init__(delay=1)
        self.directory = Path(directory).expanduser()
        self.refresh = refresh
        self.hits = 0
        self.misses = 0

    def get(self, address, **kwargs):
        payload = kwargs.get("data") or b""
        key = hashlib.sha256(address.encode() + b"\0" + payload).hexdigest()
        path = self.directory / key
        if not self.refresh and path.is_file() and time.time() - path.stat().st_mtime < 86400:
            self.hits += 1
            return path.read_bytes()
        result = super().get(address, **kwargs)
        # Do not retain Overpass's HTTP-200 timeout/error responses.
        if payload:
            try:
                if json.loads(result).get("remark"):
                    return result
            except (ValueError, AttributeError):
                return result
        write_bytes(path, result)
        self.misses += 1
        return result

    def stats(self):
        return dict(hits=self.hits, misses=self.misses)


class Progress:
    def __init__(self):
        self.stream = sys.stderr
        self.terminal = self.stream.isatty()
        self.phase = None
        self.printed = float("-inf")
        self.line = ""
        self.width = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.line:
            print("" if self.terminal else self.line, file=self.stream, flush=True)
            self.line = ""
            self.width = 0

    def __call__(self, phase, done=0, total=None):
        current = time.monotonic()
        if self.phase != phase:
            if not self.terminal:
                self.close()
            self.phase, self.started, self.initial = phase, current, done
            self.printed = float("-inf")
        if self.terminal and current - self.printed < 0.1 and done != total:
            return

        eta = "--:--"
        fraction = 0
        steps = f"{done:,}/?" if total is None else f"{done:,}/{total:,}"
        if total is not None:
            fraction = min(1, done / total) if total else 1
            if done >= total:
                eta = "00:00"
            elif done > self.initial:
                seconds = math.ceil(
                    (current - self.started) / (done - self.initial) * (total - done)
                )
                minutes, seconds = divmod(seconds, 60)
                hours, minutes = divmod(minutes, 60)
                eta = (
                    f"{hours}:{minutes:02d}:{seconds:02d}"
                    if hours
                    else f"{minutes:02d}:{seconds:02d}"
                )

        prefix, suffix = f"{phase} [", f"] {steps}  - {eta}"
        columns = shutil.get_terminal_size().columns - 1
        width = min(24, max(1, columns - len(prefix) - len(suffix)))
        full, part = divmod(int(fraction * width * 8), 8)
        bar = ("█" * full + (" ▏▎▍▌▋▊▉"[part] if part else "")).ljust(width)
        self.line = prefix + bar + suffix
        if self.terminal:
            self.line = self.line[:columns]
            print(
                "\r" + self.line.ljust(min(self.width, columns)),
                end="",
                file=self.stream,
                flush=True,
            )
            self.width = len(self.line)
        self.printed = current


@contextmanager
def atomic_path(path):
    """Leave the previous file intact if writing or replacement fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_bytes(path, data):
    with atomic_path(path) as temporary:
        temporary.write_bytes(data)


def write_json(path, data):
    with atomic_path(path) as temporary, temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
