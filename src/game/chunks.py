"""S7: split the play area into square chunks, move geometry into chunk-local coordinates, and sample ground heights."""

from dataclasses import dataclass, replace

import numpy as np

from ..common import write_bytes
from .load import inside
from .types import GROUND_CELL


@dataclass
class Chunk:
    i: int  # Column along +x.
    j: int  # Row along +z.
    rect: tuple  # (x0, z0, x1, z1)

    @property
    def id(self):
        return f"{self.i}_{self.j}"

    @property
    def center(self):
        x0, z0, x1, z1 = self.rect
        return (x0 + x1) / 2, (z0 + z1) / 2


def grid(play, size):
    """Chunks of size metres about centred on the play area's bounding box, without those entirely outside the polygon."""
    low, high = play.min(axis=0), play.max(axis=0)
    counts = np.maximum(1, np.ceil((high - low) / size - 1e-9)).astype(int)
    # Whole-metre corners keep heightfield nodes on the ground grid; one more chunk covers what rounding uncovers.
    x0, z0 = np.round((low + high) / 2 - counts * size / 2)
    counts = np.maximum(counts, np.ceil((high - (x0, z0)) / size - 1e-9).astype(int))
    chunks = [Chunk(i, j, (x0 + i * size, z0 + j * size, x0 + (i + 1) * size, z0 + (j + 1) * size))
              for j in range(counts[1]) for i in range(counts[0])]
    return [chunk for chunk in chunks if touches(play, chunk.rect)]


def touches(polygon, rect):
    """Whether a polygon and a rectangle overlap with positive area or along a border."""
    x0, z0, x1, z1 = rect
    corners = np.array([(x0, z0), (x1, z0), (x1, z1), (x0, z1)])
    if inside(polygon, *corners.T).any():
        return True
    px, pz = polygon.T
    if ((px >= x0) & (px <= x1) & (pz >= z0) & (pz <= z1)).any():
        return True
    # Remaining overlaps need a polygon edge crossing a rectangle edge.
    a, b = polygon, np.roll(polygon, -1, axis=0)
    for c, d in zip(corners, np.roll(corners, -1, axis=0)):
        if crossing(a, b, c, d).any():
            return True
    return False


def crossing(a, b, c, d):
    """Whether segments a–b (arrays) properly cross segment c–d."""
    def side(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))
    return (side(a, b, c) * side(a, b, d) < 0) & (side(c, d, a) * side(c, d, b) < 0)


def assign(chunks, points):
    """Index into chunks for each (x, z) point: the chunk holding it, else the nearest one."""
    points = np.asarray(points, float).reshape(-1, 2)
    if not chunks or not len(points):
        return np.zeros(len(points), int)
    rects = np.array([chunk.rect for chunk in chunks])
    # Distance to each rectangle, zero inside; ties go to the first chunk, so shared borders are stable.
    dx = np.maximum(np.maximum(rects[None, :, 0] - points[:, None, 0], points[:, None, 0] - rects[None, :, 2]), 0)
    dz = np.maximum(np.maximum(rects[None, :, 1] - points[:, None, 1], points[:, None, 1] - rects[None, :, 3]), 0)
    return np.argmin(dx * dx + dz * dz, axis=1)


def shift(part, center):
    """A copy of a Part with positions relative to (x, 0, z)."""
    offset = np.array([center[0], 0.0, center[1]], np.float32)
    return replace(part, positions=(np.asarray(part.positions, np.float64) - offset).astype(np.float32))


def shift_collider(collider, center):
    offset = np.array([center[0], 0.0, center[1]])
    params = dict(collider.params)
    if "center" in params:
        params["center"] = [float(v) for v in np.asarray(params["center"], float) - offset]
    vertices = None if collider.vertices is None else (np.asarray(collider.vertices, float) - offset).astype(np.float32)
    return replace(collider, vertices=vertices, params=params)


def heights(height, rect, spacing=GROUND_CELL):
    """Ground heights at grid nodes covering rect inclusive, as (rows, cols) float32 with rows along +z.

    Nodes outside the measured ground take the nearest edge value; missing values take the chunk's mean, so the
    heightfield collider is always finite.
    """
    x0, z0, x1, z1 = rect
    cols, rows = int(round((x1 - x0) / spacing)) + 1, int(round((z1 - z0) / spacing)) + 1
    x, z = np.meshgrid(x0 + np.arange(cols) * spacing, z0 + np.arange(rows) * spacing)
    values = np.asarray(height.sample(x, z), np.float64)
    if not np.isfinite(values).all():
        known = np.isfinite(values)
        values[~known] = values[known].mean() if known.any() else 0.0
    return values.astype(np.float32)


def write_heights(path, values):
    """Raw little-endian f32, row-major, rows advancing along +z."""
    write_bytes(path, np.ascontiguousarray(values, "<f4").tobytes())
