"""Data passed between game world stages: the contract every stage module codes against.

Coordinates: metres in the capture's local frame, as in mesh.glb: x east, y up, z south, centred on the capture.
y = 0 is mesh.glb's lowest point, which lies `Frame.base` metres above sea level. Ground points use (x, z) pairs.

Stages and their owners:
    load.build(folder, config, progress) -> Reference                 S1–S2 rasters and OSM
    ground.build(reference, config, progress) -> Ground                S3 ground, roads, curbs, water
    ground.chunk(ground, rect, lod) -> list[Part]                      ground geometry inside one chunk
    ground.colliders(ground, rect) -> list[Collider]                   curbs and steps inside one chunk
    buildings.build(reference, ground, config, progress) -> list[Model]   S4–S5 buildings and landmarks
    scatter.build(reference, ground, models, config, progress) -> list[Instance]   S6 trees and props
    library: material(name) -> Material, asset(name) -> Asset, style(name) -> dict, install(names, folder)
    chunks / gltf / build: assembling, writing, and scene.json

progress(phase, done=0, total=None, unit=None) reports work, as elsewhere in Aleph.
config is game.toml as a dict (may be empty), plus "quality" ("draft" or "high") and "style".
rect is (x0, z0, x1, z1) with x0 < x1 and z0 < z1.
"""

import math
from dataclasses import dataclass, field

import numpy as np

CELL = 0.25  # Metres per cell of the measured rasters.
GROUND_CELL = 0.5  # Metres per cell of the ground heightfield and surface map.
# Ground surface codes in Ground.surface, which ground.chunk turns into materials.
SURFACES = ("grass", "asphalt", "paving", "dirt", "gravel", "water", "building")


@dataclass
class Frame:
    """The capture's local frame, from mesh.glb's asset extras."""

    latitude: float
    longitude: float
    base: float  # Metres above sea level of y = 0.
    radius: float  # Planet radius the mesh was built on.

    def axes(self):
        la, lo = math.radians(self.latitude), math.radians(self.longitude)
        return ((-math.sin(lo), math.cos(lo), 0.0),
                (-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)))

    def local(self, lat, lon):
        """(x, z) arrays of points given in degrees, as mesh.glb places them."""
        la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
        p = (np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la))
        east, north = self.axes()
        x = self.radius * sum(a * b for a, b in zip(east, p))
        z = -self.radius * sum(a * b for a, b in zip(north, p))
        return x, z


@dataclass
class Raster:
    """values[row, col]; the cell centre is (x0 + (col + 0.5) * cell, z0 + (row + 0.5) * cell). NaN marks no data."""

    values: np.ndarray
    x0: float
    z0: float
    cell: float = CELL

    @property
    def shape(self):
        return self.values.shape

    def centres(self):
        """(x, z) arrays of every cell centre, shaped like values."""
        rows, cols = self.values.shape
        return np.meshgrid(self.x0 + (np.arange(cols) + 0.5) * self.cell, self.z0 + (np.arange(rows) + 0.5) * self.cell)

    def index(self, x, z):
        """Row and column arrays of the cells holding points, clipped to the grid."""
        rows, cols = self.values.shape
        col = np.clip(np.floor((np.asarray(x) - self.x0) / self.cell).astype(int), 0, cols - 1)
        row = np.clip(np.floor((np.asarray(z) - self.z0) / self.cell).astype(int), 0, rows - 1)
        return row, col

    def sample(self, x, z):
        """Bilinear values at points, clamped at the edges."""
        rows, cols = self.values.shape
        fx = np.clip((np.asarray(x, float) - self.x0) / self.cell - 0.5, 0, cols - 1)
        fz = np.clip((np.asarray(z, float) - self.z0) / self.cell - 0.5, 0, rows - 1)
        c0, r0 = np.minimum(fx.astype(int), cols - 2).clip(0), np.minimum(fz.astype(int), rows - 2).clip(0)
        c1, r1 = np.minimum(c0 + 1, cols - 1), np.minimum(r0 + 1, rows - 1)
        tx, tz = fx - c0, fz - r0
        v = self.values
        return ((v[r0, c0] * (1 - tx) + v[r0, c1] * tx) * (1 - tz)
                + (v[r1, c0] * (1 - tx) + v[r1, c1] * tx) * tz)


@dataclass
class Feature:
    """An OSM object in local coordinates."""

    id: str  # "node/1", "way/2", or "relation/3", as on openstreetmap.org.
    kind: str  # "building", "line" (roads, barriers, waterways…), "area" (land use, water, parking…), or "node".
    tags: dict
    points: np.ndarray  # (n, 2) [x, z]; rings are counter-clockwise seen from above and omit the repeated point.
    holes: list = field(default_factory=list)  # Inner rings of areas and buildings, clockwise seen from above.


@dataclass
class Reference:
    """What the capture tells us: produced by load.build."""

    folder: object  # Path of the capture folder.
    frame: Frame
    play: np.ndarray  # (n, 2) [x, z] polygon of the capture area.
    positions: np.ndarray  # (v, 3) float32 mesh.glb vertices, welded; empty without a mesh.
    triangles: np.ndarray  # (t, 3) int32
    top: Raster  # Highest mesh surface per cell (y).
    bottom: Raster  # Lowest mesh surface per cell (y).
    up: Raster  # Fraction of each cell's mesh area facing up (normal y > 0.8).
    osm: list  # [Feature]
    dem: Raster = None  # terrain.tif heights in the same y datum, without a fitted offset; None if missing.


@dataclass
class Ground:
    """Produced by ground.build; shared with buildings and scatter."""

    height: Raster  # Ground y per GROUND_CELL cell, filled everywhere inside the play area.
    surface: Raster  # Integer index into SURFACES per GROUND_CELL cell.
    data: dict = field(default_factory=dict)  # Anything else ground.chunk needs (road meshes, curbs…).


@dataclass
class Part:
    """Triangles sharing one library material.

    uvs are in metres for tiling materials (the writer divides by Material.size) and 0–1 for unit materials,
    with v pointing up, as in OpenGL; on walls u runs along the wall and v up it. The writer converts to glTF.
    colors, when given, are linear RGBA multiplied into the base colour: tints, dirt, and wear.
    """

    material: str
    positions: np.ndarray  # (n, 3) float32
    normals: np.ndarray  # (n, 3) float32
    uvs: np.ndarray  # (n, 2) float32
    indices: np.ndarray  # (m,) uint32, counter-clockwise front faces seen from outside.
    colors: np.ndarray = None  # (n, 4) float32


@dataclass
class Collider:
    """convex: vertices only; trimesh: vertices and indices; capsule and cylinder: params radius and half_height,
    standing on y around params["center"]."""

    shape: str
    vertices: np.ndarray = None
    indices: np.ndarray = None
    params: dict = field(default_factory=dict)


@dataclass
class Model:
    """A building or landmark: never cut by chunks, assigned to the chunk holding center."""

    name: str  # "building/way-123" or "landmark/way-456".
    center: tuple  # (x, z)
    lods: list  # [[Part]] for LOD0, LOD1, LOD2.
    colliders: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)  # Written to the glTF node as extras.aleph.


@dataclass
class Instance:
    """One placed library asset, such as "tree/linden" or "prop/street-lamp"."""

    asset: str
    position: tuple  # (x, y, z) of the asset's origin, on the ground.
    yaw: float = 0.0  # Radians counter-clockwise seen from above.
    scale: float = 1.0
    tint: tuple = (1.0, 1.0, 1.0)


@dataclass
class Material:
    """A library material; texture paths are relative to the library root."""

    name: str
    size: float  # Metres one texture repeat covers; 0 for unit UVs.
    base_color: str = None
    normal: str = None
    orm: str = None  # Occlusion, roughness, and metallic in R, G, B, as glTF expects.
    factor: tuple = (1.0, 1.0, 1.0, 1.0)
    roughness: float = 1.0
    metallic: float = 0.0
    alpha: str = "OPAQUE"  # "OPAQUE", "MASK", or "BLEND".
    double_sided: bool = False
    extras: dict = field(default_factory=dict)


@dataclass
class Asset:
    """A library model placed by instances; files are relative to the library root."""

    name: str
    lods: list  # glTF files for LOD0, LOD1, LOD2.
    colliders: list = field(default_factory=list)  # [Collider] in the asset's own frame.
    height: float = 0.0
    radius: float = 0.0
