"""glTF binary writer for game world meshes; textures stay in the library and are referenced by relative URI.

Part.uvs follow the OpenGL convention, v pointing up the texture (and up walls); the writer flips v for glTF,
and divides tiling UVs, given in metres, by the material's size.
"""

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..common import atomic_path

FLOAT, UINT16, UINT32 = 5126, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
POINTS, TRIANGLES = 0, 4
REPEAT, LINEAR, LINEAR_MIPMAP_LINEAR = 10497, 9729, 9987


@dataclass
class Node:
    name: str
    parts: list = field(default_factory=list)  # [Part]
    translation: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = None  # Quaternion (x, y, z, w).
    scale: tuple = None
    extras: dict = None  # Written as extras.aleph.
    children: list = field(default_factory=list)  # [Node]
    colliders: list = field(default_factory=list)  # [Collider], written as child nodes "collider/…".


def write(path, nodes, materials, library):
    """Write nodes as the scene of a .glb at path.

    materials(name) returns a types.Material; library is the library root on disk, which texture paths are relative
    to. Texture URIs are made relative to path's folder, so the library must sit at the same relative place wherever
    the files end up.
    """
    path = Path(path)
    writer = Writer(materials, Path(library), path.parent)
    roots = [writer.node(node) for node in nodes]
    gltf = writer.gltf
    gltf["scenes"] = [dict(nodes=roots)]
    gltf["scene"] = 0
    for key in [key for key, value in gltf.items() if value == []]:
        del gltf[key]
    binary = bytes(writer.buffer)
    if binary:
        gltf["buffers"] = [dict(byteLength=len(binary))]
    header = json.dumps(gltf, separators=(",", ":"), allow_nan=False, default=plain).encode()
    header += b" " * (-len(header) % 4)
    binary += b"\0" * (-len(binary) % 4)
    # A file without geometry has no binary chunk, since glTF buffers may not be empty.
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary if binary else b""
    with atomic_path(path) as temporary, temporary.open("wb") as output:
        output.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(header) + len(chunk)))
        output.write(struct.pack("<II", len(header), 0x4E4F534A) + header)
        output.write(chunk)


def plain(value):
    """numpy values in extras, as JSON numbers and lists."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class Writer:
    def __init__(self, materials, library, folder):
        self.materials, self.library, self.folder = materials, library, folder
        self.buffer = bytearray()
        self.material_index, self.image_index = {}, {}
        self.gltf = dict(asset=dict(version="2.0", generator="Aleph"), nodes=[], meshes=[], materials=[], textures=[],
                         images=[], samplers=[dict(magFilter=LINEAR, minFilter=LINEAR_MIPMAP_LINEAR, wrapS=REPEAT,
                                                   wrapT=REPEAT)],
                         accessors=[], bufferViews=[])

    def view(self, array, target):
        data = np.ascontiguousarray(array).tobytes()
        self.buffer += b"\0" * (-len(self.buffer) % 4)
        self.gltf["bufferViews"].append(dict(buffer=0, byteOffset=len(self.buffer), byteLength=len(data), target=target))
        self.buffer += data
        return len(self.gltf["bufferViews"]) - 1

    def accessor(self, array, kind, target=ARRAY_BUFFER, bounds=False):
        array = np.asarray(array)
        component = {np.dtype("<f4"): FLOAT, np.dtype("<u2"): UINT16, np.dtype("<u4"): UINT32}[array.dtype.newbyteorder("<")]
        accessor = dict(bufferView=self.view(array, target), componentType=component, count=len(array), type=kind)
        if bounds:
            accessor.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        self.gltf["accessors"].append(accessor)
        return len(self.gltf["accessors"]) - 1

    def image(self, relative):
        if relative not in self.image_index:
            uri = Path(os.path.relpath(self.library / relative, self.folder)).as_posix()
            self.gltf["images"].append(dict(uri=uri))
            self.gltf["textures"].append(dict(source=len(self.gltf["images"]) - 1, sampler=0))
            self.image_index[relative] = len(self.gltf["textures"]) - 1
        return self.image_index[relative]

    def material(self, name):
        if name in self.material_index:
            return self.material_index[name]
        m = self.materials(name)
        pbr = dict(baseColorFactor=list(m.factor), metallicFactor=m.metallic, roughnessFactor=m.roughness)
        material = dict(name=name, pbrMetallicRoughness=pbr, alphaMode=m.alpha, doubleSided=m.double_sided,
                        extras=dict(aleph=dict(m.extras, size=m.size)))
        if m.base_color:
            pbr["baseColorTexture"] = dict(index=self.image(m.base_color))
        if m.orm:
            pbr["metallicRoughnessTexture"] = dict(index=self.image(m.orm))
            material["occlusionTexture"] = dict(index=self.image(m.orm))
        if m.normal:
            material["normalTexture"] = dict(index=self.image(m.normal))
        if m.alpha == "MASK":
            material["alphaCutoff"] = 0.5
        if "emissive" in m.extras:
            material["emissiveFactor"] = list(m.extras["emissive"])
        self.gltf["materials"].append(material)
        self.material_index[name] = len(self.gltf["materials"]) - 1
        return self.material_index[name]

    def primitive(self, part):
        if not np.isfinite(np.asarray(part.positions, float)).all():
            raise ValueError(f"A {part.material} part has non-finite positions.")
        m = self.materials(part.material)
        uvs = np.asarray(part.uvs, "<f4").reshape(-1, 2)
        uvs = np.column_stack((uvs[:, 0] / m.size, -uvs[:, 1] / m.size) if m.size else (uvs[:, 0], 1 - uvs[:, 1]))
        normals = np.asarray(part.normals, "<f4").reshape(-1, 3)
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        attributes = dict(POSITION=self.accessor(np.asarray(part.positions, "<f4").reshape(-1, 3), "VEC3", bounds=True),
                          NORMAL=self.accessor(normals.astype("<f4"), "VEC3"),
                          TEXCOORD_0=self.accessor(uvs.astype("<f4"), "VEC2"))
        if part.colors is not None:
            attributes["COLOR_0"] = self.accessor(np.asarray(part.colors, "<f4").reshape(-1, 4), "VEC4")
        return dict(attributes=attributes, indices=self.indices(part.indices, len(part.positions)),
                    material=self.material(part.material), mode=TRIANGLES)

    def indices(self, indices, count):
        return self.accessor(np.asarray(indices).astype("<u2" if count <= 65535 else "<u4").ravel(), "SCALAR",
                             ELEMENT_ARRAY_BUFFER)

    def mesh(self, name, primitives):
        self.gltf["meshes"].append(dict(name=name, primitives=primitives))
        return len(self.gltf["meshes"]) - 1

    def node(self, node):
        entry = dict(name=node.name)
        if any(node.translation):
            entry["translation"] = [float(v) for v in node.translation]
        if node.rotation is not None:
            entry["rotation"] = [float(v) for v in node.rotation]
        if node.scale is not None:
            entry["scale"] = [float(v) for v in node.scale]
        if node.extras:
            entry["extras"] = dict(aleph=node.extras)
        parts = [part for part in node.parts if len(part.indices)]
        if parts:
            entry["mesh"] = self.mesh(node.name, [self.primitive(part) for part in parts])
        index = len(self.gltf["nodes"])
        self.gltf["nodes"].append(entry)
        children = [self.node(child) for child in node.children]
        children += [self.collider(f"collider/{node.name}/{i}", collider) for i, collider in enumerate(node.colliders)]
        if children:
            entry["children"] = children
        return index

    def collider(self, name, collider):
        """Hidden helper nodes: convex hulls as point clouds, trimeshes as triangles, and shapes as extras only."""
        entry = dict(name=name, extras=dict(aleph=dict(collider.params, collider=collider.shape)))
        if collider.shape in ("convex", "trimesh"):
            points = self.accessor(np.asarray(collider.vertices, "<f4").reshape(-1, 3), "VEC3", bounds=True)
            primitive = dict(attributes=dict(POSITION=points), mode=POINTS)
            if collider.shape == "trimesh":
                primitive.update(mode=TRIANGLES, indices=self.indices(collider.indices, len(collider.vertices)))
            entry["mesh"] = self.mesh(name, [primitive])
        elif "center" in collider.params:
            entry["translation"] = [float(v) for v in collider.params["center"]]
        self.gltf["nodes"].append(entry)
        return len(self.gltf["nodes"]) - 1
