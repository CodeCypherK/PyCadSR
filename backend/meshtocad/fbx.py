"""Binary FBX meshes (the format many CAD and DCC tools export), read without outside tools.

Reads every mesh geometry, places it with its model's transform chain (translation,
pre/post rotation, rotation, scaling, geometric offsets) and returns triangles in
millimetres (FBX stores the unit as centimetres per file unit).
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"Kaydara FBX Binary  \x00"


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name, self.props, self.children = name, props, children

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def all(self, name):
        return [c for c in self.children if c.name == name]


_ARR = {"f": ("<f", 4), "d": ("<d", 8), "l": ("<q", 8), "i": ("<i", 4), "b": ("<b", 1)}


def _read_prop(buf: bytes, o: int):
    t = chr(buf[o])
    o += 1
    if t == "Y":
        return struct.unpack_from("<h", buf, o)[0], o + 2
    if t == "C":
        return bool(buf[o]), o + 1
    if t == "I":
        return struct.unpack_from("<i", buf, o)[0], o + 4
    if t == "F":
        return struct.unpack_from("<f", buf, o)[0], o + 4
    if t == "D":
        return struct.unpack_from("<d", buf, o)[0], o + 8
    if t == "L":
        return struct.unpack_from("<q", buf, o)[0], o + 8
    if t in _ARR:
        n, enc, clen = struct.unpack_from("<III", buf, o)
        o += 12
        raw = buf[o:o + clen]
        o += clen
        if enc == 1:
            raw = zlib.decompress(raw)
        fmt, size = _ARR[t]
        return np.frombuffer(raw, dtype=np.dtype(fmt), count=n).copy(), o
    if t in "SR":
        n = struct.unpack_from("<I", buf, o)[0]
        o += 4
        data = buf[o:o + n]
        o += n
        return (data.decode("utf-8", "replace") if t == "S" else data), o
    raise ValueError(f"unknown FBX property type {t!r}")


def _read_node(buf: bytes, o: int, wide: bool):
    if wide:
        end, nprops, _plen = struct.unpack_from("<QQQ", buf, o)
        o += 24
    else:
        end, nprops, _plen = struct.unpack_from("<III", buf, o)
        o += 12
    nlen = buf[o]
    o += 1
    if end == 0:
        return None, o
    name = buf[o:o + nlen].decode("ascii", "replace")
    o += nlen
    props = []
    for _ in range(nprops):
        v, o = _read_prop(buf, o)
        props.append(v)
    children = []
    null = 25 if wide else 13
    while o < end - null + 1 and o < end:
        if end - o == null:
            break
        child, o = _read_node(buf, o, wide)
        if child is None:
            break
        children.append(child)
    return Node(name, props, children), end


def parse(path) -> Node:
    buf = Path(path).read_bytes()
    if not buf.startswith(MAGIC):
        raise ValueError("not a binary FBX file (ASCII FBX is not supported)")
    version = struct.unpack_from("<I", buf, 23)[0]
    wide = version >= 7500
    o = 27
    top = []
    while o < len(buf):
        node, o2 = _read_node(buf, o, wide)
        if node is None:
            break
        top.append(node)
        o = o2
    root = Node("", [version], top)
    return root


def _props70(node: Node) -> dict:
    out = {}
    p70 = node.find("Properties70") if node is not None else None
    if p70 is None:
        return out
    for p in p70.all("P"):
        if p.props:
            out[p.props[0]] = p.props[4:]
    return out


def _euler(deg, order: int = 0) -> np.ndarray:
    """FBX Euler angles (degrees) -> 3x3 matrix; order 0 = XYZ (x applied first)."""
    x, y, z = np.radians(np.asarray(deg, float))
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    seq = {0: (Rx, Ry, Rz), 1: (Rx, Rz, Ry), 2: (Ry, Rz, Rx), 3: (Ry, Rx, Rz), 4: (Rz, Rx, Ry), 5: (Rz, Ry, Rx)}
    a, b, c = seq.get(int(order), seq[0])
    return c @ b @ a


def _mat(R=None, t=None, s=None) -> np.ndarray:
    M = np.eye(4)
    if s is not None:
        M[:3, :3] = np.diag(np.asarray(s, float))
    if R is not None:
        M[:3, :3] = R @ M[:3, :3]
    if t is not None:
        M[:3, 3] = np.asarray(t, float)
    return M


def _local(p: dict) -> np.ndarray:
    v = lambda k, d: np.asarray(p.get(k, d), float)[:3]
    order = int(p.get("RotationOrder", [0])[0]) if "RotationOrder" in p else 0
    T = _mat(t=v("Lcl Translation", [0, 0, 0]))
    Roff, Rp = _mat(t=v("RotationOffset", [0, 0, 0])), _mat(t=v("RotationPivot", [0, 0, 0]))
    Rpre = _mat(R=_euler(v("PreRotation", [0, 0, 0])))
    R = _mat(R=_euler(v("Lcl Rotation", [0, 0, 0]), order))
    Rpost = _mat(R=_euler(v("PostRotation", [0, 0, 0])))
    Soff, Sp = _mat(t=v("ScalingOffset", [0, 0, 0])), _mat(t=v("ScalingPivot", [0, 0, 0]))
    S = _mat(s=v("Lcl Scaling", [1, 1, 1]))
    inv = np.linalg.inv
    return T @ Roff @ Rp @ Rpre @ R @ inv(Rpost) @ inv(Rp) @ Soff @ Sp @ S @ inv(Sp)


def _geometric(p: dict) -> np.ndarray:
    v = lambda k, d: np.asarray(p.get(k, d), float)[:3]
    return (_mat(t=v("GeometricTranslation", [0, 0, 0])) @ _mat(R=_euler(v("GeometricRotation", [0, 0, 0])))
            @ _mat(s=v("GeometricScaling", [1, 1, 1])))


def read(path, info: dict | None = None):
    """Triangles of every mesh in a binary FBX, in millimetres: (vertices, faces, part index per face, part names)."""
    root = parse(path)
    objects = root.find("Objects")
    if objects is None:
        raise ValueError("FBX has no objects")
    gs = _props70(root.find("GlobalSettings"))
    cm_per_unit = float(gs.get("UnitScaleFactor", [1.0])[0]) if "UnitScaleFactor" in gs else 1.0
    geoms, models = {}, {}
    for n in objects.children:
        if n.name == "Geometry" and len(n.props) >= 3 and n.props[2] == "Mesh":
            geoms[n.props[0]] = n
        elif n.name == "Model":
            models[n.props[0]] = n
    parent: dict[int, int] = {}
    geom_model: dict[int, int] = {}
    conns = root.find("Connections")
    for c in (conns.all("C") if conns is not None else []):
        if len(c.props) >= 3 and c.props[0] == "OO":
            child, par = c.props[1], c.props[2]
            if child in geoms and par in models:
                geom_model[child] = par
            elif child in models:
                parent[child] = par

    def world(mid):
        M = np.eye(4)
        seen = set()
        while mid in models and mid not in seen:
            seen.add(mid)
            M = _local(_props70(models[mid])) @ M
            mid = parent.get(mid, 0)
        return M

    Vs, Fs, parts, names = [], [], [], []
    off = 0
    for gid, g in geoms.items():
        vn, pn = g.find("Vertices"), g.find("PolygonVertexIndex")
        if vn is None or pn is None or not len(vn.props) or not len(pn.props):
            continue
        V = np.asarray(vn.props[0], float).reshape(-1, 3)
        idx = np.asarray(pn.props[0], np.int64)
        mid = geom_model.get(gid)
        M = world(mid) if mid is not None else np.eye(4)
        if mid is not None:
            M = M @ _geometric(_props70(models[mid]))
        V = V @ M[:3, :3].T + M[:3, 3]
        # polygons end at a negative index (~i); fan-triangulate each
        ends = np.where(idx < 0)[0]
        idx = np.where(idx < 0, ~idx, idx)
        tris = []
        start = 0
        for e in ends:
            poly = idx[start:e + 1]
            for k in range(1, len(poly) - 1):
                tris.append((poly[0], poly[k], poly[k + 1]))
            start = e + 1
        if not tris:
            continue
        T = np.asarray(tris, np.int64)
        if np.linalg.det(M[:3, :3]) < 0:  # mirrored: keep the faces pointing out
            T = T[:, ::-1]
        Vs.append(V)
        Fs.append(T + off)
        off += len(V)
        nm = models[mid].props[1].split("\x00")[0] if mid is not None and len(models[mid].props) > 1 else f"mesh{len(names)}"
        parts.append(np.full(len(T), len(names)))
        names.append(nm)
    if not Vs:
        raise ValueError("FBX has no mesh geometry")
    V = np.vstack(Vs) * (10.0 * cm_per_unit)  # to millimetres
    if info is not None:
        info.update({"version": root.props[0], "cm_per_unit": cm_per_unit, "meshes": len(names),
                     "up_axis": gs.get("UpAxis", [None])[0]})
    return V, np.vstack(Fs), np.concatenate(parts), names
