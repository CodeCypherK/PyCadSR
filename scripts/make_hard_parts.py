"""Dev-only: harder test parts (machined, scan-style, organic) with ground truth.

    .cache/devtools/Scripts/python.exe scripts/make_hard_parts.py [out_dir]

Writes for every part:  <name>.stl (pipeline input), <name>_truth.json and, where
the input is degraded or synthetic, <name>_truth.stl (clean reference surface the
reconstruction is scored against). OpenCascade parts also get <name>_truth.step.
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCP.gp import gp_Ax1, gp_Dir
from OCP.StlAPI import StlAPI_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

sys.path.insert(0, str(Path(__file__).parent))
from make_test_parts import P, box, chamfer, cut, cyl, fillet, fuse, truth, unify  # noqa: E402


# ------------------------------------------------------------------ machined parts

def housing():
    s = box(-50, -35, 0, 100, 70, 40)
    s = fillet(s, 6.0, lambda m, a: abs(abs(m[0]) - 50) < 1e-6 and abs(abs(m[1]) - 35) < 1e-6)
    pocket = box(-30, -15, 25, 60, 30, 20)
    pocket = fillet(pocket, 5.0, lambda m, a: abs(abs(m[0]) - 30) < 1e-6 and abs(abs(m[1]) - 15) < 1e-6)
    s = cut(s, pocket)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = 40 * sx, 25 * sy
            s = cut(s, cyl(3.5, 50, x, y, -5))
            s = cut(s, cyl(6.0, 10, x, y, 34))
    s = fuse(s, cyl(10, 12, 50, 0, 14, 1, 0, 0))
    s = cut(s, cyl(5, 30, 40, 0, 14, 1, 0, 0))
    slot = fuse(box(-15, -4, -1, 30, 8, 9), cyl(4, 9, -15, 0, -1))
    slot = fuse(slot, cyl(4, 9, 15, 0, -1))
    s = unify(cut(s, slot))
    s = chamfer(s, 1.5, lambda m, a: abs(m[2] - 40) < 1e-6 and (abs(m[0]) > 44 or abs(m[1]) > 29))
    return s


def revolve(profile):
    poly = BRepBuilderAPI_MakePolygon()
    for r, z in profile:
        poly.Add(P(r, 0, z))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakeRevol(face, gp_Ax1(P(0, 0, 0), gp_Dir(0, 0, 1)), 2 * np.pi).Shape()


def pulley():
    s = revolve([(8, 0), (40, 0), (40, 4), (34, 8), (40, 12), (40, 16), (18, 16), (18, 26), (8, 26)])
    s = fillet(s, 3.0, lambda m, a: abs(m[2] - 16) < 1e-6 and abs(np.hypot(m[0], m[1]) - 18) < 1e-3)
    for k in range(4):
        t = np.pi / 4 + k * np.pi / 2
        s = cut(s, cyl(5, 30, 27 * np.cos(t), 27 * np.sin(t), -5))
    s = cut(s, box(7, -2, -1, 3.5, 4, 30))
    return unify(s)


def manifold():
    s = box(-30, -20, -20, 60, 40, 40)
    s = fillet(s, 3.0, lambda m, a: sum(abs(abs(m[i]) - h) < 1e-6 for i, h in enumerate((30, 20, 20))) == 2)
    s = cut(s, cyl(6, 80, -40, 0, 0, 1, 0, 0))
    s = cut(s, cyl(9, 6, -31, 0, 0, 1, 0, 0))
    s = cut(s, cyl(4, 50, 0, -25, 0, 0, 1, 0))
    s = cut(s, cyl(3, 50, 15, 0, -25))
    return unify(s)


# ------------------------------------------------------------------ scan degradation

def scanify(mesh: trimesh.Trimesh, seed: int, noise_frac=0.0004, holes=3, hole_frac=0.04):
    """Dense remesh + noise along normals + a few missing patches, like a 3D scan."""
    rng = np.random.default_rng(seed)
    diag = float(np.linalg.norm(mesh.extents))
    V, F = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=diag / 120, max_iter=12)
    m = trimesh.Trimesh(V, F, process=True)
    m.vertices = m.vertices + m.vertex_normals * rng.normal(0, noise_frac * diag, (len(m.vertices), 1))
    centers = m.triangles_center
    keep = np.ones(len(m.faces), bool)
    for c in m.vertices[rng.choice(len(m.vertices), holes, replace=False)]:
        keep &= np.linalg.norm(centers - c, axis=1) > hole_frac * diag
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    return m


# ------------------------------------------------------------------ organic shapes

def grid_surface(fn, nu, nv, closed_u=False):
    """Vertices and quads->triangles of a parametric grid fn(u, v) on [0,1]^2."""
    us = np.linspace(0, 1, nu, endpoint=not closed_u)
    vs = np.linspace(0, 1, nv)
    U, Vv = np.meshgrid(us, vs, indexing="ij")
    X = fn(U, Vv)
    verts = X.reshape(-1, 3)
    faces = []
    for i in range(nu if closed_u else nu - 1):
        i2 = (i + 1) % nu
        for j in range(nv - 1):
            a, b, c, d = i * nv + j, i2 * nv + j, i2 * nv + j + 1, i * nv + j + 1
            faces += [[a, b, c], [a, c, d]]
    return verts, np.array(faces)


def cap(verts_ring, center):
    """Fan triangulation closing a ring of vertices."""
    n = len(verts_ring)
    V = np.vstack([verts_ring, center])
    F = np.array([[n, i, (i + 1) % n] for i in range(n)])
    return V, F


def combine(parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    m = trimesh.Trimesh(np.vstack(Vs), np.vstack(Fs), process=True)
    m.merge_vertices()
    trimesh.repair.fix_normals(m)
    return m


def bottle():
    """Surface of revolution r(z) = 14 + 3 sin(z/7), flat bottom and top."""
    r = lambda z: 14 + 3 * np.sin(z / 7)

    def side(U, V):
        t, z = 2 * np.pi * U, 60 * V
        return np.stack([r(z) * np.cos(t), r(z) * np.sin(t), z], -1)

    nu, nv = 160, 120
    V, F = grid_surface(side, nu, nv, closed_u=True)
    bottom = V.reshape(nu, nv, 3)[:, 0]
    top = V.reshape(nu, nv, 3)[:, -1]
    return combine([(V, F), cap(bottom[::-1], [0, 0, 0]), cap(top, [0, 0, 60])])


def blob():
    """Closed star-shaped blob r = 20 + 4 sin(3 theta) cos(lat)^2 (spherical coordinates about Z)."""
    def surf(U, V):
        t, lat = 2 * np.pi * U, np.pi * (V - 0.5)
        rr = 20 + 4 * np.sin(3 * t) * np.cos(lat) ** 2
        return np.stack([rr * np.cos(lat) * np.cos(t), rr * np.cos(lat) * np.sin(t), rr * np.sin(lat)], -1)

    nu, nv = 180, 90
    V, F = grid_surface(surf, nu, nv, closed_u=True)
    m = trimesh.Trimesh(V, F, process=True)
    m.merge_vertices()  # poles collapse to single vertices
    m.update_faces(m.nondegenerate_faces())
    trimesh.repair.fix_normals(m)
    return m


def saddle_shell():
    """8 mm thick plate whose faces are z = (x^2 - y^2)/80 (+8)."""
    nx, ny = 121, 101
    xs, ys = np.linspace(-30, 30, nx), np.linspace(-25, 25, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = (X ** 2 - Y ** 2) / 80
    top = np.c_[X.ravel(), Y.ravel(), Z.ravel() + 8]
    bot = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    n = nx * ny
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel()
    F = [np.c_[a, a + ny, a + ny + 1], np.c_[a, a + ny + 1, a + 1],
         np.c_[a + n, a + n + ny + 1, a + n + ny], np.c_[a + n, a + n + 1, a + n + ny + 1]]
    ring = ([k * ny for k in range(nx)] + [(nx - 1) * ny + k for k in range(1, ny)]
            + [k * ny + ny - 1 for k in range(nx - 2, -1, -1)] + [k for k in range(ny - 2, 0, -1)])
    for p, q in zip(ring, ring[1:] + ring[:1]):
        F.append(np.array([[p, p + n, q], [q, p + n, q + n]]))
    m = trimesh.Trimesh(np.vstack([top, bot]), np.vstack(F), process=True)
    trimesh.repair.fix_normals(m)
    return m


def blade():
    """Tapered, twisted NACA-2412 blade, 60 mm span, sharp trailing edge, flat root and tip."""
    def section(x):  # x in [0,1] along chord -> (upper, lower) y
        t, mc, pc = 0.12, 0.02, 0.4
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        yc = np.where(x < pc, mc / pc ** 2 * (2 * pc * x - x ** 2), mc / (1 - pc) ** 2 * (1 - 2 * pc + 2 * pc * x - x ** 2))
        return yc + yt, yc - yt

    nu, nv = 140, 80
    beta = np.linspace(0, np.pi, nu // 2)
    xs = 0.5 * (1 - np.cos(beta))  # cosine spacing, dense at the edges
    yu, yl = section(xs)
    loop_x = np.r_[xs[::-1], xs[1:-1]]
    loop_y = np.r_[yu[::-1], yl[1:-1]]

    def surf(U, V):
        k = np.clip(np.rint(U * len(loop_x)).astype(int), 0, len(loop_x) - 1)
        z = 60 * V
        chord = 40 - 0.25 * z
        tw = np.radians(30) * V
        px, py = (loop_x[k] - 0.3) * chord, loop_y[k] * chord
        return np.stack([px * np.cos(tw) - py * np.sin(tw), px * np.sin(tw) + py * np.cos(tw), z], -1)

    V, F = grid_surface(surf, len(loop_x), nv, closed_u=True)
    ring = V.reshape(len(loop_x), nv, 3)
    root, tip = ring[:, 0], ring[:, -1]
    return combine([(V, F), cap(root[::-1], root.mean(0)), cap(tip, tip.mean(0))])


# ------------------------------------------------------------------ main

def write_occ(out, name, shape, defl=0.03):
    BRepMesh_IncrementalMesh(shape, defl, False, 0.2, True)
    StlAPI_Writer().Write(shape, str(out / f"{name}.stl"))
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    w.Write(str(out / f"{name}_truth.step"))
    info = truth(shape)
    info["triangles"] = int(len(trimesh.load(out / f"{name}.stl").faces))
    (out / f"{name}_truth.json").write_text(json.dumps(info, indent=1))
    print(f"{name:16s} {info['n_faces']:3d} faces {info['faces']}  area={info['area']:.1f} tris={info['triangles']}")
    return info


def write_mesh(out, name, mesh, faces, truth_mesh=None):
    mesh.export(out / f"{name}.stl")
    ref = truth_mesh if truth_mesh is not None else mesh
    ref.export(out / f"{name}_truth.stl")
    info = {"faces": faces, "n_faces": sum(faces.values()), "area": float(ref.area),
            "volume": float(ref.volume) if ref.is_watertight else None, "triangles": int(len(mesh.faces))}
    (out / f"{name}_truth.json").write_text(json.dumps(info, indent=1))
    print(f"{name:16s} {info['n_faces']:3d} faces {faces}  area={info['area']:.1f} tris={info['triangles']}")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".cache/parts2")
    out.mkdir(parents=True, exist_ok=True)
    occ = {"housing": housing, "pulley": pulley, "manifold": manifold}
    for name, fn in occ.items():
        info = write_occ(out, name, fn())
        clean = trimesh.load(out / f"{name}.stl")
        scan = scanify(clean, seed=len(name))
        write_mesh(out, f"{name}_scan", scan, info["faces"], truth_mesh=clean)
    write_mesh(out, "bottle", bottle(), {"freeform_revolve": 1, "plane": 2})
    write_mesh(out, "blob", blob(), {"freeform": 1})
    write_mesh(out, "saddle_shell", saddle_shell(), {"freeform": 2, "plane": 4})
    write_mesh(out, "blade", blade(), {"freeform": 1, "plane": 2})


if __name__ == "__main__":
    main()
