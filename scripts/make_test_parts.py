"""Dev-only: build complex CAD test parts with OpenCascade and export ground truth.

    .cache/devtools/Scripts/python.exe scripts/make_test_parts.py [out_dir]

For each part writes <name>.stl (the tessellated input, like a CAD STL export),
<name>_truth.step (exact B-rep) and <name>_truth.json (surface types, area).
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder,
                             BRepPrimAPI_MakeSphere)
from OCP.GeomAbs import (GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane,
                         GeomAbs_Sphere, GeomAbs_Torus)
from OCP.gp import gp_Ax2, gp_Dir, gp_GTrsf, gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.StlAPI import StlAPI_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

_edge = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge
_face = getattr(TopoDS, "Face_s", None) or TopoDS.Face
TYPES = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
         GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus", GeomAbs_BSplineSurface: "bspline"}


def P(x, y, z):
    return gp_Pnt(float(x), float(y), float(z))


def ax(x, y, z, dx=0, dy=0, dz=1):
    return gp_Ax2(P(x, y, z), gp_Dir(float(dx), float(dy), float(dz)))


def box(x, y, z, dx, dy, dz):
    return BRepPrimAPI_MakeBox(P(x, y, z), dx, dy, dz).Shape()


def cyl(r, h, *a):
    return BRepPrimAPI_MakeCylinder(ax(*a), r, h).Shape()


def fuse(a, b):
    return BRepAlgoAPI_Fuse(a, b).Shape()


def cut(a, b):
    return BRepAlgoAPI_Cut(a, b).Shape()


def unify(s):
    u = ShapeUpgrade_UnifySameDomain(s, True, True, True)
    u.Build()
    return u.Shape()


def edges(shape, pred):
    out, e = [], TopExp_Explorer(shape, TopAbs_EDGE)
    while e.More():
        ed = _edge(e.Current())
        c = BRepAdaptor_Curve(ed)
        m = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
        a = c.Value(c.FirstParameter())
        if pred(np.array([m.X(), m.Y(), m.Z()]), np.array([a.X(), a.Y(), a.Z()])):
            out.append(ed)
        e.Next()
    return out


def fillet(shape, r, pred):
    mk = BRepFilletAPI_MakeFillet(shape)
    for ed in edges(shape, pred):
        mk.Add(r, ed)
    return mk.Shape()


def chamfer(shape, d, pred):
    mk = BRepFilletAPI_MakeChamfer(shape)
    for ed in edges(shape, pred):
        mk.Add(d, ed)
    return mk.Shape()


# --------------------------------------------------------------------- parts

def rounded_box():
    return fillet(box(-30, -20, -15, 60, 40, 30), 6.0, lambda m, a: True)


def bracket():
    s = unify(fuse(box(0, 0, 0, 80, 50, 10), box(0, 0, 10, 10, 50, 50)))
    s = cut(s, cyl(5, 30, 45, 15, -5))
    s = cut(s, cyl(5, 30, 45, 35, -5))
    s = cut(s, cyl(6, 30, -5, 25, 40, 1, 0, 0))
    s = fillet(s, 6.0, lambda m, a: abs(m[0] - 10) < 1e-6 and abs(m[2] - 10) < 1e-6)
    s = chamfer(s, 2.0, lambda m, a: abs(m[2] - 60) < 1e-6 and abs(m[1] - 25) < 1e-6)
    return s


def flange():
    s = unify(fuse(cyl(40, 8, 0, 0, 0), cyl(16, 30, 0, 0, 0)))
    s = cut(s, cyl(8, 40, 0, 0, -5))
    for k in range(6):
        t = 2 * np.pi * k / 6
        s = cut(s, cyl(3.5, 20, 28 * np.cos(t), 28 * np.sin(t), -5))
    s = fillet(s, 4.0, lambda m, a: abs(m[2] - 8) < 1e-6 and abs(np.hypot(m[0], m[1]) - 16) < 1e-3)
    return s


def knob():
    s = fuse(cyl(5, 21, 0, 0, 0), BRepPrimAPI_MakeCone(ax(0, 0, 20), 5, 10, 10).Shape())
    s = fuse(s, BRepPrimAPI_MakeSphere(P(0, 0, 38), 10.5).Shape())
    s = cut(s, box(-20, -20, 45, 40, 40, 20))
    return unify(s)


def ellipsoid():
    s = BRepPrimAPI_MakeSphere(P(0, 0, 0), 1.0).Shape()
    g = gp_GTrsf()
    g.SetValue(1, 1, 30.0)
    g.SetValue(2, 2, 20.0)
    g.SetValue(3, 3, 12.0)
    return BRepBuilderAPI_GTransform(s, g, True).Shape()


def wavy_plate_mesh():
    """Closed solid: flat bottom, four walls, wavy top z = 6 + 2 sin(x/9) cos(y/12)."""
    nx, ny = 121, 81
    xs, ys = np.linspace(-30, 30, nx), np.linspace(-20, 20, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = 6 + 2 * np.sin(X / 9) * np.cos(Y / 12)
    top = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    bot = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    V = np.vstack([top, bot])
    n = nx * ny
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel()
    F = [np.c_[a, a + ny, a + ny + 1], np.c_[a, a + ny + 1, a + 1]]
    F += [np.c_[a + n, a + n + ny + 1, a + n + ny], np.c_[a + n, a + n + 1, a + n + ny + 1]]
    ring = ([k * ny for k in range(nx)] + [(nx - 1) * ny + k for k in range(1, ny)]
            + [k * ny + ny - 1 for k in range(nx - 2, -1, -1)] + [k for k in range(ny - 2, 0, -1)])
    for p, q in zip(ring, ring[1:] + ring[:1]):
        F.append(np.array([[p, p + n, q], [q, p + n, q + n]]))
    m = trimesh.Trimesh(V, np.vstack(F), process=True)
    trimesh.repair.fix_normals(m)
    return m


PARTS = {"rounded_box": rounded_box, "bracket": bracket, "flange": flange, "knob": knob, "ellipsoid": ellipsoid}


def truth(shape) -> dict:
    kinds, e = Counter(), TopExp_Explorer(shape, TopAbs_FACE)
    while e.More():
        kinds[TYPES.get(BRepAdaptor_Surface(_face(e.Current())).GetType(), "other")] += 1
        e.Next()
    ps, pv = GProp_GProps(), GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, ps)
    BRepGProp.VolumeProperties_s(shape, pv)
    return {"faces": dict(kinds), "n_faces": sum(kinds.values()), "area": ps.Mass(), "volume": pv.Mass()}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".cache/parts")
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in PARTS.items():
        shape = fn()
        BRepMesh_IncrementalMesh(shape, 0.03, False, 0.2, True)
        StlAPI_Writer().Write(shape, str(out / f"{name}.stl"))
        w = STEPControl_Writer()
        w.Transfer(shape, STEPControl_AsIs)
        w.Write(str(out / f"{name}_truth.step"))
        info = truth(shape)
        mesh = trimesh.load(out / f"{name}.stl")
        info["triangles"] = int(len(mesh.faces))
        (out / f"{name}_truth.json").write_text(json.dumps(info, indent=1))
        print(f"{name:12s} {info['n_faces']:3d} faces {info['faces']}  area={info['area']:.1f}  tris={info['triangles']}")
    m = wavy_plate_mesh()
    m.export(out / "wavy_plate.stl")
    info = {"faces": {"plane": 5, "freeform": 1}, "n_faces": 6, "area": float(m.area), "volume": float(m.volume),
            "triangles": int(len(m.faces))}
    (out / "wavy_plate_truth.json").write_text(json.dumps(info, indent=1))
    print(f"{'wavy_plate':12s} {6:3d} faces {info['faces']}  area={info['area']:.1f}  tris={info['triangles']}")


if __name__ == "__main__":
    main()
