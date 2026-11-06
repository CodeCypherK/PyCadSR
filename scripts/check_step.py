"""Dev-only STEP validator (uses OpenCascade via cadquery-ocp; not shipped).

    .cache/devtools/Scripts/python.exe scripts/check_step.py file.step [--stl out.stl]

Reads the file like a CAD system would, reports surface types, B-rep validity,
area, and meshes every face to catch broken trims.
"""

import sys
from collections import Counter

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import (GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane,
                         GeomAbs_Sphere)
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

NAMES = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
         GeomAbs_Sphere: "sphere", GeomAbs_BSplineSurface: "bspline"}


def count(shape, kind):
    n, e = 0, TopExp_Explorer(shape, kind)
    while e.More():
        n += 1
        e.Next()
    return n


def main():
    path = sys.argv[1]
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        print("READ FAILED")
        return 1
    r.TransferRoots()
    shape = r.OneShape()
    print(f"solids={count(shape, TopAbs_SOLID)} shells={count(shape, TopAbs_SHELL)} faces={count(shape, TopAbs_FACE)}")
    print("valid:", BRepCheck_Analyzer(shape).IsValid())
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    print(f"area: {props.Mass():.3f}")
    BRepMesh_IncrementalMesh(shape, 0.2, False, 0.3, True)
    kinds = Counter()
    empty = []
    e = TopExp_Explorer(shape, TopAbs_FACE)
    i = 0
    tris = 0
    while e.More():
        f = (TopoDS.Face_s if hasattr(TopoDS, "Face_s") else TopoDS.Face)(e.Current())
        kinds[NAMES.get(BRepAdaptor_Surface(f).GetType(), "other")] += 1
        tri = BRep_Tool.Triangulation_s(f, TopLoc_Location())
        nt = tri.NbTriangles() if tri is not None else 0
        tris += nt
        fp = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, fp)
        if nt == 0:
            empty.append(i)
        print(f"  face {i:2d}: {NAMES.get(BRepAdaptor_Surface(f).GetType(), 'other'):8s} "
              f"area={fp.Mass():10.3f} triangles={nt}")
        i += 1
        e.Next()
    print("surface types:", dict(kinds))
    print("faces that failed to mesh:", empty or "none")
    print("total triangles:", tris)
    if "--stl" in sys.argv:
        from OCP.StlAPI import StlAPI_Writer
        out = sys.argv[sys.argv.index("--stl") + 1]
        StlAPI_Writer().Write(shape, out)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
