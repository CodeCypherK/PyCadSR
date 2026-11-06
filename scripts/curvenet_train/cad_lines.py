"""Design lines of a real CAD model: for every edge shared by two faces, how the faces meet there.

A real CAD model is often a patchwork (a test jug: 797 B-spline patches), so its faces are not its
design faces. Two patches meet either smoothly (same normal and same curvature across the edge: one
design surface split for modelling reasons) or along a design line (a crease, or a jump in the
curvature across the edge: a crest, a step, a change of shape). This writes, per shared edge, the
largest normal angle and the largest jump of the across-edge curvature along it, and groups the faces
into design surfaces (faces joined across smooth edges).

    .cache/devtools/Scripts/python.exe -I scripts/curvenet_train/cad_lines.py SOURCE.stp OUT.json
        [--angle DEG] [--kjump REL]   smooth = angle below DEG and curvature jump below REL
Face numbers follow cad_bench prep (TopExp order inside each part), so they match truth_faces.npz."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface  # noqa: E402
from OCP.BRepLProp import BRepLProp_SLProps  # noqa: E402
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402
from OCP.gp import gp_Dir  # noqa: E402

from cad_bench import parts_of, read_step, sub_shapes  # noqa: E402


def side(face, surf, p, t):
    """Unit normal (outward by the face's orientation) and the normal curvature across the edge."""
    pr = GeomAPI_ProjectPointOnSurf(p, surf)
    if pr.NbPoints() == 0:
        return None
    u, v = pr.LowerDistanceParameters()
    pr_ = BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 2, 1e-6)
    if not pr_.IsNormalDefined():
        return None
    n = pr_.Normal()
    n = np.array([n.X(), n.Y(), n.Z()])
    if face.Orientation() == TopAbs_REVERSED:
        n = -n
    k = 0.0
    if pr_.IsCurvatureDefined():
        d = np.cross(n, t)
        d /= max(np.linalg.norm(d), 1e-12)
        if pr_.IsUmbilic():
            k = pr_.MaxCurvature()
        else:
            d1, d2 = gp_Dir(), gp_Dir()
            pr_.CurvatureDirections(d1, d2)
            c = float(np.dot(d, [d1.X(), d1.Y(), d1.Z()]))
            k = pr_.MaxCurvature() * c * c + pr_.MinCurvature() * (1 - c * c)
        if face.Orientation() == TopAbs_REVERSED:
            k = -k
    return n, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--angle", type=float, default=3.0)
    ap.add_argument("--kjump", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=9)
    a = ap.parse_args()
    shape, _, _ = read_step(a.source)
    solids, shells = parts_of(shape)
    faces, base = [], 0
    edges = []
    for p in solids + shells:
        fl = [TopoDS.Face(f) for f in sub_shapes(p, TopAbs_FACE)]
        owners = {}  # edge -> the faces it bounds (shapes hash by TShape and location)
        for i, f in enumerate(fl):
            for e in sub_shapes(f, TopAbs_EDGE):
                lst = owners.setdefault(hash(e), [])
                for e_, nums in lst:
                    if e_.IsSame(e):
                        if all(q != base + i for q, _ in nums):
                            nums.append((base + i, f))
                        break
                else:
                    lst.append((e, [(base + i, f)]))
        for e, nums in (x for lst in owners.values() for x in lst):
            e = TopoDS.Edge(e)
            if len(nums) != 2 or BRep_Tool.Degenerated_s(e):
                continue
            c = BRepAdaptor_Curve(e)
            t0, t1 = c.FirstParameter(), c.LastParameter()
            (ia, fa), (ib, fb) = nums
            sa, sb = BRep_Tool.Surface_s(fa), BRep_Tool.Surface_s(fb)
            ang, kj, kmax, ln = [], [], [], 0.0
            prev = None
            for s in np.linspace(0.02, 0.98, a.samples):
                tt = t0 + s * (t1 - t0)
                pnt = c.Value(tt)
                d1 = c.DN(tt, 1)
                tv = np.array([d1.X(), d1.Y(), d1.Z()])
                if np.linalg.norm(tv) < 1e-12:
                    continue
                tv /= np.linalg.norm(tv)
                xa, xb = side(fa, sa, pnt, tv), side(fb, sb, pnt, tv)
                q = np.array([pnt.X(), pnt.Y(), pnt.Z()])
                if prev is not None:
                    ln += float(np.linalg.norm(q - prev))
                prev = q
                if xa is None or xb is None:
                    continue
                (na, ka), (nb, kb) = xa, xb
                ang.append(float(np.degrees(np.arccos(np.clip(abs(np.dot(na, nb)), -1, 1)))))
                kk = max(abs(ka), abs(kb), 1e-9)
                kj.append(float(abs(ka - kb) / kk) if kk > 1.0 / 2000 else 0.0)  # flat both sides: no jump
                kmax.append(float(kk))
            if not ang:
                continue
            edges.append(dict(a=ia, b=ib, angle=max(ang), angle_med=float(np.median(ang)), kjump=float(np.median(kj)),
                              kjump_max=max(kj), r_min=1.0 / max(max(kmax), 1e-9), length=ln))
        faces += fl
        base += len(fl)
    # design surfaces: faces joined across smooth edges
    par = list(range(base))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for e in edges:
        e["smooth"] = bool(e["angle"] < a.angle and e["kjump"] < a.kjump)
        if e["smooth"]:
            par[find(e["a"])] = find(e["b"])
    roots = sorted({find(i) for i in range(base)})
    group = [roots.index(find(i)) for i in range(base)]
    json.dump(dict(source=a.source, angle=a.angle, kjump=a.kjump, faces=base, edges=edges, group=group), open(a.out, "w"))
    A = np.array([e["angle"] for e in edges])
    K = np.array([e["kjump"] for e in edges])
    print(f"{base} faces, {len(edges)} shared edges: angle p50 {np.median(A):.2f} p90 {np.percentile(A, 90):.2f} deg, "
          f"curvature jump p50 {np.median(K):.2f} p90 {np.percentile(K, 90):.2f}; "
          f"{sum(e['smooth'] for e in edges)} smooth -> {len(roots)} design surfaces")


if __name__ == "__main__":
    main()
