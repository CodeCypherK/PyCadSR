"""Dev-only benchmark on real CAD assemblies (needs .cache/devtools: cadquery-ocp, trimesh, scipy).

    python scripts/cad_bench.py prep  SOURCE.step OUT_DIR [--rel 2e-4] [--angle 0.25]
        Mesh every solid like a CAD export would and write what the app gets fed:
        OUT_DIR/NAME.glb   one node per part (glTF, metres)
        OUT_DIR/truth.json per part: surface types, area, bounding box (mm)
        OUT_DIR/truth.stl  fine tessellation of the source (mm) for scoring

    python scripts/cad_bench.py check OURS.step OUT_DIR
        Open our STEP like a CAD tool (OpenCASCADE reader with shape healing) and
        report: parts and bodies, solids vs open shells, invalid shapes, faces that
        fail to mesh, surface types against the source, and accuracy (exact
        point-to-surface distance both ways, % of the assembly's size).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
from compare_step import TYPES, _face, _memory_guard  # noqa: E402
from score_mesh import surface_distance  # noqa: E402

from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Surface  # noqa: E402
from OCP.BRepBndLib import BRepBndLib  # noqa: E402
from OCP.BRepCheck import BRepCheck_Analyzer  # noqa: E402
from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: E402
from OCP.Bnd import Bnd_Box  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402
from OCP.STEPControl import STEPControl_Reader  # noqa: E402
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopLoc import TopLoc_Location  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

_solid = getattr(TopoDS, "Solid_s", None) or TopoDS.Solid
_shell = getattr(TopoDS, "Shell_s", None) or TopoDS.Shell


def read_step(path):
    r = STEPControl_Reader()
    status = r.ReadFile(str(path))
    n = r.TransferRoots()
    return r.OneShape(), int(status), int(n)


def sub_shapes(shape, kind, avoid=None):
    out, e = [], (TopExp_Explorer(shape, kind, avoid) if avoid is not None else TopExp_Explorer(shape, kind))
    while e.More():
        out.append(e.Current())
        e.Next()
    return out


def parts_of(shape):
    """Solids, plus shells that are not inside a solid (surface bodies)."""
    solids = [_solid(s) for s in sub_shapes(shape, TopAbs_SOLID)]
    free_shells = [_shell(s) for s in sub_shapes(shape, TopAbs_SHELL, TopAbs_SOLID)]
    return solids, free_shells


def face_types(shape):
    c = Counter()
    for f in sub_shapes(shape, TopAbs_FACE):
        c[TYPES.get(BRepAdaptor_Surface(_face(f)).GetType(), "other")] += 1
    return dict(c)


def area(shape):
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, p)
    return float(p.Mass())


def bbox(shape):
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b)
    lo, hi = b.CornerMin(), b.CornerMax()
    return [lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()]


def face_params(f):
    """Surface type and exact parameters of a face (in assembly coordinates, mm)."""
    ad = BRepAdaptor_Surface(f)
    kind = TYPES.get(ad.GetType(), "other")
    v = lambda d: [d.X(), d.Y(), d.Z()]
    out = {"type": kind}
    if kind == "plane":
        pl = ad.Plane()
        n, o = v(pl.Axis().Direction()), v(pl.Location())
        out.update(normal=n, offset=float(np.dot(n, o)))
    elif kind == "cylinder":
        c = ad.Cylinder()
        out.update(axis=v(c.Axis().Direction()), point=v(c.Location()), radius=c.Radius())
    elif kind == "cone":
        c = ad.Cone()
        out.update(axis=v(c.Axis().Direction()), apex=v(c.Apex()), half_angle=c.SemiAngle(), ref_radius=c.RefRadius())
    elif kind == "sphere":
        s = ad.Sphere()
        out.update(center=v(s.Location()), radius=s.Radius())
    elif kind == "torus":
        tt = ad.Torus()
        out.update(axis=v(tt.Axis().Direction()), center=v(tt.Location()), R=tt.MajorRadius(), r=tt.MinorRadius())
    return out


def triangles(shape, defl, angle=0.3, per_face=True, skip_invalid=False, stats=None, labels=None):
    """Triangles of a shape (each face meshed on its own so one bad face cannot stall the rest).
    Returns (triangles (n,3,3), faces total, faces without triangles). With skip_invalid,
    faces that fail BRepCheck are not meshed (the mesher can hang on them) and are
    counted per surface type in stats["invalid"]."""
    tris, n_faces, failed = [], 0, 0
    if not per_face:
        BRepMesh_IncrementalMesh(shape, defl, False, angle, True)
    trace = os.environ.get("CAD_BENCH_TRACE")
    for f in sub_shapes(shape, TopAbs_FACE):
        f = _face(f)
        n_faces += 1
        if trace:  # the face being meshed, so a mesher hang can be traced to it
            Path(trace).write_text(f"{n_faces - 1} {TYPES.get(BRepAdaptor_Surface(f).GetType(), 'other')}")
        if skip_invalid and not BRepCheck_Analyzer(f).IsValid():
            failed += 1
            if stats is not None:
                stats.setdefault("invalid", Counter())[TYPES.get(BRepAdaptor_Surface(f).GetType(), "other")] += 1
            continue
        try:
            if per_face:
                BRepMesh_IncrementalMesh(f, defl, False, angle, False)
            loc = TopLoc_Location()
            T = BRep_Tool.Triangulation_s(f, loc)
            if T is None or T.NbTriangles() == 0:
                failed += 1
                continue
            tr = loc.Transformation()
            P = np.array([(lambda p: (p.X(), p.Y(), p.Z()))(T.Node(i).Transformed(tr))
                          for i in range(1, T.NbNodes() + 1)])
            idx = np.array([T.Triangle(i).Get() for i in range(1, T.NbTriangles() + 1)]) - 1
            tris.append(P[idx])
            if labels is not None:
                labels.append(np.full(len(idx), n_faces - 1, np.int64))
        except Exception:  # noqa: BLE001
            failed += 1
    return (np.concatenate(tris) if tris else np.zeros((0, 3, 3))), n_faces, failed


def tri_mesh(T):
    return trimesh.Trimesh(T.reshape(-1, 3), np.arange(len(T) * 3).reshape(-1, 3), process=True)


def prep(args):
    shape, status, n = read_step(args.source)
    solids, shells = parts_of(shape)
    parts = solids + shells
    diag = float(np.linalg.norm(np.diff(np.array(bbox(shape)).reshape(2, 3), axis=0)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    info, fine = [], []
    face_rows, fine_face = [], []  # per-face truth: part, type, exact parameters
    for k, p in enumerate(parts):
        name = f"part_{k + 1}"
        T, nf, bad = triangles(p, args.rel * diag, args.angle, per_face=False)
        if len(T) == 0:
            continue
        m = tri_mesh(T)
        scene.add_geometry(m.copy().apply_scale(0.001), node_name=name, geom_name=name)  # glTF is metres
        lab = []
        Tf, _, _ = triangles(p, 0.15 * args.rel * diag, 0.1, per_face=False, labels=lab)
        base = len(face_rows)
        for f in sub_shapes(p, TopAbs_FACE):
            f = _face(f)
            row = face_params(f)
            row.update(part=len(info), area=area(f))
            face_rows.append(row)
        fine.append(Tf if len(Tf) else T)
        fine_face.append(np.concatenate(lab) + base if lab else np.zeros(0, np.int64))
        info.append({"name": name, "kind": "solid" if k < len(solids) else "shell", "faces": face_types(p),
                     "area": area(p), "bbox": bbox(p), "mesh_triangles": int(len(m.faces))})
    name = Path(args.source).stem
    scene.export(out / f"{name}.glb")
    tri_mesh(np.concatenate(fine)).export(out / "truth.stl")
    allT = np.concatenate(fine)
    np.savez_compressed(out / "truth_faces.npz", centroid=allT.mean(axis=1), corners=allT,
                        face=np.concatenate(fine_face))
    (out / "truth_faces.json").write_text(json.dumps(face_rows))
    tot = Counter()
    for i in info:
        tot.update(i["faces"])
    doc = {"source": str(args.source), "diag_mm": diag, "parts": info, "faces_total": dict(tot),
           "area_total": float(sum(i["area"] for i in info)), "solids": len(solids), "shells": len(shells),
           "mesh_triangles": int(sum(i["mesh_triangles"] for i in info))}
    (out / "truth.json").write_text(json.dumps(doc, indent=1))
    print(f"{name}: {len(solids)} solids, {len(shells)} free shells, faces {dict(tot)}, "
          f"{doc['mesh_triangles']:,} triangles in the GLB, size {diag:.1f} mm")


def check(args):
    _memory_guard(float(os.environ.get("CAD_BENCH_MEM_GB", "20")))
    truth = json.loads((Path(args.out) / "truth.json").read_text())
    diag = truth["diag_mm"]
    shape, status, n = read_step(args.ours)
    solids, shells = parts_of(shape)
    invalid_solids = sum(not BRepCheck_Analyzer(s).IsValid() for s in solids)
    invalid_shells = sum(not BRepCheck_Analyzer(s).IsValid() for s in shells)
    stats = {}
    T, nf, failed = triangles(shape, 3e-4 * diag, 0.3, per_face=True, skip_invalid=True, stats=stats)
    ft = face_types(shape)
    print(f"read status {status}, roots {n}; parts: {len(solids)} solids ({invalid_solids} invalid), "
          f"{len(shells)} open shells ({invalid_shells} invalid); truth: {truth['solids']} solids")
    print(f"faces {nf}, failed {failed} (invalid by type {dict(stats.get('invalid', {}))}); "
          f"types ours {ft} | source {truth['faces_total']}")
    report = {"solids": len(solids), "invalid_solids": int(invalid_solids), "open_shells": len(shells),
              "invalid_shells": int(invalid_shells), "true_solids": truth["solids"], "faces": nf, "failed_faces": failed,
              "invalid_by_type": dict(stats.get("invalid", {})), "types": ft}
    if len(T):
        # no vertex merging (not needed for distances, heavy on big assemblies)
        rec = trimesh.Trimesh(T.reshape(-1, 3), np.arange(len(T) * 3).reshape(-1, 3), process=False)
        src = trimesh.load(Path(args.out) / "truth.stl", force="mesh", process=False)
        a, _ = trimesh.sample.sample_surface(src, 150_000, seed=1)
        b, _ = trimesh.sample.sample_surface(rec, 150_000, seed=2)
        me = 0.008 * diag
        d_tr = surface_distance(rec, a, max_edge=me) / diag * 100
        d_rt = surface_distance(src, b, max_edge=me) / diag * 100
        print(f"area {rec.area:.1f} vs {truth['area_total']:.1f} mm2 ({(rec.area - truth['area_total']) / truth['area_total'] * 100:+.1f}%)")
        report.update(area_err_pct=float((rec.area - truth["area_total"]) / truth["area_total"] * 100),
                      source_to_ours_mean_pct=float(d_tr.mean()), source_to_ours_p99_pct=float(np.percentile(d_tr, 99)),
                      missing_pct=float((d_tr > 0.2).mean() * 100), ours_to_source_mean_pct=float(d_rt.mean()),
                      ours_to_source_max_pct=float(d_rt.max()))
        print(f"source->ours mean {d_tr.mean():.4f}% p95 {np.percentile(d_tr, 95):.4f}% p99 {np.percentile(d_tr, 99):.3f}% "
              f"missing(>0.2%) {(d_tr > 0.2).mean() * 100:.2f}% | ours->source mean {d_rt.mean():.4f}% "
              f"p99 {np.percentile(d_rt, 99):.3f}% max {d_rt.max():.2f}%  (size {diag:.1f} mm)")
    (Path(args.ours).parent / "check.json").write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("source")
    p.add_argument("out")
    p.add_argument("--rel", type=float, default=2e-4)
    p.add_argument("--angle", type=float, default=0.25)
    c = sub.add_parser("check")
    c.add_argument("ours")
    c.add_argument("out")
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else check(a)


if __name__ == "__main__":
    main()
