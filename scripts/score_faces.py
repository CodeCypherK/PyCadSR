"""Dev-only: surface accuracy of a STEP file, healing and meshing each face on its own.

    .cache/devtools/Scripts/python.exe scripts/score_faces.py TRUTH_MESH FILE.step [--scale 0.001]

For big exports where whole-model shape healing fails: every face is repaired
and triangulated separately (as a tolerant CAD importer would), then the result
is compared with the source surface in both directions. --scale converts the
STEP's length unit back to the mesh's (OpenCASCADE reads lengths in mm).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from compare_step import _face, _memory_guard  # noqa: E402

from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: E402
from OCP.ShapeFix import ShapeFix_Shape  # noqa: E402
from OCP.STEPControl import STEPControl_Reader  # noqa: E402
from OCP.TopAbs import TopAbs_FACE  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopLoc import TopLoc_Location  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("step")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    _memory_guard(14.0)
    truth = trimesh.load(args.truth, force="mesh")
    diag = float(np.linalg.norm(truth.extents))
    r = STEPControl_Reader()
    r.ReadFile(args.step)
    r.TransferRoots()
    shape = r.OneShape()
    defl = diag * 6e-4 / args.scale
    tris, n_faces, n_meshed, n_fixfail = [], 0, 0, 0
    e = TopExp_Explorer(shape, TopAbs_FACE)
    while e.More():
        f = _face(e.Current())
        n_faces += 1
        try:
            fx = ShapeFix_Shape(f)
            fx.Perform()
            g = fx.Shape()
        except Exception:  # noqa: BLE001
            n_fixfail += 1
            g = f
        try:
            BRepMesh_IncrementalMesh(g, defl, False, 0.3, False)
            ex = TopExp_Explorer(g, TopAbs_FACE)
            while ex.More():
                ff = _face(ex.Current())
                loc = TopLoc_Location()
                T = BRep_Tool.Triangulation_s(ff, loc)
                if T is not None and T.NbTriangles() > 0:
                    tr = loc.Transformation()
                    P = np.array([[*(lambda p: (p.X(), p.Y(), p.Z()))(T.Node(i).Transformed(tr))]
                                  for i in range(1, T.NbNodes() + 1)])
                    idx = np.array([T.Triangle(i).Get() for i in range(1, T.NbTriangles() + 1)]) - 1
                    tris.append(P[idx] * args.scale)
                    n_meshed += 1
                ex.Next()
        except Exception:  # noqa: BLE001
            pass
        if n_faces % 10000 == 0:
            print(f"  .. {n_faces} faces", flush=True)
        e.Next()
    T = np.concatenate(tris)
    rec = trimesh.Trimesh(T.reshape(-1, 3), np.arange(len(T) * 3).reshape(-1, 3), process=False)
    a, _ = trimesh.sample.sample_surface(truth, 400_000, seed=1)
    b, _ = trimesh.sample.sample_surface(rec, 400_000, seed=2)
    d_tr = cKDTree(b).query(a)[0] / diag * 100
    d_rt = cKDTree(a).query(b)[0] / diag * 100
    print(f"faces {n_faces}, meshed {n_meshed}, healing threw on {n_fixfail}")
    print(f"area {rec.area:.1f} vs {truth.area:.1f} ({(rec.area - truth.area) / truth.area * 100:+.1f}%)")
    print(f"truth->recon mean {d_tr.mean():.3f}% p95 {np.percentile(d_tr, 95):.3f}% "
          f"uncovered(>0.5%) {(d_tr > 0.5).mean() * 100:.2f}% | recon->truth mean {d_rt.mean():.3f}% "
          f"p95 {np.percentile(d_rt, 95):.3f}% max {d_rt.max():.2f}%  (% of size {diag:.3g})")


if __name__ == "__main__":
    main()
