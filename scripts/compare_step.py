"""Dev-only: compare a reconstructed STEP against the source part.

    .cache/devtools/Scripts/python.exe scripts/compare_step.py PARTS_DIR NAME RESULT_DIR

Reports surface types, B-rep validity, area, and the geometric deviation in both
directions (truth -> reconstruction catches missing surface, reconstruction ->
truth catches surface that should not be there), as a fraction of part size.
"""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import (GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane,
                         GeomAbs_Sphere, GeomAbs_Torus)
from OCP.GProp import GProp_GProps
from OCP.STEPControl import STEPControl_Reader
from OCP.StlAPI import StlAPI_Writer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from scipy.spatial import cKDTree

_face = getattr(TopoDS, "Face_s", None) or TopoDS.Face
TYPES = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
         GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus", GeomAbs_BSplineSurface: "bspline"}


def _memory_guard(limit_gb: float = 16.0):
    """OpenCASCADE can run away on huge STEP files: stop this process instead of the machine."""
    import ctypes
    import os
    import threading
    import time
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                   [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                                                   "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                                   "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                   "PagefileUsage", "PeakPagefileUsage")]

    def watch():
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        h = k32.GetCurrentProcess()
        c = PMC()
        c.cb = ctypes.sizeof(c)
        while True:
            psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
            if c.PagefileUsage > limit_gb * 2 ** 30:
                print(f"compare_step: stopped, memory above {limit_gb:.0f} GB", flush=True)
                os._exit(3)
            time.sleep(1)

    if os.name == "nt":
        threading.Thread(target=watch, daemon=True).start()


def stage(msg):
    print(f"  .. {msg}", flush=True)


def read(path):
    r = STEPControl_Reader()
    r.ReadFile(str(path))
    r.TransferRoots()
    return r.OneShape()


def tessellate(shape, defl):
    BRepMesh_IncrementalMesh(shape, defl, False, 0.2, True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.stl"
        StlAPI_Writer().Write(shape, str(p))
        return trimesh.load(p, force="mesh")


def main():
    _memory_guard()
    parts, name, res = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    truth = json.loads((parts / f"{name}_truth.json").read_text())
    # degraded / synthetic inputs are scored against their clean reference surface
    ref = parts / f"{name}_truth.stl"
    src = trimesh.load(ref if ref.exists() else parts / f"{name}.stl", force="mesh")
    diag = float(np.linalg.norm(src.extents))
    stage("reading STEP")
    shape = read(res / f"{name}.step")
    kinds, e = Counter(), TopExp_Explorer(shape, TopAbs_FACE)
    while e.More():
        kinds[TYPES.get(BRepAdaptor_Surface(_face(e.Current())).GetType(), "other")] += 1
        e.Next()
    stage("surface integral")
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    stage("tessellating")
    rec = tessellate(shape, diag * 6e-4)
    # area from the tessellation: OCC's surface integral is unreliable on trimmed
    # surfaces of revolution with spline profiles
    rec_area = float(rec.area)
    stage("sampling and distances")
    a, _ = trimesh.sample.sample_surface_even(src, 40000, seed=1)
    b, _ = trimesh.sample.sample_surface_even(rec, 40000, seed=2)
    dense_rec, _ = trimesh.sample.sample_surface(rec, 400000, seed=3)
    dense_src, _ = trimesh.sample.sample_surface(src, 400000, seed=4)
    d_tr = cKDTree(dense_rec).query(a)[0] / diag * 100   # truth -> reconstruction
    d_rt = cKDTree(dense_src).query(b)[0] / diag * 100   # reconstruction -> truth
    stage("validity check")
    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    report = {
        "truth_faces": truth["faces"], "recon_faces": dict(kinds),
        "valid": valid,
        "area_truth": truth["area"], "area_recon": rec_area, "area_recon_occ_integral": props.Mass(),
        "area_err_pct": (rec_area - truth["area"]) / truth["area"] * 100,
        "truth_to_recon_pct": {"mean": float(d_tr.mean()), "p95": float(np.percentile(d_tr, 95)),
                               "p99": float(np.percentile(d_tr, 99)), "max": float(d_tr.max()),
                               "uncovered_pct": float((d_tr > 1.0).mean() * 100)},
        "recon_to_truth_pct": {"mean": float(d_rt.mean()), "p95": float(np.percentile(d_rt, 95)),
                               "p99": float(np.percentile(d_rt, 99)), "max": float(d_rt.max())},
    }
    (res / "compare.json").write_text(json.dumps(report, indent=1))
    tr, rt = report["truth_to_recon_pct"], report["recon_to_truth_pct"]
    print(f"{name}: valid={report['valid']}  truth {truth['faces']}  ->  recon {dict(kinds)}")
    print(f"   area {rec_area:.1f} vs {truth['area']:.1f} ({report['area_err_pct']:+.2f}%)   "
          f"deviation (% of size) truth->recon mean {tr['mean']:.3f} p95 {tr['p95']:.3f} max {tr['max']:.2f} "
          f"uncovered {tr['uncovered_pct']:.2f}% | recon->truth mean {rt['mean']:.3f} p95 {rt['p95']:.3f} max {rt['max']:.2f}")


if __name__ == "__main__":
    main()
