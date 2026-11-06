"""Dev-only: tessellate every face of a STEP file on its own and report the failures.

    .cache/devtools/Scripts/python.exe scripts/check_faces.py FILE.step [deflection]

Prints, per surface type, how many faces fail to triangulate or are invalid, and
the slowest faces, so a broken export can be traced to the faces that cause it.
"""

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_step import TYPES, _face, _memory_guard  # noqa: E402

from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Surface  # noqa: E402
from OCP.BRepCheck import BRepCheck_Analyzer  # noqa: E402
from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: E402
from OCP.STEPControl import STEPControl_Reader  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopLoc import TopLoc_Location  # noqa: E402


def count(shape, kind):
    n, e = 0, TopExp_Explorer(shape, kind)
    while e.More():
        n += 1
        e.Next()
    return n


def main():
    _memory_guard(12.0)
    path = sys.argv[1]
    defl = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    r = STEPControl_Reader()
    r.ReadFile(path)
    r.TransferRoots()
    shape = r.OneShape()
    stats = defaultdict(Counter)
    slow = []
    examples = defaultdict(list)
    e = TopExp_Explorer(shape, TopAbs_FACE)
    i = 0
    t_all = time.perf_counter()
    while e.More():
        f = _face(e.Current())
        kind = TYPES.get(BRepAdaptor_Surface(f).GetType(), "other")
        n_edges, n_wires = count(f, TopAbs_EDGE), count(f, TopAbs_WIRE)
        t = time.perf_counter()
        BRepMesh_IncrementalMesh(f, defl, False, 0.3, False)
        dt = time.perf_counter() - t
        tri = BRep_Tool.Triangulation_s(f, TopLoc_Location())
        ok_tri = tri is not None and tri.NbTriangles() > 0
        valid = BRepCheck_Analyzer(f).IsValid()
        st = stats[kind]
        st["faces"] += 1
        st["no_triangles"] += not ok_tri
        st["invalid"] += not valid
        if not ok_tri or not valid:
            key = ("no_tri" if not ok_tri else "") + ("+invalid" if not valid else "")
            if len(examples[(kind, key)]) < 5:
                examples[(kind, key)].append((i, n_wires, n_edges))
        slow.append((dt, i, kind, n_wires, n_edges))
        i += 1
        if i % 5000 == 0:
            print(f"  .. {i} faces, {time.perf_counter() - t_all:.0f}s", flush=True)
        e.Next()
    print(f"{i} faces in {time.perf_counter() - t_all:.0f}s")
    for kind, st in sorted(stats.items(), key=lambda kv: -kv[1]["faces"]):
        print(f"  {kind:9s} faces {st['faces']:6d}  no triangles {st['no_triangles']:6d}  invalid {st['invalid']:6d}")
    print("examples (face index, wires, edges):")
    for (kind, key), ex in examples.items():
        print(f"  {kind} {key}: {ex}")
    slow.sort(reverse=True)
    print("slowest:", [(round(d, 2), idx, k, w, n) for d, idx, k, w, n in slow[:8]])


if __name__ == "__main__":
    main()
