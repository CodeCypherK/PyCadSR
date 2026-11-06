"""Dev-only (used by assembly_suite.py): load a mesh, segment, export STEP; write pipeline.json.

    runtime/python/python.exe -I scripts/suite_pipeline.py MESH OUT_DIR [--strategy auto]
"""

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from meshtocad import protocol  # noqa: E402
from meshtocad.server import App  # noqa: E402

mesh, out = sys.argv[1], Path(sys.argv[2])
strategy = sys.argv[sys.argv.index("--strategy") + 1] if "--strategy" in sys.argv else "auto"
(out / "work").mkdir(parents=True, exist_ok=True)
info = {"notes": []}


def hook(m):
    if m["type"] == "error":
        print("ERROR", m.get("msg"), m.get("trace", "")[-3000:], flush=True)
        info["error"] = m.get("msg")
    elif m["type"] == "log":
        print(m["msg"], flush=True)
    elif m["type"] == "mesh_loaded":
        info.update(parts=m["stats"].get("parts"), bodies=m["stats"].get("bodies"), triangles=m["stats"]["n_faces"])
    elif m["type"] == "segments_ready":
        info["features"] = sum(1 for r in m["regions"] if r.get("feature"))
    elif m["type"] == "exported":
        st = m["info"].get("step", {})
        info["step"] = {k: v for k, v in st.items() if k != "warnings"}
        info["notes"] = [w for w in st.get("warnings", []) if not w.startswith("region")][:20]


protocol.set_hook(hook)
app = App(out / "work")
app.cmd_load_mesh({"path": mesh})
t = time.perf_counter()
app.cmd_segment({"strategy": strategy})
info["segment_s"] = time.perf_counter() - t
s = app.s
info["regions"] = len(s.regions)
info["kinds"] = dict(Counter(r.prim for r in s.regions))
info["exact"] = sum(1 for r in s.regions if (s.results.get(r.id) or {}).get("status") == "exact")
(out / "session.pkl").write_bytes(pickle.dumps(s))
t = time.perf_counter()
units = "m" if mesh.lower().endswith((".glb", ".gltf")) else "mm"
app.cmd_export({"out_dir": str(out), "name": Path(mesh).stem, "formats": ["step"], "units": units})
info["export_s"] = time.perf_counter() - t
(out / "pipeline.json").write_text(json.dumps(info, indent=1, default=str))
print(json.dumps(info, default=str), flush=True)
