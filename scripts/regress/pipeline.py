"""The app's full pipeline on one mesh, as the desktop app runs it: load, segment, export STEP.
    runtime python -I pipeline.py BACKEND MESH OUT_DIR   -> OUT_DIR/<mesh stem>.step + pipeline.json"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from meshtocad import protocol  # noqa: E402
from meshtocad.server import App  # noqa: E402

mesh, out = sys.argv[2], Path(sys.argv[3])
(out / "work").mkdir(parents=True, exist_ok=True)
info = {"mesh": mesh}


def hook(m):
    if m["type"] == "error":
        print("ERROR", m.get("msg"), m.get("trace", "")[-3000:], flush=True)
        info["error"] = m.get("msg")
    elif m["type"] == "log":
        print(m["msg"], flush=True)
    elif m["type"] == "exported":
        st = m["info"].get("step", {})
        info["step"] = {k: v for k, v in st.items() if k != "warnings"}
        info["warnings"] = st.get("warnings", [])


protocol.set_hook(hook)
app = App(out / "work")
t = time.perf_counter()
app.cmd_load_mesh({"path": mesh})
app.cmd_segment({"strategy": "auto"})
info["segment_s"] = round(time.perf_counter() - t, 1)
info["regions"] = len(app.s.regions)
info["kinds"] = dict(Counter(r.prim for r in app.s.regions))
t = time.perf_counter()
units = "m" if mesh.lower().endswith((".glb", ".gltf")) else "mm"
app.cmd_export({"out_dir": str(out), "name": Path(mesh).stem, "formats": ["step"], "units": units})
info["export_s"] = round(time.perf_counter() - t, 1)
(out / "pipeline.json").write_text(json.dumps(info, indent=1, default=str))
