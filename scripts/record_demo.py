"""Record a real backend session for the browser preview (dev only).

    runtime/python/python.exe scripts/record_demo.py [--mesh file.stl] [--iterations 30]

Runs the actual pipeline (including PySR) and stores every event, grouped by
the command that produced it, in app/public/demo/events.json together with the
binary files the UI reads. `npm run dev` in a normal browser then replays it.
The app build never includes this folder (vite copyPublicDir: false).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from meshtocad import protocol  # noqa: E402
from meshtocad.selftest import make_test_mesh  # noqa: E402
from meshtocad.server import App  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--iterations", type=int, default=30)
    args = ap.parse_args()

    out = ROOT / "app" / "public" / "demo"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    work = ROOT / ".cache" / "demo_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    mesh = Path(args.mesh) if args.mesh else work / "test_shapes.stl"
    if not args.mesh:
        make_test_mesh().export(mesh)

    groups: dict[str, list] = {"_boot": []}
    state = {"cmd": "_boot", "t": time.time()}

    def hook(msg: dict) -> None:
        now = time.time()
        dt, state["t"] = now - state["t"], now
        files = msg.get("files")
        if isinstance(files, dict) and msg["type"] != "exported":
            for k, p in files.items():
                shutil.copy(p, out / Path(p).name)
                files[k] = f"/demo/{Path(p).name}"
        groups.setdefault(state["cmd"], []).append({"dt": round(dt, 3), "ev": msg})
        if msg["type"] in ("julia_status", "sr_done", "segments_ready"):
            print(f"[{state['cmd']}] {msg['type']} {msg.get('state', '')} {msg.get('region', '')}", flush=True)

    protocol.set_hook(hook)
    app = App(work)
    app.worker.start()
    while app.worker.state not in ("ready", "error"):
        time.sleep(0.2)
    steps = [
        ("load_mesh", {"path": str(mesh)}),
        ("sample", {"n_points": 40000}),
        ("segment", {}),
        ("discover", {"settings": {"iterations": args.iterations, "tolerance": 0.004}}),
        ("export", {"out_dir": str(work / "export"), "name": mesh.stem, "formats": ["step", "stl", "json"]}),
    ]
    for cmd, a in steps:
        state["cmd"], state["t"] = cmd, time.time()
        getattr(app, f"cmd_{cmd}")(a)
    app.worker.shutdown()
    (out / "events.json").write_text(json.dumps(groups, separators=(",", ":")))
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"recorded {sum(len(v) for v in groups.values())} events, {size:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
