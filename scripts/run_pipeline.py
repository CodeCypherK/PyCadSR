"""Run the whole MeshToCad pipeline headless on one mesh (dev/benchmark).

    runtime/python/python.exe scripts/run_pipeline.py --mesh part.stl --out DIR [--iterations 60]

Writes DIR/<name>.step etc. plus DIR/report.json with per-region results and timings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from meshtocad import protocol  # noqa: E402
from meshtocad.server import App  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--points", type=int, default=60000)
    ap.add_argument("--tol", type=float, default=0.005, help="segmentation tolerance, fraction of size")
    ap.add_argument("--angle", type=float, default=30)
    args = ap.parse_args()

    mesh = Path(args.mesh)
    out = Path(args.out)
    (out / "work").mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    iters: dict[int, int] = {}

    def hook(msg):
        events.append(msg)
        if msg["type"] == "sr_iter":
            iters[msg["region"]] = msg["iteration"]
        elif msg["type"] == "error":
            print("ERROR", msg.get("msg"), msg.get("trace", "")[-1500:], file=sys.stderr)

    protocol.set_hook(hook)
    app = App(out / "work")
    app.worker.start()
    t = {}
    t0 = time.perf_counter()
    app.cmd_load_mesh({"path": str(mesh)})
    app.cmd_sample({"n_points": args.points})
    t["sample"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    app.cmd_segment({"tol": args.tol, "angle_deg": args.angle})
    t["segment"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    app.cmd_discover({"settings": {"iterations": args.iterations, "tolerance": 0.004}})
    t["discover"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    app.cmd_export({"out_dir": str(out), "name": mesh.stem, "formats": ["step", "stl", "json"]})
    t["export"] = time.perf_counter() - t1
    app.worker.shutdown()

    seg = next(e for e in events if e["type"] == "segments_ready")
    done = {e["region"]: e for e in events if e["type"] == "sr_done"}
    exp = next((e for e in events if e["type"] == "exported"), {})
    regions = []
    for r in seg["regions"]:
        d = done.get(r["id"], {})
        sm = d.get("summary", {})
        regions.append({"id": r["id"], "prim": r["prim"], "n_points": r["n_points"], "frame": r["frame"]["kind"],
                        "cad": (sm.get("classification") or {}).get("type"), "status": d.get("reason"),
                        "iterations": iters.get(r["id"]), "rms": sm.get("rms"), "equation": sm.get("world")})
    report = {"mesh": str(mesh), "timings": t, "regions": regions, "export": exp.get("info"),
              "errors": [e.get("msg") for e in events if e["type"] == "error"]}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{mesh.stem}: {len(regions)} regions  "
          + "  ".join(f"{k} {v:.1f}s" for k, v in t.items()))
    for r in regions:
        rms = f"{r['rms']:.4f}" if r["rms"] is not None else "-"
        print(f"  R{r['id']:<3d} {r['prim']:9s}->{str(r['cad']):9s} n={r['n_points']:6d} it={r['iterations']} "
              f"rms={rms:8s} {str(r['equation'])[:90]}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
