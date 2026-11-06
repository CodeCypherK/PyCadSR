"""Headless end-to-end test:  ``python -m meshtocad.selftest [--no-sr] [--out DIR]``

Builds a mesh of known shapes, runs sampling -> segmentation -> symbolic
regression -> export, and checks the discovered surface types.
Also used by ``scripts/build_runtime.py`` to warm up the bundled Julia.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh

from . import protocol


def make_test_mesh() -> trimesh.Trimesh:
    parts = []
    cyl = trimesh.creation.cylinder(radius=10, height=30, sections=72)
    parts.append(cyl)
    cap = trimesh.creation.capsule(height=24, radius=7, count=[48, 48])
    cap.apply_translation([40, 0, 0])
    parts.append(cap)
    cone = trimesh.creation.cone(radius=12, height=22, sections=72)
    cone.apply_translation([0, 45, -11])
    parts.append(cone)
    box = trimesh.creation.box(extents=[20, 14, 10])
    box.apply_translation([45, 45, 0])
    parts.append(box)
    sph = trimesh.creation.icosphere(subdivisions=4, radius=9)
    sph.apply_translation([-40, 0, 0])
    parts.append(sph)
    # freeform sheet z = 3 sin(x/6) cos(y/6)
    n = 60
    xs, ys = np.meshgrid(np.linspace(-15, 15, n), np.linspace(-15, 15, n), indexing="ij")
    zs = 3 * np.sin(xs / 6) * np.cos(ys / 6)
    V = np.c_[xs.ravel(), ys.ravel(), zs.ravel()] + [-40, 45, 0]
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel()
    F = np.r_[np.c_[a, a + n, a + n + 1], np.c_[a, a + n + 1, a + 1]]
    parts.append(trimesh.Trimesh(V, F))
    return trimesh.util.concatenate(parts)


def make_bottle(nt: int = 120, nz: int = 90) -> trimesh.Trimesh:
    """A freeform bottle (mm): superelliptic body swelling with height, a shoulder into a round
    neck with an open mouth, a shallow domed bottom; mirror-symmetric."""
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)

    def section(z):
        a = 62 + 8 * np.sin(np.pi * np.clip(z / 200, 0, 1))
        b = 38 + 6 * np.sin(np.pi * np.clip(z / 200, 0, 1))
        r = (np.abs(np.cos(th) / a) ** 3 + np.abs(np.sin(th) / b) ** 3) ** (-1 / 3)
        w = np.clip((z - 200) / 40, 0, 1)
        w = w * w * (3 - 2 * w)
        return (1 - w) * r + w * 15.0
    zs = np.r_[np.linspace(6, 200, int(nz * 0.55), endpoint=False), np.linspace(200, 240, int(nz * 0.3), endpoint=False),
               np.linspace(240, 270, int(nz * 0.15))]
    rim = section(6.0)
    rings = []
    for k in range(1, 30):
        s_ = k / 30
        if s_ < 0.9:
            rr, z = s_ / 0.9 * (rim - 6.0), 3.0 * (1 - (s_ / 0.9) ** 2)
        else:
            u = (s_ - 0.9) / 0.1 * (np.pi / 2)
            rr, z = (rim - 6.0) + 6.0 * np.sin(u), 6.0 - 6.0 * np.cos(u)
        rings.append(np.c_[rr * np.cos(th), rr * np.sin(th), np.full(nt, z)])
    rings += [np.c_[section(z) * np.cos(th), section(z) * np.sin(th), np.full(nt, z)] for z in zs]
    V = np.vstack([[[0.0, 0.0, 3.0]]] + rings)
    F = [(0, 1 + i, 1 + (i + 1) % nt) for i in range(nt)]
    for r in range(len(rings) - 1):
        o0, o1 = 1 + r * nt, 1 + (r + 1) * nt
        for i in range(nt):
            j = (i + 1) % nt
            F += [(o0 + i, o1 + i, o1 + j), (o0 + i, o1 + j, o0 + j)]
    m = trimesh.Trimesh(V, np.array(F), process=False)
    m.fix_normals()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sr", action="store_true", help="skip symbolic regression")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--points", type=int, default=40000)
    args = ap.parse_args()

    from .server import App

    events: list[dict] = []

    def hook(msg):
        events.append(msg)
        t = msg["type"]
        if t in ("julia_status", "log", "error", "progress"):
            print(f"  [{t}] {msg.get('state', '')} {msg.get('msg', '')}", file=sys.stderr, flush=True)
        elif t == "sr_iter":
            hof = msg["hof"][msg["selected"]]
            print(f"  [sr_iter] region {msg['region']} it {msg['iteration']} "
                  f"loss={hof['loss']:.3g} c={hof['complexity']} eq={hof['equation']} "
                  f"({msg['dt']:.2f}s)", file=sys.stderr, flush=True)
        elif t == "sr_done":
            sm = msg["summary"]
            print(f"  [sr_done] region {msg['region']} {msg['reason']}: {sm.get('world')} "
                  f"-> {sm.get('classification', {}).get('type')}", file=sys.stderr, flush=True)

    protocol.set_hook(hook)
    out = Path(args.out or tempfile.mkdtemp(prefix="meshtocad_selftest_"))
    out.mkdir(parents=True, exist_ok=True)
    mesh_path = out / "test_shapes.stl"
    make_test_mesh().export(mesh_path)

    app = App(out / "work")
    (out / "work").mkdir(exist_ok=True)
    if not args.no_sr:
        app.worker.start()
    t0 = time.perf_counter()
    app.cmd_load_mesh({"path": str(mesh_path)})
    app.cmd_sample({"n_points": args.points})
    app.cmd_segment({})
    regs = next(e for e in events if e["type"] == "segments_ready")["regions"]
    print(f"segmentation: {len(regs)} regions in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    for r in regs:
        print(f"  region {r['id']:2d}: {r['prim']:9s} n={r['n_points']:6d} frame={r['frame']['kind']}",
              file=sys.stderr)
    kinds = sorted(r["prim"] for r in regs)
    expected = {"plane": 9, "cylinder": 2, "sphere": 3, "cone": 1, "freeform": 1}
    ok = True
    for k, n in expected.items():
        got = kinds.count(k)
        flag = "ok" if got >= n else "MISSING"
        ok &= got >= n
        print(f"  expect >= {n} {k}: got {got} [{flag}]", file=sys.stderr)

    if not args.no_sr:
        app.cmd_discover({"settings": {"iterations": args.iterations, "tolerance": 0.004}})
        done = [e for e in events if e["type"] == "sr_done"]
        print(f"symbolic regression: {len(done)} regions", file=sys.stderr)
        errs = [e for e in events if e["type"] == "error"]
        ok &= not errs and len(done) == len(regs)
    app.cmd_export({"out_dir": str(out / "export"), "name": "test_shapes",
                    "formats": ["step", "stl", "json"]})
    exp = next((e for e in events if e["type"] == "exported"), None)
    print(f"export: {exp and exp['files']}", file=sys.stderr)
    print(f"export info: {exp and exp['info']}", file=sys.stderr)
    ok &= exp is not None
    # curve-built faces (curvenet) on a freeform bottle: mirror, neck, trimmed B-spline faces, STEP
    try:
        from . import curvenet
        bm = make_bottle()
        t1 = time.perf_counter()
        cn = curvenet.build(bm.vertices, bm.faces, None, 1.0, log=lambda m: None)
        (out / "export").mkdir(parents=True, exist_ok=True)
        info = curvenet.write_step(cn, out / "export" / "bottle.step", "bottle", "mm")
        good = info["faces"] >= 4 and cn["brep"]["loop_problems"] == 0
        print(f"curve-built faces: {info['faces']} faces, worst fit {cn['brep']['max_fit']:.3f} mm, "
              f"{time.perf_counter() - t1:.1f}s [{'ok' if good else 'BAD'}]", file=sys.stderr)
        ok &= good
    except Exception as exc:
        print(f"curve-built faces FAILED: {exc}", file=sys.stderr)
        ok = False
    app.worker.shutdown()
    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}  ({time.perf_counter() - t0:.1f}s)  output: {out}",
          file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
