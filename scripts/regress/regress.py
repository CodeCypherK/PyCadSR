"""MeshToCad regression tests: one command to check a backend change before it goes into the app.

    python scripts/regress/regress.py seg   [--backend DIR] [--list all|quick|FILE] [--tag T] [--workers 16] [--labels]
    python scripts/regress/regress.py step  [--backend DIR] [--list quick|FILE] [--tag T] [--lanes 6]
    python scripts/regress/regress.py selftest [--backend DIR]
    python scripts/regress/regress.py compare seg|step TAG_A TAG_B
    python scripts/regress/regress.py baseline seg|step TAG      (make TAG the baseline new runs are compared with)

seg   segments every model (613 real + synthetic parts with known true CAD faces) and scores
      it against the true faces: missing pieces, faces spilling onto other surfaces, wrong
      surface types, extra faces (one true surface split into several), mixed faces, saw-teeth.
step  runs the app's full pipeline (load, segment, export STEP) and opens the STEP like a CAD
      tool: failing faces, open shells, solids, missing area against the source.
Runs land in .cache/regress/TAG; each run is compared with scripts/regress/baselines/.
Python: the app runtime (runtime/python) for seg/selftest; step also needs .cache/devtools.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "runtime/python/python.exe"
DEVPY = ROOT / ".cache/devtools/Scripts/python.exe"
RUNS = ROOT / ".cache/regress"
BASE = HERE / "baselines"
TIMES = ROOT / ".cache/r30/times.json"
SEG_WEIGHTS = {"incomplete": 1.0, "impure": 2.0, "wrong_type": 1.0, "extra": 0.3, "mixed": 1.0, "jagged": 0.02}


def read_list(spec: str, default_dir: Path) -> list[str]:
    p = Path(spec)
    if not p.exists():
        p = default_dir / f"{spec}.txt"
    return [ln.strip().split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


# ----------------------------------------------------------------------------- seg
def run_seg(backend: Path, names: list[str], out: Path, workers: int, labels: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    times = json.loads(TIMES.read_text()) if TIMES.exists() else {}
    lanes = [[] for _ in range(workers)]
    load = [0.0] * workers
    for n in sorted(names, key=lambda n: -times.get(n, 10.0)):
        k = load.index(min(load))
        lanes[k].append(n)
        load[k] += times.get(n, 10.0)
    label_dir = out / "labels"
    if labels:
        label_dir.mkdir(exist_ok=True)
    procs = []
    for k, lane in enumerate(lanes):
        if not lane:
            continue
        f = out / f"seg_w{k}.jsonl"
        f.unlink(missing_ok=True)
        procs.append(subprocess.Popen([str(PY), "-I", str(HERE / "seg_worker.py"), str(backend), str(f),
                                       str(label_dir) if labels else "-"] + lane,
                                      stdout=subprocess.DEVNULL, stderr=open(out / f"seg_w{k}.err", "w")))
    t0 = time.time()
    # generous: other jobs on the machine slow every model down, and a kill loses the lane's rest
    limit = max(1800.0, 12.0 * max(load))
    while any(p.poll() is None for p in procs):
        if time.time() - t0 > limit:
            for p in procs:
                if p.poll() is None:
                    p.kill()
            break
        time.sleep(2.0)
    res = {}
    for f in out.glob("seg_w*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            res[r["name"]] = r
    for n in names:
        res.setdefault(n, {"name": n, "error": "timed out"})
    (out / "seg.json").write_text(json.dumps({"backend": str(backend), "wall_s": round(time.time() - t0, 1),
                                              "results": res}, indent=0))
    return res


def group_of(name: str) -> str:
    if name.startswith("syn_"):
        parts = name.split("_")  # syn_box_1_023_c -> box_c
        return f"{parts[1]}_{parts[-1]}"
    return "real"


def seg_summary(res: dict) -> dict:
    ok = [r for r in res.values() if "error" not in r]
    terms = {k: sum(r[k] for r in ok) / max(len(ok), 1) for k in SEG_WEIGHTS}
    return {"models": len(res), "errors": len(res) - len(ok),
            "score": sum(r["score"] for r in ok) / max(len(ok), 1),
            "weighted": {k: SEG_WEIGHTS[k] * v for k, v in terms.items()}, "terms": terms,
            "faces": sum(r["faces"] for r in ok), "true_faces": sum(r["true_faces"] for r in ok)}


def compare_seg(a: dict, b: dict, label_a="A", label_b="B", show=12) -> int:
    common = sorted(set(a) & set(b))
    worse_total = 0
    print(f"seg: {label_a} -> {label_b}, {len(common)} models in both")
    groups = sorted({group_of(n) for n in common})
    for g in groups:
        ns = [n for n in common if group_of(n) == g and "error" not in a[n] and "error" not in b[n]]
        if not ns:
            continue
        sa = sum(a[n]["score"] for n in ns) / len(ns)
        sb = sum(b[n]["score"] for n in ns) / len(ns)
        ch = [n for n in ns if a[n].get("hash") != b[n].get("hash")]
        better = sum(b[n]["score"] < a[n]["score"] - 1e-9 for n in ch)
        worse = sum(b[n]["score"] > a[n]["score"] + 1e-9 for n in ch)
        flag = "  <-- worse" if sb > sa + 0.002 else ""
        print(f"  {g:12s} n={len(ns):3d} score {sa:.4f} -> {sb:.4f}  changed {len(ch):3d} better {better:3d} worse {worse:3d}{flag}")
    ok = [n for n in common if "error" not in a[n] and "error" not in b[n]]
    sa = seg_summary({n: a[n] for n in ok})
    sb = seg_summary({n: b[n] for n in ok})
    print(f"  overall score {sa['score']:.4f} -> {sb['score']:.4f}; faces {sa['faces']} -> {sb['faces']} (true {sa['true_faces']})")
    print("  weighted terms: " + ", ".join(f"{k} {sa['weighted'][k]:.4f}->{sb['weighted'][k]:.4f}" for k in SEG_WEIGHTS))
    errs = [n for n in common if "error" in b[n] and "error" not in a[n]]
    if errs:
        print(f"  NEW ERRORS ({len(errs)}): " + ", ".join(errs[:10]))
        worse_total += len(errs)
    real = [n for n in ok if group_of(n) == "real"]
    d = sorted(((b[n]["score"] - a[n]["score"], n) for n in real if a[n].get("hash") != b[n].get("hash")))
    if d:
        print("  real models, biggest changes:")
        for dv, n in d[:show]:
            if dv < 0:
                print(f"    better {n:44s} {a[n]['score']:.3f} -> {b[n]['score']:.3f}  faces {a[n]['faces']} -> {b[n]['faces']}")
        for dv, n in d[::-1][:show]:
            if dv > 0:
                print(f"    WORSE  {n:44s} {a[n]['score']:.3f} -> {b[n]['score']:.3f}  faces {a[n]['faces']} -> {b[n]['faces']}")
                worse_total += 1
    return worse_total


# ----------------------------------------------------------------------------- step
def mesh_of(name: str) -> tuple[Path, Path]:
    """(input mesh, truth folder) of a STEP-test model."""
    for bench in (ROOT / ".cache/user_bench", ROOT / ".cache/cad_bench", ROOT / ".cache/cad_bench_synth"):
        d = bench / name
        if d.exists():
            for ext in ("*.glb", "*.fbx", "*.stl"):
                hits = sorted(d.glob(ext))
                if hits:
                    return hits[0], d
    raise FileNotFoundError(name)


def run_step(backend: Path, names: list[str], out: Path, lanes: int, timeout: int = 3600) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    times = json.loads(TIMES.read_text()) if TIMES.exists() else {}
    queue = sorted(names, key=lambda n: -times.get(n, 30.0))
    running: list[tuple[subprocess.Popen, str, float, str]] = []
    res: dict = {}

    def start(n):
        o = out / n
        o.mkdir(exist_ok=True)
        mesh, truth = mesh_of(n)
        step = o / f"{mesh.stem}.step"
        cmd = (f'"{PY}" -I "{HERE / "pipeline.py"}" "{backend}" "{mesh}" "{o}" > "{o / "pipeline.log"}" 2>&1 && '
               f'"{DEVPY}" -I "{ROOT / "scripts/cad_bench.py"}" check "{step}" "{truth}" > "{o / "check.log"}" 2>&1')
        p = subprocess.Popen(cmd, shell=True, env={**__import__("os").environ, "CAD_BENCH_MEM_GB": "6"})
        running.append((p, n, time.time(), str(o)))

    def drain(lanes_now: int):
        while queue or running:
            while queue and len(running) < lanes_now:
                start(queue.pop(0))
            time.sleep(2.0)
            for item in list(running):
                p, n, t0, o = item
                if p.poll() is None and time.time() - t0 < timeout:
                    continue
                if p.poll() is None:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                running.remove(item)
                cj = Path(o) / "check.json"
                r = json.loads(cj.read_text()) if cj.exists() else {"error": "no check (pipeline or checker failed / timed out)"}
                r["seconds"] = round(time.time() - t0, 1)
                res[n] = r
                print(f"  {n:44s} " + (f"failing {r['failed_faces']} open {r.get('open_shells')} solids {r.get('solids')}/"
                                       f"{r.get('true_solids')} missing {r.get('missing_pct', 0):.3f}" if "error" not in r else r["error"]),
                      flush=True)

    drain(lanes)
    # a busy machine can starve a model into a timeout or a silent exit: give those one more
    # run on their own before calling them failures
    again = [n for n, r in res.items() if "error" in r]
    if again:
        print(f"  re-running {len(again)} model(s) with no result: {', '.join(again)}", flush=True)
        for n in again:
            for f in ("check.json", "check.log"):
                (out / n / f).unlink(missing_ok=True)
        queue.extend(again)
        drain(min(lanes, 3))
    (out / "step.json").write_text(json.dumps({"backend": str(backend), "results": res}, indent=0))
    return res


def compare_step(a: dict, b: dict, label_a="A", label_b="B") -> int:
    common = sorted(set(a) & set(b))
    g = lambda r, k: r.get(k, 0) or 0  # noqa: E731
    ok = [n for n in common if "error" not in a[n] and "error" not in b[n]]
    print(f"step: {label_a} -> {label_b}, {len(ok)} models checked in both")
    for kind, sel in (("real", lambda n: not n.startswith("syn_")), ("synthetic", lambda n: n.startswith("syn_"))):
        ns = [n for n in ok if sel(n)]
        if not ns:
            continue
        tot = lambda r, k: sum(g(r[n], k) for n in ns)  # noqa: E731
        print(f"  {kind:9s} n={len(ns):3d} failing {tot(a, 'failed_faces')} -> {tot(b, 'failed_faces')}, "
              f"open shells {tot(a, 'open_shells')} -> {tot(b, 'open_shells')}, solids {tot(a, 'solids')} -> {tot(b, 'solids')}, "
              f"missing mean {tot(a, 'missing_pct') / len(ns):.3f}% -> {tot(b, 'missing_pct') / len(ns):.3f}%, "
              f"faces {tot(a, 'faces')} -> {tot(b, 'faces')}")
    worse = 0
    for n in ok:
        fa, fb = g(a[n], "failed_faces") + g(a[n], "open_shells"), g(b[n], "failed_faces") + g(b[n], "open_shells")
        ma, mb = g(a[n], "missing_pct"), g(b[n], "missing_pct")
        if fb > fa or mb > ma + 0.05:
            worse += 1
            print(f"    WORSE  {n:44s} failing+open {fa} -> {fb}  missing {ma:.3f} -> {mb:.3f}")
        elif fb < fa or mb < ma - 0.05:
            print(f"    better {n:44s} failing+open {fa} -> {fb}  missing {ma:.3f} -> {mb:.3f}")
    lost = [n for n in common if "error" in b[n] and "error" not in a[n]]
    for n in lost:
        worse += 1
        print(f"    NO RESULT {n}: {b[n]['error']}")
    return worse


# ----------------------------------------------------------------------------- main
def load_run(suite: str, tag: str) -> dict:
    p = RUNS / tag / f"{suite}.json"
    if not p.exists():
        p = BASE / f"{suite}.json" if tag == "baseline" else p
    return json.loads(p.read_text())["results"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["seg", "step", "selftest", "compare", "baseline"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--backend", default=str(ROOT / "backend"))
    ap.add_argument("--list", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lanes", type=int, default=6)
    ap.add_argument("--labels", action="store_true", help="seg: keep each model's faces for analysis")
    a = ap.parse_args()
    backend = Path(a.backend).resolve()
    tag = a.tag or time.strftime("%Y%m%d_%H%M%S")
    if a.cmd == "seg":
        names = read_list(a.list or "all", HERE)
        print(f"seg: {len(names)} models, backend {backend}, run .cache/regress/{tag}")
        res = run_seg(backend, names, RUNS / tag, a.workers, a.labels)
        s = seg_summary(res)
        print(f"  score {s['score']:.4f} over {s['models']} models ({s['errors']} errors); "
              + ", ".join(f"{k} {v:.4f}" for k, v in s["weighted"].items()))
        if (BASE / "seg.json").exists():
            sys.exit(1 if compare_seg(load_run("seg", "baseline"), res, "baseline", tag) else 0)
    elif a.cmd == "step":
        names = read_list(a.list or "step_quick", HERE)
        print(f"step: {len(names)} models, backend {backend}, run .cache/regress/{tag}")
        res = run_step(backend, names, RUNS / tag, a.lanes)
        if (BASE / "step.json").exists():
            sys.exit(1 if compare_step(load_run("step", "baseline"), res, "baseline", tag) else 0)
    elif a.cmd == "selftest":
        r = subprocess.run([str(PY), "-m", "meshtocad.selftest", "--no-sr"], cwd=backend, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or ["(no output)"]
        print(tail[0])
        sys.exit(0 if "SELFTEST PASSED" in r.stdout else 1)
    elif a.cmd == "compare":
        suite, ta, tb = a.args
        fn = compare_seg if suite == "seg" else compare_step
        fn(load_run(suite, ta), load_run(suite, tb), ta, tb)
    elif a.cmd == "baseline":
        suite, t = a.args
        BASE.mkdir(exist_ok=True)
        shutil.copy(RUNS / t / f"{suite}.json", BASE / f"{suite}.json")
        print(f"baseline {suite} = run {t}")


if __name__ == "__main__":
    main()
