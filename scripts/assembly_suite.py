"""Dev-only: run the whole pipeline on every prepared CAD assembly and summarise.

    runtime/python/python.exe scripts/assembly_suite.py [names...] [--skip-step]

Expects `.cache/cad_bench/<name>/` from `scripts/cad_bench.py prep`. For each
assembly: segment (app pipeline, no symbolic regression) and export STEP, score
the surfaces against the true CAD faces (`seg_eval.py`), and open the STEP like a
CAD kernel (`cad_bench.py check`, OpenCASCADE from .cache/devtools). Results go to
`.cache/cad_run/<name>/` and a table to `.cache/cad_run/summary.md`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime" / "python" / "python.exe"
DEVTOOLS = ROOT / ".cache" / "devtools" / "Scripts" / "python.exe"
BENCH = ROOT / ".cache" / "cad_bench"
RUN = ROOT / ".cache" / "cad_run"
PIPE = Path(__file__).resolve().parent / "suite_pipeline.py"


def run(cmd, log, timeout):
    t = time.perf_counter()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        try:
            r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=ROOT)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return code, time.perf_counter() - t


def main():
    tag = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tag=")), "")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip_step = "--skip-step" in sys.argv
    names = args or sorted(p.name for p in BENCH.iterdir() if (p / "truth.json").exists())
    rows = []
    for name in names:
        glb = next((BENCH / name).glob("*.glb"))
        out = RUN / name
        out.mkdir(parents=True, exist_ok=True)
        for f in ("seg_eval.json", "check.json", "pipeline.json"):
            (out / f).unlink(missing_ok=True)
        print(f"== {name}", flush=True)
        code, t_pipe = run([RUNTIME, "-I", PIPE, glb, out], out / "pipeline.log", 3600)
        print(f"   pipeline {code} in {t_pipe:.0f}s", flush=True)
        code_e, _ = run([RUNTIME, "-I", ROOT / "scripts" / "seg_eval.py", out / "session.pkl", BENCH / name],
                        out / "seg_eval.log", 1800)
        code_c, t_chk = (None, 0)
        if not skip_step and (out / f"{glb.stem}.step").exists():
            code_c, t_chk = run([DEVTOOLS, "-I", ROOT / "scripts" / "cad_bench.py", "check", out / f"{glb.stem}.step",
                                 BENCH / name], out / "check.log", 3600)
        load = lambda f: json.loads((out / f).read_text()) if (out / f).exists() else {}
        rows.append({"name": name, "pipeline": load("pipeline.json"), "seg": load("seg_eval.json"),
                     "step": load("check.json"), "codes": [code, code_e, code_c]})
        print(f"   seg {code_e}, step check {code_c} ({t_chk:.0f}s)", flush=True)
    # table
    lines = ["| assembly | parts | true faces | our surfaces | exact eq. | segment s | type ok | split faces | mixed regions |"
             " STEP solids/parts | STEP faces failing | area err | mean dev | missing |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p, s, c = r["pipeline"], r["seg"], r["step"]
        f = lambda d, k, fmt="{:.1f}": (fmt.format(d[k]) if k in d and d[k] is not None else "-")
        lines.append(
            f"| {r['name']} | {p.get('parts', '-')} | {s.get('true_faces', '-')} | {s.get('regions', '-')} | "
            f"{p.get('exact', '-')} | {f(p, 'segment_s', '{:.0f}')} | {f(s, 'type_accuracy_pct')}% | "
            f"{f(s, 'over_split_faces_pct')}% | {s.get('merged_regions', '-')} | "
            f"{c.get('solids', '-')}/{p.get('parts', '-')} | {c.get('failed_faces', '-')}/{c.get('faces', '-')} | "
            f"{f(c, 'area_err_pct')}% | {f(c, 'source_to_ours_mean_pct', '{:.4f}')}% | {f(c, 'missing_pct', '{:.2f}')}% |")
    # totals: face-weighted averages over everything that ran
    seg = [r["seg"] for r in rows if r["seg"]]
    stp = [r["step"] for r in rows if r["step"]]
    pip = [r["pipeline"] for r in rows if r["pipeline"]]
    if seg:
        w = np.array([s_["true_faces"] for s_ in seg], float)
        acc = float(np.average([s_["type_accuracy_pct"] for s_ in seg], weights=w))
        split = float(np.average([s_["over_split_faces_pct"] for s_ in seg], weights=w))
        mixed = sum(s_["merged_regions"] for s_ in seg)
        solids = sum(c.get("solids", 0) for c in stp)
        parts = sum(p.get("parts", 0) or 0 for p in pip)
        failing = sum(c.get("failed_faces", 0) for c in stp)
        faces = sum(c.get("faces", 0) for c in stp)
        miss = float(np.mean([c["missing_pct"] for c in stp if "missing_pct" in c])) if stp else float("nan")
        errors = sum(1 for r in rows if r["codes"][0] != 0 or "error" in r["pipeline"])
        lines.append(f"| **all {len(rows)}** ({errors} errors) | {parts} | {int(w.sum())} | "
                     f"{sum(s_['regions'] for s_ in seg)} | - | - | {acc:.1f}% | {split:.1f}% | {mixed} | "
                     f"{solids}/{parts} | {failing}/{faces} | - | - | {miss:.2f}% |")
    suffix = f"_{tag}" if tag else ""
    (RUN / f"summary{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RUN / f"summary{suffix}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
