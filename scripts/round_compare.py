"""Dev-only: per-model comparison of two suite run folders (round A vs round B) -> JSON + table.

    runtime/python/python.exe scripts/round_compare.py RUN_A RUN_B OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np


def load(d: Path):
    rows = {}
    for m in sorted(p for p in d.iterdir() if p.is_dir() and (p / "seg_eval.json").exists()):
        rd = lambda f: json.loads((m / f).read_text()) if (m / f).exists() else {}
        s, c, p = rd("seg_eval.json"), rd("check.json"), rd("pipeline.json")
        rows[m.name] = {
            "true_faces": s.get("true_faces"), "surfaces": s.get("regions"), "type_pct": s.get("type_accuracy_pct"),
            "split_pct": s.get("over_split_faces_pct"), "mixed": s.get("merged_regions"), "parts": p.get("parts"),
            "solids": c.get("solids"), "failed": c.get("failed_faces"), "faces": c.get("faces"),
            "missing_pct": c.get("missing_pct"), "area_err_pct": c.get("area_err_pct"), "segment_s": p.get("segment_s"),
        }
    return rows


def totals(rows, names):
    seg = [rows[n] for n in names if rows[n]["type_pct"] is not None]
    w = np.array([r["true_faces"] for r in seg], float)
    stp = [rows[n] for n in names if rows[n]["faces"] is not None]
    return {
        "models": len(names), "true_faces": int(w.sum()), "surfaces": int(sum(r["surfaces"] for r in seg)),
        "type_pct": float(np.average([r["type_pct"] for r in seg], weights=w)),
        "split_pct": float(np.average([r["split_pct"] for r in seg], weights=w)),
        "solids": int(sum(r["solids"] or 0 for r in stp)), "parts": int(sum(rows[n]["parts"] or 0 for n in names)),
        "failed": int(sum(r["failed"] or 0 for r in stp)), "faces": int(sum(r["faces"] or 0 for r in stp)),
        "missing_pct": float(np.mean([r["missing_pct"] for r in stp if r["missing_pct"] is not None])),
    }


def main():
    a, b = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    names = sorted(set(a) & set(b))
    out = {"names": names, "a": {n: a[n] for n in names}, "b": {n: b[n] for n in names},
           "totals_a": totals(a, names), "totals_b": totals(b, names)}
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
    print(json.dumps({"a": out["totals_a"], "b": out["totals_b"]}, indent=1))
    print(f"{'model':40} {'type% a->b':>14} {'surf a->b':>12} {'fail a->b':>12} {'miss% a->b':>14}")
    for n in sorted(names, key=lambda n: (b[n]["type_pct"] or 0) - (a[n]["type_pct"] or 0)):
        f = lambda r, k, fmt: (fmt.format(r[k]) if r[k] is not None else "-")
        print(f"{n[:40]:40} {f(a[n],'type_pct','{:.1f}'):>6}->{f(b[n],'type_pct','{:.1f}'):<6} "
              f"{f(a[n],'surfaces','{}'):>5}->{f(b[n],'surfaces','{}'):<5} {f(a[n],'failed','{}'):>5}->{f(b[n],'failed','{}'):<5} "
              f"{f(a[n],'missing_pct','{:.2f}'):>6}->{f(b[n],'missing_pct','{:.2f}'):<6}")


if __name__ == "__main__":
    main()
