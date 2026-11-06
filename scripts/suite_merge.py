"""Dev-only: merge tagged suite summaries (summary_<tag>.json) into one table with totals.

    runtime/python/python.exe scripts/suite_merge.py TAG [TAG ...] [--out NAME]
"""

import json
import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent.parent / ".cache" / "cad_run"


def main():
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), "merged")
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = []
    for t in tags:
        f = RUN / f"summary_{t}.json"
        if f.exists():
            rows += json.loads(f.read_text())
    seg = [r["seg"] for r in rows if r["seg"]]
    stp = [r["step"] for r in rows if r["step"]]
    pip = [r["pipeline"] for r in rows if r["pipeline"]]
    w = np.array([s["true_faces"] for s in seg], float)
    tot = {
        "models": len(rows),
        "pipeline_errors": sum(1 for r in rows if r["codes"][0] != 0 or "error" in (r["pipeline"] or {})),
        "parts": sum(p.get("parts", 0) or 0 for p in pip),
        "true_faces": int(w.sum()),
        "surfaces": sum(s["regions"] for s in seg),
        "type_accuracy_pct": float(np.average([s["type_accuracy_pct"] for s in seg], weights=w)) if seg else None,
        "split_faces_pct": float(np.average([s["over_split_faces_pct"] for s in seg], weights=w)) if seg else None,
        "mixed_regions": sum(s["merged_regions"] for s in seg),
        "step_solids": sum(c.get("solids", 0) for c in stp),
        "step_failed_faces": sum(c.get("failed_faces", 0) for c in stp),
        "step_faces": sum(c.get("faces", 0) for c in stp),
        "missing_pct_mean": float(np.mean([c["missing_pct"] for c in stp if "missing_pct" in c])) if stp else None,
        "segment_s_total": float(sum(p.get("segment_s", 0) for p in pip)),
    }
    (RUN / f"{out}.json").write_text(json.dumps({"totals": tot, "rows": rows}, indent=1))
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
