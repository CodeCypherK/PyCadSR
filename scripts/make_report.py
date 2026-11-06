"""Dev-only: build the benchmark report page (HTML, images embedded) from suite run folders.

    runtime/python/python.exe scripts/make_report.py OUT.html

Round 6 (round 5's segmentation exported again with the round-6 STEP writer) against
round 5, round 5 against round 4 on every prepared model (the 84-model main set and the
120 expanded FreeCAD-library models), plus the trend on the main set since round 1.
Folders: round 3 .cache/cad_run, round 4 .cache/r4/run, round 5 .cache/r6/run with
models rerun on the final code in .cache/r7/run taking precedence, and STEP checks of
sessions re-exported with the final exporter in .cache/r5final/run over those; round 6
STEP checks in .cache/r9/run_final. Images from .cache/report/*.jpg.
"""

from __future__ import annotations

import base64
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R3, R4 = ROOT / ".cache" / "cad_run", ROOT / ".cache" / "r4" / "run"
R5A, R5B = ROOT / ".cache" / "r6" / "run", ROOT / ".cache" / "r7" / "run"
R5C = ROOT / ".cache" / "r5final" / "run"  # sessions re-exported with the final exporter
R6 = ROOT / ".cache" / "r9" / "run_final"  # round 6: the same sessions, round-6 STEP writer
R7 = ROOT / ".cache" / "r10" / "run_final"  # round 7: round 6 plus exact circle edges
R8 = ROOT / ".cache" / "r11" / "run_final"  # round 8: round 7 plus our own seams on cone bands
R9 = ROOT / ".cache" / "r12" / "run_final"  # round 9: round 8 plus shared axes for coaxial turned faces
R10 = ROOT / ".cache" / "r15" / "run_final"  # round 10: fresh segmentation and the round-10 writer
R11 = ROOT / ".cache" / "r16" / "run_final"  # round 11: wrap-split segmentation and the round-11 writer
IMG = ROOT / ".cache" / "report"


def load(d: Path, name: str):
    m = d / name
    rd = lambda f: json.loads((m / f).read_text()) if (m / f).exists() else {}
    s, c, p = rd("seg_eval.json"), rd("check.json"), rd("pipeline.json")
    if not s:
        return None
    return {"true": s.get("true_faces"), "surf": s.get("regions"), "type": s.get("type_accuracy_pct"),
            "split": s.get("over_split_faces_pct"), "parts": p.get("parts") or 0, "solids": c.get("solids"),
            "failed": c.get("failed_faces"), "faces": c.get("faces"), "missing": c.get("missing_pct"),
            "seg_s": p.get("segment_s"), "error": bool(p.get("error"))}


def load5(name: str):
    r = load(R5B, name) or load(R5A, name)
    c = load(R5C, name)
    if r and c and c["faces"] is not None:  # same segmentation, final STEP export
        r = {**r, **{k: c[k] for k in ("solids", "failed", "faces", "missing")}}
    return r


def load6(name: str):
    r, c = load5(name), load(R6, name)
    if r and c and c["faces"] is not None:
        return {**r, **{k: c[k] for k in ("solids", "failed", "faces", "missing")}}
    return None


def load7(name: str):
    r, c = load5(name), load(R7, name)
    if r and c and c["faces"] is not None:
        return {**r, **{k: c[k] for k in ("solids", "failed", "faces", "missing")}}
    return None


def load8(name: str):
    r, c = load5(name), load(R8, name)
    if r and c and c["faces"] is not None:
        return {**r, **{k: c[k] for k in ("solids", "failed", "faces", "missing")}}
    return None


def load9(name: str):
    r, c = load5(name), load(R9, name)
    if r and c and c["faces"] is not None:
        return {**r, **{k: c[k] for k in ("solids", "failed", "faces", "missing")}}
    return None


def load10(name: str):
    r, c = load5(name), load(R10, name)
    if r and c and c["faces"] is not None:
        return {**r, **{k: c[k] for k in ("solids", "failed", "faces", "missing", "type", "split", "surf")}}
    return None


def load11(name: str):
    r, c = load5(name), load(R11, name)
    if r and c and c["faces"] is not None:
        return {**r, **{k: c[k] for k in ("solids", "failed", "faces", "missing", "type", "split", "surf")}}
    return None


def quick(d: Path, names):
    """Faces whose quick (default) OpenCASCADE area is off by more than 1 %, summed."""
    n_ = 0
    for n in names:
        p = d / n / "quick.json"
        if p.exists():
            n_ += json.loads(p.read_text())["quick_off_faces"]
    return n_


def export_note(d: Path, names, pattern: str) -> int:
    n_ = 0
    for n in names:
        p = d / n / "export.log"
        if p.exists():
            m = re.search(pattern, p.read_text(encoding="utf-8", errors="replace"))
            n_ += int(m.group(1)) if m else 0
    return n_


def count_entity(d: Path, names, what: str) -> int:
    n_ = 0
    for n in names:
        p = d / n / f"{n}.step"
        if p.exists():
            n_ += p.read_text(encoding="utf-8", errors="replace").count(f"={what}(")
    return n_


def step_stats(d: Path, names):
    """Circle edges and bytes over the models' STEP files."""
    circles = size = 0
    for n in names:
        p = d / n / f"{n}.step"
        if p.exists():
            t = p.read_text(encoding="utf-8", errors="replace")
            circles += t.count("=CIRCLE(")
            size += p.stat().st_size
    return circles, size


def totals(rows):
    rows = [r for r in rows if r]
    w = np.array([r["true"] for r in rows], float)
    stp = [r for r in rows if r["faces"] is not None]
    return {"models": len(rows), "true": int(w.sum()), "surf": int(sum(r["surf"] for r in rows)),
            "type": float(np.average([r["type"] for r in rows], weights=w)),
            "split": float(np.average([r["split"] for r in rows], weights=w)),
            "solids": int(sum(r["solids"] or 0 for r in stp)), "parts": int(sum(r["parts"] for r in stp)),
            "failed": int(sum(r["failed"] or 0 for r in stp)), "faces": int(sum(r["faces"] or 0 for r in stp)),
            "missing": float(np.mean([r["missing"] for r in stp if r["missing"] is not None])),
            "errors": int(sum(r["error"] for r in rows))}


def img(name: str) -> str:
    p = IMG / name
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def pretty(n: str) -> str:
    big = {"engine4": "Engine (4-cylinder, 15 parts)", "inline4": "Inline-4 engine", "lamp": "Desk lamp",
           "soarm": "SO-ARM100 robot arm", "minisb": "Voron Mini Stealthburner"}
    if n in big:
        return big[n]
    return n.split("_", 1)[1].replace("_", " ") if "_" in n else n


def f1(v, suf=""):
    return "–" if v is None else f"{v:.1f}{suf}"


def iv(v):
    return "–" if v is None else f"{v:,}"


def bars(title, unit, vals, better, fmt):
    """One small chart: a bar per round, one scale, values labelled."""
    W, H, pad_l, pad_r, top, bh, gap = 340, 30 + 34 * len(vals), 92, 70, 26, 22, 12
    mx = max(v for _, v in vals) or 1.0
    out = [f'<figure class="mini"><figcaption>{html.escape(title)} <span class="unit">{html.escape(unit)} · {better}</span></figcaption>',
           f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}">']
    for i, (lab, v) in enumerate(vals):
        y = top + i * (bh + gap)
        w = (W - pad_l - pad_r) * v / mx
        cls = f"b{max(0, i + 4 - len(vals))}"
        out.append(f'<text x="{pad_l - 10}" y="{y + bh * 0.68}" class="lab" text-anchor="end">{html.escape(lab)}</text>')
        out.append(f'<rect x="{pad_l}" y="{y}" width="{max(w, 2):.1f}" height="{bh}" rx="3" class="{cls}"><title>{html.escape(lab)}: {fmt(v)}</title></rect>')
        out.append(f'<text x="{pad_l + max(w, 2) + 8:.1f}" y="{y + bh * 0.68}" class="val">{fmt(v)}</text>')
    out.append(f'<line x1="{pad_l}" y1="{top - 6}" x2="{pad_l}" y2="{H - 8}" class="axis"/></svg></figure>')
    return "\n".join(out)


def main():
    out = Path(sys.argv[1])
    names = sorted(p.name for p in R5A.iterdir() if p.is_dir() and (p / "seg_eval.json").exists())
    names = [n for n in names if (R4 / n / "seg_eval.json").exists()]
    main_set = [n for n in names if not n.startswith("fd_") and (R3 / n / "seg_eval.json").exists()]
    a4 = {n: load(R4, n) for n in names}
    a5 = {n: load5(n) for n in names}
    a3 = {n: load(R3, n) for n in main_set}
    # STEP figures only over models checked in both rounds being compared
    both = [n for n in names if a4[n]["faces"] is not None and a5[n]["faces"] is not None]
    t4, t5 = totals([a4[n] for n in both]), totals([a5[n] for n in both])
    s4, s5 = totals(a4.values()), totals(a5.values())
    for k in ("type", "split", "true", "surf", "models", "errors"):
        t4[k], t5[k] = s4[k], s5[k]
    mboth = [n for n in main_set if a3[n]["faces"] is not None and a4[n]["faces"] is not None and a5[n]["faces"] is not None]
    m3, m4, m5 = (totals([d[n] for n in main_set]) for d in (a3, a4, a5))
    ms3, ms4, ms5 = (totals([d[n] for n in mboth]) for d in (a3, a4, a5))
    r1 = {"type": 93.0, "split": 6.5, "missing": 6.04}  # round 1, 79 library models, earlier scorer
    a6 = {n: load6(n) for n in names}
    both6 = [n for n in names if a5[n] and a6[n] and a5[n]["faces"] is not None and a6[n]["missing"] is not None
             and a5[n]["missing"] is not None]
    u5, u6 = totals([a5[n] for n in both6]), totals([a6[n] for n in both6])
    m6both = [n for n in mboth if a6[n]]
    # STEP charts over the main-set models checked in every round 3-6
    ms3b, ms4b, ms5b, ms6 = (totals([d[n] for n in m6both]) for d in (a3, a4, a5, a6))

    charts = "\n".join([
        bars("Surface type right", "% of area", [("Round 1", r1["type"]), ("Round 3", m3["type"]), ("Round 4", m4["type"]),
                                                 ("Round 5", m5["type"])], "higher is better", lambda v: f"{v:.1f}%"),
        bars("Faces split in pieces", "% of faces", [("Round 1", r1["split"]), ("Round 3", m3["split"]), ("Round 4", m4["split"]),
                                                     ("Round 5", m5["split"])], "lower is better", lambda v: f"{v:.1f}%"),
        bars("STEP faces failing", f"{len(m6both)} models", [("Round 3", ms3b["failed"]), ("Round 4", ms4b["failed"]),
                                                            ("Round 5", ms5b["failed"]), ("Round 6", ms6["failed"])],
             "lower is better", lambda v: f"{v:,.0f}"),
        bars("Area missing in STEP", f"% mean, {len(m6both)} models", [("Round 3", ms3b["missing"]), ("Round 4", ms4b["missing"]),
                                                                    ("Round 5", ms5b["missing"]), ("Round 6", ms6["missing"])],
             "lower is better", lambda v: f"{v:.2f}%"),
    ])

    def arrow(x, y, fmt, lower_better=True, tol=0.5):
        if x is None or y is None:
            return "", f"{fmt(x)} → {fmt(y)}"
        d = (y - x) if not lower_better else (x - y)
        cls = "up" if d > tol else "down" if d < -tol else ""
        return cls, f"{fmt(x)} → {fmt(y)}"

    def row(n):
        x, y = a4[n], a5[n]
        ct, tt = arrow(x["type"], y["type"], lambda v: f1(v, "%"), lower_better=False)
        cs, ts = arrow(x["surf"], y["surf"], iv, tol=0)
        cf, tf = arrow(x["failed"], y["failed"], iv, tol=0)
        cm, tm_ = arrow(x["missing"], y["missing"], lambda v: f1(v, "%"))
        set_ = "main" if not n.startswith("fd_") else "expanded"
        return (f'<tr><td class="name">{html.escape(pretty(n))}<span class="set">{set_}</span></td><td>{y["true"]}</td>'
                f'<td class="{cs}">{ts}</td><td class="{ct}">{tt}</td><td class="{cf}">{tf}</td><td class="{cm}">{tm_}</td>'
                f'<td>{y["solids"] if y["solids"] is not None else "–"}/{y["parts"]}</td></tr>')

    def gain(n):
        x, y = a4[n], a5[n]
        return ((y["type"] or 0) - (x["type"] or 0)) + 0.05 * ((x["failed"] or 0) - (y["failed"] or 0)) + \
            0.01 * ((x["surf"] or 0) - (y["surf"] or 0))

    def change6(n):  # STEP only: fewer failing faces and less missing area is better
        x, y = a5[n], a6[n]
        return ((x["failed"] or 0) - (y["failed"] or 0)) + 2.0 * ((x["missing"] or 0) - (y["missing"] or 0))

    def row6(n):
        x, y = a5[n], a6[n]
        cf, tf = arrow(x["failed"], y["failed"], iv, tol=0)
        cm, tm_ = arrow(x["missing"], y["missing"], lambda v: f1(v, "%"), tol=0.3)
        cs, ts = arrow(x["solids"], y["solids"], iv, lower_better=False, tol=0)
        set_ = "main" if not n.startswith("fd_") else "expanded"
        return (f'<tr><td class="name">{html.escape(pretty(n))}<span class="set">{set_}</span></td><td>{y["faces"]:,}</td>'
                f'<td class="{cf}">{tf}</td><td class="{cm}">{tm_}</td><td class="{cs}">{ts}</td></tr>')

    ch6 = {n: change6(n) for n in both6}
    a7 = {n: load7(n) for n in names}
    both7 = [n for n in names if a6[n] and a7[n] and a6[n]["missing"] is not None and a7[n]["missing"] is not None]
    v6, v7 = totals([a6[n] for n in both7]), totals([a7[n] for n in both7])

    def change7(n):
        x, y = a6[n], a7[n]
        return ((x["failed"] or 0) - (y["failed"] or 0)) + 2.0 * ((x["missing"] or 0) - (y["missing"] or 0))

    ch7 = {n: change7(n) for n in both7}
    better7 = sum(1 for v in ch7.values() if v > 0.5)
    worse7 = sum(1 for v in ch7.values() if v < -0.5)
    circ6, size6 = step_stats(R6, both7)
    circ7, size7 = step_stats(R7, both7)
    with_circ = sum(1 for n in both7 if step_stats(R7, [n])[0] > 0)

    def row7(n):
        x, y = a6[n], a7[n]
        cf, tf = arrow(x["failed"], y["failed"], iv, tol=0)
        cm, tm_ = arrow(x["missing"], y["missing"], lambda v: f1(v, "%"), tol=0.3)
        set_ = "main" if not n.startswith("fd_") else "expanded"
        return (f'<tr><td class="name">{html.escape(pretty(n))}<span class="set">{set_}</span></td>'
                f'<td>{step_stats(R7, [n])[0]:,}</td><td class="{cf}">{tf}</td><td class="{cm}">{tm_}</td></tr>')

    moved7 = sorted([n for n in both7 if abs(ch7[n]) > 0.5], key=lambda n: -ch7[n])
    a8 = {n: load8(n) for n in names}
    both8 = [n for n in names if a7[n] and a8[n] and a7[n]["missing"] is not None and a8[n]["missing"] is not None]
    w7, w8 = totals([a7[n] for n in both8]), totals([a8[n] for n in both8])
    ch8 = {n: ((a7[n]["failed"] or 0) - (a8[n]["failed"] or 0)) + 2.0 * ((a7[n]["missing"] or 0) - (a8[n]["missing"] or 0))
           for n in both8}
    better8 = sum(1 for v in ch8.values() if v > 0.5)
    worse8 = sum(1 for v in ch8.values() if v < -0.5)
    lines8 = count_entity(R8, both8, "LINE")
    circ8a, circ8b = count_entity(R7, both8, "CIRCLE"), count_entity(R8, both8, "CIRCLE")
    moved8 = sorted([n for n in both8 if abs(ch8[n]) > 0.5], key=lambda n: -ch8[n])
    note8 = "; ".join(f'{html.escape(pretty(n))} {a7[n]["failed"]} → {a8[n]["failed"]} failing faces, '
                      f'{a7[n]["missing"]:.2f}% → {a8[n]["missing"]:.2f}% missing' for n in moved8) or "none"
    a9 = {n: load9(n) for n in names}
    both9 = [n for n in names if a8[n] and a9[n] and a8[n]["missing"] is not None and a9[n]["missing"] is not None
             and (R8 / n / "quick.json").exists() and (R9 / n / "quick.json").exists()]
    x8, x9 = totals([a8[n] for n in both9]), totals([a9[n] for n in both9])
    ch9 = {n: ((a8[n]["failed"] or 0) - (a9[n]["failed"] or 0)) + 2.0 * ((a8[n]["missing"] or 0) - (a9[n]["missing"] or 0))
           for n in both9}
    better9 = sum(1 for v in ch9.values() if v > 0.5)
    worse9 = sum(1 for v in ch9.values() if v < -0.5)
    circ9a, circ9b = count_entity(R8, both9, "CIRCLE"), count_entity(R9, both9, "CIRCLE")
    quick9a, quick9b = quick(R8, both9), quick(R9, both9)
    shared9 = export_note(R9, both9, r'"(\d+) turned faces refit on shared axes"')
    rings9 = export_note(R9, both9, r'"(\d+) junction loops between coaxial faces')
    moved9 = sorted([n for n in both9 if abs(ch9[n]) > 0.5], key=lambda n: -ch9[n])
    note9 = "; ".join(f'{html.escape(pretty(n))} {a8[n]["failed"]} → {a9[n]["failed"]} failing faces, '
                      f'{a8[n]["missing"]:.2f}% → {a9[n]["missing"]:.2f}% missing' for n in moved9) or "none"
    a10 = {n: load10(n) for n in names}
    both10 = [n for n in names if a9[n] and a10[n] and a9[n]["missing"] is not None and a10[n]["missing"] is not None
              and (R9 / n / "quick.json").exists() and (R10 / n / "quick.json").exists()]
    y9, y10 = totals([a9[n] for n in both10]), totals([a10[n] for n in both10])
    ch10 = {n: ((a9[n]["failed"] or 0) - (a10[n]["failed"] or 0)) + 2.0 * ((a9[n]["missing"] or 0) - (a10[n]["missing"] or 0))
            for n in both10}
    better10 = sum(1 for v in ch10.values() if v > 0.5)
    worse10 = sum(1 for v in ch10.values() if v < -0.5)
    circ10a, circ10b = count_entity(R9, both10, "CIRCLE"), count_entity(R10, both10, "CIRCLE")
    quick10a, quick10b = quick(R9, both10), quick(R10, both10)
    top10 = sorted([n for n in both10 if ch10[n] > 0.5], key=lambda n: -ch10[n])[:8]
    note10b = "; ".join(f'{html.escape(pretty(n))} {a9[n]["failed"]} → {a10[n]["failed"]} failing faces, '
                        f'{a9[n]["missing"]:.1f}% → {a10[n]["missing"]:.1f}% missing' for n in top10) or "none"
    worst10 = sorted([n for n in both10 if ch10[n] < -0.5], key=lambda n: ch10[n])
    note10w = "; ".join(f'{html.escape(pretty(n))} {a9[n]["failed"]} → {a10[n]["failed"]} failing faces, '
                        f'{a9[n]["missing"]:.2f}% → {a10[n]["missing"]:.2f}% missing' for n in worst10) or "none"
    a11 = {n: load11(n) for n in names}
    both11 = [n for n in names if a10[n] and a11[n] and a10[n]["missing"] is not None and a11[n]["missing"] is not None]
    z10, z11 = totals([a10[n] for n in both11]), totals([a11[n] for n in both11])
    ch11 = {n: ((a10[n]["failed"] or 0) - (a11[n]["failed"] or 0)) + 2.0 * ((a10[n]["missing"] or 0) - (a11[n]["missing"] or 0))
            for n in both11}
    better11 = sum(1 for v in ch11.values() if v > 0.5)
    worse11 = sum(1 for v in ch11.values() if v < -0.5)
    top11 = sorted([n for n in both11 if ch11[n] > 0.5], key=lambda n: -ch11[n])[:6]
    note11b = "; ".join(f'{html.escape(pretty(n))} {a10[n]["failed"]} → {a11[n]["failed"]} failing faces' for n in top11) or "none"
    worst11 = sorted([n for n in both11 if ch11[n] < -0.5], key=lambda n: ch11[n])
    note11w = "; ".join(f'{html.escape(pretty(n))} {a10[n]["failed"]} → {a11[n]["failed"]} failing faces, '
                        f'{a10[n]["missing"]:.2f}% → {a11[n]["missing"]:.2f}% missing' for n in worst11) or "none"
    better6 = sum(1 for v in ch6.values() if v > 0.5)
    worse6 = sum(1 for v in ch6.values() if v < -0.5)
    moved = sorted([n for n in both6 if abs(ch6[n]) > 0.5], key=lambda n: -ch6[n])
    order = sorted(names, key=lambda n: -gain(n))
    better = sum(1 for n in names if gain(n) > 0.05)
    worse = sum(1 for n in names if gain(n) < -0.05)
    tpl = (ROOT / "scripts" / "report_template.html").read_text(encoding="utf-8")
    fill = {
        "A_MODELS": str(t5["models"]), "A_TRUE": f'{t5["true"]:,}', "A_NBOTH": str(len(both)),
        "A_TYPE": f'{t5["type"]:.1f}%', "A_TYPE4": f'{t4["type"]:.1f}%',
        "A_SURF": f'{t5["surf"]:,}', "A_SURF4": f'{t4["surf"]:,}',
        "A_FAILED": f'{t5["failed"]:,}', "A_FAILED4": f'{t4["failed"]:,}', "A_FACES": f'{t5["faces"]:,}',
        "A_MISSING": f'{t5["missing"]:.2f}%', "A_MISSING4": f'{t4["missing"]:.2f}%',
        "A_SOLIDS": f'{t5["solids"]:,}/{t5["parts"]:,}', "A_SOLIDS4": f'{t4["solids"]:,}/{t4["parts"]:,}',
        "A_ERRORS": str(t5["errors"]), "A_BETTER": str(better), "A_WORSE": str(worse),
        "M_MODELS": str(m5["models"]), "M_NBOTH": str(len(m6both)),
        "R6_MODELS": str(len(both6)), "R6_FAILED": f'{u6["failed"]:,}', "R6_FAILED5": f'{u5["failed"]:,}',
        "R6_FACES": f'{u6["faces"]:,}', "R6_MISSING": f'{u6["missing"]:.2f}%', "R6_MISSING5": f'{u5["missing"]:.2f}%',
        "R6_SOLIDS": f'{u6["solids"]:,}', "R6_SOLIDS5": f'{u5["solids"]:,}', "R6_BETTER": str(better6),
        "R6_WORSE": str(worse6), "ROWS6": "\n".join(row6(n) for n in moved),
        "R6_WORST_NOTE": ("Worse than round 5: " + "; ".join(
            f'{html.escape(pretty(n))} ({a5[n]["failed"]} → {a6[n]["failed"]} failing faces, '
            f'{a5[n]["missing"]:.1f}% → {a6[n]["missing"]:.1f}% missing)'
            for n in sorted((n for n in both6 if ch6[n] < -0.5), key=lambda n: ch6[n])[:3]) + "."
            if worse6 else ""),
        "IMG_PIN6": img("pin6_before_after.jpg"), "IMG_BOTTLE6": img("bottle6_before_after.jpg"),
        "R7_MODELS": str(len(both7)), "R7_FAILED": f'{v7["failed"]:,}', "R7_FAILED6": f'{v6["failed"]:,}',
        "R7_MISSING": f'{v7["missing"]:.2f}%', "R7_MISSING6": f'{v6["missing"]:.2f}%',
        "R7_CIRCLES": f"{circ7:,}", "R7_WITHCIRC": str(with_circ),
        "R7_SIZE": f"{(size7 - size6) / max(size6, 1) * 100:+.1f}%",
        "R7_BETTER": str(better7), "R7_WORSE": str(worse7),
        "ROWS7": "\n".join(row7(n) for n in moved7) or '<tr><td class="name" colspan="4">No model moved by more than the threshold.</td></tr>',
        "IMG_FLANGE7": img("flange7_circles.jpg"),
        "R8_MODELS": str(len(both8)), "R8_FAILED": f'{w8["failed"]:,}', "R8_FAILED7": f'{w7["failed"]:,}',
        "R8_MISSING": f'{w8["missing"]:.2f}%', "R8_MISSING7": f'{w7["missing"]:.2f}%',
        "R8_SEAMS": f"{lines8:,}", "R8_CIRCLES": f"{circ8b:,}", "R8_CIRCLES7": f"{circ8a:,}",
        "R8_BETTER": str(better8), "R8_WORSE": str(worse8), "R8_MOVED": note8,
        "IMG_BOTTLE8": img("bottle8_seams.jpg"),
        "R9_MODELS": str(len(both9)), "R9_FAILED": f'{x9["failed"]:,}', "R9_FAILED8": f'{x8["failed"]:,}',
        "R9_MISSING": f'{x9["missing"]:.2f}%', "R9_MISSING8": f'{x8["missing"]:.2f}%',
        "R9_CIRCLES": f"{circ9b:,}", "R9_CIRCLES8": f"{circ9a:,}", "R9_QUICK": f"{quick9b:,}", "R9_QUICK8": f"{quick9a:,}",
        "R9_SHARED": f"{shared9:,}", "R9_RINGS": f"{rings9:,}",
        "R9_BETTER": str(better9), "R9_WORSE": str(worse9), "R9_MOVED": note9,
        "IMG_BOTTLE9": img("bottle9_axes.jpg"),
        "R10_MODELS": str(len(both10)), "R10_FAILED": f'{y10["failed"]:,}', "R10_FAILED9": f'{y9["failed"]:,}',
        "R10_MISSING": f'{y10["missing"]:.2f}%', "R10_MISSING9": f'{y9["missing"]:.2f}%',
        "R10_SOLIDS": f'{y10["solids"]:,}', "R10_SOLIDS9": f'{y9["solids"]:,}',
        "R10_TYPE": f'{y10["type"]:.1f}%', "R10_TYPE9": f'{y9["type"]:.1f}%',
        "R10_CIRCLES": f"{circ10b:,}", "R10_CIRCLES9": f"{circ10a:,}", "R10_QUICK": f"{quick10b:,}", "R10_QUICK9": f"{quick10a:,}",
        "R10_BETTER": str(better10), "R10_WORSE": str(worse10), "R10_TOP": note10b, "R10_WORST": note10w,
        "IMG_T8": img("t8_helix.jpg"),
        "R11_MODELS": str(len(both11)), "R11_FAILED": f'{z11["failed"]:,}', "R11_FAILED10": f'{z10["failed"]:,}',
        "R11_MISSING": f'{z11["missing"]:.2f}%', "R11_MISSING10": f'{z10["missing"]:.2f}%',
        "R11_SOLIDS": f'{z11["solids"]:,}', "R11_SOLIDS10": f'{z10["solids"]:,}',
        "R11_TYPE": f'{z11["type"]:.1f}%', "R11_TYPE10": f'{z10["type"]:.1f}%',
        "R11_BETTER": str(better11), "R11_WORSE": str(worse11), "R11_TOP": note11b, "R11_WORST": note11w,
        "IMG_ENC11": img("encoder11.jpg"),
        "CHARTS": charts, "ROWS": "\n".join(row(n) for n in order),
        "IMG_BEND": img("bend_before_after.jpg"), "IMG_SPRING": img("springs_before_after.jpg"),
        "IMG_PIN": img("drawing_pin_before_after.jpg"), "IMG_APP": img("app5.jpg"),
    }
    for k, v in fill.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    out.write_text(tpl, encoding="utf-8")
    print(json.dumps({"all4": t4, "all5": t5, "main3": m3, "main4": m4, "main5": m5,
                      "step_main": [ms3, ms4, ms5, ms6], "r6": {"n": len(both6), "r5": u5, "r6": u6,
                      "better": better6, "worse": worse6, "moved": [(n, round(ch6[n], 2)) for n in moved]},
                      "r11": {"n": len(both11), "r10": z10, "r11": z11, "better": better11, "worse": worse11,
                              "moved": sorted(((n, round(ch11[n], 2)) for n in both11 if abs(ch11[n]) > 0.5),
                                              key=lambda x: -x[1])},
                      "r10": {"n": len(both10), "r9": y9, "r10": y10, "better": better10, "worse": worse10,
                              "circles9": circ10a, "circles10": circ10b, "quick9": quick10a, "quick10": quick10b,
                              "moved": sorted(((n, round(ch10[n], 2)) for n in both10 if abs(ch10[n]) > 0.5),
                                              key=lambda x: -x[1])},
                      "r9": {"n": len(both9), "r8": x8, "r9": x9, "better": better9, "worse": worse9,
                             "circles8": circ9a, "circles9": circ9b, "quick8": quick9a, "quick9": quick9b,
                             "shared": shared9, "rings": rings9, "moved": [(n, round(ch9[n], 2)) for n in moved9]},
                      "r8": {"n": len(both8), "r7": w7, "r8": w8, "better": better8, "worse": worse8,
                             "seams": lines8, "circles7": circ8a, "circles8": circ8b,
                             "moved": [(n, round(ch8[n], 2)) for n in moved8]},
                      "r7": {"n": len(both7), "r6": v6, "r7": v7, "better": better7, "worse": worse7,
                             "circles": circ7, "with_circles": with_circ, "size6": size6, "size7": size7,
                             "moved": [(n, round(ch7[n], 2)) for n in moved7]},
                      "better": better, "worse": worse}, indent=1, default=str))
    print("wrote", out, f"{out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
