"""What the extra faces are: every face of ours that is a fragment of a true CAD face another face
of ours mainly covers, by kind, with its share of the trial's 'extra' term.
    runtime python -I census.py RUN_TAG [--real|--synthetic] [--examples N]
Needs a seg run made with --labels (.cache/regress/RUN_TAG/labels)."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / ".cache/r30/data"
sys.path.insert(0, str(HERE))
from score import tri_area  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--real", action="store_true")
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--examples", type=int, default=3)
a = ap.parse_args()
run = ROOT / ".cache/regress" / a.tag
labs = sorted((run / "labels").glob("*.npz"))
if a.real:
    labs = [p for p in labs if not p.stem.startswith("syn_")]
if a.synthetic:
    labs = [p for p in labs if p.stem.startswith("syn_")]

cls_w = defaultdict(float)       # class -> share of mean 'extra'
cls_n = defaultdict(int)         # class -> pieces
cls_ex = defaultdict(list)       # class -> (weight, model, true face, piece tris)
models = 0
for p in labs:
    name = p.stem
    d = dict(np.load(DATA / f"{name}.npz", allow_pickle=False))
    z = np.load(p, allow_pickle=False)
    L, K = z["label"].astype(np.int64), z["kinds"]
    V, F = d["V"], d["F"]
    A = tri_area(V, F)
    T0 = np.asarray(d["tface"], np.int64)
    side = np.asarray(d["tside"], np.int64) if "tside" in d else np.ones(len(T0), np.int64)
    T = 2 * T0 + (side < 0)
    ftype = np.asarray(d["face_type"])
    nL, nT = int(L.max()) + 1, int(T.max()) + 1
    key = T * nL + L
    u, inv = np.unique(key, return_inverse=True)
    a_ = np.bincount(inv, weights=A)
    c_ = np.bincount(inv)
    tf, rf = u // nL, u % nL
    face_tris = np.bincount(T, minlength=nT)
    face_area = np.bincount(T, weights=A, minlength=nT)
    nreal = max(int((face_tris >= 3).sum()), 1)
    o = np.lexsort((-a_, rf))
    first = np.r_[True, rf[o][1:] != rf[o][:-1]]
    dom_t = tf[o][first]          # main true face of each face of ours
    dom_r = rf[o][first]
    reg_area = np.bincount(L, weights=A, minlength=nL)
    reg_tris = np.bincount(L, minlength=nL)
    # which of our faces share a mesh edge (pieces apart from their main face are often a
    # second copy of an instanced part: the scorer limit in the README)
    E = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    ef = np.tile(np.arange(len(F)), 3)
    o_e = np.lexsort((E[:, 1], E[:, 0]))
    E, ef = E[o_e], ef[o_e]
    dup = (E[1:] == E[:-1]).all(1)
    la, lb = L[ef[:-1][dup]], L[ef[1:][dup]]
    m_ = la != lb
    touch = set(zip(la[m_].tolist(), lb[m_].tolist())) | set(zip(lb[m_].tolist(), la[m_].tolist()))
    by_t = defaultdict(list)
    for r, t in zip(dom_r, dom_t):
        by_t[int(t)].append(int(r))
    models += 1
    for t, rs in by_t.items():
        if len(rs) < 2:
            continue
        rs.sort(key=lambda r: -reg_area[r])
        main = rs[0]
        tt = str(ftype[t // 2])
        for r in rs[1:]:
            share = reg_area[r] / max(face_area[t], 1e-300)
            size = "speck" if reg_tris[r] <= 2 else ("sliver" if share < 0.02 else "tile")
            same = "main ok" if str(K[main]) == tt or (tt == "bspline" and str(K[main]) == "freeform") else f"main {K[main]}"
            near = "touches" if (r, main) in touch else "apart"
            c = f"true {tt:8s} | piece {str(K[r]):8s} | {size:6s} | {same:13s} | {near}"
            w = 1.0 / nreal
            cls_w[c] += w
            cls_n[c] += 1
            cls_ex[c].append((w, name, t // 2, int(reg_tris[r])))
tot = sum(cls_w.values())
print(f"{models} models; mean extra = {tot / max(models, 1):.4f} (x0.3 in the score)")
print(f"{'share':>6s} {'pieces':>6s}  class")
for c, w in sorted(cls_w.items(), key=lambda x: -x[1])[:30]:
    ex = sorted(cls_ex[c], reverse=True)[: a.examples]
    exs = ", ".join(f"{m}#{t}({n}t)" for _, m, t, n in ex)
    print(f"{w / tot * 100:5.1f}% {cls_n[c]:6d}  {c}   e.g. {exs}")
json.dump({c: {"share": w / tot, "pieces": cls_n[c]} for c, w in cls_w.items()},
          open(run / "census.json", "w"), indent=1)
