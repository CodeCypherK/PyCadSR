"""Train the curve network's learned border decision (keep a border between two smooth regions as a
design line, or merge them) and export it as JSON trees for backend/meshtocad/curvenet/learned.py.

    runtime/python/python.exe -I scripts/curvenet_train/train.py OUT.json DATA.jsonl [DATA.jsonl ...]
        [--holdout REGEX]   parts matching REGEX are never trained on, only reported
        [--holdout-frac F]  also hold out this fraction of parts (by a hash of the name)
        [--backend DIR]     backend whose curvenet defines the features (default backend)
        [--rules DIR]       also teach the hand rule: write rule_threshold.json (the crest-radius threshold
                            re-tuned on the data) and rule_d2/rule_d3.json (tiny trees taught by the model),
                            same format as the model, so any of them can stand in for border_model.json
Rows come from mine.py. Accuracy is reported by part-level cross-validation (weighted by the rows'
weights), next to the hand rule the model replaces (merge when the border's median crest radius is over
30 mm and it is not a pocket's rim).

The loop that makes it better over time (round 32 built border_model.json this way, three rounds):
  1. mine.py --truth on parts with known true faces (first merge round, hand rule merging); train.
  2. put the model in a copy of the backend and mine again with --all-rounds: the model makes the merges,
     every border of every round is labelled, so it learns from the situations its own merges create;
     the user's approved layouts come in with --approved --approved-lab (high --weight). Train on all rows.
  3. repeat 2 with the new model. Before copying a model to backend/meshtocad/curvenet/border_model.json,
     check it on held-out parts, on the approved jug (all its faces must stay), the bottles and the test
     parts (full builds); new parts and newly approved layouts are simply more rows.
Real CAD models (round 33, the model in the backend now): cad_lines.py finds how a model's faces meet
(smooth joins vs design lines) and groups them into design surfaces; cad_truth.py labels each triangle of
its mesh with them; mine.py --truth on that, with --all-rounds, gives rows like any other truth. Round 33
trained on 115 benchmark models (.cache/cad_bench, weighted 3x; 57 more held out for testing) plus the generated parts and the approved
jug. A real CAD model whose faces are an auto-surfaced patchwork (a test jug: 797 patches, smooth
everywhere, edges unrelated to its design lines) carries no design lines; use an approved layout for it.
Approved layouts must also be learned on the mesh exactly as the app reads it (app_mesh.py -> .npz, which
mine.py takes as MESH): the round 33 model kept the jug's 22 faces on an STL copy but lost 3 on the app's
own .glb (float32 copy, borders near the cut-off). The model in the backend (v5) adds those rows and the
jug's copies (rotated, rescaled, re-meshed; weight 5) to the round 33 data."""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text



def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def base(part):
    """The part a row comes from: augmented (@aN) and second-mesh (~b) copies belong to it."""
    return part.split("@")[0].split("~")[0]


def hand_rule(x):
    """1 = keep (the rule the model replaces)."""
    r_med = 10 ** x[0]
    pocket_rim = x[9] * x[10] < 0
    return int(r_med <= 30.0 or pocket_rim)


def fit(X, y, w):
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.08, l2_regularization=1.0,
                                         min_samples_leaf=8, random_state=0)
    clf.fit(X, y, sample_weight=w)
    return clf


def export(clf, features):
    trees = []
    for it in clf._predictors:
        nd = it[0].nodes
        trees.append({"f": nd["feature_idx"].astype(int).tolist(), "t": nd["num_threshold"].astype(float).tolist(),
                      "l": nd["left"].astype(int).tolist(), "r": nd["right"].astype(int).tolist(),
                      "leaf": nd["is_leaf"].astype(bool).tolist(), "v": nd["value"].astype(float).tolist(),
                      "ml": nd["missing_go_to_left"].astype(bool).tolist()})
    return {"features": features, "base": float(np.ravel(clf._baseline_prediction)[0]), "trees": trees, "threshold": 0.5}


def tree_json(t, features):
    """A sklearn decision tree in the model's JSON format (one tree, leaves are log-odds)."""
    tr = t.tree_
    val = tr.value[:, 0, :]
    p = np.clip(val[:, 1] / np.maximum(val.sum(1), 1e-12), 0.02, 0.98)
    leaf = tr.children_left < 0
    return {"features": features, "base": 0.0, "threshold": 0.5,
            "trees": [{"f": np.where(leaf, 0, tr.feature).astype(int).tolist(), "t": tr.threshold.astype(float).tolist(),
                       "l": tr.children_left.astype(int).tolist(), "r": tr.children_right.astype(int).tolist(),
                       "leaf": leaf.tolist(), "v": np.log(p / (1 - p)).tolist(), "ml": [True] * tr.node_count}]}


def teach_rules(out_dir, clf, X, y, w, hold, features):
    """The hand rule, taught by the data and the model: the crest-radius threshold re-tuned, and tiny
    trees (a few readable cases) fitted to the model's own answers."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    Xh = np.array([r["x"] for r in hold], float) if hold else np.zeros((0, X.shape[1]))
    yh = np.array([r["y"] for r in hold], int)

    def report(name, f):
        tr_ = float(((f(X) == y) * w).sum() / w.sum())
        ho = f"held-out {np.mean(f(Xh) == yh):.3f}" if len(yh) else ""
        print(f"   {name:34s} train {tr_:.3f}  {ho}")
    print("rules taught for the hand rule:")
    report("hand rule now (30 mm, pockets)", lambda Z: np.array([hand_rule(z) for z in Z], int))
    best = None
    for R in np.geomspace(3, 1000, 120):
        p_ = (10 ** X[:, 0] <= R).astype(int)
        s_ = float(((p_ == y) * w).sum())
        if best is None or s_ > best[0]:
            best = (s_, R)
    R = best[1]
    thr = {"features": features, "base": 0.0, "threshold": 0.5,
           "trees": [{"f": [0, 0, 0], "t": [float(np.log10(R)), 0.0, 0.0], "l": [1, -1, -1], "r": [2, -1, -1],
                      "leaf": [False, True, True], "v": [0.0, 3.0, -3.0], "ml": [True] * 3}]}
    json.dump(thr, open(os.path.join(out_dir, "rule_threshold.json"), "w"))
    report(f"re-tuned threshold ({R:.0f} mm)", lambda Z: (10 ** Z[:, 0] <= R).astype(int))
    teach = (clf.predict_proba(X)[:, 1] >= 0.5).astype(int)
    for d in (2, 3):
        t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=50, random_state=0).fit(X, teach, sample_weight=w)
        json.dump(tree_json(t, features), open(os.path.join(out_dir, f"rule_d{d}.json"), "w"))
        report(f"rule taught by the model, depth {d}", lambda Z, t=t: t.predict(Z))
        for ln in export_text(t, feature_names=list(features), decimals=2).splitlines():
            print("      " + ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--backend", default="backend")
    ap.add_argument("--rules", default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="also hold out this fraction of parts, chosen by a hash of the part name")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    from meshtocad.curvenet.islands import BORDER_FEATURES
    rows = load(a.data)
    def held(r):
        if r["src"] == "approved":
            return False  # the user's own layouts are always learned from
        b = base(r["part"])
        return bool((a.holdout and re.search(a.holdout, b)) or
                    (a.holdout_frac and int(hashlib.md5(("h" + b).encode()).hexdigest(), 16) % 1000 < 1000 * a.holdout_frac))
    hold = [r for r in rows if held(r)]
    rows = [r for r in rows if not held(r)]
    X = np.array([r["x"] for r in rows], float)
    y = np.array([r["y"] for r in rows], int)
    w = np.array([r.get("w", 1.0) for r in rows], float)
    parts = np.array([base(r["part"]) for r in rows])  # augmented and re-meshed copies stay with their part
    print(f"{len(rows)} rows from {len(set(parts))} parts ({y.mean():.0%} keep); held out {len(hold)} rows")
    # part-level cross-validation
    fold = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % a.folds for p in parts])
    pred = np.zeros(len(y), int)
    for k in range(a.folds):
        tr, te = fold != k, fold == k
        if te.any() and len(set(y[tr])) > 1:
            pred[te] = (fit(X[tr], y[tr], w[tr]).predict_proba(X[te])[:, 1] >= 0.5).astype(int)
    rule = np.array([hand_rule(x) for x in X])

    def acc(p, m=slice(None)):
        return float(((p[m] == y[m]) * w[m]).sum() / max(w[m].sum(), 1e-12))
    print(f"cross-validated accuracy: model {acc(pred):.3f}  hand rule {acc(rule):.3f}")
    by_src = defaultdict(list)
    for i, r in enumerate(rows):
        by_src[r["src"]].append(i)
    for s_, idx in by_src.items():
        idx = np.array(idx)
        print(f"   {s_:9s} {len(idx):5d} rows: model {acc(pred, idx):.3f}  hand rule {acc(rule, idx):.3f}")
    clf = fit(X, y, w)
    model = export(clf, BORDER_FEATURES)
    if hold:
        Xh = np.array([r["x"] for r in hold], float)
        yh = np.array([r["y"] for r in hold], int)
        ph = (clf.predict_proba(Xh)[:, 1] >= 0.5).astype(int)
        rh = np.array([hand_rule(x) for x in Xh])
        print(f"held-out parts: model {np.mean(ph == yh):.3f}  hand rule {np.mean(rh == yh):.3f} ({len(hold)} rows)")
        fam = defaultdict(list)
        for i, r in enumerate(hold):
            fam[re.sub(r"_\d+$", "", base(r["part"]))].append(i)
        for f_, idx in sorted(fam.items()):
            idx = np.array(idx)
            print(f"   {f_:24s} model {np.mean(ph[idx] == yh[idx]):.2f}  rule {np.mean(rh[idx] == yh[idx]):.2f}  ({len(idx)} borders)")
    json.dump(model, open(a.out, "w"))
    if a.rules:
        teach_rules(a.rules, clf, X, y, w, hold, BORDER_FEATURES)
    # the exported trees must give the model's own answers
    from meshtocad.curvenet import learned
    p_sk = clf.predict_proba(X[:200])[:, 1]
    raw = np.full(min(200, len(X)), model["base"])
    for t in model["trees"]:
        for i, x in enumerate(X[:200]):
            n = 0
            while not t["leaf"][n]:
                v = x[t["f"][n]]
                n = t["l"][n] if ((t["ml"][n]) if v != v else v <= t["t"][n]) else t["r"][n]
            raw[i] += t["v"][n]
    p_js = 1 / (1 + np.exp(-raw))
    print(f"exported {len(model['trees'])} trees to {a.out}; matches sklearn: {bool(np.allclose(p_js, p_sk, atol=1e-9))}")
    _ = learned


if __name__ == "__main__":
    main()
