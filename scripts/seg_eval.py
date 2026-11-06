"""Dev-only: score surface finding against the true CAD faces (from scripts/cad_bench.py prep).

    .cache/devtools/Scripts/python.exe scripts/seg_eval.py SESSION.pkl TRUTH_DIR [--detail]

Every sample point is matched to the true CAD face it lies on. True faces that
are the same surface (a cylinder split into two halves at its seam, coplanar
faces) are grouped into one true *surface*, because merging those is correct.

Reports, point-weighted unless noted:
  type accuracy     our region's surface type equals the true surface's type
                    (freeform counts as right on a B-spline)
  over-split        true surfaces cut into 2+ regions (each holding 10 %+ of it)
  merged            regions spanning 2+ true surfaces (each 10 %+ of the region)
  parameters        radius / axis / normal errors where a region matches one
                    true analytic surface of the same type
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from score_mesh import closest_on_triangles  # noqa: E402

SAME = {"freeform": "bspline"}


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


def surface_key(f, diag):
    """Faces with the same key are one surface (tolerances relative to the part size)."""
    t, q = f["type"], 2e-4 * diag
    rnd = lambda x, s=q: tuple(np.round(np.asarray(x, float) / s).astype(int).tolist()) if np.ndim(x) else int(round(x / s))
    if t == "plane":
        n = unit(f["normal"])
        n = n if n[np.argmax(np.abs(n))] > 0 else -n
        d = f["offset"] * (1 if unit(f["normal"]) @ n > 0 else -1)
        return (t, rnd(n, 1e-3), rnd(d))
    if t in ("cylinder", "cone", "torus"):
        a = unit(f["axis"])
        a = a if a[np.argmax(np.abs(a))] > 0 else -a
        p = np.asarray(f.get("point", f.get("apex", f.get("center"))), float)
        foot = p - (p @ a) * a  # axis line position independent of the point along it
        extra = {"cylinder": ("radius",), "cone": ("half_angle",), "torus": ("R", "r")}[t]
        vals = tuple(rnd(f[k], 1e-3 if k == "half_angle" else q) for k in extra)
        key_p = rnd(p) if t != "cylinder" else rnd(foot)
        return (t, rnd(a, 1e-3), key_p) + vals
    if t == "sphere":
        return (t, rnd(f["center"]), rnd(f["radius"]))
    return (t, id(f))  # B-splines: every face its own surface


def group_faces(faces, diag):
    """Union-find of faces describing the same surface (same type, parameters within tolerance)."""
    n = len(faces)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    d_tol, a_tol = 2e-4 * diag, np.cos(np.radians(0.05))
    by_type = defaultdict(list)
    for i, f in enumerate(faces):
        by_type[f["type"]].append(i)
    for t, idx in by_type.items():
        if t in ("bspline", "other"):
            continue
        for ii, i in enumerate(idx):
            fi = faces[i]
            for j in idx[ii + 1:]:
                fj = faces[j]
                same = False
                if t == "plane":
                    ni, nj = unit(fi["normal"]), unit(fj["normal"])
                    c = float(ni @ nj)
                    same = abs(c) > a_tol and abs(fi["offset"] - np.sign(c) * fj["offset"]) < d_tol
                elif t in ("cylinder", "cone", "torus"):
                    ai, aj = unit(fi["axis"]), unit(fj["axis"])
                    if abs(float(ai @ aj)) < a_tol:
                        continue
                    pi = np.asarray(fi.get("point", fi.get("apex", fi.get("center"))), float)
                    pj = np.asarray(fj.get("point", fj.get("apex", fj.get("center"))), float)
                    v = pj - pi
                    off_axis = np.linalg.norm(v - (v @ ai) * ai)
                    if t == "cylinder":
                        same = off_axis < d_tol and abs(fi["radius"] - fj["radius"]) < d_tol
                    elif t == "cone":
                        same = np.linalg.norm(v) < d_tol and abs(fi["half_angle"] - fj["half_angle"]) < 1e-4
                    else:
                        same = np.linalg.norm(v) < d_tol and abs(fi["R"] - fj["R"]) < d_tol and abs(fi["r"] - fj["r"]) < d_tol
                elif t == "sphere":
                    same = np.linalg.norm(np.subtract(fi["center"], fj["center"])) < d_tol and \
                        abs(fi["radius"] - fj["radius"]) < d_tol
                if same:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def main():
    sess, tdir = Path(sys.argv[1]), Path(sys.argv[2])
    detail = "--detail" in sys.argv
    s = pickle.loads(sess.read_bytes())
    truth = json.loads((tdir / "truth.json").read_text())
    faces = json.loads((tdir / "truth_faces.json").read_text())
    tf = np.load(tdir / "truth_faces.npz")
    diag = truth["diag_mm"]
    # Surfaces of revolution / extrusion (OpenCASCADE "other") sweep a spline profile:
    # freeform faces like B-splines
    for f in faces:
        if f["type"] == "other":
            f["type"] = "bspline"
    # Some CAD systems store planes as B-spline surfaces: a flat "B-spline" face is a plane
    for fi, f in enumerate(faces):
        if f["type"] == "bspline":
            pts = tf["corners"][tf["face"] == fi].reshape(-1, 3)
            if len(pts) >= 3:
                c = pts.mean(axis=0)
                _, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
                if np.abs((pts - c) @ vt[2]).max() < 1e-5 * diag:
                    f["type"], f["normal"], f["offset"] = "plane", vt[2].tolist(), float(vt[2] @ c)
    mesh = s.mesh
    scale = diag / float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    # snap to the nearest unit factor (OpenCASCADE's bounding box is padded, so the ratio is only rough)
    scale = min((1.0, 10.0, 25.4, 1000.0), key=lambda k: abs(np.log(scale / k)))
    # judge per mesh triangle (area-weighted): its centre lies inside one true face
    P = mesh.triangles_center * scale
    W = mesh.area_faces * scale ** 2
    best_d = np.full(len(P), np.inf)
    best_f = np.zeros(len(P), np.int64)
    # big truth triangles (flat faces meshed by a few) are split, so the nearest
    # centres found below include the triangle a point actually lies on
    import trimesh as _tm
    Cv = tf["corners"].reshape(-1, 3)
    Cf = np.arange(len(Cv)).reshape(-1, 3)
    Vs, Fs, idx = _tm.remesh.subdivide_to_size(Cv, Cf, max_edge=0.01 * diag, return_index=True)
    tf = {"corners": Vs[Fs], "centroid": Vs[Fs].mean(axis=1), "face": tf["face"][idx]}
    C = tf["corners"]
    # match within the same part: touching parts share coincident faces, and identical
    # copies of a part sit on the same surfaces (our parts carry the names part_<k>)
    tri_part = np.array([faces[f]["part"] for f in tf["face"]])
    names = list(getattr(s, "part_names", []) or [])
    fp = getattr(s, "face_part", None)
    ours_t = np.full(len(P), -1)
    if fp is not None and names and all(n.startswith("part_") for n in names):
        ours_t = np.array([int(names[p].split("_")[-1]) - 1 for p in fp])
    groups = [(np.arange(len(P)), np.arange(len(tri_part)))] if (ours_t < 0).any() else \
        [(np.where(ours_t == k)[0], np.where(tri_part == k)[0]) for k in np.unique(ours_t)]
    for qi, ti in groups:
        if len(qi) == 0 or len(ti) == 0:
            continue
        tree = cKDTree(tf["centroid"][ti])
        _, cand = tree.query(P[qi], k=min(8, len(ti)))
        cand = np.atleast_2d(cand.reshape(len(qi), -1))
        for j in range(cand.shape[1]):
            t = C[ti[cand[:, j]]]
            q = closest_on_triangles(P[qi], t[:, 0], t[:, 1], t[:, 2])
            d = np.linalg.norm(q - P[qi], axis=1)
            m = d < best_d[qi]
            best_d[qi[m]] = d[m]
            best_f[qi[m]] = tf["face"][ti[cand[m, j]]]
    # true faces -> true surfaces: faces of the same type whose parameters agree within
    # tolerance (halves of a cylinder split at a seam, coplanar faces) are one surface
    surf_of_face = group_faces(faces, diag)
    keys = {i: i for i in np.unique(surf_of_face)}
    surf_type = {int(sf): faces[int(np.where(surf_of_face == sf)[0][0])]["type"] for sf in keys}
    ts = surf_of_face[best_f]
    labels = s.face_labels
    regions = s.regions
    ok = labels >= 0
    print(f"{len(P):,} triangles, {len(regions)} regions; true: {len(faces)} faces = {len(keys)} surfaces "
          f"{dict(Counter(surf_type.values()))}; centre-to-truth distance median {np.median(best_d):.4f} mm")
    # type accuracy (point-weighted)
    prim = np.array([r.prim for r in regions])
    ours = prim[labels[ok]]
    theirs = np.array([surf_type[x] for x in ts[ok]])
    ours_n = np.array([SAME.get(x, x) for x in ours])
    w = W[ok]
    acc = w[ours_n == theirs].sum() / w.sum() * 100
    conf = Counter()
    for a, b, x in zip(theirs, ours, w):
        conf[(a, b)] += x
    print(f"type accuracy {acc:.1f}% of area")
    by_true = defaultdict(Counter)
    for (a, b), n in conf.items():
        by_true[a][b] += n
    for a, c in sorted(by_true.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        print(f"   true {a:9s} {tot / W.sum() * 100:5.1f}% of area -> " + ", ".join(f"{b} {n / tot * 100:.0f}%" for b, n in c.most_common(4)))
    # over-split: true surfaces covered by several regions
    pairs = Counter()
    for a, b, x in zip(ts[ok], labels[ok], W[ok]):
        pairs[(a, b)] += x
    per_surf = defaultdict(list)
    per_reg = defaultdict(list)
    for (sf, rg), n in pairs.items():
        per_surf[sf].append((n, rg))
        per_reg[rg].append((n, sf))
    min_area = 1e-6 * diag ** 2
    big_surfs = [sf for sf, lst in per_surf.items() if sum(n for n, _ in lst) >= min_area]
    over = [sf for sf in big_surfs
            if sum(1 for n, _ in per_surf[sf] if n >= 0.1 * sum(m for m, _ in per_surf[sf])) >= 2]
    merged = [rg for rg, lst in per_reg.items()
              if sum(1 for n, _ in lst if n >= 0.1 * sum(m for m, _ in lst)) >= 2]
    print(f"over-split: {len(over)} of {len(big_surfs)} true surfaces ({len(over) / max(len(big_surfs), 1) * 100:.0f}%) "
          f"by type {dict(Counter(surf_type[x] for x in over))}")
    print(f"merged: {len(merged)} of {len(per_reg)} regions span 2+ true surfaces "
          f"by our type {dict(Counter(prim[x] for x in merged))}")
    # per true FACE (a CAD face is one connected piece; identical planes elsewhere are other faces)
    fpairs = Counter()
    for a, b, x in zip(best_f[ok], labels[ok], W[ok]):
        fpairs[(a, b)] += x
    per_face = defaultdict(list)
    for (fa, rg), n in fpairs.items():
        per_face[fa].append(n)
    big_faces = [fa for fa, lst in per_face.items() if sum(lst) >= min_area]
    over_f = [fa for fa in big_faces if sum(1 for n in per_face[fa] if n >= 0.1 * sum(per_face[fa])) >= 2]
    print(f"over-split faces: {len(over_f)} of {len(big_faces)} true faces ({len(over_f) / max(len(big_faces), 1) * 100:.0f}%) "
          f"by type {dict(Counter(faces[fa]['type'] for fa in over_f))}")
    # regions per true surface (how fragmented)
    frag = np.array([len([1 for n, _ in per_surf[sf] if n >= 0.02 * sum(m for m, _ in per_surf[sf])]) for sf in big_surfs])
    if len(frag):
        print(f"regions per true surface: mean {frag.mean():.2f}, max {frag.max()}")
    # parameter accuracy on clean matches
    errs = defaultdict(list)
    for rg, lst in per_reg.items():
        tot = sum(n for n, _ in lst)
        n, sf = max(lst)
        if n < 0.8 * tot:
            continue
        r = regions[rg]
        t = surf_type[sf]
        if SAME.get(r.prim, r.prim) != t:
            continue
        fi = int(np.where(surf_of_face == sf)[0][0])
        f = faces[fi]
        p = r.params
        if r.prim == "freeform" or not p:
            continue
        if t == "plane":
            errs["plane normal (deg)"].append(np.degrees(np.arccos(min(1, abs(unit(p["normal"]) @ unit(f["normal"]))))))
            errs["plane offset (mm)"].append(abs(abs(unit(p["normal"]) @ (np.asarray(p["point"]) * scale)) - abs(f["offset"])))
        elif t == "cylinder":
            errs["cylinder radius (mm)"].append(abs(p["radius"] * scale - f["radius"]))
            errs["cylinder axis (deg)"].append(np.degrees(np.arccos(min(1, abs(unit(p["axis"]) @ unit(f["axis"]))))))
        elif t == "cone":
            errs["cone half angle (deg)"].append(abs(np.degrees(p["half_angle"]) - np.degrees(f["half_angle"])))
        elif t == "sphere":
            errs["sphere radius (mm)"].append(abs(p["radius"] * scale - f["radius"]))
        elif t == "torus":
            errs["torus R (mm)"].append(abs(p["R"] * scale - f["R"]))
            errs["torus r (mm)"].append(abs(p["r"] * scale - f["r"]))
    summary = {"triangles": int(len(P)), "regions": len(regions), "true_faces": len(faces), "true_surfaces": len(keys),
               "type_accuracy_pct": float(acc), "over_split_faces_pct": float(len(over_f) / max(len(big_faces), 1) * 100),
               "merged_regions": len(merged), "params": {}}
    for k, v in errs.items():
        v = np.asarray(v)
        summary["params"][k] = {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
        print(f"   {k:22s} n={len(v):4d} median {np.median(v):.4f} p90 {np.percentile(v, 90):.4f} max {v.max():.3f}")
    (sess.parent / "seg_eval.json").write_text(json.dumps(summary, indent=1))
    if detail:
        for sf in sorted(over, key=lambda x: -sum(n for n, _ in per_surf[x]))[:15]:
            lst = sorted(per_surf[sf], reverse=True)
            print(f"   over-split surface {sf} ({surf_type[sf]}): " + ", ".join(f"R{rg}:{prim[rg]}:{n}" for n, rg in lst[:6]))


if __name__ == "__main__":
    main()
