"""Mine labelled border examples for the curve network's learned border decision.

For one part: run the curve network's own preparation and layout (curvenet.prepare + island_curves in
dump mode), take every candidate border of the first merge round with its features, and label it:
  --truth tri_face.npy   CAD truth: keep (1) when the two regions' calm cores lie on different true
                         faces, merge (0) when they lie on the same one (cores mixed: skipped)
  --approved             a layout the user approved (the hand rules' result): keep when the two
                         regions end up in different faces of it (this run's final layout, or the one
                         saved in --approved-lab, so the model's own runs can be labelled by it)
  --all-rounds           every border of every merge round, not just the first (the faces as they are
                         by then), so the model also learns from the situations its own merges create
    runtime/python/python.exe -I scripts/curvenet_train/mine.py BACKEND MESH OUT.jsonl [--truth F] [--approved]
              [--name NAME] [--weight W] [--mm-per-unit S]
MESH may be an .npz from app_mesh.py: the mesh exactly as the app hands it to the curve network (mirrored
half, mirror plane, scale), so an approved layout is learned on what the app really builds from.
"""
import argparse
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


def split_with_parent(V, F, emax, passes=12):
    """Long edges split at midpoints (no cracks), keeping each new triangle's original triangle."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    par = np.arange(len(F))
    for _ in range(passes):
        E = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
        Eu, inv = np.unique(E, axis=0, return_inverse=True)
        sp = np.linalg.norm(V[Eu[:, 0]] - V[Eu[:, 1]], axis=1) > emax
        if not sp.any():
            break
        mid = np.full(len(Eu), -1, np.int64)
        mid[sp] = len(V) + np.arange(int(sp.sum()))
        V = np.vstack([V, 0.5 * (V[Eu[sp, 0]] + V[Eu[sp, 1]])])
        m = mid[np.asarray(inv).ravel()].reshape(-1, 3)
        a, b, c = F[:, 0], F[:, 1], F[:, 2]
        m0, m1, m2 = m[:, 0], m[:, 1], m[:, 2]
        h0, h1, h2 = m0 >= 0, m1 >= 0, m2 >= 0
        outs, pars = [], []

        def add(k, *tris):
            for t in tris:
                outs.append(t[k])
                pars.append(par[k])
        add(~(h0 | h1 | h2), F)
        k = h0 & h1 & h2
        add(k, np.c_[a, m0, m2], np.c_[m0, b, m1], np.c_[m2, m1, c], np.c_[m0, m1, m2])
        add(h0 & ~h1 & ~h2, np.c_[a, m0, c], np.c_[m0, b, c])
        add(~h0 & h1 & ~h2, np.c_[a, b, m1], np.c_[a, m1, c])
        add(~h0 & ~h1 & h2, np.c_[a, b, m2], np.c_[m2, b, c])
        add(h0 & h1 & ~h2, np.c_[m0, b, m1], np.c_[a, m0, m1], np.c_[a, m1, c])
        add(~h0 & h1 & h2, np.c_[m2, m1, c], np.c_[a, b, m1], np.c_[a, m1, m2])
        add(h0 & ~h1 & h2, np.c_[a, m0, m2], np.c_[m0, b, c], np.c_[m0, c, m2])
        F, par = np.vstack(outs), np.concatenate(pars)
    return V, F, par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend")
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--truth")
    ap.add_argument("--approved", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--mm-per-unit", type=float, default=1.0)
    ap.add_argument("--eval", choices=["rule", "model"], help="score the final layout of the hand rule or the model against the truth")
    ap.add_argument("--all-rounds", action="store_true")
    ap.add_argument("--approved-lab", help=".npz (from --save-lab) holding the approved layout as lab1")
    ap.add_argument("--save-lab", help="also save the working mesh and its face layouts (before / after merging) to this .npz")
    ap.add_argument("--aug", type=int, default=None,
                    help="augmentation seed: random rotation, position, scale 0.4-2.5x and (40 %%) scan-like vertex noise")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    import trimesh
    from meshtocad import curvenet
    from meshtocad.curvenet.islands import BORDER_FEATURES, island_curves

    t0 = time.time()
    mirror = None
    if a.mesh.endswith(".npz"):  # the mesh as the app hands it over (app_mesh.py): with its mirror plane
        z = np.load(a.mesh)
        V0, F0 = np.asarray(z["V"], float), np.asarray(z["F"], np.int64)
        mirror = json.loads(str(z["mirror"]))
        a.mm_per_unit = float(z["scale"])
    else:
        m = trimesh.load(a.mesh, process=False, force="mesh")
        m.merge_vertices()  # keeps the triangles and their order (truth is per triangle)
        V0, F0 = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    aug = {}
    if a.aug is not None:  # the triangles and their order stay, so the truth labels still hold
        rng = np.random.default_rng(a.aug)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        qa, qb, qc, qd = q
        Rm = np.array([[qa * qa + qb * qb - qc * qc - qd * qd, 2 * (qb * qc - qa * qd), 2 * (qb * qd + qa * qc)],
                       [2 * (qb * qc + qa * qd), qa * qa - qb * qb + qc * qc - qd * qd, 2 * (qc * qd - qa * qb)],
                       [2 * (qb * qd - qa * qc), 2 * (qc * qd + qa * qb), qa * qa - qb * qb - qc * qc + qd * qd]])
        sc = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))
        V0 = (V0 - V0.mean(0)) @ Rm.T * sc + rng.uniform(-200, 200, 3)
        sig = 0.0
        if rng.random() < 0.4:
            el = np.linalg.norm(V0[F0[:, 0]] - V0[F0[:, 1]], axis=1)
            sig = float(rng.uniform(0.02, 0.25) * np.median(el))
            V0 = V0 + rng.normal(scale=sig, size=V0.shape)
        aug = dict(seed=a.aug, scale=sc, noise=sig)
    w = curvenet.prepare(V0, F0, mirror, a.mm_per_unit, log=lambda s: None)
    X, F = w["X"], w["F"]
    dump = []
    island_curves(X, F, w["feat"], w["necks"], w["tm"], w["half"], log=lambda s: None,
                  USE_MODEL=(a.eval != "rule"), DUMP=dump, DUMP_ONLY=True, DUMP_ALL=a.all_rounds)
    lab0 = next(d["lab0"] for d in dump if "lab0" in d)
    calm = next(d["calm"] for d in dump if "calm" in d)
    lab1 = next(d["lab1"] for d in dump if "lab1" in d)
    pairs = [d for d in dump if "x" in d]
    merges = {d["round"]: d["merge"] for d in dump if "merge" in d}
    A = 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
    name = a.name or a.mesh.replace("\\", "/").split("/")[-2]
    if a.aug is not None:
        name = f"{name}@a{a.aug}"
    ids = np.unique(lab0)
    pos = {int(l_): i for i, l_ in enumerate(ids)}
    li = np.searchsorted(ids, lab0)

    def per_region(vals, mask=None):
        """Area-weighted histogram of an integer label per first-round region."""
        m = np.ones(len(F), bool) if mask is None else mask
        H = np.zeros((len(ids), int(vals.max()) + 1))
        np.add.at(H, (li[m], vals[m]), A[m])
        return H

    def rounds():
        """Each pair with the first-round regions inside its two faces at that round."""
        owner = {int(l_): int(l_) for l_ in ids}
        done = -1
        for d in sorted(pairs, key=lambda d: d.get("round", 0)):
            r_ = d.get("round", 0)
            while done + 1 < r_:
                done += 1
                if done in merges:
                    k_, g_ = merges[done]
                    for l_, o_ in owner.items():
                        if o_ == g_:
                            owner[l_] = k_
            ma = [pos[l_] for l_, o_ in owner.items() if o_ == d["a"]]
            mb = [pos[l_] for l_, o_ in owner.items() if o_ == d["b"]]
            yield d, ma, mb

    lab1i = np.unique(lab1, return_inverse=True)[1].ravel()
    Hfin = per_region(lab1i)
    rows = []
    src = None
    if a.truth or a.save_lab:
        # the working triangles in the part's coordinates -> their original triangle
        P = (X[F].mean(1) @ w["R"] + w["o_mm"]) / w["scale"]
        diag = float(np.linalg.norm(np.ptp(V0, axis=0)))
        Vs, Fs, par = split_with_parent(V0, F0, 0.004 * diag)
        _, j = cKDTree(Vs[Fs].mean(1)).query(P)
        src = par[j]
    if a.truth:
        tf = np.load(a.truth)
        if len(tf) != len(F0):
            raise SystemExit(f"truth has {len(tf)} labels for {len(F0)} triangles")
        tri_truth = np.asarray(tf[src], np.int64)  # -> the original triangle's true face
        Hc, Ha = per_region(tri_truth, calm), per_region(tri_truth)
        for d, ma, mb in rounds():
            ha, hb = Hc[ma].sum(0), Hc[mb].sum(0)
            ha = ha if ha.sum() > 0 else Ha[ma].sum(0)
            hb = hb if hb.sum() > 0 else Ha[mb].sum(0)
            if min(ha.max() / max(ha.sum(), 1e-12), hb.max() / max(hb.sum(), 1e-12)) < 0.6:
                continue  # a face whose core is mixed: no clear answer
            rows.append(dict(part=name, src="truth", a=d["a"], b=d["b"], x=d["x"], n=d["n"], round=d.get("round", 0),
                             y=int(ha.argmax() != hb.argmax()), w=a.weight,
                             final=int(Hfin[ma].sum(0).argmax() != Hfin[mb].sum(0).argmax())))
    if a.approved:
        Happ = Hfin
        if a.approved_lab:
            ap_ = np.load(a.approved_lab)
            if not np.array_equal(ap_["F"], F):
                raise SystemExit("the approved layout is on another working mesh")
            Happ = per_region(np.unique(ap_["lab1"], return_inverse=True)[1].ravel())
        for d, ma, mb in rounds():
            rows.append(dict(part=name, src="approved", a=d["a"], b=d["b"], x=d["x"], n=d["n"], round=d.get("round", 0),
                             y=int(Happ[ma].sum(0).argmax() != Happ[mb].sum(0).argmax()), w=a.weight,
                             final=int(Hfin[ma].sum(0).argmax() != Hfin[mb].sum(0).argmax())))
    if a.eval and a.truth:
        ok = [r["final"] == r["y"] for r in rows if r["src"] == "truth" and r["round"] == 0]
        print(f"EVAL {name} {a.eval}: final layout agrees with the truth on {sum(ok)}/{len(ok)} borders, "
              f"{len(np.unique(lab1))} faces")
    if a.save_lab:
        np.savez_compressed(a.save_lab, X=X, F=F, lab0=lab0, lab1=lab1, src=src)  # src: original triangle
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(pairs)} candidate borders, {len(rows)} labelled ({sum(r['y'] for r in rows)} keep), "
          f"features {len(BORDER_FEATURES)}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
