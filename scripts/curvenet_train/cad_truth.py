"""Per-triangle truth for a mesh from its real CAD model: each triangle's CAD face (nearest triangle of
the fine CAD tessellation, truth_faces.npz from cad_bench prep), and its design surface (the CAD faces
grouped across smooth edges by cad_lines.py, at any curvature-jump threshold).

    python -I scripts/curvenet_train/cad_truth.py MESH TRUTH_FACES.npz LINES.json OUT_PREFIX
        [--mm-per-unit S] [--kjump K ...]  writes OUT_PREFIX_face.npy and OUT_PREFIX_k<K>.npy"""
import argparse
import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def groups(lines, kjump, angle=3.0):
    par = list(range(lines["faces"]))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for e in lines["edges"]:
        if e["angle"] < angle and e["kjump"] < kjump:
            par[find(e["a"])] = find(e["b"])
    roots = {r: k for k, r in enumerate(sorted({find(i) for i in range(lines["faces"])}))}
    return np.array([roots[find(i)] for i in range(lines["faces"])], np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("truth")
    ap.add_argument("lines")
    ap.add_argument("out")
    ap.add_argument("--mm-per-unit", type=float, default=1.0)
    ap.add_argument("--kjump", type=float, nargs="*", default=[0.3, 0.4, 0.5])
    a = ap.parse_args()
    m = trimesh.load(a.mesh, process=False, force="mesh")
    m.merge_vertices()
    C = np.asarray(m.vertices, float)[np.asarray(m.faces)].mean(1) * a.mm_per_unit
    z = np.load(a.truth)
    d, j = cKDTree(z["centroid"]).query(C)
    face = np.asarray(z["face"])[j]
    np.save(a.out + "_face.npy", face)
    lines = json.load(open(a.lines))
    rows = a.truth.replace(".npz", ".json")
    try:
        n_rows = len(json.load(open(rows)))
    except OSError:
        n_rows = lines["faces"]
    if n_rows != lines["faces"] or face.max() >= lines["faces"]:
        raise SystemExit(f"face numbers differ: {n_rows} faces in {rows}, {lines['faces']} in the lines")
    msg =[f"{len(C)} triangles, nearest CAD triangle p99 {np.percentile(d, 99):.3f} mm, {len(np.unique(face))} CAD faces"]
    for k in a.kjump:
        g = groups(lines, k)[face]
        np.save(f"{a.out}_k{k:g}.npy", g)
        msg.append(f"k<{k:g}: {len(np.unique(g))} design surfaces")
    print("; ".join(msg))


if __name__ == "__main__":
    main()
