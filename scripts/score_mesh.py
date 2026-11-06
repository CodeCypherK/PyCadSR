"""Dev-only: how closely a reconstructed mesh follows the source surface.

    .cache/devtools/Scripts/python.exe scripts/score_mesh.py TRUTH_MESH RECON_MESH [...] [--scale=1000]

--scale multiplies the reconstruction (a GLB is in metres, an STL source often in mm).

Exact point-to-triangle distances in both directions, as % of the source's
bounding-box diagonal: truth -> recon finds missing surface, recon -> truth finds
surface that should not be there. (Point-to-point distances between random
samples have a floor set by the sample spacing, which hides small differences.)
"""

import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def closest_on_triangles(P, A, B, C):
    """Closest points on triangles (A,B,C) to P (all (n,3)); Ericson's region tests."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = P - B
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = P - C
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(all="ignore"):
        den = va + vb + vc
        v = vb / den
        w = vc / den
        out = A + ab * v[:, None] + ac * w[:, None]
        # edges
        t_ab = d1 / (d1 - d3)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out[m] = A[m] + ab[m] * t_ab[m, None]
        t_ac = d2 / (d2 - d6)
        m2 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out[m2] = A[m2] + ac[m2] * t_ac[m2, None]
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        m3 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        out[m3] = B[m3] + (C[m3] - B[m3]) * t_bc[m3, None]
    # vertices
    out[(d1 <= 0) & (d2 <= 0)] = A[(d1 <= 0) & (d2 <= 0)]
    out[(d3 >= 0) & (d4 <= d3)] = B[(d3 >= 0) & (d4 <= d3)]
    out[(d6 >= 0) & (d5 <= d6)] = C[(d6 >= 0) & (d5 <= d6)]
    bad = ~np.isfinite(out).all(axis=1)
    out[bad] = A[bad]
    return out


def surface_distance(mesh, P, k=12, max_edge=None):
    """Distance from points to the mesh surface (candidates from triangle centroids)."""
    m = mesh
    if max_edge is not None and m.edges_unique_length.max() > max_edge:
        V, F = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=max_edge)
        m = trimesh.Trimesh(V, F, process=False)
    T = m.triangles
    tree = cKDTree(T.mean(axis=1))
    _, cand = tree.query(P, k=k)
    best = np.full(len(P), np.inf)
    for j in range(k):
        t = T[cand[:, j]]
        q = closest_on_triangles(P, t[:, 0], t[:, 1], t[:, 2])
        best = np.minimum(best, np.linalg.norm(q - P, axis=1))
    return best


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--scale=")]
    scale = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--scale=")), 1.0))
    truth = trimesh.load(args[0], force="mesh")
    diag = float(np.linalg.norm(truth.extents))
    max_edge = 0.004 * diag
    a, _ = trimesh.sample.sample_surface(truth, 300_000, seed=1)
    print(f"truth: {len(truth.faces):,} triangles, area {truth.area:.1f}, size {diag:.4g}")
    for path in args[1:]:
        rec = trimesh.load(path, force="mesh")
        rec.apply_scale(scale)
        b, _ = trimesh.sample.sample_surface(rec, 300_000, seed=2)
        d_tr = surface_distance(rec, a, max_edge=max_edge) / diag * 100
        d_rt = surface_distance(truth, b, max_edge=max_edge) / diag * 100
        print(f"{path.replace(chr(92), '/').split('/')[-1]}: {len(rec.faces):,} triangles, area {rec.area:.1f} "
              f"({(rec.area - truth.area) / truth.area * 100:+.1f}%)")
        print(f"   truth->recon mean {d_tr.mean():.4f}% p95 {np.percentile(d_tr, 95):.4f}% p99 "
              f"{np.percentile(d_tr, 99):.3f}% | recon->truth mean {d_rt.mean():.4f}% p95 "
              f"{np.percentile(d_rt, 95):.4f}% p99 {np.percentile(d_rt, 99):.3f}% max {d_rt.max():.2f}%")


if __name__ == "__main__":
    main()
