"""Experiment: a structural prior from slices - base shapes, low relief, high relief, objects.

    .cache/devtools/Scripts/python.exe scripts/relief_prior.py MESH OUT_DIR [--pitch 0.05] [--up y]
        [--massing 0.3] [--low 0.1]

From a dense point cloud (the mesh is only sampled):

1. Thick slices along all three axes, unified (dense points, low slice density):
   the solid the outlines enclose.
2. Large basic shapes: that solid opened then closed at the massing scale, which
   removes anything thinner than the scale (cornices, columns, balconies) and fills
   anything narrower (window recesses). What is left is the massing.
3. Every point gets its signed distance to the massing surface:
     base         on the massing surface
     low relief   within --low of it (panels, shallow recesses, mouldings)
     high relief  further out or in, but attached
     object       a separate piece (its own connected cluster of points)
4. Writes per-class renders and area shares, and the labels for later use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from slice_experiment import AX, solid_along  # noqa: E402

CLASSES = ["base", "low relief", "high relief", "object"]
COLORS = np.array([[0.62, 0.66, 0.74], [0.30, 0.64, 0.90], [0.90, 0.45, 0.29], [0.37, 0.82, 0.48]])


def open_close(U: np.ndarray, r_cells: float) -> np.ndarray:
    """Morphological opening then closing with a ball of radius r (via distance transforms)."""
    inner = ndimage.distance_transform_edt(U)
    eroded = inner > r_cells
    opened = ndimage.distance_transform_edt(~eroded) <= r_cells
    dil = ndimage.distance_transform_edt(~opened) <= r_cells
    return ndimage.distance_transform_edt(dil) > r_cells


def majority(U: np.ndarray, r_cells: float) -> np.ndarray:
    """Cells where most of the surrounding cube (side 2r+1) is solid: the base surface
    settles where most of the material is, so a wall with pilasters, cornices and
    window recesses keeps its wall plane."""
    size = int(2 * round(r_cells) + 1)
    return ndimage.uniform_filter(U.astype(np.float32), size=size, mode="constant") > 0.5


def signed_distance(M: np.ndarray, pitch: float) -> np.ndarray:
    """Positive outside the massing, negative inside (model units)."""
    out = ndimage.distance_transform_edt(~M)
    inn = ndimage.distance_transform_edt(M)
    return (np.where(M, -(inn - 0.5), out - 0.5)) * pitch


def objects_from_points(P: np.ndarray, spacing: float, min_share: float = 0.02) -> np.ndarray:
    """True for points in clusters that are separate from the main structure."""
    sub = np.arange(len(P))
    tree = cKDTree(P)
    pairs = tree.query_pairs(2.5 * spacing, output_type="ndarray")
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    _, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    big = sizes >= min_share * len(P)
    return ~big[lab[sub]]


def render(P, colors, up, view, path, px=900):
    """Z-buffered point splat seen along `view` (from the high end), up axis vertical."""
    from PIL import Image

    hor = [a for a in range(3) if a not in (view, up)][0]
    u, v, d = P[:, hor], P[:, up], P[:, view]
    lo_u, hi_u, lo_v, hi_v = u.min(), u.max(), v.min(), v.max()
    s = (px - 1) / max(hi_u - lo_u, hi_v - lo_v)
    iu = ((u - lo_u) * s).astype(np.int64)
    iv = ((hi_v - v) * s).astype(np.int64)
    W, H = iu.max() + 1, iv.max() + 1
    key = iv * W + iu
    order = np.lexsort((-d, key))  # nearest (largest depth coordinate) first per pixel
    k_sorted = key[order]
    first = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
    sel = order[first]
    img = np.full((H * W, 3), 0.11)
    dd = (d[sel] - d.min()) / max(np.ptp(d), 1e-9)
    img[key[sel]] = colors[sel] * (0.55 + 0.45 * dd[:, None])
    img = (img.reshape(H, W, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--pitch", type=float, default=0.05, help="slice thickness (low slice density)")
    ap.add_argument("--up", default="y")
    ap.add_argument("--massing", type=float, default=0.3, help="features thinner than 2x this are relief")
    ap.add_argument("--low", type=float, default=0.1, help="relief shallower than this is low relief")
    ap.add_argument("--points", type=int, default=6_000_000)
    ap.add_argument("--method", default="majority", choices=["majority", "openclose"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    T = lambda: f"[{time.perf_counter() - t0:6.1f}s]"

    mesh = trimesh.load(args.mesh, force="mesh")
    P, fi = trimesh.sample.sample_surface(mesh, args.points, seed=7)
    Nr = mesh.face_normals[fi]
    h = args.pitch
    lo = P.min(0) - 2.37 * h
    shape = tuple(int(v) for v in np.floor((P.max(0) - lo) / h).astype(np.int64) + 3)
    print(T(), f"{len(P):,} points, slices {h} thick, grid {shape}", flush=True)

    U = np.ones(shape, bool)
    for a in range(3):
        U &= solid_along(P, Nr, lo, h, shape, a)
    print(T(), f"unified solid from slices: {U.sum():,} cells", flush=True)

    # A hollow object (a building with an interior) has two sets of large shapes:
    # the outer massing (cavities filled) and the cavity's own massing inside.
    filled = ndimage.binary_fill_holes(U)
    cavity = filled & ~U
    r = args.massing / h
    smooth = majority if args.method == "majority" else open_close
    M_out = smooth(filled, r)
    M_in = smooth(cavity, r) if cavity.any() else np.zeros_like(U)
    M = M_out & ~M_in
    print(T(), f"massing: outer {M_out.sum():,} cells, cavity {M_in.sum():,} cells "
               f"(solid {U.sum():,}, enclosed cavity {cavity.sum():,})", flush=True)
    ijk = np.clip(np.floor((P - lo) / h).astype(np.int64), 0, np.array(shape) - 1)
    d_out = signed_distance(M_out, h)[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    d_in = (-signed_distance(M_in, h)[ijk[:, 0], ijk[:, 1], ijk[:, 2]] if M_in.any()
            else np.full(len(P), np.inf))
    # positive = material standing out of the base shape it belongs to
    d = np.where(np.abs(d_out) <= np.abs(d_in), d_out, d_in)

    spacing = float(np.sqrt(mesh.area / len(P)))
    sub = np.random.default_rng(0).choice(len(P), min(len(P), 1_500_000), replace=False)
    obj_sub = objects_from_points(P[sub], spacing * np.sqrt(len(P) / len(sub)))
    obj = np.zeros(len(P), bool)
    _, j = cKDTree(P[sub]).query(P, k=1)
    obj = obj_sub[j]
    print(T(), "separate objects found", flush=True)

    cls = np.where(np.abs(d) <= h, 0, np.where(np.abs(d) <= args.low, 1, 2))
    cls[obj] = 3
    share = {CLASSES[c]: float((cls == c).mean() * 100) for c in range(4)}
    out_side = {"relief standing out": float(((cls >= 1) & (cls <= 2) & (d > 0)).mean() * 100),
                "relief cut in": float(((cls >= 1) & (cls <= 2) & (d < 0)).mean() * 100)}
    print(T(), "surface share: " + ", ".join(f"{k} {v:.1f}%" for k, v in share.items()), flush=True)
    print(T(), "relief: " + ", ".join(f"{k} {v:.1f}%" for k, v in out_side.items()), flush=True)

    up = AX.index(args.up)
    others = [a for a in range(3) if a != up]
    col = COLORS[cls]
    for name, view in (("front", others[1]), ("side", others[0])):
        render(P, col, up, view, out / f"prior_{name}.png")
    # the massing alone, as a depth render
    from slice_experiment import depth_render
    depth_render(M, others[1], out / "massing_front.png", "massing")
    depth_render(M, up, out / "massing_top.png", "massing")
    np.save(out / "classes.npy", cls.astype(np.int8))
    (out / "prior.json").write_text(json.dumps({"share_pct": share, "relief_pct": out_side,
                                                "pitch": h, "massing": args.massing, "low": args.low}, indent=1))
    print(T(), "done", flush=True)


if __name__ == "__main__":
    main()
