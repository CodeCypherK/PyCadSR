"""Experiment: describe a shape by its cross-sections along three axes, then unify.

    runtime/python/python.exe scripts/slice_experiment.py MESH OUT_DIR [--pitch 0.025] [--up y]

Works from a point cloud only (the mesh is used just to sample points, and at the
end to score the result):

1. Cut the cloud into thin stacked slices along the vertical axis.
2. In each slice, find the closed outlines and what they enclose with the
   even-odd rule, so a ring (a courtyard, a donut) keeps its hole.
3. Describe each outline with math: straight lines (Douglas-Peucker on the
   traced contour, snapped to the axes) and circular arcs where a run of short
   edges turns steadily.
4. Consecutive slices with the same outline merge into one extrusion step, so
   the whole axis becomes a short piecewise description in the slicing coordinate.
5. Repeat along the other two axes.
6. Unify: a point is inside when all three descriptions agree,
   F(x, y, z) = max(F_y(x, z | y), F_x(y, z | x), F_z(x, y | z)) <= 0.

Writes slice plots, depth renders of each solid, a voxel-boundary STL of the
unified solid, the outline math as JSON, and an accuracy report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

AX = "xyz"


# ------------------------------------------------------------------ slices

def fill_even_odd(curve: np.ndarray) -> np.ndarray:
    """Solid of a 2D slice: curve pixels plus the regions an odd number of curves deep."""
    free = ~curve
    lab_f, nf = ndimage.label(free)  # 4-connected empty regions
    lab_c, nc = ndimage.label(curve, structure=np.ones((3, 3)))  # 8-connected curves
    if nf == 0:
        return curve.copy()
    # which empty regions touch which curves
    pairs = []
    for sh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        a = lab_f
        b = np.roll(lab_c, sh, axis=(0, 1))
        m = (a > 0) & (b > 0)
        pairs.append(np.c_[a[m], b[m]])
    pairs = np.unique(np.vstack(pairs), axis=0) if pairs else np.zeros((0, 2), int)
    border = np.unique(np.r_[lab_f[0], lab_f[-1], lab_f[:, 0], lab_f[:, -1]])
    depth_f = np.full(nf + 1, -1)
    depth_c = np.full(nc + 1, -1)
    frontier = border[border > 0]
    depth_f[frontier] = 0
    d = 0
    while len(frontier):
        curves = np.unique(pairs[np.isin(pairs[:, 0], frontier), 1])
        curves = curves[depth_c[curves] < 0]
        depth_c[curves] = d
        nxt = np.unique(pairs[np.isin(pairs[:, 1], curves), 0])
        nxt = nxt[depth_f[nxt] < 0]
        d += 1
        depth_f[nxt] = d
        frontier = nxt
    depth_f[depth_f < 0] = 1  # enclosed but unreachable: treat as inside
    inside = (depth_f % 2 == 1)[lab_f] & free
    return inside | curve


def fill_by_normals(curve: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """Solid of a 2D slice from the outline and the points' normals.

    Normals point out of the material, so an empty region is inside when the
    outline around it faces away from it. This keeps a bolt hole in a flat face
    open, which the even-odd rule cannot (the face is a filled blob in its slice).
    Regions with no usable normals fall back to the even-odd rule.
    """
    free = ~curve
    lab, nf = ndimage.label(free)
    if nf == 0:
        return curve.copy()
    vote = np.zeros(nf + 1)
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        c = np.roll(curve, (-dr, -dc), axis=(0, 1))  # c[e] = curve[e + d]
        m = (lab > 0) & c
        w = np.roll(n0, (-dr, -dc), axis=(0, 1))[m] * dr + np.roll(n1, (-dr, -dc), axis=(0, 1))[m] * dc
        vote += np.bincount(lab[m], weights=w, minlength=nf + 1)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    inside = vote > 0
    undecided = np.abs(vote) < 1e-9
    if undecided[1:].any():
        eo = fill_even_odd(curve)
        inside_eo = np.zeros(nf + 1, bool)
        sel = lab > 0
        np.logical_or.at(inside_eo, lab[sel], eo[sel])
        inside[undecided] = inside_eo[undecided]
    inside[border] = False
    inside[0] = False
    return inside[lab] | curve


def solid_along(P: np.ndarray, Nr: np.ndarray, lo: np.ndarray, h: float, shape, axis: int) -> np.ndarray:
    """Stack of filled slices along one axis, straight from the points."""
    out = np.zeros(shape, bool)
    uv = [b for b in range(3) if b != axis]
    k = np.floor((P[:, axis] - lo[axis]) / h).astype(np.int64)
    # steep surfaces also show in the neighbouring slices, so thin walls stay
    # continuous; near-flat faces only count in their own slice
    steep = np.abs(Nr[:, axis]) < 0.7
    order = np.argsort(k, kind="stable")
    ks = k[order]
    close = np.ones((3, 3), bool)
    n_u, n_v = shape[uv[0]], shape[uv[1]]
    for i in range(shape[axis]):
        a, b = np.searchsorted(ks, [i - 1, i + 2])
        sel = order[a:b]
        sel = sel[(k[sel] == i) | steep[sel]]
        if len(sel) == 0:
            continue
        r = np.floor((P[sel, uv[0]] - lo[uv[0]]) / h).astype(np.int64)
        c = np.floor((P[sel, uv[1]] - lo[uv[1]]) / h).astype(np.int64)
        curve = np.zeros((n_u, n_v), bool)
        curve[r, c] = True
        n0 = np.zeros((n_u, n_v))
        n1 = np.zeros((n_u, n_v))
        np.add.at(n0, (r, c), Nr[sel, uv[0]])
        np.add.at(n1, (r, c), Nr[sel, uv[1]])
        closed = ndimage.binary_closing(curve, close, border_value=0) | curve
        sl = [slice(None)] * 3
        sl[axis] = i
        out[tuple(sl)] = fill_by_normals(closed, n0, n1)
    return out


# --------------------------------------------------------------- outline math

def douglas_peucker(P: np.ndarray, eps: float) -> np.ndarray:
    """Indices of the kept vertices."""
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = P[i + 1:j] - a
        dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return np.where(keep)[0]


def fit_circle(P: np.ndarray):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = float(np.sqrt(max(sol[2] + c @ c, 0)))
    return c, r, float(np.abs(np.linalg.norm(P - c, axis=1) - r).max())


def outline_math(img: np.ndarray, origin: np.ndarray, pitch: float, tol: float):
    """Closed outlines of a filled slice as lines and arcs, in model units."""
    import contourpy

    gen = contourpy.contour_generator(z=img.astype(np.float32), line_type="Separate")
    loops = []
    lines = [ln for ln in gen.lines(0.5) if len(ln) >= 4]
    # outer outlines and holes run opposite ways; the largest outline is an outer one
    areas = [0.5 * float(np.sum(ln[:-1, 0] * ln[1:, 1] - ln[1:, 0] * ln[:-1, 1])) for ln in lines]
    outer_sign = np.sign(areas[int(np.argmax(np.abs(areas)))]) if areas else 1.0
    for line, a_px in zip(lines, areas):
        if len(line) < 4:
            continue
        P = (line[:, ::-1] + 0.5) * pitch + origin  # (row, col) cell centres -> model (u, v)
        area = 0.5 * float(np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1]))
        if abs(area) < 4 * pitch ** 2:
            continue
        ki = douglas_peucker(P, tol)
        V = P[ki].copy()
        # snap almost-axis-aligned edges
        for k in range(len(V) - 1):
            d = V[k + 1] - V[k]
            if abs(d[0]) < 0.5 * pitch:
                V[k + 1, 0] = V[k, 0]
            elif abs(d[1]) < 0.5 * pitch:
                V[k + 1, 1] = V[k, 1]
        V[-1] = V[0]
        # arcs: runs of >= 3 edges turning the same way by moderate angles whose raw
        # contour points fit one circle within tolerance
        n = len(V) - 1
        e = np.diff(V, axis=0)
        ang = np.arctan2(e[:, 1], e[:, 0])
        turn = (np.r_[ang[1:], ang[:1]] - ang + np.pi) % (2 * np.pi) - np.pi  # at vertex k+1
        edges, k = [], 0
        while k < n:
            j = k
            s = 0.0
            while j + 1 < n and np.radians(2) < abs(turn[j]) < np.radians(50) and (s == 0 or np.sign(turn[j]) == s):
                s = np.sign(turn[j])
                j += 1
            if j - k >= 2:
                raw = P[ki[k]:ki[j + 1] + 1]
                c, r, err = fit_circle(raw)
                if err < tol and r < 0.5 * float(np.ptp(P, axis=0).max()) + 10 * pitch:
                    th = np.unwrap(np.arctan2(raw[:, 1] - c[1], raw[:, 0] - c[0]))
                    edges.append({"arc": [V[k].tolist(), V[j + 1].tolist()], "center": c.tolist(), "r": r,
                                  "start": float(th[0]), "sweep": float(th[-1] - th[0])})
                    k = j + 1
                    continue
            edges.append({"line": [V[k].tolist(), V[k + 1].tolist()]})
            k += 1
        loops.append({"hole": bool(np.sign(a_px) != outer_sign), "area": abs(area), "edges": edges})
    return loops


def line_eq(p, q) -> str:
    (u0, v0), (u1, v1) = p, q
    if abs(u1 - u0) < 1e-9:
        return f"u = {u0:.4g}"
    if abs(v1 - v0) < 1e-9:
        return f"v = {v0:.4g}"
    a, b = v1 - v0, u0 - u1
    return f"{a:.4g} u + {b:.4g} v = {a * u0 + b * v0:.4g}"


# ------------------------------------------------------------------ rendering

def depth_render(V: np.ndarray, axis: int, path: Path, title: str):
    from PIL import Image

    # look from the high end of `axis` (from above when it is the vertical axis)
    Vr = np.flip(V, axis=axis)
    hit = Vr.any(axis=axis)
    first = np.argmax(Vr, axis=axis).astype(float)
    n = V.shape[axis]
    depth = np.where(hit, 1 - first / max(n - 1, 1), 0)
    if axis == 0:      # (y, z): y up
        depth, hit = np.flipud(depth), np.flipud(hit)
    elif axis == 2:    # (x, y) -> (y, x): y up
        depth, hit = np.flipud(depth.T), np.flipud(hit.T)
    else:              # (x, z) -> (z, x): plan view
        depth, hit = depth.T, hit.T
    gy, gx = np.gradient(depth)
    shade = np.clip(0.35 + 0.65 * depth + 25 * (gx - gy), 0.05, 1)
    rgb = np.stack([shade * 0.86, shade * 0.89, shade * 0.95], -1)
    rgb[~hit] = (0.11, 0.11, 0.12)
    img = (rgb * 255).astype(np.uint8)
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(path)


def boundary_points(V: np.ndarray, origin: np.ndarray, pitch: float, n: int, rng):
    """Random points on the faces between inside and outside cells, and their total area."""
    Vp = np.pad(V, 1).astype(np.int8)
    found, total = [], 0
    for ax in range(3):
        idx = np.argwhere(np.diff(Vp, axis=ax) != 0)
        total += len(idx)
        found.append((ax, idx))
    out = []
    for ax, idx in found:
        k = int(round(n * len(idx) / max(total, 1)))
        if k == 0 or len(idx) == 0:
            continue
        pick = idx[rng.choice(len(idx), min(k, len(idx)), replace=False)].astype(float)
        pick += rng.random(pick.shape) - 0.5  # anywhere on the face ...
        pick[:, ax] = np.round(pick[:, ax]) + 0.5  # ... which sits between two cells
        out.append((pick - 0.5) * pitch + origin)
    return np.vstack(out), total * pitch ** 2


def boundary_mesh(V: np.ndarray, origin: np.ndarray, pitch: float) -> trimesh.Trimesh:
    """Quads between inside and outside voxels."""
    Vp = np.pad(V, 1)
    verts, faces = [], []
    corner = {
        0: [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
        1: [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],
        2: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
    }
    nv = 0
    for ax in range(3):
        d = np.diff(Vp.astype(np.int8), axis=ax)
        for sign in (1, -1):
            idx = np.argwhere(d == sign).astype(float)
            if len(idx) == 0:
                continue
            idx[:, ax] += 1
            base = idx - 1  # undo the padding
            q = np.stack([base + np.array(c) for c in corner[ax]], 1)
            q[:, :, ax] = base[:, None, ax]
            if sign == -1:
                q = q[:, ::-1]
            verts.append(q.reshape(-1, 3))
            k = np.arange(len(idx))[:, None] * 4 + nv
            faces.append(np.r_[np.c_[k, k + 1, k + 2], np.c_[k, k + 2, k + 3]])
            nv += 4 * len(idx)
    Vt = np.vstack(verts) * pitch + origin
    m = trimesh.Trimesh(Vt, np.vstack(faces), process=False)
    m.merge_vertices()
    return m


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--pitch", type=float, default=0.025)
    ap.add_argument("--up", default="y")
    ap.add_argument("--points", type=int, default=0, help="0 = about 3 per surface cell")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    T = lambda: f"[{time.perf_counter() - t0:6.1f}s]"

    mesh = trimesh.load(args.mesh, force="mesh")
    h = args.pitch
    n_pts = args.points or int(6 * mesh.area / h ** 2)
    P, fi = trimesh.sample.sample_surface(mesh, n_pts, seed=7)
    Nr = mesh.face_normals[fi]
    print(T(), f"point cloud: {len(P):,} points with normals, cell {h}", flush=True)

    # a quarter-cell offset keeps flat faces at round coordinates off the slice boundaries
    lo = P.min(0) - 2.37 * h
    shape = tuple(int(v) for v in np.floor((P.max(0) - lo) / h).astype(np.int64) + 3)
    print(T(), f"grid {shape}", flush=True)

    up = AX.index(args.up)
    order = [up] + [a for a in range(3) if a != up]
    solids, report, maths = {}, {}, {}
    for a in order:
        Sa = solid_along(P, Nr, lo, h, shape, a)
        solids[a] = Sa
        # outline math per slice, merged into extrusion steps
        steps, prev, prev_img = [], None, None
        n_vert = 0
        uv_axes = [b for b in range(3) if b != a]
        for i in range(shape[a]):
            sl = [slice(None)] * 3
            sl[a] = i
            img = Sa[tuple(sl)]
            if not img.any():
                prev_img = None
                continue
            if prev_img is not None:
                xor = np.count_nonzero(img ^ prev_img)
                if xor <= 0.002 * max(np.count_nonzero(img), 1) + 2:
                    steps[-1]["to"] = float(lo[a] + (i + 1) * h)
                    continue
            loops = outline_math(img, lo[uv_axes], h, 0.75 * h)
            n_vert += sum(len(lp["edges"]) for lp in loops)
            steps.append({"from": float(lo[a] + i * h), "to": float(lo[a] + (i + 1) * h), "loops": loops})
            prev_img = img
        maths[AX[a]] = steps
        n_loops = sum(len(s["loops"]) for s in steps)
        n_holes = sum(lp["hole"] for s in steps for lp in s["loops"])
        n_arcs = sum("arc" in e for s in steps for lp in s["loops"] for e in lp["edges"])
        report[AX[a]] = {"slices": int(shape[a]), "steps": len(steps), "loops": n_loops, "holes": int(n_holes),
                         "edges": int(n_vert), "arcs": int(n_arcs), "solid_cells": int(Sa.sum())}
        print(T(), f"along {AX[a]}: {shape[a]} slices -> {len(steps)} extrusion steps, {n_loops} outlines "
                   f"({n_holes} holes), {n_vert} edges ({n_arcs} arcs)", flush=True)
        depth_render(Sa, up, out / f"solid_{AX[a]}_top.png", f"slices along {AX[a]}")
        depth_render(Sa, order[1], out / f"solid_{AX[a]}_front.png", f"slices along {AX[a]}")

    U = solids[0] & solids[1] & solids[2]
    print(T(), f"unified: {U.sum():,} solid cells", flush=True)
    for a in range(3):
        report[AX[a]]["removed_by_unify_pct"] = float(100 * (1 - U.sum() / max(solids[a].sum(), 1)))
    depth_render(U, up, out / "unified_top.png", "unified")
    depth_render(U, order[1], out / "unified_front.png", "unified")
    depth_render(U, order[2], out / "unified_side.png", "unified")

    # accuracy: boundary of each solid against the true surface
    truth, _ = trimesh.sample.sample_surface(mesh, 1_000_000, seed=11)
    tt = cKDTree(truth)
    diag = float(np.linalg.norm(mesh.extents))
    rng = np.random.default_rng(3)
    for name, V in [(f"along_{AX[a]}", solids[a]) for a in order] + [("unified", U)]:
        b, b_area = boundary_points(V, lo, h, 300_000, rng)
        d_rt = tt.query(b)[0]
        d_tr = cKDTree(b).query(truth[::5])[0]
        report[name] = {"recon_to_truth_mean_pct": float(d_rt.mean() / diag * 100),
                        "recon_to_truth_p95_pct": float(np.percentile(d_rt, 95) / diag * 100),
                        "truth_to_recon_mean_pct": float(d_tr.mean() / diag * 100),
                        "truth_uncovered_pct": float((d_tr > 2 * h).mean() * 100),
                        "boundary_area": float(b_area)}
        print(T(), f"{name}: recon->truth mean {report[name]['recon_to_truth_mean_pct']:.3f}% "
                   f"p95 {report[name]['recon_to_truth_p95_pct']:.3f}% | truth->recon mean "
                   f"{report[name]['truth_to_recon_mean_pct']:.3f}%, more than 2 cells away "
                   f"{report[name]['truth_uncovered_pct']:.1f}%  (cell area {b_area:.0f} vs {mesh.area:.0f})", flush=True)
    # a viewable mesh of the unified solid (coarsened on big grids)
    Uv, hv = U, h
    while Uv.size > 40e6 or np.count_nonzero(np.diff(Uv.astype(np.int8), axis=0)) > 1.5e6:
        s = [d - d % 2 for d in Uv.shape]
        Uv = Uv[:s[0], :s[1], :s[2]].reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).mean(axis=(1, 3, 5)) >= 0.5
        hv *= 2
    boundary_mesh(Uv, lo, hv).export(out / "unified_voxels.stl")
    print(T(), f"wrote unified_voxels.stl (cell {hv:g})", flush=True)

    # slice plots: a few vertical slices with points and the fitted outline math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = maths[AX[up]]
    picks = [steps[int(k)] for k in np.linspace(0, len(steps) - 1, 6)] if steps else []
    uv = [b for b in range(3) if b != up]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor="#1b1b1d")
    for ax, st in zip(axs.ravel(), picks):
        mid = 0.5 * (st["from"] + st["to"])
        m = np.abs(P[:, up] - min(mid, st["from"] + 0.5 * h)) < 0.5 * h
        ax.scatter(P[m, uv[0]], P[m, uv[1]], s=0.3, c="#9a9a9a", linewidths=0)
        for lp in st["loops"]:
            col = "#e5734a" if lp["hole"] else "#4aa3e5"
            for e in lp["edges"]:
                if "line" in e:
                    (u0, v0), (u1, v1) = e["line"]
                    ax.plot([u0, u1], [v0, v1], color=col, lw=1)
                else:
                    c, r = np.array(e["center"]), e["r"]
                    tt_ = np.linspace(e["start"], e["start"] + e["sweep"], 64)
                    ax.plot(c[0] + r * np.cos(tt_), c[1] + r * np.sin(tt_), color="#5fd17a", lw=1.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#1b1b1d")
        ax.tick_params(colors="#888")
        n_e = sum(len(lp["edges"]) for lp in st["loops"])
        ax.set_title(f"{AX[up]} = {st['from']:.2f} .. {st['to']:.2f}   {len(st['loops'])} outlines, {n_e} edges",
                     color="#ddd", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "slices_vertical.png", dpi=110, facecolor=fig.get_facecolor())

    # one example of the math for the report
    if steps:
        mid = steps[len(steps) // 2]
        lp = max(mid["loops"], key=lambda l: l["area"])
        report["example"] = {"slab": [mid["from"], mid["to"]], "axis": AX[up], "uv": [AX[b] for b in uv],
                             "first_edges": [line_eq(*e["line"]) if "line" in e else
                                             f"(u - {e['center'][0]:.4g})^2 + (v - {e['center'][1]:.4g})^2 = {e['r']:.4g}^2"
                                             for e in lp["edges"][:12]]}
    (out / "outline_math.json").write_text(json.dumps(maths))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(T(), "done", flush=True)


if __name__ == "__main__":
    main()
