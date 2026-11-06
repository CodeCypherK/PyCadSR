"""Coarse-to-fine decomposition: whole shape -> directions -> planes -> details.

Instead of picking local frames for small point clusters, start from the object
as a whole and refine, the way a person reads a building:

    level 0  the overall shape, e.g. a box  max(|x|/a, |y|/b, |z|/c) = 1
    level 1  dominant surface directions (the box's faces, bevels)
    level 2  planes at distinct depths along those directions, taken exactly
             from the triangles (a polygonal model already contains them)
    level 3  what is left (pipes, rounded edges, freeform), handed to the
             point-based primitive search and symbolic regression

Every level only refines what the level above did not explain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

AXES = np.eye(3)


@dataclass
class Plane:
    normal: np.ndarray
    offset: float
    faces: np.ndarray
    area: float


@dataclass
class Hierarchy:
    frame: np.ndarray              # 3x3, rows = global axes (level 0)
    center: np.ndarray
    half: np.ndarray               # box half-extents along the frame axes
    expression: str
    latex: str
    directions: list = field(default_factory=list)   # (normal, share) level 1
    planes: list = field(default_factory=list)       # level 2
    plane_of_face: np.ndarray | None = None          # -1 = not planar (level 3)
    level_labels: dict = field(default_factory=dict)  # level -> per-face label


def mesh_suits_hierarchy(mesh: trimesh.Trimesh) -> tuple[bool, float]:
    """Use coarse-to-fine when most of the surface is exactly planar along few directions."""
    dirs, rem = dominant_directions(mesh.face_normals, mesh.area_faces, max_dirs=24)
    covered = 1.0 - mesh.area_faces[rem].sum() / max(mesh.area_faces.sum(), 1e-300)
    return covered >= 0.5, float(covered)


def dominant_directions(N: np.ndarray, A: np.ndarray, cos_seed: float = np.cos(np.radians(2.0)),
                        min_share: float = 0.003, max_dirs: int = 32):
    """Area-weighted peaks on the sphere of triangle normals (mean-shift refined).

    Returns ([(direction, faces, share)], remaining_mask). Directions within half a
    degree of a world axis are snapped to it.
    """
    total = float(A.sum()) or 1.0
    remaining = np.ones(len(N), bool)
    out = []
    while len(out) < max_dirs:
        idx = np.where(remaining)[0]
        if len(idx) == 0:
            break
        # vote on a 101^3 grid of normal directions (one integer key per face)
        key = np.round(N[idx] * 50).astype(np.int64) + 50
        flat = (key[:, 0] * 101 + key[:, 1]) * 101 + key[:, 2]
        best = int(np.argmax(np.bincount(flat, weights=A[idx], minlength=101 ** 3)))
        d = np.array([best // 101 ** 2, (best // 101) % 101, best % 101], float) - 50
        d /= np.linalg.norm(d) or 1.0
        for _ in range(3):
            near = idx[(N[idx] @ d) > cos_seed]
            if len(near) == 0:
                break
            d = (N[near] * A[near, None]).sum(0)
            d /= np.linalg.norm(d) or 1.0
        k = int(np.argmax(np.abs(AXES @ d)))
        if abs(AXES[k] @ d) > np.cos(np.radians(0.5)):
            d = AXES[k] * np.sign(AXES[k] @ d)
        near = idx[(N[idx] @ d) > cos_seed]
        share = float(A[near].sum() / total)
        if share < min_share:
            break
        out.append((d, near, share))
        remaining[near] = False
    return out, remaining


def global_frame(dirs) -> np.ndarray:
    """Three orthogonal axes from the strongest directions (world axes when they dominate)."""
    if not dirs:
        return AXES.copy()
    a = dirs[0][0]
    b = None
    for d, _, _ in dirs[1:]:
        if abs(d @ a) < np.cos(np.radians(80)):
            b = d - (d @ a) * a
            break
    if b is None:
        b = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        b = b - (b @ a) * a
    b /= np.linalg.norm(b)
    c = np.cross(a, b)
    F = np.vstack([a, b, c])
    # order like the world axes when they coincide, for readable expressions
    order = [int(np.argmax(np.abs(F @ AXES[i]))) for i in range(3)]
    if sorted(order) == [0, 1, 2]:
        F = np.vstack([F[order[i]] * np.sign(F[order[i]] @ AXES[i]) for i in range(3)])
    return F


def box_expression(center, half, frame) -> tuple[str, str]:
    names = ["x", "y", "z"]
    world = np.allclose(np.abs(frame), np.eye(3), atol=1e-6)
    parts, tparts = [], []
    for i in range(3):
        if world:
            v = names[i]
            c = center[i]
            if abs(c) <= 1e-9:
                arg = v
            else:
                arg = f"{v} {'-' if c > 0 else '+'} {abs(c):.4g}"
            targ = arg
        else:
            arg = targ = f"u_{i + 1}"
        parts.append(f"|{arg}|/{half[i]:.4g}")
        tparts.append(rf"\frac{{|{targ}|}}{{{half[i]:.4g}}}")
    return "max(" + ", ".join(parts) + ") = 1", r"\max\left(" + ", ".join(tparts) + r"\right) = 1"


def find_planes(mesh: trimesh.Trimesh, dirs, tol: float, cos_assign: float = np.cos(np.radians(0.5))):
    """Planes at distinct depths along each dominant direction, from exact triangle positions."""
    N, A, C = mesh.face_normals, mesh.area_faces, mesh.triangles_center
    V, F = mesh.vertices, mesh.faces
    planes: list[Plane] = []
    for d, near, _ in dirs:
        strict = (N[near] @ d) > cos_assign  # truly coplanar, not a curved facet nearly facing d
        faces = near[strict]
        if len(faces) == 0:
            continue
        off = C[faces] @ d
        order = np.argsort(off)
        o = off[order]
        cut = np.r_[0, np.where(np.diff(o) > tol)[0] + 1, len(o)]
        offs, members = [], []
        for s, e in zip(cut[:-1], cut[1:]):
            f = faces[order[s:e]]
            offs.append(float(np.average(o[s:e], weights=A[f])))
            members.append([f])
        offs = np.array(offs)
        # Tiny triangles of a detailed model often have normals a little off (export
        # rounding) although they lie on the plane. Keep a face when every corner is
        # within tol of one of these planes: that is exact, whatever its normal says.
        loose = near[~strict]
        if len(loose):
            vo = V[F[loose]] @ d
            mid = vo.mean(axis=1)
            srt = np.argsort(offs)
            j = np.searchsorted(offs[srt], mid)
            cand = np.stack([np.clip(j - 1, 0, len(offs) - 1), np.clip(j, 0, len(offs) - 1)], axis=1)
            pick = cand[np.arange(len(mid)), np.abs(offs[srt][cand] - mid[:, None]).argmin(axis=1)]
            best = srt[pick]
            ok = np.abs(vo - offs[best][:, None]).max(axis=1) <= tol
            for k in np.unique(best[ok]):
                members[k].append(loose[ok & (best == k)])
        for off_k, parts in zip(offs, members):
            f = np.concatenate(parts)
            planes.append(Plane(d.copy(), float(off_k), f, float(A[f].sum())))
    planes.sort(key=lambda p: -p.area)
    plane_of_face = np.full(len(mesh.faces), -1, np.int64)
    for i, p in enumerate(planes):
        plane_of_face[p.faces] = i
    return planes, plane_of_face


def build(mesh: trimesh.Trimesh, sag: float = 0.0, progress=None) -> Hierarchy:
    progress = progress or (lambda *a: None)
    N, A = mesh.face_normals, mesh.area_faces
    diag = float(np.linalg.norm(mesh.extents)) or 1.0

    progress(0.05, "Level 1: dominant surface directions")
    dirs, remaining = dominant_directions(N, A)
    frame = global_frame(dirs)

    progress(0.15, "Level 0: overall shape")
    local = mesh.vertices @ frame.T
    lo, hi = local.min(0), local.max(0)
    center_local = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    center = center_local @ frame
    expr, tex = box_expression(center, half, frame)

    progress(0.25, "Level 2: planes at each depth")
    tol = max(1e-4 * diag, 2 * sag)
    planes, plane_of_face = find_planes(mesh, dirs, tol)

    h = Hierarchy(frame, center, half, expr, tex,
                  [(d, share) for d, _, share in dirs], planes, plane_of_face)
    # per-face labels for each level, for the live coarse-to-fine view
    h.level_labels[0] = np.zeros(len(mesh.faces), np.int64)
    lab1 = np.full(len(mesh.faces), len(dirs), np.int64)
    for i, (_, near, _) in enumerate(dirs):
        lab1[near] = i
    h.level_labels[1] = lab1
    lab2 = plane_of_face.copy()
    lab2[lab2 < 0] = len(planes)
    h.level_labels[2] = lab2
    return h


def box_mesh(h: Hierarchy) -> tuple[np.ndarray, np.ndarray]:
    """Level-0 preview: the fitted box as triangles."""
    b = trimesh.creation.box(extents=2 * h.half)
    V = b.vertices @ h.frame + h.center
    return V, b.faces


def plane_patches(mesh: trimesh.Trimesh, plane_of_face: np.ndarray) -> np.ndarray:
    """Connected pieces of each plane (used to report how many CAD faces the planes make)."""
    adj = mesh.face_adjacency
    a, b = adj[:, 0], adj[:, 1]
    same = (plane_of_face[a] == plane_of_face[b]) & (plane_of_face[a] >= 0)
    n = len(mesh.faces)
    _, comp = connected_components(csr_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(n, n)),
                                   directed=False)
    return comp


def sample_faces(mesh: trimesh.Trimesh, faces: np.ndarray, counts: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """counts[i] random points on triangle faces[i] (uniform barycentric)."""
    rep = np.repeat(faces, counts)
    r1, r2 = rng.random(len(rep)), rng.random(len(rep))
    s = np.sqrt(r1)
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
    T = mesh.triangles[rep]
    P = T[:, 0] * w0[:, None] + T[:, 1] * w1[:, None] + T[:, 2] * w2[:, None]
    return P, rep


def allocate(area: np.ndarray, budget: int, minimum: np.ndarray | None, rng) -> np.ndarray:
    """Area-proportional sample counts with stochastic rounding (and optional per-face minimum)."""
    if budget <= 0 or len(area) == 0:
        return np.zeros(len(area), np.int64)
    expect = area / max(area.sum(), 1e-300) * budget
    counts = np.floor(expect).astype(np.int64)
    counts += rng.random(len(area)) < (expect - counts)
    if minimum is not None:
        counts = np.maximum(counts, minimum)
    return counts


def cube_map_split(N: np.ndarray, nbrs: np.ndarray, idx: np.ndarray, axes: np.ndarray, min_pts: int,
                   cos_merge: float = np.cos(np.radians(75)), smooth_iters: int = 3, P: np.ndarray | None = None):
    """Split a freeform region into pieces that each face one side of the overall box.

    Each piece is a height field over that side's direction, so it gets a bounded
    equation w = f(u, v) instead of a closed blob around the whole region (a flat
    relief panel wrapped in r = f(theta, lat) comes out as a donut). Points take the
    box side their normal points to most, labels are smoothed on the neighbour
    graph, and pieces are merged back while the result is still a height field.

    nbrs: (n, k) neighbour indices into the full cloud (a point may list itself).
    Returns [(point indices, direction)].
    """
    D = np.vstack([axes, -axes])
    m = len(idx)
    local = np.full(len(N), -1, np.int64)
    local[idx] = np.arange(m)
    nb = local[nbrs[idx]]
    rows = np.repeat(np.arange(m), nb.shape[1])
    cols = nb.ravel()
    keep = (cols >= 0) & (cols != rows)
    rows, cols = rows[keep], cols[keep]
    lab = np.argmax(N[idx] @ D.T, axis=1)
    for _ in range(smooth_iters):
        votes = np.zeros((m, 6))
        np.add.at(votes, (rows, lab[cols]), 1.0)
        votes[np.arange(m), lab] += 1.5
        lab = votes.argmax(axis=1)
    same = lab[rows] == lab[cols]
    _, comp = connected_components(csr_matrix((np.ones(int(same.sum())), (rows[same], cols[same])),
                                              shape=(m, m)), directed=False)
    side = np.zeros(comp.max() + 1, np.int64)
    side[comp] = lab
    Nr = N[idx]
    # merge small pieces into a neighbour while that neighbour stays a height field
    for _ in range(4):
        sizes = np.bincount(comp)
        cross = comp[rows] != comp[cols]
        if not cross.any():
            break
        pa, pb = comp[rows[cross]], comp[cols[cross]]
        key, cnt = np.unique(pa * (comp.max() + 1) + pb, return_counts=True)
        a_of, b_of = key // (comp.max() + 1), key % (comp.max() + 1)
        target = np.arange(len(sizes))
        changed = False
        for c in np.argsort(sizes):
            if sizes[c] == 0 or target[c] != c:
                continue
            sel = a_of == c
            if not sel.any():
                continue
            order = np.argsort(-cnt[sel])
            mine = comp == c
            for nbc in b_of[sel][order]:
                t = nbc
                while target[t] != t:
                    t = target[t]
                if t == c:
                    continue
                ok = (Nr[mine] @ D[side[t]] > cos_merge).mean() >= 0.97
                if ok or sizes[c] < min_pts:
                    target[c] = t
                    sizes[t] += sizes[c]
                    changed = True
                    break
        if not changed:
            break
        for c in range(len(target)):
            t = c
            while target[t] != t:
                t = target[t]
            target[c] = t
        comp = target[comp]
    # crumbs with no neighbour in the region join the spatially nearest real piece
    sizes = np.bincount(comp)
    small = sizes[comp] < min_pts
    if P is not None and small.any() and (~small).any():
        from scipy.spatial import cKDTree
        big = np.where(~small)[0]
        _, j = cKDTree(P[idx[big]]).query(P[idx[small]])
        comp[small] = comp[big[j]]
    out = []
    for c in np.unique(comp):
        part = idx[comp == c]
        out.append((part, D[side[c]].copy()))
    return out
