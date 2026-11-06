"""Mirror symmetry: a part designed symmetric is modelled as one half and its mirror image.

find_mirror   finds a plane the surface is mirrored in (from the surface itself, so it works
              whatever the triangles look like) and says how well the two halves agree.
symmetrize    cuts the mesh on that plane, keeps one half (averaged with the mirror image of
              the other half when they differ a little) and adds its exact mirror image.
mirror_segment segments the kept half and mirrors its faces: a face the plane runs through
              stays one face when it continues smoothly across, every other face gets a
              mirrored twin, so the faces, edges and surfaces come out exactly symmetric.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# the halves must agree everywhere: half the surface within 0.1 % of the part's size and
# nearly all of it within 0.5 % (a hole or boss on one side only breaks that at once)
MIRROR_P50 = 1e-3
MIRROR_P99 = 5e-3
MIRROR_MAX = 1e-2
MIRROR_TURN_DEG = 20.0  # a feature on one side only turns the surface: its walls face elsewhere
DIRECTIONS = ("normal", "axis")


def _reflect_pts(X, n, d):
    return X - 2.0 * ((X @ n) - d)[..., None] * n


def _reflect_dir(v, n):
    return v - 2.0 * (v @ n) * n


def _area_samples(V, F, k, rng, centroids=False):
    """k points spread by area (and every triangle's centre too, if asked), their facet
    normals, the triangle areas and the triangle of each point."""
    A = V[F]
    cr = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = cr / np.maximum(2 * area[:, None], 1e-300)
    t = rng.choice(len(F), k, p=area / area.sum())
    if centroids:
        t = np.r_[t, np.arange(len(F))]
        k = len(t)
    r1, r2 = rng.random(k), rng.random(k)
    s = np.sqrt(r1)
    if centroids:
        s[-len(F):], r2[-len(F):] = 2 / 3, 0.5  # barycentre
    P = A[t, 0] * (1 - s)[:, None] + A[t, 1] * (s * (1 - r2))[:, None] + A[t, 2] * (s * r2)[:, None]
    return P, nrm[t], area, t


def _ref_count(nf):
    return int(min(3_000_000, max(200_000, 6 * nf)))


def _sag_radius(V, F):
    """Per triangle: how far its chord can lie inside the curved surface (chord x turn / 8)
    and the surface's local radius (chord / turn; flat = inf), from the vertex normals."""
    VN = _vertex_normals(V, F)
    sag = np.zeros(len(F))
    rad = np.full(len(F), np.inf)
    for a_, b_ in ((0, 1), (1, 2), (2, 0)):
        L = np.linalg.norm(V[F[:, a_]] - V[F[:, b_]], axis=1)
        th = np.arccos(np.clip((VN[F[:, a_]] * VN[F[:, b_]]).sum(1), -1, 1))
        sag = np.maximum(sag, L * th / 8)
        rad = np.minimum(rad, np.where(th > 1e-9, L / np.maximum(th, 1e-12), np.inf))
    return sag, rad


def _closest_on_tris(p, a, b, c):
    """Closest points on triangles abc to points p (all (..., 3)); Ericson's regions."""
    dot = lambda x, y: (x * y).sum(-1)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-300, 1e-300, den)
    out = a + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]  # inside
    done = np.zeros(p.shape[:-1], bool)

    def put(m, val):
        nonlocal out, done
        m = m & ~done
        out = np.where(m[..., None], val, out)
        done |= m

    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    t = d1 / np.where(np.abs(d1 - d3) < 1e-300, 1e-300, d1 - d3)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * t[..., None])
    put((d6 >= 0) & (d5 <= d6), c)
    t = d2 / np.where(np.abs(d2 - d6) < 1e-300, 1e-300, d2 - d6)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * t[..., None])
    e1, e2 = d4 - d3, d5 - d6
    t = e1 / np.where(np.abs(e1 + e2) < 1e-300, 1e-300, e1 + e2)
    put((va <= 0) & (e1 >= 0) & (e2 >= 0), b + (c - b) * t[..., None])
    return out


def _nearest_on_mesh(X, V, F, tree, tq, k=8):
    """Closest point on the mesh (V, F) to each X, among the triangles of the k samples
    nearest to it (tree over samples, tq their triangles): (points, triangles, distances)."""
    out = np.empty_like(X)
    tri = np.empty(len(X), np.int64)
    dist = np.empty(len(X))
    for s0 in range(0, len(X), 50_000):
        x = X[s0:s0 + 50_000]
        _, j = tree.query(x, k=k)
        t = tq[j]
        A, B, C = V[F[t, 0]], V[F[t, 1]], V[F[t, 2]]
        cp = _closest_on_tris(np.repeat(x[:, None], k, 1), A, B, C)
        dd = np.linalg.norm(cp - x[:, None], axis=2)
        m = dd.argmin(1)
        r = np.arange(len(x))
        out[s0:s0 + len(x)], tri[s0:s0 + len(x)], dist[s0:s0 + len(x)] = cp[r, m], t[r, m], dd[r, m]
    return out, tri, dist


def find_mirror(V, F, log=None) -> dict | None:
    """The mirror plane {normal, d, p50, p99, p999, max} (distances in model units), or None."""
    from scipy.optimize import minimize

    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if len(F) < 50:
        return None
    rng = np.random.default_rng(0)
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    # a coarse cloud to find the plane, a dense one (every triangle) to measure it
    Q, QN, area, _ = _area_samples(V, F, 200_000, rng)
    tree = cKDTree(Q)
    Qd, _, _, tQd = _area_samples(V, F, _ref_count(len(F)), rng, centroids=True)
    P, PN, _, _ = _area_samples(V, F, 20_000, rng)
    C = V[F].mean(axis=1)
    c = (C * area[:, None]).sum(0) / area.sum()

    def dist(n, d, pts):
        R = _reflect_pts(pts, n, d)
        _, j = tree.query(R)
        return np.abs(((R - Q[j]) * QN[j]).sum(1))

    def cost(n, d, pts):
        x = np.sort(dist(n, d, pts))
        return float(x[: int(0.8 * len(x))].mean())  # trimmed: a local difference must not steer it

    X = C - c
    _, U = np.linalg.eigh((X * area[:, None]).T @ X)
    cands = [np.eye(3)[k] for k in range(3)] + [U[:, k] for k in range(3)]
    g = (1 + 5 ** 0.5) / 2
    for i in range(120):  # a half sphere of directions
        z = 1 - (i + 0.5) / 120
        r = np.sqrt(1 - z * z)
        cands.append(np.array([r * np.cos(2 * np.pi * i / g), r * np.sin(2 * np.pi * i / g), z]))
    few = P[:3000]
    quick = sorted(((cost(n, float(c @ n), few), i) for i, n in enumerate(cands)))[:5]
    best = None
    for _, i in quick:
        n0 = cands[i] / np.linalg.norm(cands[i])
        u = np.cross(n0, [1.0, 0, 0] if abs(n0[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n0, u)

        def f(x, n0=n0, u=u, v=v):
            n = n0 + x[0] * u + x[1] * v
            n /= np.linalg.norm(n)
            return cost(n, float(c @ n0) + x[2] * diag, P[:6000])

        r = minimize(f, np.zeros(3), method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-9 * diag, "initial_simplex":
                              np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.01]]), "maxiter": 400})
        n = n0 + r.x[0] * u + r.x[1] * v
        n /= np.linalg.norm(n)
        d = float(c @ n0) + r.x[2] * diag
        if best is None or r.fun < best[2]:
            best = (n, d, r.fun)
    n, d, fbest = best
    # designers mirror in a principal plane of the model: snap to one when it fits as well
    k = int(np.argmax(np.abs(n)))
    if abs(n[k]) > np.cos(np.radians(1.0)):
        e = np.sign(n[k]) * np.eye(3)[k]
        for dd in (0.0, float(d)):
            if cost(e, dd, P[:6000]) <= 1.1 * fbest + 1e-9 * diag:
                n, d = e, dd
                break
    _, tri, x = _nearest_on_mesh(_reflect_pts(P, n, d), V, F, cKDTree(Qd), tQd)
    sag, _ = _sag_radius(V, F)
    FN = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    FN /= np.maximum(np.linalg.norm(FN, axis=1, keepdims=True), 1e-300)
    RN = PN - 2 * (PN @ n)[:, None] * n
    turn = np.degrees(np.arccos(np.clip(np.abs((RN * FN[tri]).sum(1)), 0, 1)))
    tol = 3e-5 * diag
    exact = float(np.mean(x <= 1.5 * sag[tri] + tol))
    agree = float(np.mean(turn <= MIRROR_TURN_DEG))
    res = {"normal": n, "d": float(d), "p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99)),
           "p999": float(np.percentile(x, 99.9)), "max": float(x.max()), "diag": diag, "exact": exact >= 0.995}
    # exactly symmetric: the halves differ by no more than the facets do; nearly: by a
    # little everywhere, facing the same way (a hole, boss or thread on one side only differs
    # by its own size and its walls face elsewhere)
    near = (res["p50"] <= MIRROR_P50 * diag and res["p99"] <= MIRROR_P99 * diag
            and res["max"] <= MIRROR_MAX * diag and agree >= 0.99)
    ok = res["exact"] or near
    if log:
        log(f"mirror plane {np.round(n, 4)} at {d:.4g}: halves agree within {res['p50']:.3g} (median), "
            f"{res['p99']:.3g} (99 %), {res['max']:.3g} (max); {100 * exact:.1f} % within facet accuracy, "
            f"{100 * agree:.1f} % facing the same way -> "
            + ("exactly symmetric" if res["exact"] else "nearly symmetric" if near else "not symmetric"))
    return res if ok else None


def _vertex_normals(V, F):
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], cr)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)


def _cut_point(p0, p1, n0, n1, s0, s1, n, d, smooth):
    """Where the edge p0-p1 crosses the plane. On a curved surface the chord runs inside it:
    follow the curve the vertex normals describe (cubic Hermite) instead."""
    t = s0 / (s0 - s1)
    chord = p0 + t * (p1 - p0)
    if not smooth:
        return chord
    e = p1 - p0
    L = np.linalg.norm(e)
    T0 = e - (e @ n0) * n0
    T1 = e - (e @ n1) * n1
    if np.linalg.norm(T0) < 1e-12 or np.linalg.norm(T1) < 1e-12:
        return chord
    T0 *= L / np.linalg.norm(T0)
    T1 *= L / np.linalg.norm(T1)
    for _ in range(8):
        t2, t3 = t * t, t * t * t
        H = (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * T0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * T1
        dH = (6 * t2 - 6 * t) * p0 + (3 * t2 - 4 * t + 1) * T0 + (-6 * t2 + 6 * t) * p1 + (3 * t2 - 2 * t) * T1
        g, dg = H @ n - d, dH @ n
        if abs(dg) < 1e-300:
            return chord
        t = t - g / dg
        if not 0.0 <= t <= 1.0:
            return chord
    if np.linalg.norm(H - chord) > 0.25 * L:
        return chord
    return H - ((H @ n) - d) * n


def symmetrize(mesh, mirror: dict, tol: float = 0.0, log=None):
    """(mirrored mesh, info): the + side of the plane (averaged with the mirror image of the
    - side when they differ by more than tol) and its exact mirror image. Triangles 0..h-1
    are the kept half, h..2h-1 their mirror images in the same order."""
    n = np.asarray(mirror["normal"], float)
    d = float(mirror["d"])
    V = np.asarray(mesh.vertices, float).copy()
    F = np.asarray(mesh.faces, np.int64)
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    eps = 1e-7 * diag
    s = V @ n - d
    near = np.abs(s) < eps
    V[near] -= s[near, None] * n
    s[near] = 0.0
    VN = _vertex_normals(V, F)
    FN = mesh.face_normals
    # smooth edges: both triangles on it within 20 degrees; vertex normals that agree with them
    edges = {}
    adj = np.asarray(mesh.face_adjacency, np.int64)
    adj_e = np.sort(np.asarray(mesh.face_adjacency_edges, np.int64), axis=1)
    ang = np.asarray(mesh.face_adjacency_angles, float)
    for (a, b), (i, j), w in zip(adj, adj_e, ang):
        edges[(int(i), int(j))] = (int(a), int(b), float(w))
    newV = [V]
    extra = []  # coordinates of the cut points, in order
    cut_id = {}
    nv = len(V)

    def pos(i):
        return V[i] if i < len(V) else extra[i - len(V)]

    def cut(i, j):
        nonlocal nv
        key = (min(i, j), max(i, j))
        if key in cut_id:
            return cut_id[key]
        smooth = False
        if key in edges:
            a, b, w = edges[key]
            if w < np.radians(20):
                ok = [VN[i] @ FN[a], VN[i] @ FN[b], VN[j] @ FN[a], VN[j] @ FN[b]]
                smooth = min(ok) > np.cos(np.radians(25))
        p = _cut_point(V[i], V[j], VN[i], VN[j], s[i], s[j], n, d, smooth)
        newV.append(p[None])
        extra.append(p)
        cut_id[key] = nv
        nv += 1
        return cut_id[key]

    keep = []
    for f in F:
        ss = s[f]
        if (ss >= 0).all():
            if (ss == 0).all():
                continue  # lying in the plane: zero-thickness, nothing to mirror
            keep.append(f)
            continue
        if (ss <= 0).all():
            continue
        poly = []
        for q in range(3):
            i, j = int(f[q]), int(f[(q + 1) % 3])
            if s[i] >= 0:
                poly.append(i)
            if s[i] * s[j] < 0:
                poly.append(cut(i, j))
        if len(poly) == 3:
            keep.append(np.array(poly))
        elif len(poly) == 4:
            P4 = np.array([pos(i) for i in poly])
            if np.linalg.norm(P4[0] - P4[2]) <= np.linalg.norm(P4[1] - P4[3]):
                keep += [np.array([poly[0], poly[1], poly[2]]), np.array([poly[0], poly[2], poly[3]])]
            else:
                keep += [np.array([poly[0], poly[1], poly[3]]), np.array([poly[1], poly[2], poly[3]])]
    VA = np.vstack(newV)
    FA = np.array(keep, np.int64)
    used = np.unique(FA)
    remap = np.full(len(VA), -1, np.int64)
    remap[used] = np.arange(len(used))
    VA, FA = VA[used], remap[FA]
    sA = VA @ n - d
    on = np.abs(sA) < eps
    VA[on] -= sA[on, None] * n
    moved = 0.0
    if not mirror.get("exact", False) and tol > 0:
        # the halves differ a little: meet in the middle (half the difference each), the
        # move smoothed over the surface so the facets stay as flat as they were
        rng = np.random.default_rng(1)
        Vm = _reflect_pts(V, n, d)
        Fm = F[:, ::-1]
        Q, QN, _, tq = _area_samples(Vm, Fm, _ref_count(len(Fm)), rng, centroids=True)
        # a facet on a curved surface is a chord inside it, by its sag (chord x turn / 8):
        # a vertex that far from the mirror image's facets is still on the mirrored surface
        sag, _ = _sag_radius(Vm, Fm)
        cp, ti, dev = _nearest_on_mesh(VA, Vm, Fm, cKDTree(Q), tq)
        dev[dev <= 1.5 * sag[ti] + tol] = 0.0
        delta = -0.5 * (VA - cp) * (dev > 0)[:, None]
        delta[on] -= (delta[on] @ n)[:, None] * n
        E = np.r_[FA[:, [0, 1]], FA[:, [1, 2]], FA[:, [2, 0]]]
        deg = np.bincount(E[:, 0], minlength=len(VA)) + np.bincount(E[:, 1], minlength=len(VA))
        for _ in range(4):
            acc = np.zeros_like(delta)
            np.add.at(acc, E[:, 0], delta[E[:, 1]])
            np.add.at(acc, E[:, 1], delta[E[:, 0]])
            delta = 0.5 * delta + 0.5 * acc / np.maximum(deg, 1)[:, None]
        delta[on] -= (delta[on] @ n)[:, None] * n
        VA = VA + delta
        moved = float(np.linalg.norm(delta, axis=1).max()) if (dev != 0).any() else 0.0
        if not moved:
            VA = VA - delta
    # the mirror image: plane vertices are shared, the rest reflected, triangles turned over
    h = len(FA)
    other = np.where(~on)[0]
    vmap = np.arange(len(VA))
    vmap[other] = len(VA) + np.arange(len(other))
    VB = _reflect_pts(VA[other], n, d)
    FB = vmap[FA][:, ::-1]
    out = trimesh.Trimesh(np.vstack([VA, VB]), np.vstack([FA, FB]), process=False)
    tri_mirror = np.r_[np.arange(h) + h, np.arange(h)]
    info = {"normal": n, "d": d, "half": h, "tri_mirror": tri_mirror, "moved": moved,
            "p50": mirror.get("p50"), "p99": mirror.get("p99"), "max": mirror.get("max")}
    if log:
        log(f"mirror symmetry: kept one half ({h:,} triangles, {len(cut_id):,} cut on the plane"
            + (f", halves met in the middle, moved up to {moved:.3g}" if moved else "") + "), mirrored the other")
    return out, info


# ---------------------------------------------------------------- segmentation
def reflect_params(prm: dict, n, d) -> dict:
    out = {}
    for k, v in prm.items():
        a = np.asarray(v) if isinstance(v, (list, tuple, np.ndarray)) else None
        if a is not None and a.dtype.kind == "f" and a.shape == (3,):
            out[k] = _reflect_dir(a, n) if k in DIRECTIONS else _reflect_pts(a, n, d)
        else:
            out[k] = v
    return out


def symmetric_params(kind: str, prm: dict, n, d):
    """The surface made exactly symmetric in the plane, if it nearly is (else None)."""
    def onto(p):
        return p - ((p @ n) - d) * n

    def inplane(v):
        w = v - (v @ n) * n
        return w / np.linalg.norm(w)

    q = dict(prm)
    if kind == "plane":
        nn = np.asarray(prm["normal"], float)
        if abs(nn @ n) < 0.05:
            q["normal"], q["point"] = inplane(nn), onto(np.asarray(prm["point"], float))
            return q
        return None
    ax = np.asarray(prm["axis"], float) if "axis" in prm else None
    if kind == "sphere":
        q["center"] = onto(np.asarray(prm["center"], float))
        return q
    if kind in ("cylinder", "torus", "cone"):
        key = {"cylinder": "point", "torus": "center", "cone": "apex"}[kind]
        p = np.asarray(prm[key], float)
        if abs(ax @ n) < 0.05:  # axis in the plane
            q["axis"], q[key] = inplane(ax), onto(p)
            return q
        if abs(ax @ n) > 0.998 and kind != "cone":  # axis across the plane
            q["axis"] = np.sign(ax @ n) * n
            if kind == "torus":
                q[key] = onto(p)
            return q
    return None


def mirror_segment(mesh, info: dict, sharp_deg: float = 30.0, groups=None, sag: float = 0.0, progress=None,
                   log=None):
    """Face labels and regions [(kind, params, faces)] for a mesh made by symmetrize."""
    from .meshseg import TriMesh, _height_field, mesh_segment

    n, d, h = np.asarray(info["normal"], float), float(info["d"]), int(info["half"])
    tm_ = info["tri_mirror"]
    half = trimesh.Trimesh(mesh.vertices, mesh.faces[:h], process=False)
    half.remove_unreferenced_vertices()
    gh = None if groups is None else np.asarray(groups)[:h]
    _, regs, tol = mesh_segment(half, sharp_deg, gh, sag=sag, progress=progress)
    tm = TriMesh(mesh, sharp_deg, groups)
    adj = np.asarray(mesh.face_adjacency, np.int64)
    ang = np.asarray(mesh.face_adjacency_angles, float)
    across = (adj[:, 0] < h) != (adj[:, 1] < h)
    ca, cw = np.where(adj[across, 0] < h, adj[across, 0], adj[across, 1]), ang[across]
    touch_ang = {}
    for a, w in zip(ca, cw):
        touch_ang.setdefault(int(a), []).append(float(w))
    out = []
    merged = 0
    for kind, prm, faces in regs:
        faces = np.asarray(faces, np.int64)
        mf = tm_[faces]
        hit = [w for f in faces if int(f) in touch_ang for w in touch_ang[int(f)]]
        one = None
        if hit:
            U = np.r_[faces, mf]
            special = any(k.startswith(("helix", "tube")) for k in prm)
            if kind == "freeform" and not special:
                hit = np.degrees(hit)
                if np.median(hit) < 10 and max(hit) < 35 and _height_field(tm, U, 65.0):
                    one = (kind, dict(prm), U)
            elif kind != "freeform":
                q = symmetric_params(kind, prm, n, d)
                if q is not None and float(tm.vert_residual(kind, q, U).max()) <= 2 * tol:
                    one = (kind, q, U)
        if one is not None:
            out.append(one)
            merged += 1
            continue
        if any(k.startswith(("helix", "tube")) for k in prm):
            out.append((kind, prm, faces))
            out.append(("freeform", {}, mf))  # mirrored thread or pipe: a plain spline patch
            continue
        out.append((kind, prm, faces))
        out.append((kind, reflect_params(prm, n, d), mf))
    label = np.full(len(mesh.faces), -1, np.int64)
    for k, (_, _, faces) in enumerate(out):
        label[faces] = k
    if log:
        log(f"mirror symmetry: {len(regs)} faces found on one half; {merged} continue across the plane, "
            f"{len(out)} faces in all")
    return label, out, tol
