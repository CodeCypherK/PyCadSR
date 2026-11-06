"""Surface finding on the triangles of a CAD-exported mesh.

A CAD tessellation is not a cloud of noisy samples: every triangle belongs to
exactly one CAD face, so face borders run along triangle edges, and every vertex
lies on its true surface to float precision (the tessellator evaluates the
surface). Segmenting the triangles directly, and judging a triangle by its
*vertices*, separates faces that meet tangentially (a fillet and the plane it
rolls into) with a tolerance of microns instead of the chord error.

1. Smooth patches: triangles connected across edges sharper than `sharp_deg` are
   different faces.
2. Curved primitives: from seeds on curved edges, a hypothesis (cylinder, cone,
   sphere, torus) is fitted to the neighbourhood (facet normals give the axis,
   vertices give exact radii), refined by geometric least squares on vertices,
   and grown over adjacent triangles whose three vertices lie on it.
3. Planes: what is left and exactly coplanar.
4. Freeform: what fits nothing (B-spline faces), by connectivity.
5. Adjacent regions that are the same surface (a cylinder grown from two seeds,
   halves split at a seam) are merged.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .segmentation import (PENALTY, _axis_from_normals, _fit_circle_2d, _meridian, _unit, fit_cone,
                           fit_cylinder, fit_plane, fit_sphere, refine_geometric, residual)

CURVED_KINDS = ("cylinder", "cone", "sphere", "torus")
GROW_PENALTY = {"cylinder": 1.0, "cone": 0.9, "sphere": 0.88, "torus": 0.82}
# choose all faces together after the greedy passes (globalsel.py)
GLOBAL_SELECT = True
# the mesh-noise estimate: a low percentile of the local quadric fits, because on an
# exact CAD mesh coarse curved facets inflate the median far above the real precision
NOISE_PERCENTILE = 20.0


class TriMesh:
    """Triangle adjacency (CSR, sharp edges removed) and per-face geometry."""

    def __init__(self, mesh, sharp_deg: float, groups=None):
        self.V = np.asarray(mesh.vertices, float)
        self.F = np.asarray(mesh.faces, np.int64)
        self.FN = np.asarray(mesh.face_normals, float)
        self.C = self.V[self.F].mean(axis=1)
        self.A = np.asarray(mesh.area_faces, float)
        n = len(self.F)
        adj = np.asarray(mesh.face_adjacency, np.int64)
        ang = np.asarray(mesh.face_adjacency_angles, float)
        keep = ang < np.radians(sharp_deg)
        if groups is not None:
            keep &= groups[adj[:, 0]] == groups[adj[:, 1]]
        a, b, w = adj[keep, 0], adj[keep, 1], ang[keep]
        self.G = csr_matrix((np.r_[w, w] + 1e-12, (np.r_[a, b], np.r_[b, a])), shape=(n, n))
        self.G.sort_indices()
        self.indptr, self.indices, self.angle = self.G.indptr, self.G.indices, self.G.data - 1e-12
        _, self.patch = connected_components(self.G, directed=False)
        # every neighbour within the same body, sharp edges included (for slivers)
        ka = np.ones(len(adj), bool) if groups is None else groups[adj[:, 0]] == groups[adj[:, 1]]
        aa, bb = adj[ka, 0], adj[ka, 1]
        self.Gall = csr_matrix((np.ones(2 * len(aa)), (np.r_[aa, bb], np.r_[bb, aa])), shape=(n, n))
        lo, hi = self.V.min(0), self.V.max(0)
        self.diag = float(np.linalg.norm(hi - lo)) or 1.0

    def neighbours(self, faces: np.ndarray) -> np.ndarray:
        if len(faces) == 0:
            return faces
        st, en = self.indptr[faces], self.indptr[faces + 1]
        cnt = en - st
        idx = np.repeat(st - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(int(cnt.sum()))
        return np.unique(self.indices[idx])

    def verts(self, faces: np.ndarray) -> np.ndarray:
        return self.V[np.unique(self.F[faces])]

    def vert_residual(self, kind, prm, faces) -> np.ndarray:
        """Largest distance of each face's three corners from the surface."""
        P = self.V[self.F[faces]].reshape(-1, 3)
        d, _ = residual(kind, prm, P, np.zeros_like(P))
        return d.reshape(-1, 3).max(axis=1)

    def normal_ok(self, kind, prm, faces, cos_min) -> np.ndarray:
        _, c = residual(kind, prm, self.C[faces], self.FN[faces])
        return c >= cos_min


def _vertex_noise(tm: TriMesh, samples: int = 2000) -> float:
    """How precisely the vertices sit on smooth surfaces: around sample vertices, the
    rms residual of a local quadratic height field through the vertices of two rings of
    triangles in the same smooth patch
    (median over samples, which ignores neighbourhoods across sharp edges). Float
    noise for an exact CAD export; the real error for a rounded or sloppy one."""
    V = tm.V
    if len(tm.F) < 8:
        return 0.0
    rng = np.random.default_rng(0)
    # neighbourhoods inside one smooth patch: two rings of triangles around a sample
    # triangle (nearest vertices of a coarse mesh would reach across sharp edges)
    picks = rng.choice(len(tm.F), min(samples, len(tm.F)), replace=False)
    res = []
    for f in picks:
        ring = tm.neighbours(np.array([f]))
        ring = np.unique(np.r_[f, ring, tm.neighbours(ring)]) if len(ring) else np.array([f])
        row = np.unique(tm.F[ring])
        if len(row) < 8:
            continue
        Q = V[row]
        c = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        L = (Q - c) @ vt.T
        u, w, h = L[:, 0], L[:, 1], L[:, 2]
        A = np.c_[u * u, u * w, w * w, u, w, np.ones_like(u)]
        coef, *_ = np.linalg.lstsq(A, h, rcond=None)
        res.append(float(np.sqrt(np.mean((A @ coef - h) ** 2))))
    return float(np.percentile(res, NOISE_PERCENTILE)) if res else 0.0


def _hypotheses(tm: TriMesh, faces: np.ndarray):
    """Curved-surface candidates from a neighbourhood of triangles."""
    C, N = tm.C[faces], tm.FN[faces]
    P = tm.verts(faces)
    out = []
    w, _ = np.linalg.eigh(N.T @ N / len(N))
    if w[0] < 0.02 and w[1] > 1e-4:  # normals all perpendicular to one direction: a cylinder
        a = fit_cylinder(C, N)["axis"]
        c = _cyl_from_normals(tm, faces, a)
        if c is not None:
            out.append(("cylinder", c))
        e1, e2 = _perp2(a)
        # also the circle through the exact vertices (well posed when they span the arc)
        c2, r = _fit_circle_2d(np.c_[P @ e1, P @ e2])
        out.append(("cylinder", {"point": c2[0] * e1 + c2[1] * e2, "axis": a, "radius": r}))
    sph = fit_sphere(P)
    if sph is not None and sph["radius"] < 3 * tm.diag:
        out.append(("sphere", sph))
    cone = fit_cone(C, N)
    if cone is not None:
        out.append(("cone", cone))
    ax = _axis_from_normals(C, N)
    if ax is not None:
        a, q = ax
        rho, z, _, _ = _meridian(P, np.zeros_like(P), a, q)
        (cr, cz), r = _fit_circle_2d(np.c_[rho, z])
        if cr > 0 and r > 0 and r < 3 * tm.diag:
            out.append(("torus", {"center": q + cz * a, "axis": a, "R": float(cr), "r": float(r)}))
    return out


def _refine(kind, prm, P, tol):
    """Geometric refit on vertices; small sets are repeated to pass the refit's size floor."""
    if len(P) < 20:
        P = np.repeat(P, int(np.ceil(20 / max(len(P), 1))), axis=0)
    return refine_geometric(kind, prm, P, tol / 3)


def _quick_refit(tm, kind, prm, faces):
    """Closed-form refit on exact vertices (and facet normals for axes): cheap, and
    exact for spheres and cylinders; cones and tori get close enough to settle."""
    P = tm.verts(faces)
    C, N = tm.C[faces], tm.FN[faces]
    if kind == "sphere":
        return fit_sphere(P)
    if kind == "cylinder":
        a = fit_cylinder(C, N, prm)["axis"]
        c = _cyl_from_normals(tm, faces, a)
        return c if c is not None else prm
    if kind == "cone":
        c = fit_cone(C, N, prm)
        if c is None:
            return prm
        # axis from the facet planes; apex and angle from the exact vertices: r = a z + b
        a, q = c["axis"], c["apex"]
        v = P - q
        z = v @ a
        rho = np.linalg.norm(v - np.outer(z, a), axis=1)
        A = np.c_[z, np.ones_like(z)]
        (k, b), *_ = np.linalg.lstsq(A, rho, rcond=None)
        if abs(k) < 1e-9:
            return prm
        if k < 0:  # opening the other way: flip the axis
            a, k, z = -a, -k, -z
            b = b
        z0 = -b / k
        return {"apex": q + z0 * a, "axis": a, "half_angle": float(np.arctan(k))}
    if kind == "torus":
        ax = _axis_from_normals(C, N)
        a, q = ax if ax is not None else (prm["axis"], prm["center"])
        # profile circle from the exact vertices in the meridian half-plane
        rho, z, _, _ = _meridian(P, np.zeros_like(P), a, q)
        (cr, cz), r = _fit_circle_2d(np.c_[rho, z])
        if not (cr > 0 and r > 0):
            return prm
        return {"center": q + cz * a, "axis": a, "R": float(cr), "r": float(r)}
    return prm


def _planar(tm, faces, tol) -> bool:
    """Indistinguishable from a plane: then it is a plane, not a huge cylinder or torus."""
    P = tm.verts(faces)
    if len(P) < 3:
        return True
    return bool((tm.vert_residual("plane", fit_plane(P), faces) <= tol).all())


def _surface_normals(kind, prm, P):
    if kind == "cylinder":
        a = prm["axis"]
        v = P - prm["point"]
        n = v - np.outer(v @ a, a)
    elif kind == "sphere":
        n = P - prm["center"]
    elif kind == "cone":
        a, t_ = prm["axis"], prm["half_angle"]
        v = P - prm["apex"]
        rad = v - np.outer(v @ a, a)
        rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-300)
        n = rad * np.cos(t_) - np.outer(np.full(len(P), np.sin(t_)), a)
    elif kind == "torus":
        a, R = prm["axis"], prm["R"]
        v = P - prm["center"]
        h = v @ a
        rad = v - np.outer(h, a)
        rho = np.maximum(np.linalg.norm(rad, axis=1), 1e-300)
        n = rad / rho[:, None] * (rho - R)[:, None] + np.outer(h, a)
    else:
        return None
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def _too_coarse(tm, kind, prm, faces, max_deg: float = 25.0) -> bool:
    """Facets far coarser than any tessellation of this surface: a sphere through the
    two rims of a cylinder band fits every vertex, but each band triangle would then
    span 60+ degrees of the sphere. Area-weighted mean angle between each triangle's
    normal and the surface normal at its corners (half a strip angle on a real one)."""
    T = tm.V[tm.F[faces]]
    fn = tm.FN[faces]
    worst = np.ones(len(faces))  # smallest |cos| per triangle over its corners
    for k in range(3):
        n = _surface_normals(kind, prm, T[:, k])
        if n is None:
            return False
        c = np.abs(np.einsum("ij,ij->i", n, fn))
        if kind == "cone":  # the apex has no normal: a pointed cone's fan meets there
            v = T[:, k] - prm["apex"]
            c[np.linalg.norm(v, axis=1) < 1e-6 * tm.diag] = 1.0
        worst = np.minimum(worst, c)
    ang = np.degrees(np.arccos(np.clip(worst, -1.0, 1.0)))
    return bool(np.average(ang, weights=tm.A[faces] + 1e-300) > max_deg)


def _one_facet(tm, faces, frac: float = 0.7) -> bool:
    """Mostly one flat facet: a pin's flat side and its two tapered tips have corners
    on one circle (any symmetric trapezoid does), so a cylinder fits them exactly.
    A tessellated curve spreads its area over many narrow strips instead."""
    if len(faces) < 3:
        return True
    sub = tm.G[faces][:, faces].tocoo()
    m = (sub.data - 1e-12) <= np.radians(0.05)  # coplanar neighbours
    k = len(faces)
    g = csr_matrix((np.ones(int(m.sum())), (sub.row[m], sub.col[m])), shape=(k, k))
    _, lab = connected_components(g, directed=False)
    area = np.bincount(lab, weights=tm.A[faces])
    return bool(area.max() >= frac * area.sum())


def _settle(tm, kind, prm, ring, seed, tol, loose):
    """Loose inliers -> refit -> tighter inliers ... until the micron tolerance.
    Closed-form refits while loose, one geometric least-squares refit at the end."""
    thr = loose
    for _ in range(4):
        res = tm.vert_residual(kind, prm, ring)
        inl = ring[res <= thr]
        if len(inl) < 3 or seed not in inl:
            return None
        if thr <= tol:
            break
        new = _quick_refit(tm, kind, prm, inl)
        if _valid(kind, new, tm.diag):
            prm = new
        thr = max(tol, thr / 8)
    if _planar(tm, inl, tol):
        return None
    new = _refine(kind, prm, tm.verts(inl), tol)
    return new if _valid(kind, new, tm.diag) else None


def _perp2(a):
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(t - (t @ a) * a)
    return e1, np.cross(a, e1)


def _valid(kind, prm, diag):
    if prm is None:
        return False
    r = prm.get("radius", prm.get("r"))
    if r is not None and not (1e-5 * diag < r < 3 * diag):
        return False
    if kind == "torus" and not (prm["R"] > 0):
        return False
    if kind == "cone" and not (np.radians(1) < prm["half_angle"] < np.radians(89)):
        return False
    return all(np.all(np.isfinite(np.asarray(v, float))) for v in prm.values())


def _ring_curved(tm: TriMesh, seed: int, free: np.ndarray, size: int, curved: np.ndarray) -> np.ndarray:
    """Neighbourhood grown only across curved edges (plus each triangle's flat quad
    partner): it follows a narrow fillet along its arc instead of spilling onto the
    flat faces it rolls into."""
    got = [seed]
    seen = {seed}
    front = [seed]
    while front and len(got) < size:
        nxt = []
        for f in front:
            for k in range(tm.indptr[f], tm.indptr[f + 1]):
                g = int(tm.indices[k])
                if g in seen or not free[g]:
                    continue
                # across a curved edge, or to a flat partner that itself lies on curvature
                if curved[k] or (tm.angle[k] <= np.radians(0.3) and tm.on_curve[g]):
                    seen.add(g)
                    got.append(g)
                    nxt.append(g)
                    if len(got) >= size:
                        break
            if len(got) >= size:
                break
        front = nxt
    return np.array(got, np.int64)


def _axis_hypotheses(tm: TriMesh, ring: np.ndarray, axes: list, tol: float):
    """Surfaces of revolution around axes already found nearby. A fillet around a
    hole, the shoulder of a turned part: the axis is the hole's or the body's, which
    a small patch cannot pin down by itself. The axis must pass through the patch's
    facet normal lines; then the profile in the meridian plane is a circle (torus),
    a sloped line (cone) or a parallel line (cylinder)."""
    out = []
    if not axes:
        return out
    C, N = tm.C[ring], tm.FN[ring]
    P = tm.verts(ring)
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    for a, q in axes:
        cr = np.cross(N, a)
        nrm = np.linalg.norm(cr, axis=1)
        okn = nrm > 1e-3
        if okn.sum() < 3:
            continue
        dist = np.abs(np.einsum("ij,ij->i", C[okn] - q, cr[okn])) / nrm[okn]
        if np.median(dist) > 0.05 * size + 10 * tol:  # normal lines do not meet this axis
            continue
        v = P - q
        z = v @ a
        rho = np.linalg.norm(v - np.outer(z, a), axis=1)
        (cr_, cz), r = _fit_circle_2d(np.c_[rho, z])
        if cr_ > 0 and 0 < r < 3 * tm.diag:
            out.append(("torus", {"center": q + cz * a, "axis": a, "R": float(cr_), "r": float(r)}))
        k, b = np.polyfit(z, rho, 1) if np.ptp(z) > 1e-12 else (0.0, float(rho.mean()))
        if abs(k) < 1e-3:
            out.append(("cylinder", {"point": q, "axis": a, "radius": float(rho.mean())}))
        elif abs(k) < 50:
            sgn = 1 if k > 0 else -1
            out.append(("cone", {"apex": q + (-b / k) * a, "axis": sgn * a, "half_angle": float(np.arctan(abs(k)))}))
    return out


def _cyl_from_normals(tm: TriMesh, faces: np.ndarray, a: np.ndarray):
    """Cylinder with axis direction a through the common point of the facet normal
    lines (each facet of a cylinder has its normal through the axis), radius from the
    exact vertices. Exact even when the vertices lie on only a few generator lines,
    where fitting a circle through the vertices is ill-conditioned."""
    a = _unit(a)
    e1, e2 = _perp2(a)
    c = np.c_[tm.C[faces] @ e1, tm.C[faces] @ e2]
    d = np.c_[tm.FN[faces] @ e1, tm.FN[faces] @ e2]
    dn = np.linalg.norm(d, axis=1)
    ok = dn > 0.5
    if ok.sum() < 2:
        return None
    m = np.c_[-d[ok, 1], d[ok, 0]] / dn[ok, None]  # line normals in the cross-section
    rhs = np.einsum("ij,ij->i", m, c[ok])
    M = m.T @ m
    if np.linalg.cond(M) > 1e8:
        return None
    x = np.linalg.solve(M, m.T @ rhs)
    P = tm.V[np.unique(tm.F[faces])]
    r = float(np.median(np.linalg.norm(np.c_[P @ e1, P @ e2] - x, axis=1)))
    return {"point": x[0] * e1 + x[1] * e2, "axis": a, "radius": r}


def _pair_cylinders(tm: TriMesh, seed: int, ring: np.ndarray, max_pairs: int = 6):
    """Cylinders from two facets that both lie on curvature: the axis is perpendicular
    to both normals and passes where their normal lines meet."""
    out = []
    if not tm.on_curve[seed]:
        return out
    ns = tm.FN[seed]
    others = ring[(ring != seed) & tm.on_curve[ring]]
    ang = np.arccos(np.clip(tm.FN[others] @ ns, -1, 1))
    sel = (ang > np.radians(1.0)) & (ang < np.radians(60))
    for j in others[sel][np.argsort(-ang[sel])][:max_pairs]:
        a = np.cross(ns, tm.FN[j])
        if np.linalg.norm(a) < 1e-9:
            continue
        c = _cyl_from_normals(tm, np.array([seed, j]), a)
        if c is not None:
            out.append(("cylinder", c))
    return out


def _ring(tm: TriMesh, seed: int, free: np.ndarray, size: int = 40) -> np.ndarray:
    got = np.array([seed])
    front = got
    while len(got) < size and len(front):
        nb = tm.neighbours(front)
        nb = nb[free[nb]]
        nb = np.setdiff1d(nb, got, assume_unique=False)
        if len(nb) == 0:
            break
        front = nb[: max(1, size - len(got))]
        got = np.r_[got, front]
    return got


SIMPLER = {"sphere": ("cylinder", "cone"), "torus": ("cylinder", "cone", "sphere"), "cone": ("cylinder",)}


def _simplest(tm: TriMesh, kind, prm, faces, tol, cos_min):
    """The simplest surface type that fits the region as well. A band one facet row
    high has all its vertices on two circles, which a sphere fits exactly too; CAD
    (and we) prefer the cylinder."""
    for alt in SIMPLER.get(kind, ()):
        C, N = tm.C[faces], tm.FN[faces]
        if alt == "cylinder":
            guess = _cyl_from_normals(tm, faces, fit_cylinder(C, N)["axis"])
        elif alt == "cone":
            guess = _quick_refit(tm, "cone", fit_cone(C, N), faces) if fit_cone(C, N) is not None else None
        else:
            guess = fit_sphere(tm.verts(faces))
        if guess is None or not _valid(alt, guess, tm.diag):
            continue
        new = _refine(alt, guess, tm.verts(faces), tol)
        if _valid(alt, new, tm.diag) and _fits(tm, alt, new, faces, tol, cos_min):
            return alt, new
    return kind, prm


def _small_change(kind, p0, p1, rel: float = 0.02) -> bool:
    """A refit that refines the surface (radius and position within 2 % of the radius,
    axis within 1 degree), not one that drifts to another surface."""
    if kind == "plane":
        return True
    r0 = p0.get("radius", p0.get("r"))
    r1 = p1.get("radius", p1.get("r"))
    if r0 is not None and abs(r1 - r0) > rel * abs(r0):
        return False
    if kind == "torus" and abs(p1["R"] - p0["R"]) > rel * max(p0["R"], p0["r"]):
        return False
    scale = abs(r0) if r0 is not None else 1.0
    key = {"cylinder": "point", "sphere": "center", "torus": "center", "cone": "apex"}[kind]
    v = np.asarray(p1[key]) - np.asarray(p0[key])
    if "axis" in p0:
        a = _unit(p0["axis"])
        if abs(float(a @ _unit(p1["axis"]))) < np.cos(np.radians(1.0)):
            return False
        if kind == "cylinder":  # sliding along the axis is no change
            v = v - (v @ a) * a
    if kind == "cone":  # the apex of a shallow cone is far away; judge the angle
        return abs(p1["half_angle"] - p0["half_angle"]) < np.radians(0.5)
    return float(np.linalg.norm(v)) <= rel * scale


def _grow(tm: TriMesh, kind, prm, seed_faces, free, tol, cos_min):
    """Grow over neighbours whose corners lie on the surface. When growth stops,
    refit on everything grown so far and try again: a first guess from a few
    triangles (a few strips of a long fillet) is a near miss that only converges
    onto the true surface once more of the face constrains it."""
    region = np.asarray(seed_faces)
    front = region
    next_refit = max(60, 2 * len(region))
    regrows = 0
    while True:
        while len(front):
            nb = tm.neighbours(front)
            nb = nb[free[nb]]
            nb = np.setdiff1d(nb, region)
            if len(nb) == 0:
                break
            ok = (tm.vert_residual(kind, prm, nb) <= tol) & tm.normal_ok(kind, prm, nb, cos_min)
            front = nb[ok]
            region = np.r_[region, front]
            if len(region) >= next_refit:
                new = _refine(kind, prm, tm.verts(region), tol)
                if _valid(kind, new, tm.diag):
                    prm = new
                next_refit = 2 * len(region)
        if regrows >= 6:
            break
        regrows += 1
        new = _refine(kind, prm, tm.verts(region), tol)
        if not _valid(kind, new, tm.diag):
            break
        # the refit must still hold everything already grown, and only refine the
        # surface: a big change means it is creeping onto a tangent neighbour
        if not (tm.vert_residual(kind, new, region) <= tol).all() or not _small_change(kind, prm, new):
            break
        prm = new
        nb = tm.neighbours(region)
        nb = np.setdiff1d(nb[free[nb]], region)
        if len(nb) == 0:
            break
        ok = (tm.vert_residual(kind, prm, nb) <= tol) & tm.normal_ok(kind, prm, nb, cos_min)
        if not ok.any():
            break
        front = nb[ok]
        region = np.r_[region, front]
    return region, prm


def mesh_segment(mesh, sharp_deg: float = 30.0, groups=None, tol_rel: float = 3e-5, sag: float = 0.0,
                 progress=None):
    """Face labels and regions [(kind, params, faces)] for a clean (CAD-exported) mesh."""
    progress = progress or (lambda *a: None)
    tm = TriMesh(mesh, sharp_deg, groups)
    n = len(tm.F)
    # the vertex tolerance: microns for an exact CAD export, the mesh's own precision
    # for a rounded or sloppy one (else its faces would shatter into fragments)
    noise = _vertex_noise(tm)
    tol = max(tol_rel * tm.diag, min(5 * noise, 2e-4 * tm.diag), 1e-9)
    # first guesses come from facet centroids, which sit off a curved surface by the
    # chord error: inliers are picked loosely first, then tightened to `tol`
    loose = max(30 * tol, 3 * sag)
    cos_min = np.cos(np.radians(35))
    label = np.full(n, -1, np.int64)
    regions: list[tuple[str, dict, np.ndarray]] = []
    free = np.ones(n, bool)
    # seeds: triangles on a curved (not flat, not sharp) edge, biggest smooth patches first
    curved_edge = tm.angle > np.radians(0.3)
    rows = np.repeat(np.arange(n), np.diff(tm.indptr))
    seed_mask = np.zeros(n, bool)
    seed_mask[rows[curved_edge]] = True
    tm.on_curve = seed_mask  # triangles touching a curved edge
    patch_area = np.bincount(tm.patch, weights=tm.A)
    seeds = np.where(seed_mask)[0]
    seeds = seeds[np.lexsort((seeds, -patch_area[tm.patch[seeds]]))]
    tried = np.zeros(n, bool)
    axes: list = []  # axes of the curved surfaces found so far (turned parts, holes)
    fails = {}  # consecutive failed seeds per smooth patch
    for k, s in enumerate(seeds):
        if not free[s] or tried[s]:
            continue
        pt = int(tm.patch[s])
        if fails.get(pt, 0) >= 300:
            # 300 seeds in a row found nothing here (a helical coil, a sculpted area):
            # leave the rest of this patch to the freeform step
            continue
        if k % 2000 == 0:
            progress(0.6 * k / max(len(seeds), 1), f"Curved surfaces: seed {k + 1}/{len(seeds)}")
        tried[s] = True
        cands = []
        # a small neighbourhood first (it stays on one face), a wider one if the
        # normals do not yet pin an axis down (a large radius, a few facets)
        # neighbourhoods: along the curvature first (stays on a narrow fillet), then
        # plain rings; hypotheses from the whole neighbourhood and from facet pairs
        rings = [("curved", 12), ("curved", 30), ("plain", 12), ("plain", 40)]
        for how, size in rings:
            ring = _ring_curved(tm, s, free, size, curved_edge) if how == "curved" else _ring(tm, s, free, size)
            if len(ring) < 4:
                continue
            extra = _axis_hypotheses(tm, ring, axes, tol) if how == "curved" or size == 12 else []
            for kind, prm in _hypotheses(tm, ring) + _pair_cylinders(tm, s, ring) + extra:
                if not _valid(kind, prm, tm.diag):
                    continue
                prm = _settle(tm, kind, prm, ring, s, tol, loose)
                if prm is None:
                    continue
                ok = (tm.vert_residual(kind, prm, ring) <= tol) & tm.normal_ok(kind, prm, ring, cos_min)
                if ok[0] and ok.sum() >= 3:  # the seed itself must be on it
                    cands.append((ok.sum() * PENALTY[kind], kind, prm, ring[ok]))
            if any(len(c[3]) >= 6 for c in cands):
                break
        if not cands:
            fails[pt] = fails.get(pt, 0) + 1
            continue
        # On a few triangles a cone (one more parameter) fits a cylinder's facets as
        # well as the cylinder does, then stops growing. Grow every candidate and keep
        # the one that covers the most: the true surface covers its whole face.
        best = None
        for _, kind, prm, start in sorted(cands, key=lambda c: -c[0])[:4]:
            faces, prm = _grow(tm, kind, prm, start, free, tol, cos_min)
            if _too_coarse(tm, kind, prm, faces):
                continue
            # a more flexible surface also covers a sliver of the tangent neighbour (a
            # torus spans a narrow cylinder band plus the start of its fillet): it must
            # win clearly, not by a few triangles
            score = len(faces) * GROW_PENALTY[kind]
            if best is None or score > best[0]:
                best = (score, kind, prm, faces)
        if best is None:
            fails[pt] = fails.get(pt, 0) + 1
            continue
        _, kind, prm, faces = best
        # a curved patch must actually curve: a few coplanar triangles are a plane
        if len(faces) < 3 or _planar(tm, faces, tol) or _one_facet(tm, faces):
            fails[pt] = fails.get(pt, 0) + 1
            continue
        fails[pt] = 0
        kind, prm = _simplest(tm, kind, prm, faces, tol, cos_min)
        if kind in ("cylinder", "cone", "torus") and len(faces) >= 6:
            a_ = _unit(prm["axis"])
            q_ = np.asarray(prm.get("point", prm.get("apex", prm.get("center"))), float)
            dup = False
            for a0, q0 in axes:
                v_ = q_ - q0
                if abs(float(a0 @ a_)) > np.cos(np.radians(0.5)) and np.linalg.norm(v_ - (v_ @ a0) * a0) < 10 * tol:
                    dup = True
                    break
            if not dup:
                axes.append((a_, q_))
                if len(axes) > 60:
                    axes.pop(0)
        prm = _refine(kind, prm, tm.verts(faces), tol)
        label[faces] = len(regions)
        free[faces] = False
        regions.append((kind, prm, faces))
    if FILLET_RAILS:
        # coarse fillets the seeds missed: cylinders from parallel rails
        for prm, faces in _fillet_rails(tm, free, tol, cos_min, regions):
            label[faces] = len(regions)
            free[faces] = False
            regions.append(("cylinder", prm, faces))
    progress(0.65, "Planes")
    # planes: remaining triangles joined across flat edges, checked for coplanarity
    flat = tm.angle <= np.radians(0.3)
    a, b = rows[flat], tm.indices[flat]
    m = free[a] & free[b]
    Gp = csr_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(n, n))
    _, comp = connected_components(Gp, directed=False)
    rem = np.where(free)[0]
    order = np.argsort(comp[rem], kind="stable")
    starts = np.searchsorted(comp[rem][order], np.unique(comp[rem]))
    bounds = np.r_[starts, len(rem)]
    singles = []
    for i in range(len(starts)):
        faces = rem[order[bounds[i]:bounds[i + 1]]]
        if len(faces) <= 2:
            # one or two flat triangles with unassigned curved neighbours are a facet
            # of a curved surface nothing fitted, not a plane face
            nb = np.setdiff1d(tm.neighbours(faces), faces)
            if free[nb].any():
                continue
        prm = fit_plane(tm.verts(faces), tm.FN[faces])
        if (tm.vert_residual("plane", prm, faces) <= tol).all():
            label[faces] = len(regions)
            free[faces] = False
            regions.append(("plane", prm, faces))
        else:
            singles.append(faces)
    _reclaim_planes(tm, regions, label, free, tol)
    regions, label = _compact(regions, label)
    progress(0.8, "Freeform faces")
    # freeform: whatever fits nothing, by connectivity (B-spline faces)
    rem = np.where(free)[0]
    if len(rem):
        a, b = rows, tm.indices
        m = free[a] & free[b]
        Gf = csr_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(n, n))
        _, comp = connected_components(Gf, directed=False)
        for c in np.unique(comp[rem]):
            faces = rem[comp[rem] == c]
            label[faces] = len(regions)
            free[faces] = False
            # a group skipped as a possible facet of a curve (a pin's flat side between
            # two tapered tips) that is exactly flat on its own is a plane after all
            prm = fit_plane(tm.verts(faces), tm.FN[faces])
            if (tm.vert_residual("plane", prm, faces) <= tol).all():
                regions.append(("plane", prm, faces))
            else:
                regions.append(("freeform", {}, faces))
    regions, label = _merge_same(tm, regions, label, tol)
    progress(0.9, "Joining pieces of the same surface")
    regions, label = _absorb_and_merge(tm, regions, label, tol, cos_min)
    regions, label = _absorb_tiny(tm, regions, label, max(10 * tol, 2 * sag), tol=tol)
    regions, label = _freeform_clusters(tm, regions, label)
    regions, label = _thin_chains(tm, regions, label, tol, noise)
    regions, label = _absorb_exact(tm, regions, label, tol, noise, cos_min)
    regions, label = _continuity_merge(tm, regions, label, tol, noise)
    regions, label = _absorb_spots(tm, regions, label, tol)
    regions, label = _promote_freeform(tm, regions, label, tol, noise)
    regions, label = _absorb_exact(tm, regions, label, tol, noise, cos_min)
    if GLOBAL_SELECT:
        from .globalsel import global_select
        from .protocol import log
        progress(0.92, "Choosing all faces together")
        regions, label = global_select(tm, regions, label, tol, max(10 * tol, 2 * sag), log=log)
    regions, label = _tubes(tm, regions, label, tol, noise)
    if HELICAL_FACES:
        from .protocol import log as _log
        progress(0.95, "Threads and helical faces")
        regions, label = _helical_faces(tm, regions, label, tol, log=_log)
    regions, label = _split_wrapping(tm, regions, label)
    regions, label = _split_chords(tm, regions, label, tol)
    if ROD_DECOMPOSE:
        from .protocol import log as _log
        regions, label = _rod_decompose(tm, regions, label, tol, cos_min, log=_log)
    if SPLIT_MIXED:
        from .protocol import log as _log
        regions, label = _split_mixed(tm, regions, label, tol, cos_min, log=_log)
    if SMOOTH_BORDERS:
        from .protocol import log as _log
        regions, label = _merge_same_any(tm, regions, label, tol, log=_log)
        regions, label = _join_on_surface(tm, regions, label, tol, log=_log)
        if JOIN_BENDS:
            regions, label = _join_bends(tm, regions, label, tol, cos_min, log=_log)
        regions, label = _smooth_borders(tm, regions, label, tol, cos_min, log=_log)
        regions, label = _merge_same_any(tm, regions, label, tol, log=_log)
    if ABSORB_SPECKS:
        from .protocol import log as _log
        regions, label = _absorb_specks(tm, regions, label, tol, log=_log)
        # each pass clears pieces that blocked the other: a speck between a sliver and the
        # face it lies on hides most of their shared border, and a joined sliver leaves
        # specks sitting in a notch of that face
        settled = 0
        for _ in range(SETTLE_ROUNDS):
            n0 = len(regions)
            regions, label = _join_on_surface(tm, regions, label, tol, log=_log)
            regions, label = _absorb_specks(tm, regions, label, tol, log=_log)
            if len(regions) == n0:
                break
            settled += 1
        if settled and SMOOTH_BORDERS:
            regions, label = _smooth_borders(tm, regions, label, tol, cos_min, log=_log)
            regions, label = _merge_same_any(tm, regions, label, tol, log=_log)
            regions, label = _absorb_specks(tm, regions, label, tol, log=_log)
    return label, regions, tol


def _split_chords(tm: TriMesh, regions, label, tol: float):
    """Flat facets taken for part of a curved face. A flat cut across a cylinder (a
    D-shaft's flat, a notch in a round body) is tessellated as one or two big triangles
    whose corners all lie on the cylinder, so they fit it to the micron and get grown
    into it; the B-rep face then has a loop that doubles back on itself. Each such
    triangle spans far more of the surface than the face's own strips (the angle
    between the surface normals at its corners) and sags far deeper below it. They
    are taken out; a flat group of them becomes a plane (merged into a coplanar
    neighbouring plane), anything else a freeform region."""
    regions = list(regions)
    label = label.copy()
    taken = []
    for ri, (kind, prm, faces) in enumerate(regions):
        if kind not in CURVED_KINDS or len(faces) < 8:
            continue
        T = tm.V[tm.F[faces]]
        ns = [_surface_normals(kind, prm, T[:, k]) for k in range(3)]
        if ns[0] is None:
            continue
        span = np.zeros(len(faces))
        for i, j in ((0, 1), (1, 2), (0, 2)):
            c = np.clip(np.einsum("ij,ij->i", ns[i], ns[j]), -1.0, 1.0)
            span = np.maximum(span, np.arccos(c))
        if kind == "cone":  # the apex has no normal: a pointed cone's fan meets there
            near = np.zeros(len(faces), bool)
            for k in range(3):
                near |= np.linalg.norm(T[:, k] - prm["apex"], axis=1) < 1e-6 * tm.diag
            span[near] = 0.0
        sag = residual(kind, prm, tm.C[faces], np.zeros((len(faces), 3)))[0]
        out = ((span > max(3 * float(np.percentile(span, 75)), np.radians(8)))
               & (sag > max(10 * float(np.percentile(sag, 75)), 10 * tol)))
        if not out.any() or out.sum() > 0.2 * len(faces):
            continue
        regions[ri] = (kind, prm, faces[~out])
        taken.append(faces[out])
    if not taken:
        return regions, label
    taken = np.concatenate(taken)
    label[taken] = -1
    sub = tm.Gall[taken][:, taken]
    _, comp = connected_components(sub, directed=False)
    for c in np.unique(comp):
        faces = taken[comp == c]
        P = tm.verts(faces)
        prm = fit_plane(P, tm.FN[faces])
        if len(P) >= 3 and (tm.vert_residual("plane", prm, faces) <= tol).all():
            # a coplanar plane next to it is the rest of the same flat face
            nb = np.unique(label[np.setdiff1d(tm.neighbours(faces), faces)])
            joined = False
            for r in nb[nb >= 0]:
                k2, p2, f2 = regions[r]
                if k2 != "plane":
                    continue
                both = np.r_[f2, faces]
                pb = fit_plane(tm.verts(both), tm.FN[both])
                if (tm.vert_residual("plane", pb, both) <= tol).all():
                    regions[r] = ("plane", pb, both)
                    label[faces] = r
                    joined = True
                    break
            if not joined:
                label[faces] = len(regions)
                regions.append(("plane", prm, faces))
        else:
            label[faces] = len(regions)
            regions.append(("freeform", {}, faces))
    return regions, label


def _promote_freeform(tm: TriMesh, regions, label, tol: float, noise: float):
    """A leftover freeform region may be one exact primitive the seeds never caught
    (a tank wall with a 4.5 degree taper looks like a cylinder on any small patch,
    and a cylinder misses it): fit the whole region; exact -> that surface."""
    nV = len(tm.V)
    fl = np.repeat(label, 3)
    lo = np.full(nV, np.iinfo(np.int64).max)
    hi = np.full(nV, -1)
    np.minimum.at(lo, tm.F.ravel(), fl)
    np.maximum.at(hi, tm.F.ravel(), fl)
    planes = []
    for i, (k, prm, faces) in enumerate(regions):
        if k == "plane":
            v = np.unique(tm.F[faces])
            v = v[(lo[v] == i) & (hi[v] == i)]
            if len(v) >= 3:
                d, _ = residual("plane", prm, tm.V[v], np.zeros((len(v), 3)))
                planes.append(float(np.sqrt(np.mean(d ** 2))))
    thr = max(4 * (float(np.median(planes)) if planes else noise), 0.03 * tol)
    out = list(regions)
    changed = False
    for i, (k, prm, faces) in enumerate(regions):
        if k != "freeform" or len(faces) < 6:
            continue
        res = _exact_over(tm, faces, tol, thr)
        if res is not None:
            out[i] = res
            changed = True
    return (out, label) if changed else (regions, label)


def _tubes(tm: TriMesh, regions, label, tol: float, noise: float):
    """A coil spring or a bent wire is one smooth face: a circle swept along a spine.
    Its pieces each fit some torus or cylinder roughly, or nothing: smooth groups of
    such soft pieces are tested as one tube (every vertex at one distance from a
    spine curve, all the way round); a tube becomes one freeform region carrying its
    radius and spine."""
    from .tube import TubeSurface, fit_tube
    R = len(regions)
    nV = len(tm.V)
    fl = np.repeat(label, 3)
    lo = np.full(nV, np.iinfo(np.int64).max)
    hi = np.full(nV, -1)
    np.minimum.at(lo, tm.F.ravel(), fl)
    np.maximum.at(hi, tm.F.ravel(), fl)

    def inner_rms(i):
        k, p, faces = regions[i]
        v = np.unique(tm.F[faces])
        v = v[(lo[v] == i) & (hi[v] == i)]
        if len(v) < 3:
            return None
        d, _ = residual(k, p, tm.V[v], np.zeros((len(v), 3)))
        return float(np.sqrt(np.mean(d ** 2)))

    planes = [x for i, (k, _, _) in enumerate(regions) if k == "plane" and (x := inner_rms(i)) is not None]
    thr = max(4 * (float(np.median(planes)) if planes else noise), 0.03 * tol)
    # candidates: every curved or freeform piece; exact ones are counted, so a real
    # pipe (a few exact cylinders and tori) is left alone
    soft = np.zeros(R, bool)
    exact = np.zeros(R, bool)
    for i, (k, p, faces) in enumerate(regions):
        if k == "freeform":
            soft[i] = "tube_r" not in p
        elif k in CURVED_KINDS:
            soft[i] = True
            x = inner_rms(i)
            v = np.unique(tm.F[faces])
            exact[i] = x is not None and x <= thr and int(((lo[v] == i) & (hi[v] == i)).sum()) >= 12
    if not soft.any():
        return regions, label
    coo = tm.G.tocoo()
    la, lb = label[coo.row], label[coo.col]
    m = (la >= 0) & (lb >= 0) & (la != lb) & soft[np.maximum(la, 0)] & soft[np.maximum(lb, 0)]
    parent = np.arange(R)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in set(zip(la[m].tolist(), lb[m].tolist())):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(R)])
    out = list(regions)
    gone = np.zeros(R, bool)
    changed = False
    for root in np.unique(roots[soft]):
        members = np.where((roots == root) & soft)[0]
        faces = np.concatenate([regions[i][2] for i in members])
        if len(faces) < 200:
            continue
        if len(members) < 3 and not any(regions[i][0] == "freeform" and not _height_field(tm, regions[i][2])
                                         for i in members):
            continue
        area = np.array([float(tm.A[regions[i][2]].sum()) for i in members])
        ex = exact[members]
        if ex.sum() <= 4 and area[ex].sum() > 0.5 * area.sum():
            continue  # a few exact faces carry it: real cylinders and tori
        try:
            fit = fit_tube(tm.V, tm.F[faces], tm.FN[faces], tm.A[faces], tol)
        except Exception:
            fit = None
        if fit is None or fit["p99"] > max(5 * tol, 0.03 * fit["r"]) or fit["rms"] > max(2 * tol, 0.01 * fit["r"]):
            continue
        tube = TubeSurface(None, fit["r"], fit["tck"], n=4000)
        _, ph = tube.angles(tm.verts(faces))
        sp_ = np.sort(ph)
        gap = max(float(np.diff(sp_).max()), float(sp_[0] + 2 * np.pi - sp_[-1]))
        if gap > np.pi / 3:
            continue  # part of a tube only (a rolling-ball fillet): left as it is
        prm = tube.to_json()
        out[members[0]] = ("freeform", prm, faces)
        gone[members[1:]] = True
        changed = True
    if not changed:
        return regions, label
    out = [r for i, r in enumerate(out) if not gone[i]]
    lab2 = np.full(len(label), -1, np.int64)
    for k, (_, _, f) in enumerate(out):
        lab2[f] = k
    return out, lab2


def _height_field(tm: TriMesh, faces, max_deg: float = 80.0) -> bool:
    """Every facet faces within max_deg of the patch's mean normal: the patch is a
    height field w = f(u, v) over that direction (one B-spline face can hold it)."""
    N, A = tm.FN[faces], tm.A[faces]
    m = (N * A[:, None]).sum(axis=0)
    nm = np.linalg.norm(m)
    if nm <= 0.5 * A.sum():
        return False
    c = N @ (m / nm)
    big = A > 1e-3 * A.mean()  # ignore degenerate slivers
    return bool(c[big].min() > np.cos(np.radians(max_deg))) if big.any() else True


def _split_wrapping(tm: TriMesh, regions, label, min_frac: float = 0.01):
    """One smooth freeform surface that wraps round (a rolling fillet all the way
    round an outline) is still one surface, but no single spline patch over one
    direction can hold it: CAD splits it into a few faces. Split by the direction
    the facets face (six directions from the patch's own normal axes), smooth the
    split on the triangle graph, then join neighbouring pieces back while the union
    stays a height field."""
    out = []
    changed = False
    for k, prm, faces in regions:
        # a patch whose facets turn more than 65 degrees from its mean is a height field
        # only in name: seen that edge-on, a rounded strip running round a drawer front
        # folded its spline and its edges crossed
        # (a long strip meshed with a handful of triangles across, a coarse round along
        # an outline, is better left whole: cut into pieces, their seams fail instead)
        coarse = len(faces) < 200 and float(np.ptp(tm.V[np.unique(tm.F[faces])], axis=0).max()) > 0.3 * tm.diag
        if (k != "freeform" or len(faces) < 20 or "tube_r" in prm or "helix_lead" in prm
                or _height_field(tm, faces, 80.0 if coarse else 65.0)):
            out.append((k, prm, faces))
            continue
        N, A = tm.FN[faces], tm.A[faces]
        _, E = np.linalg.eigh((N * A[:, None]).T @ N)
        D = np.vstack([E.T, -E.T])
        lab = np.argmax(N @ D.T, axis=1)
        sub = tm.Gall[faces][:, faces].tocoo()
        r_, c_ = sub.row, sub.col
        n = len(faces)
        for _ in range(3):
            votes = np.zeros((n, 6))
            np.add.at(votes, (r_, lab[c_]), 1.0)
            votes[np.arange(n), lab] += 1.5
            lab = votes.argmax(axis=1)
        same = lab[r_] == lab[c_]
        _, comp = connected_components(csr_matrix((np.ones(int(same.sum())), (r_[same], c_[same])), shape=(n, n)),
                                       directed=False)
        pieces = {int(c): np.where(comp == c)[0] for c in np.unique(comp)}
        # tiny pieces join the neighbour they share most border with
        area = {c: float(A[ix].sum()) for c, ix in pieces.items()}
        total = sum(area.values())
        for c in sorted(pieces, key=lambda c: area[c]):
            if c not in pieces or area[c] >= min_frac * total:
                continue
            m = (comp[r_] == c) & (comp[c_] != c)
            if not m.any():
                continue
            nb, cnt = np.unique(comp[c_[m]], return_counts=True)
            tgt = int(nb[np.argmax(cnt)])
            pieces[tgt] = np.r_[pieces[tgt], pieces.pop(c)]
            comp[pieces[tgt]] = tgt
            area[tgt] += area.pop(c)
        # join neighbours back while still a height field (largest borders first)
        merged = True
        while merged and len(pieces) > 1:
            merged = False
            m = comp[r_] != comp[c_]
            keys, cnt = np.unique(np.sort(np.c_[comp[r_[m]], comp[c_[m]]], axis=1), axis=0, return_counts=True)
            for (a, b) in keys[np.argsort(-cnt)]:
                a, b = int(a), int(b)
                if a not in pieces or b not in pieces:
                    continue
                u = np.r_[pieces[a], pieces[b]]
                if _height_field(tm, faces[u], 80.0 if coarse else 65.0):
                    pieces[a] = u
                    comp[pieces.pop(b)] = a
                    merged = True
                    break
        if len(pieces) == 1:
            out.append((k, prm, faces))
            continue
        changed = True
        out += [("freeform", {}, faces[ix]) for ix in pieces.values()]
    if not changed:
        return regions, label
    lab2 = np.full(len(label), -1, np.int64)
    for i, (_, _, f) in enumerate(out):
        lab2[f] = i
    return out, lab2


def _absorb_spots(tm: TriMesh, regions, label, tol: float, rounds: int = 12):
    """Spotty patches on a freeform face: a small cylinder or torus piece that fits only
    approximately (a stretch of a rolling-ball fillet along a curved edge) and meets a
    much larger freeform face smoothly along a good part of its border joins it.
    Repeated, so a run of such pieces is taken in from the freeform side."""
    R = len(regions)
    kind = [r[0] for r in regions]
    faces_of = [r[2] for r in regions]
    area = np.array([float(tm.A[f].sum()) for f in faces_of])
    approx = np.zeros(R, bool)
    for i, (k, p, faces) in enumerate(regions):
        if k in CURVED_KINDS:
            P = tm.verts(faces)
            d, _ = residual(k, p, P, np.zeros_like(P))
            approx[i] = float(np.sqrt(np.mean(d ** 2))) > 0.1 * tol
    if not approx.any() or "freeform" not in kind:
        return regions, label
    lab = label.copy()
    changed = False
    for _ in range(rounds):
        coo_all, coo_s = tm.Gall.tocoo(), tm.G.tocoo()
        la, lb = lab[coo_all.row], lab[coo_all.col]
        m = (la >= 0) & (lb >= 0) & (la != lb)
        tot = np.bincount(la[m], minlength=R)
        sa, sb = lab[coo_s.row], lab[coo_s.col]
        ms = (sa >= 0) & (sb >= 0) & (sa != sb)
        keys, cnt = np.unique(sa[ms].astype(np.int64) * R + sb[ms], return_counts=True)
        moved = False
        for key, c in zip(keys.tolist(), cnt.tolist()):
            a, b = divmod(key, R)
            if not approx[a] or kind[b] != "freeform" or area[a] >= 0.25 * area[b]:
                continue
            if c < 0.3 * tot[a]:
                continue
            lab[faces_of[a]] = b
            faces_of[b] = np.r_[faces_of[b], faces_of[a]]
            faces_of[a] = faces_of[a][:0]
            area[b] += area[a]
            area[a] = 0.0
            approx[a] = False
            moved = changed = True
        if not moved:
            break
    if not changed:
        return regions, label
    out = [(kind[i], regions[i][1], faces_of[i]) for i in range(R) if len(faces_of[i])]
    lab2 = np.full(len(label), -1, np.int64)
    for k, (_, _, f) in enumerate(out):
        lab2[f] = k
    return out, lab2


def _exact_over(tm: TriMesh, faces, tol: float, thr: float):
    """One analytic surface over the whole of a smooth group, if one fits it exactly
    (a wire bend found as cylinder slices is one torus), else None. Simplest first."""
    rng = np.random.default_rng(0)
    sample = faces if len(faces) <= 300 else rng.choice(faces, 300, replace=False)
    P = tm.verts(faces)
    # interior vertices of the group (every incident triangle in it)
    inside = np.zeros(len(tm.F), bool)
    inside[faces] = True
    bad = np.zeros(len(tm.V), bool)
    bad[tm.F[~inside].ravel()] = True
    vin = np.unique(tm.F[faces])
    vin = vin[~bad[vin]]
    # judged on interior vertices; a band one facet row wide has none, then all of
    # its vertices must fit (edge noise included)
    Pin = tm.V[vin] if len(vin) >= 6 else P
    best = None
    for kind, prm in _hypotheses(tm, sample):
        if not _valid(kind, prm, tm.diag):
            continue
        prm = _refine(kind, prm, P, tol)
        if not _valid(kind, prm, tm.diag):
            continue
        d, _ = residual(kind, prm, Pin, np.zeros_like(Pin))
        rms = float(np.sqrt(np.mean(d ** 2)))
        if rms <= thr and (tm.vert_residual(kind, prm, faces) <= tol).all() and                 not _too_coarse(tm, kind, prm, faces) and not _one_facet(tm, faces):
            rank = ("cylinder", "cone", "sphere", "torus").index(kind) if kind in CURVED_KINDS else 9
            if best is None or rank < best[0]:
                best = (rank, kind, prm)
    return (best[1], best[2], faces) if best else None


def _absorb_exact(tm: TriMesh, regions, label, tol: float, noise: float, cos_min: float, rounds: int = 4):
    """A surface pinned down exactly by many interior vertices takes in what lies on
    it as exactly (float noise), spreading outward: the slices of a pipe bend that each
    fit some cylinder within tolerance, and stray tori or freeform scraps along it, all
    lie on the bend's one torus. An approximate primitive that one exact surface
    explains is refit to it first."""
    R = len(regions)
    if R < 2:
        return regions, label
    nV = len(tm.V)
    kind = [r[0] for r in regions]
    prm = [r[1] for r in regions]
    faces_of = [np.asarray(r[2]) for r in regions]
    lab = np.full(len(tm.F), -1, np.int64)
    for i, f in enumerate(faces_of):
        lab[f] = i
    # vertices on a sharp edge (or a body boundary) come from edge curves and carry
    # the source's edge tolerance; every other vertex sits on its surface exactly
    deg_s = np.diff(tm.G.indptr)
    deg_a = np.diff(tm.Gall.indptr)
    crease = np.zeros(nV, bool)
    crease[tm.F[deg_a > deg_s].ravel()] = True
    crease[tm.F[deg_a < 3].ravel()] = True

    lo = np.empty(nV, np.int64)
    hi = np.empty(nV, np.int64)

    def refresh():
        fl = np.repeat(lab, 3)
        lo.fill(np.iinfo(np.int64).max)
        hi.fill(-1)
        np.minimum.at(lo, tm.F.ravel(), fl)
        np.maximum.at(hi, tm.F.ravel(), fl)

    refresh()

    def interior(i):
        v = np.unique(tm.F[faces_of[i]])
        return v[(lo[v] == i) & (hi[v] == i)]

    def inner_rms(i):
        v = interior(i)
        if len(v) < 3:
            return None, len(v)
        d, _ = residual(kind[i], prm[i], tm.V[v], np.zeros((len(v), 3)))
        return float(np.sqrt(np.mean(d ** 2))), len(v)

    plane_rms = [r for i in range(R) if kind[i] == "plane" and (r := inner_rms(i)[0]) is not None]
    thr = max(4 * (float(np.median(plane_rms)) if plane_rms else noise), 0.03 * tol)

    def is_exact(i):
        if kind[i] == "freeform" or not len(faces_of[i]):
            return False
        r, n = inner_rms(i)
        return r is not None and n >= 12 and r <= thr

    exact = np.array([is_exact(i) for i in range(R)])
    if not exact.any():
        return regions, label
    changed = False

    def lies_on(q: int, f) -> bool:
        if (tm.vert_residual(kind[q], prm[q], f) > tol).any():
            return False
        if not tm.normal_ok(kind[q], prm[q], f, cos_min).all():
            return False
        v = np.unique(tm.F[f])
        v = v[~crease[v]] if (~crease[v]).sum() >= 3 else v
        d, _ = residual(kind[q], prm[q], tm.V[v], np.zeros((len(v), 3)))
        return float(np.sqrt(np.mean(d ** 2))) <= thr

    import heapq
    refit_tried = np.full(R, -1)  # face count when an exact refit was last tried
    for _ in range(rounds):
        moved = False
        refresh()
        # 0. an approximate primitive that one exact surface explains (a torus found
        # from a few slices, slightly off) takes that surface
        for i in range(R):
            if kind[i] not in CURVED_KINDS or exact[i] or len(faces_of[i]) < 6 or refit_tried[i] == len(faces_of[i]):
                continue
            refit_tried[i] = len(faces_of[i])
            rr, n = inner_rms(i)
            if rr is None or n < 12:
                continue
            res = _exact_over(tm, faces_of[i], tol, thr)
            if res is not None:
                kind[i], prm[i] = res[0], res[1]
                exact[i] = is_exact(i)
                moved = moved or bool(exact[i])
        # 1. whole neighbours that lie on an exact surface
        coo = tm.G.tocoo()
        la, lb = lab[coo.row], lab[coo.col]
        m = (la >= 0) & (lb >= 0) & (la != lb)
        nbrs: dict[int, set] = {}
        for x, y in zip(la[m].tolist(), lb[m].tolist()):
            nbrs.setdefault(x, set()).add(y)
        size = np.array([len(f) for f in faces_of])
        owner = np.arange(R)
        heap = [(-int(size[q]), q, r) for q in np.where(exact)[0].tolist() for r in nbrs.get(q, ())]
        heapq.heapify(heap)
        while heap:
            _, q, r = heapq.heappop(heap)
            q, r = int(owner[q]), int(owner[r])
            if q == r or not exact[q] or len(faces_of[r]) == 0:
                continue
            if exact[r] and size[r] > size[q]:
                continue
            if not lies_on(q, faces_of[r]):
                continue
            lab[faces_of[r]] = q
            faces_of[q] = np.r_[faces_of[q], faces_of[r]]
            faces_of[r] = faces_of[r][:0]
            size[q] += size[r]
            size[r] = 0
            owner[owner == r] = q
            exact[r] = False
            moved = True
            for n in nbrs.get(r, ()):
                n = int(owner[n])
                if n != q:
                    nbrs.setdefault(q, set()).add(n)
                    nbrs.setdefault(n, set()).add(q)
                    heapq.heappush(heap, (-int(size[q]), q, n))
        if not moved:
            break
        changed = True
    if not changed:
        return regions, label
    out = [(kind[i], prm[i], faces_of[i]) for i in range(R) if len(faces_of[i])]
    lab2 = np.full(len(label), -1, np.int64)
    for k, (_, _, f) in enumerate(out):
        lab2[f] = k
    return out, lab2


def _thin_chains(tm: TriMesh, regions, label, tol: float, noise: float):
    """A band one facet row wide has no interior vertices to pin its shape: a slice of
    a pipe bend fits a cylinder within tolerance, so a bend comes out as a stack of
    cylinder slices. Chains of such bands meeting smoothly are fitted as a whole (the
    whole bend pins the torus down); one exact surface over all of them replaces them."""
    nV = len(tm.V)
    fl = np.repeat(label, 3)
    lo = np.full(nV, np.iinfo(np.int64).max)
    hi = np.full(nV, -1)
    np.minimum.at(lo, tm.F.ravel(), fl)
    np.maximum.at(hi, tm.F.ravel(), fl)
    R = len(regions)
    thin = np.zeros(R, bool)
    planes = []
    for i, (k, prm, faces) in enumerate(regions):
        v = np.unique(tm.F[faces])
        v = v[(lo[v] == i) & (hi[v] == i)]
        if k in CURVED_KINDS:
            thin[i] = len(v) < 3
        elif k == "plane" and len(v) >= 3:
            d, _ = residual("plane", prm, tm.V[v], np.zeros((len(v), 3)))
            planes.append(float(np.sqrt(np.mean(d ** 2))))
    if thin.sum() < 3:
        return regions, label
    thr = max(4 * (float(np.median(planes)) if planes else noise), 0.03 * tol)
    coo_all, coo_s = tm.Gall.tocoo(), tm.G.tocoo()
    la, lb = label[coo_all.row], label[coo_all.col]
    m = (la >= 0) & (lb >= 0) & (la < lb)
    every = dict(zip(*np.unique(la[m].astype(np.int64) * R + lb[m], return_counts=True)))
    sa, sb = label[coo_s.row], label[coo_s.col]
    ms = (sa >= 0) & (sb >= 0) & (sa < sb)
    keys, cnt = np.unique(sa[ms].astype(np.int64) * R + sb[ms], return_counts=True)
    parent = np.arange(R)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key, c in zip(keys.tolist(), cnt.tolist()):
        a, b = divmod(int(key), R)
        if thin[a] and thin[b] and c >= 0.9 * every.get(key, 1):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(R)])
    out = list(regions)
    gone = np.zeros(R, bool)
    changed = False
    for r in np.unique(roots[thin]):
        members = np.where(roots == r)[0]
        if len(members) < 3:
            continue
        faces = np.concatenate([regions[i][2] for i in members])
        res = _exact_over(tm, faces, tol, thr)
        if res is None:
            continue
        out[members[0]] = res
        gone[members[1:]] = True
        changed = True
    if not changed:
        return regions, label
    out = [r for i, r in enumerate(out) if not gone[i]]
    lab2 = np.full(len(label), -1, np.int64)
    for k, (_, _, f) in enumerate(out):
        lab2[f] = k
    return out, lab2


def _continuity_merge(tm: TriMesh, regions, label, tol: float, noise: float):
    """One smooth face is one surface. The pieces of a freeform face (a rolling-ball
    fillet along a curved edge, a turned spline profile) each fit some cylinder, torus
    or sphere approximately, but never exactly the way a true analytic face fits its
    own tessellation (float noise). Neighbouring approximate pieces and freeform
    regions that meet smoothly join into one freeform surface; exact ones stay."""
    # how exactly a true surface fits its own vertices: the flat faces show the
    # coordinates' rounding (the local-quadric noise estimate also picks up the
    # curvature of coarse curved facets, so it overstates this on all-curved parts)
    # Judged on interior vertices only: a mesher puts those exactly on the surface,
    # while edge vertices come from edge curves and carry the source's edge tolerance.
    nV = len(tm.V)
    fl = np.repeat(label, 3)
    lo = np.full(nV, np.iinfo(np.int64).max)
    hi = np.full(nV, -1)
    np.minimum.at(lo, tm.F.ravel(), fl)
    np.maximum.at(hi, tm.F.ravel(), fl)

    def inner_rms(kind, prm, faces, rid):
        v = np.unique(tm.F[faces])
        v = v[(lo[v] == rid) & (hi[v] == rid)]
        if len(v) < 3:
            return None  # no interior: nothing tells exact from approximate
        d, _ = residual(kind, prm, tm.V[v], np.zeros((len(v), 3)))
        return float(np.sqrt(np.mean(d ** 2)))

    plane_rms = [r for i, (k, p, faces) in enumerate(regions)
                 if k == "plane" and (r := inner_rms("plane", p, faces, i)) is not None]
    floor = float(np.median(plane_rms)) if plane_rms else noise
    thr = max(4 * floor, 0.03 * tol)
    R = len(regions)
    soft = np.zeros(R, bool)
    for i, (k, p, faces) in enumerate(regions):
        if k == "freeform":
            soft[i] = True
        elif k in CURVED_KINDS:
            r = inner_rms(k, p, faces, i)
            soft[i] = r is not None and r > thr
    if soft.sum() < 2:
        return regions, label

    def border_counts(G):
        coo = G.tocoo()
        la, lb = label[coo.row], label[coo.col]
        m = (la >= 0) & (lb >= 0) & (la < lb)
        u, c = np.unique(la[m].astype(np.int64) * R + lb[m], return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))

    every, smooth = border_counts(tm.Gall), border_counts(tm.G)
    parent = np.arange(R)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key, n_all in every.items():
        a, b = divmod(key, R)
        # tangent all along the border (no crease): the same smooth face
        if soft[a] and soft[b] and smooth.get(key, 0) >= 0.9 * n_all:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(R)])
    if len(np.unique(roots)) == R:
        return regions, label
    out = []
    for r in np.unique(roots):
        members = np.where(roots == r)[0]
        if len(members) == 1:
            out.append(regions[members[0]])
        else:
            faces = np.concatenate([regions[i][2] for i in members])
            out.append(_exact_over(tm, faces, tol, thr) or ("freeform", {}, faces))
    lab = np.full(len(label), -1, np.int64)
    for k, (_, _, faces) in enumerate(out):
        lab[faces] = k
    return out, lab


def _absorb_tiny(tm: TriMesh, regions, label, loose: float, max_tri: int = 2, tol: float = 0.0):
    """Not every face of a poor tessellation is a surface: a region of one or two
    triangles, or a sliver strip narrower than twice the loose tolerance, that meets its neighbours smoothly (no sharp edge around it: not a
    real small face like a chamfer strip) joins the neighbour whose surface fits it
    best, within a looser tolerance."""
    size = np.array([len(r[2]) for r in regions])

    def width(faces):  # 2 x area / perimeter: a 5-micron strip is a sliver however many triangles
        e = np.sort(tm.F[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        u, c = np.unique(e, axis=0, return_counts=True)
        b = u[c == 1]
        per = float(np.linalg.norm(tm.V[b[:, 0]] - tm.V[b[:, 1]], axis=1).sum()) if len(b) else 0.0
        return 2 * float(tm.A[faces].sum()) / per if per > 0 else np.inf

    sliver_w = 2 * loose
    # triangles thinner than the tolerance (needles a mesher leaves along an edge):
    # their normal is noise, so their edges all look sharp
    E = tm.V[tm.F]
    longest = np.linalg.norm(E[:, [1, 2, 0]] - E, axis=2).max(axis=1)
    degen = 2 * tm.A < max(tol, 1e-300) * np.maximum(longest, 1e-300)
    tiny = np.array([i for i in range(len(regions))
                     if size[i] <= max_tri or (size[i] <= 200 and width(regions[i][2]) < sliver_w)], np.int64)
    if len(tiny) == 0:
        return regions, label
    regs = [list(r) for r in regions]
    alive = np.ones(len(regs), bool)
    for r in tiny:
        faces = regs[r][2]
        flat_out = bool(degen[faces].all())
        if len(faces) > max_tri or flat_out:  # a sliver: its neighbours across sharp edges too
            nb = np.unique(tm.Gall[faces].indices)
        else:  # a tiny patch: only smooth neighbours (else it is a real small face)
            nb = tm.neighbours(faces)
        nb = np.setdiff1d(nb, faces)
        owners = [int(o) for o in np.unique(label[nb]) if o >= 0 and o != r and alive[o]]
        if not owners:
            continue
        best, best_d = None, np.inf
        # a strip with real width (not a few microns) must also face the same way as
        # the surface it joins: a narrow tilted face (a pin's taper) is a face of its own
        wide = len(faces) > max_tri and tol > 0 and width(faces) > 10 * tol
        for o in owners:
            k, prm, _ = regs[o]
            if k == "freeform" or not prm:
                continue
            if wide:
                _, c = residual(k, prm, tm.C[faces], tm.FN[faces])
                if float(np.average(c, weights=tm.A[faces] + 1e-300)) < np.cos(np.radians(10)):
                    continue
            d = float(tm.vert_residual(k, prm, faces).max())
            if d < best_d:
                best, best_d = o, d
        if best is not None and (best_d <= loose or flat_out):
            regs[best][2] = np.r_[regs[best][2], faces]
            label[faces] = best
            alive[r] = False
    out = [tuple(regs[i]) for i in range(len(regs)) if alive[i]]
    lab = np.full(len(label), -1, np.int64)
    for k, (_, _, faces) in enumerate(out):
        lab[faces] = k
    return out, lab


def _coaxial_chain(tm: TriMesh, regs) -> bool:
    """Do these pieces share one axis (a hole with its chamfer and fillets, a turned
    shoulder)? Then they are real features, not fragments of a freeform face."""
    lines = []  # (point, direction or None for a sphere centre, area)
    for k, p, faces in regs:
        a = float(tm.A[faces].sum())
        if k == "cylinder":
            lines.append((np.asarray(p["point"]), _unit(p["axis"]), a))
        elif k == "cone":
            lines.append((np.asarray(p["apex"]), _unit(p["axis"]), a))
        elif k == "torus":
            lines.append((np.asarray(p["center"]), _unit(p["axis"]), a))
        elif k == "sphere":
            lines.append((np.asarray(p["center"]), None, a))
        else:
            lines.append((None, None, a))
    axes = [l for l in lines if l[1] is not None]
    if not axes:
        return False
    total = sum(l[2] for l in lines)
    tol = 2e-3 * tm.diag
    for q, d, _ in axes:  # try each axis as the common one
        on = 0.0
        for p, e, a in lines:
            if p is None:
                continue
            v = p - q
            off = np.linalg.norm(v - (v @ d) * d)
            if off < tol and (e is None or abs(float(e @ d)) > np.cos(np.radians(1.0))):
                on += a
        if on >= 0.7 * total:
            return True
    return False


def _freeform_clusters(tm: TriMesh, regions, label, small: int = 40, min_members: int = 4):
    """A freeform (B-spline) face fits no primitive as a whole, but at micron
    tolerance any few of its triangles fit some sphere or torus: it comes out as a
    cluster of many small curved pieces touching each other. A fillet chain is a few
    sizeable pieces. Clusters of 4+ small curved pieces become one freeform region,
    together with any freeform leftovers they touch."""
    size = np.array([len(r[2]) for r in regions])
    kind = [r[0] for r in regions]
    frag = np.array([(k in CURVED_KINDS and n < small) or k == "freeform" for k, n in zip(kind, size)])
    pairs = _region_pairs(tm, label)
    if len(pairs) == 0:
        return regions, label
    m = frag[pairs[:, 0]] & frag[pairs[:, 1]]
    R = len(regions)
    G = csr_matrix((np.ones(int(m.sum())), (pairs[m, 0], pairs[m, 1])), shape=(R, R))
    _, comp = connected_components(G, directed=False)
    groups: dict[int, list[int]] = {}
    for r in np.where(frag)[0]:
        groups.setdefault(int(comp[r]), []).append(int(r))
    out, lab = [], np.full(len(label), -1, np.int64)
    used = np.zeros(R, bool)
    for members in groups.values():
        n_curved = sum(1 for r in members if kind[r] in CURVED_KINDS)
        if n_curved >= min_members and not _coaxial_chain(tm, [regions[r] for r in members]):
            faces = np.concatenate([regions[r][2] for r in members])
            lab[faces] = len(out)
            out.append(("freeform", {}, faces))
            used[members] = True
    for r in range(R):
        if not used[r]:
            lab[regions[r][2]] = len(out)
            out.append(regions[r])
    return out, lab


def _region_pairs(tm: TriMesh, label: np.ndarray) -> np.ndarray:
    """Pairs of regions that touch across a non-sharp edge."""
    rows = np.repeat(np.arange(len(tm.F)), np.diff(tm.indptr))
    la, lb = label[rows], label[tm.indices]
    m = (la != lb) & (la >= 0) & (lb >= 0)
    return np.unique(np.sort(np.c_[la[m], lb[m]], axis=1), axis=0)


def _close_params(kind, p1, p2, tol) -> bool:
    """Cheap pre-check before a joint refit: could these be the same surface?"""
    ang = lambda u, v: abs(float(np.dot(_unit(u), _unit(v))))
    rel = lambda x, y: abs(x - y) <= 0.02 * max(abs(x), abs(y)) + 3 * tol
    if kind == "plane":
        return ang(p1["normal"], p2["normal"]) > np.cos(np.radians(0.5)) and \
            abs(float(np.dot(_unit(p1["normal"]), p2["point"] - p1["point"]))) < 30 * tol
    if kind == "cylinder":
        return ang(p1["axis"], p2["axis"]) > np.cos(np.radians(1.0)) and rel(p1["radius"], p2["radius"])
    if kind == "sphere":
        return rel(p1["radius"], p2["radius"]) and np.linalg.norm(p1["center"] - p2["center"]) < 0.02 * p1["radius"] + 3 * tol
    if kind == "cone":
        return ang(p1["axis"], p2["axis"]) > np.cos(np.radians(1.0)) and abs(p1["half_angle"] - p2["half_angle"]) < 0.01
    if kind == "torus":
        return ang(p1["axis"], p2["axis"]) > np.cos(np.radians(1.0)) and rel(p1["R"], p2["R"]) and rel(p1["r"], p2["r"])
    return False


def _fits(tm, kind, prm, faces, tol, cos_min) -> bool:
    if kind == "freeform" or not prm:
        return False
    return bool((tm.vert_residual(kind, prm, faces) <= tol).all() and tm.normal_ok(kind, prm, faces, cos_min).all())


def _absorb_and_merge(tm: TriMesh, regions, label, tol, cos_min):
    """Pieces of one surface found from different seeds become one region.

    A few triangles of a cylinder also fit many other cylinders, so a face can come
    out as fragments with slightly different (or plainly wrong) parameters. A piece
    whose vertices all lie on a neighbour's surface joins it; neighbours of the same
    kind whose union one refit explains within tolerance merge.
    """
    regs = [list(r) for r in regions]
    alive = [True] * len(regs)
    for _ in range(4):
        changed = False
        pairs = _region_pairs(tm, label)
        nbrs: dict[int, list[int]] = {}
        for i, j in pairs:
            nbrs.setdefault(int(i), []).append(int(j))
            nbrs.setdefault(int(j), []).append(int(i))
        size = np.array([len(r[2]) for r in regs])
        # 1. absorb small pieces into a neighbour that already explains them
        for r in np.argsort(size):
            r = int(r)
            if not alive[r] or r not in nbrs:
                continue
            for nb in sorted(nbrs[r], key=lambda x: -size[x]):
                if not alive[nb] or size[nb] < size[r]:
                    continue
                kind, prm, _ = regs[nb]
                if _fits(tm, kind, prm, regs[r][2], tol, cos_min):
                    regs[nb][2] = np.r_[regs[nb][2], regs[r][2]]
                    label[regs[r][2]] = nb
                    alive[r] = False
                    size[nb] += size[r]
                    changed = True
                    break
        # 2. merge same-kind neighbours that one refit explains together
        pairs = _region_pairs(tm, label)
        order = np.argsort(-(size[pairs[:, 0]] + size[pairs[:, 1]])) if len(pairs) else []
        for k in order:
            i, j = int(pairs[k, 0]), int(pairs[k, 1])
            if not (alive[i] and alive[j]) or regs[i][0] != regs[j][0] or regs[i][0] == "freeform":
                continue
            kind = regs[i][0]
            if not _close_params(kind, regs[i][1], regs[j][1], tol):
                continue
            big, small = (i, j) if size[i] >= size[j] else (j, i)
            faces = np.r_[regs[big][2], regs[small][2]]
            prm = fit_plane(tm.verts(faces), tm.FN[faces]) if kind == "plane" else \
                _refine(kind, regs[big][1], tm.verts(faces), tol)
            if _valid(kind, prm, tm.diag) and _fits(tm, kind, prm, faces, tol, cos_min):
                regs[big][1], regs[big][2] = prm, faces
                label[regs[small][2]] = big
                alive[small] = False
                size[big] += size[small]
                changed = True
        if not changed:
            break
    out = [tuple(regs[i]) for i in range(len(regs)) if alive[i]]
    lab = np.full(len(label), -1, np.int64)
    for k, (_, _, faces) in enumerate(out):
        lab[faces] = k
    return out, lab


ABSORB_SPECKS = True
SPECK_MAX_TRI = 6
SETTLE_ROUNDS = 4  # rounds of joining slivers and specks after the main passes
EXACT_KINDS = ("plane",) + CURVED_KINDS


def _absorb_specks(tm: TriMesh, regions, label, tol: float, max_tri: int = SPECK_MAX_TRI, log=None):
    """Faces of up to max_tri triangles whose corners all lie on a neighbouring exact face's
    surface join that face (the one they share most edges with, then the closest fit).

    A single triangle is always exactly flat, so a facet of a coarse torus, or a corner
    triangle left over by the passes before, ends up as a 'plane' face of its own: a
    stray triangle no CAD model has. A real tiny face (a chamfer, a resistor's end cap)
    is not on its neighbour's surface and stays; nor does a flat face whose corners lie on
    the surface but whose middle sags deeper below it than the face's own facets."""
    regions = list(regions)
    label = label.copy()
    G = tm.Gall
    moved = 0
    for r in range(len(regions)):
        kind, prm, faces = regions[r]
        if not 0 < len(faces) <= max_tri:
            continue
        shared: dict[int, int] = {}
        for f in faces:
            for g in G.indices[G.indptr[f]:G.indptr[f + 1]]:
                q = int(label[g])
                if q != r and q >= 0:
                    shared[q] = shared.get(q, 0) + 1
        border = sum(shared.values())
        best = None
        for q, n_ in shared.items():
            k2, p2, f2 = regions[q]
            if k2 not in EXACT_KINDS or len(f2) <= max_tri:
                continue
            # it must fill a notch of that face (half its border or more along it): a piece
            # touching it along one edge would stick out of it as a spike the writer collapses
            if 2 * n_ < border:
                continue
            res = float(tm.vert_residual(k2, p2, faces).max())
            if res > tol:
                continue
            # its middle no deeper below the surface than the face's own facets of that size
            # would sag (sag grows with a facet's area): a big flat face whose corners happen
            # to lie on a cylinder (a chord cut) sags far deeper than that
            sag = float(residual(k2, p2, tm.C[faces], np.zeros((len(faces), 3)))[0].max())
            own_f = f2 if len(f2) <= 400 else f2[np.linspace(0, len(f2) - 1, 400).astype(int)]
            own = residual(k2, p2, tm.C[own_f], np.zeros((len(own_f), 3)))[0]
            grow = max(1.0, float(tm.A[faces].max()) / max(float(tm.A[own_f].mean()), 1e-300))
            if sag > max(2 * tol, 3 * float(np.percentile(own, 90)) * grow):
                continue
            if best is None or (n_, -res) > (best[1], -best[2]):
                best = (q, n_, res)
        if best is None:
            continue
        q = best[0]
        k2, p2, f2 = regions[q]
        regions[q] = (k2, p2, np.concatenate([f2, faces]))
        label[faces] = q
        regions[r] = (kind, prm, faces[:0])
        moved += 1
    if moved and log is not None:
        log(f"{moved} stray faces of a few triangles joined the face whose surface they lie on")
    return _compact(regions, label) if moved else (regions, label)


def _compact(regions, label):
    """Drop regions left without triangles; renumber the labels."""
    keep = [r for r in regions if len(r[2])]
    lab = np.full(len(label), -1, np.int64)
    for k, (_, _, faces) in enumerate(keep):
        lab[faces] = k
    return keep, lab


def _reclaim_planes(tm: TriMesh, regions, label, free, tol):
    """Flat triangles a curved region took from a plane go back to it.

    Near the line where a fillet rolls into a plane, the plane's sliver triangles lie
    within microns of the fillet's cylinder or torus and get grown into it, cutting
    the plane apart. A triangle joined to a plane region across a flat edge, that
    fits that plane within tolerance, belongs to the plane.
    """
    rows = np.repeat(np.arange(len(tm.F)), np.diff(tm.indptr))
    flat = tm.angle <= np.radians(0.3)
    kinds = [r[0] for r in regions]
    for _ in range(50):
        la, lb = label[rows[flat]], label[tm.indices[flat]]
        a, b = rows[flat], tm.indices[flat]
        ok = (la >= 0) & (lb >= 0) & (la != lb)
        a, b, la, lb = a[ok], b[ok], la[ok], lb[ok]
        if len(la) == 0:
            break
        is_plane = np.array([k == "plane" for k in kinds], bool)
        to_plane = is_plane[la] & ~is_plane[lb]
        if not to_plane.any():
            break
        moved = 0
        for pid in np.unique(la[to_plane]):
            cand = np.unique(b[to_plane & (la == pid)])
            prm = regions[pid][1]
            fit = cand[tm.vert_residual("plane", prm, cand) <= tol]
            if len(fit) == 0:
                continue
            for owner in np.unique(label[fit]):
                mine = fit[label[fit] == owner]
                k, p, faces = regions[owner]
                regions[owner] = (k, p, np.setdiff1d(faces, mine))
            k, p, faces = regions[pid]
            regions[pid] = (k, p, np.r_[faces, fit])
            label[fit] = pid
            moved += len(fit)
        if moved == 0:
            break
    # regions emptied by this are dropped later (no faces)


def _same_surface(k1, p1, k2, p2, tol, diag):
    if k1 != k2 or k1 == "freeform":
        return False
    ang = lambda u, v: abs(float(np.dot(_unit(u), _unit(v))))
    if k1 == "plane":
        return ang(p1["normal"], p2["normal"]) > np.cos(np.radians(0.05)) and \
            abs(float(np.dot(p1["normal"], p2["point"] - p1["point"]))) < 3 * tol
    if k1 == "cylinder":
        if ang(p1["axis"], p2["axis"]) < np.cos(np.radians(0.1)) or abs(p1["radius"] - p2["radius"]) > 3 * tol:
            return False
        v = p2["point"] - p1["point"]
        return np.linalg.norm(v - (v @ p1["axis"]) * p1["axis"]) < 3 * tol
    if k1 == "sphere":
        return np.linalg.norm(p1["center"] - p2["center"]) < 3 * tol and abs(p1["radius"] - p2["radius"]) < 3 * tol
    if k1 == "cone":
        return ang(p1["axis"], p2["axis"]) > np.cos(np.radians(0.1)) and \
            np.linalg.norm(p1["apex"] - p2["apex"]) < 30 * tol and abs(p1["half_angle"] - p2["half_angle"]) < 1e-3
    if k1 == "torus":
        return ang(p1["axis"], p2["axis"]) > np.cos(np.radians(0.1)) and \
            np.linalg.norm(p1["center"] - p2["center"]) < 3 * tol and abs(p1["R"] - p2["R"]) < 3 * tol and \
            abs(p1["r"] - p2["r"]) < 3 * tol
    return False


def _merge_same(tm: TriMesh, regions, label, tol):
    """Merge adjacent regions that are the same surface."""
    rows = np.repeat(np.arange(len(tm.F)), np.diff(tm.indptr))
    la, lb = label[rows], label[tm.indices]
    m = (la != lb) & (la >= 0) & (lb >= 0)
    pairs = np.unique(np.sort(np.c_[la[m], lb[m]], axis=1), axis=0)
    parent = list(range(len(regions)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ki, pi, _ = regions[i]
        kj, pj, _ = regions[j]
        if _same_surface(ki, pi, kj, pj, tol, tm.diag):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = sorted({find(i) for i in range(len(regions))})
    new_id = {r: k for k, r in enumerate(roots)}
    merged = []
    for r in roots:
        members = [i for i in range(len(regions)) if find(i) == r]
        faces = np.concatenate([regions[i][2] for i in members])
        kind, prm, _ = regions[r]
        if len(members) > 1 and kind != "freeform":
            new = refine_geometric(kind, prm, tm.verts(faces), tol / 3) if kind != "plane" else \
                fit_plane(tm.verts(faces), tm.FN[faces])
            prm = new if _valid(kind, new, tm.diag) else prm
        merged.append((kind, prm, faces))
    lab = np.full(len(label), -1, np.int64)
    for k, (_, _, faces) in enumerate(merged):
        lab[faces] = k
    return merged, lab


HELICAL_FACES = True


def _helical_faces(tm: TriMesh, regions, label, tol: float, log=None):
    """A thread, an auger or a worm: a smooth area that is one profile swept along a
    helix. Its faces are not height fields, so earlier passes cut them into strips and
    sprinkle tiny planes wherever a few facets happen to lie flat. When the freeform
    triangles of a connected area (with the small planes among them) all lie on one
    screw motion, they are regrouped into whole helical faces: pieces between the
    corners of the profile, each fitted with its own profile curve, grown over the
    neighbouring triangles their surface holds and joined where one profile holds two.
    Each becomes one freeform region carrying its helical surface (`helix_lead` etc.),
    exact to the mesh."""
    from scipy.spatial import cKDTree

    from .helix import _coords, _thickness, fit_helix_fixed, screw_motion
    from .segmentation import _perp

    ntri = np.array([len(f) for _, _, f in regions])
    ff = np.array([k == "freeform" and "tube_r" not in p and "helix_lead" not in p for k, p, _ in regions], bool)
    if not ff.any():
        return regions, label
    small = np.array([k == "plane" for k, _, _ in regions], bool) & (ntri <= 200)
    cand_reg = ff | small
    ok_lab = label >= 0
    cand = np.zeros(len(label), bool)
    cand[ok_lab] = cand_reg[label[ok_lab]]
    tri = np.nonzero(cand)[0]
    if len(tri) < 300:
        return regions, label
    _, comp = connected_components(tm.Gall[tri][:, tri], directed=False)
    new = []  # (helix surface, faces)
    taken = np.zeros(len(label), bool)
    for cc in np.unique(comp):
        faces = tri[comp == cc]
        ffaces = faces[ff[label[faces]]]
        if len(ffaces) < 300:
            continue
        vid = np.unique(tm.F[ffaces])
        Nv = np.zeros((len(tm.V), 3))
        np.add.at(Nv, tm.F[ffaces].ravel(), np.repeat(tm.FN[ffaces] * tm.A[ffaces, None], 3, axis=0))
        N = Nv[vid] / np.maximum(np.linalg.norm(Nv[vid], axis=1, keepdims=True), 1e-300)
        try:
            sm = screw_motion(tm.V[vid], N, tol)
        except Exception:
            sm = None
        if sm is None:
            continue
        a, c, L, _ = sm
        # thickness of every vertex of the area (freeform and small planes) on that screw
        av = np.unique(tm.F[faces])
        rho, q, _ = _coords(tm.V[av], a, c, L)
        if abs(L) < 0.02 * float(rho.max()):
            # a lead next to nothing: a surface of revolution, which any screw of tiny
            # lead about its axis maps onto one curve (the finest real thread advances
            # several percent of its radius per turn)
            continue
        X = np.c_[rho, q]
        _, nn = cKDTree(X).query(X, k=10)
        thin = np.zeros(len(tm.V), bool)
        thin[av] = np.abs(_thickness(X, nn[:, 1:])) <= 0.1 * tol
        # corners of the profile (where a groove meets a crest): seen from distinct profile
        # positions (a row of vertices round the turns lands on one point), a corner's
        # neighbourhood is a V, not a line
        corner = np.zeros(len(tm.V), bool)
        corner[av] = _profile_corners(X, tol)
        tri_thin = thin[tm.F[faces]].all(axis=1) & ~corner[tm.F[faces]].any(axis=1)
        fa = tm.A[ffaces]
        if float(fa[thin[tm.F[ffaces]].all(axis=1)].sum()) < 0.8 * float(fa.sum()):
            continue  # not one screw motion: some other freeform surface
        # it must wind round its axis: a full turn at least
        e1 = _perp(a)
        e2 = np.cross(a, e1)
        v = tm.V[vid] - c
        th = np.arctan2(v @ e2, v @ e1)
        s_ = th + 2 * np.pi * np.round(((v @ a) - L * th / (2 * np.pi) - np.median(q)) / L)
        if np.ptp(s_) < 2 * np.pi:
            continue
        # seed pieces: all-thin triangles between sharp edges
        seed = faces[tri_thin]
        Gs = tm.G[seed][:, seed]
        _, pc = connected_components(Gs, directed=False)
        pieces = []
        n_bad = 0  # pieces with no single clean profile (a thread's run-out, a seam band)
        for p in np.unique(pc):
            pf = seed[pc == p]
            if len(pf) < 30:
                continue
            hx = fit_helix_fixed(tm.V[np.unique(tm.F[pf])], a, c, L, tol)
            if hx is not None and hx.max_res > tol:
                # a few stray triangles (where a thread runs out) off the profile: drop them
                rt = hx.residual(tm.V[tm.F[pf]].reshape(-1, 3)).reshape(-1, 3).max(axis=1)
                good = rt <= tol
                if good.sum() >= 0.95 * len(pf):
                    hx2 = fit_helix_fixed(tm.V[np.unique(tm.F[pf[good]])], a, c, L, tol)
                    if hx2 is not None and hx2.max_res <= tol:
                        hx, pf = hx2, pf[good]
            if hx is None or hx.max_res > tol:
                n_bad += 1
                continue
            pieces.append([hx, pf])
        if not pieces:
            continue
        # grow each piece over neighbouring area triangles its surface holds
        owner = np.full(len(label), -1, np.int64)
        for k_, (_, pf) in enumerate(pieces):
            owner[pf] = k_
        inarea = np.zeros(len(label), bool)
        inarea[faces] = True
        for rnd in range(4):
            n_before = int((owner >= 0).sum())
            _grow_pieces(tm, pieces, owner, inarea, tol)
            if rnd == 3 or int((owner >= 0).sum()) == n_before and rnd > 0:
                break
            # refit each piece on what it holds now: its profile then reaches further
            for k_ in range(len(pieces)):
                pf = np.nonzero(owner == k_)[0]
                hx2 = fit_helix_fixed(tm.V[np.unique(tm.F[pf])], a, c, L, tol)
                if hx2 is not None and hx2.max_res <= tol:
                    pieces[k_][0] = hx2
        # neighbouring pieces on one profile (one face cut apart by a flat spot) join, and
        # an earlier region left over between them (a band along a seam of the source
        # mesh, whose own rows give no clean profile) joins a piece when the two fit as one
        merged = True
        while merged:
            merged = False
            for g in np.unique(label[faces[owner[faces] < 0]]):
                left = faces[(label[faces] == g) & (owner[faces] < 0)]
                ow = owner[np.unique(tm.Gall[left].indices)]
                ow = ow[ow >= 0]
                for k_ in np.argsort(-np.bincount(ow, minlength=len(pieces)))[:3] if len(ow) else []:
                    if not np.any(ow == k_):
                        break
                    both = np.r_[np.nonzero(owner == k_)[0], left]
                    hxu = fit_helix_fixed(tm.V[np.unique(tm.F[both])], a, c, L, tol)
                    if hxu is not None and hxu.max_res <= tol:
                        owner[left] = k_
                        pieces[k_][0] = hxu
                        merged = True
                        break
            G2 = tm.Gall.tocoo()
            oa, ob = owner[G2.row], owner[G2.col]
            m = (oa >= 0) & (ob >= 0) & (oa != ob)
            pairs, cnt = np.unique(np.sort(np.c_[oa[m], ob[m]], axis=1), axis=0, return_counts=True)
            for (p1, p2) in pairs[np.argsort(-cnt)]:
                both = np.nonzero((owner == p1) | (owner == p2))[0]
                hxu = fit_helix_fixed(tm.V[np.unique(tm.F[both])], a, c, L, tol)
                if hxu is not None and hxu.max_res <= tol:
                    owner[owner == p2] = p1
                    pieces[p1][0] = hxu
                    merged = True
                    break
        _grow_pieces(tm, pieces, owner, inarea, tol)
        # crumbs of the area left between helical faces (a few triangles where a profile
        # ends) join the face they border most if it holds them nearly: a crumb face
        # breaks a CAD reader's shell
        rest = faces[owner[faces] < 0]
        if len(rest):
            sub = tm.Gall[rest][:, rest]
            _, rc = connected_components(sub, directed=False)
            for r_ in np.unique(rc):
                cf = rest[rc == r_]
                if len(cf) > 8:
                    continue
                nb = tm.Gall[cf].indices
                ow = owner[nb]
                ow = ow[ow >= 0]
                if len(ow) == 0:
                    continue
                k_ = int(np.bincount(ow).argmax())
                if pieces[k_][0].residual(tm.V[np.unique(tm.F[cf])]).max() <= 3 * tol:
                    owner[cf] = k_
        # an earlier region partly taken must not be left as a crumb: its last few
        # triangles join the helical face beside them if it nearly holds them, else the
        # region gets back what was taken
        for g in np.unique(label[faces]):
            mine = faces[label[faces] == g]
            left, took = mine[owner[mine] < 0], mine[owner[mine] >= 0]
            if len(left) == 0 or len(took) == 0 or len(left) > 10:
                continue
            ow = owner[np.unique(tm.Gall[left].indices)]
            ow = ow[ow >= 0]
            if len(ow):
                k_ = int(np.bincount(ow).argmax())
                if pieces[k_][0].residual(tm.V[np.unique(tm.F[left])]).max() <= 3 * tol:
                    owner[left] = k_
                    continue
            owner[took] = -1
        # worth it only when the area ends up with fewer faces, nearly all of it helical:
        # a thread whose flanks already are one region each is left as it is, and so is
        # an area only partly regrouped (its leftovers would be extra fragments)
        held = faces[owner[faces] >= 0]
        n_before = len(np.unique(label[faces]))
        n_after = len(np.unique(owner[held])) + len(np.unique(label[faces[owner[faces] < 0]]))
        if n_after >= n_before or len(held) < 0.75 * len(faces):
            if log:
                log(f"helical faces: lead {L:.6g}, area of {len(faces)} triangles left as it is "
                    f"({n_before} faces -> {n_after}, {len(held)} triangles held)")
            continue
        n_new = len(new)
        for k_, (hx, _) in enumerate(pieces):
            pf = np.nonzero(owner == k_)[0]
            if len(pf) == 0:
                continue
            hx2 = fit_helix_fixed(tm.V[np.unique(tm.F[pf])], a, c, L, tol)
            if hx2 is not None and hx2.max_res <= tol:
                hx = hx2
            new.append((hx, pf))
            taken[pf] = True
        if log:
            log(f"helical faces: lead {L:.6g}, {len(new) - n_new} faces from {len(faces)} triangles "
                f"({int(taken[faces].sum())} regrouped; {n_bad} seed pieces without a clean profile)")
    if not new:
        return regions, label
    out = []
    for k, prm, f in regions:
        f2 = f[~taken[f]]
        if len(f2):
            out.append((k, prm, f2))
    for hx, pf in new:
        prm = hx.to_json()
        prm["helix_rms"] = float(hx.rms)
        out.append(("freeform", prm, pf))
    lab2 = np.full(len(label), -1, np.int64)
    for i, (_, _, f) in enumerate(out):
        lab2[f] = i
    return out, lab2


def _grow_pieces(tm: TriMesh, pieces, owner, inarea, tol):
    """Grow helical pieces ([surface, faces]) over free neighbouring triangles of the area
    whose three corners their surface holds within tol, ring by ring from the newly
    added triangles; a triangle two pieces could take goes to the closer surface."""
    tried = [set() for _ in pieces]
    frontier = [np.nonzero(owner == k_)[0] for k_ in range(len(pieces))]
    while True:
        best_res = {}
        for k_, (hx, _) in enumerate(pieces):
            if len(frontier[k_]) == 0:
                continue
            nb = np.unique(tm.Gall[frontier[k_]].indices)
            nb = nb[inarea[nb] & (owner[nb] < 0)]
            nb = np.array([x for x in nb.tolist() if x not in tried[k_]], np.int64)
            if len(nb) == 0:
                continue
            tried[k_].update(nb.tolist())
            Pv = tm.V[tm.F[nb]].reshape(-1, 3)
            sv, xv = hx.coords(Pv)
            r = np.linalg.norm(hx.point(sv, xv) - Pv, axis=1).reshape(-1, 3).max(axis=1)
            # only where the profile already is (plus a sliver): its straight continuation
            # past the end can pass through a neighbouring face's corner rows
            lo_, hi_ = hx.x_rng
            m_ = 0.02 * (hi_ - lo_)
            xv = xv.reshape(-1, 3)
            r[(xv.min(axis=1) < lo_ - m_) | (xv.max(axis=1) > hi_ + m_)] = np.inf
            # and facing the way the surface does there
            cn = np.abs(np.einsum("ij,ij->i", hx.normals(tm.C[nb]), tm.FN[nb]))
            r[cn < np.cos(np.radians(30))] = np.inf
            for x, rr in zip(nb[r <= tol].tolist(), r[r <= tol].tolist()):
                if x not in best_res or rr < best_res[x][0]:
                    best_res[x] = (rr, k_)
        if not best_res:
            return
        frontier = [[] for _ in pieces]
        for x, (_, k_) in best_res.items():
            owner[x] = k_
            frontier[k_].append(x)
        frontier = [np.array(f_, np.int64) for f_ in frontier]


def _profile_corners(X, tol, max_ratio: float = 0.2):
    """Profile points (rows of X) at a corner of the profile curve set: around each
    distinct position, the neighbours within a small radius spread across two
    directions (a V) instead of along one line."""
    from scipy.spatial import cKDTree

    key = np.round(X / max(0.01 * tol, 1e-300)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    U = X[first]
    if len(U) < 8:
        return np.zeros(len(X), bool)
    kd = cKDTree(U)
    dk, _ = kd.query(U, k=min(4, len(U)))
    spacing = float(np.median(dk[:, -1]))
    ext = float(np.ptp(U, axis=0).max())
    r1 = max(0.01 * ext, 2.5 * spacing)
    out = np.zeros(len(U), bool)
    for i, nb in enumerate(kd.query_ball_point(U, r1)):
        if len(nb) < 4:
            continue
        D = U[nb] - U[i]
        # spread across the main direction of the neighbourhood, relative to its length
        C = D.T @ D / len(nb)
        w = np.linalg.eigvalsh(C)
        out[i] = np.sqrt(max(w[0], 0.0) / max(w[1], 1e-300)) > max_ratio
    return out[inv]


# a region whose long edges leave its surface mixes several true faces: re-explain them
SPLIT_MIXED = True
SPLIT_SAG_DELTA = 3.0  # x the mesh's own chord sag (99th percentile over curved regions)
SPLIT_SAG_TOL = 10.0  # never below this many vertex tolerances


def _mid_sag(tm: TriMesh, kind, prm, faces) -> np.ndarray:
    """Per triangle, the largest distance of its edge midpoints from the surface. On the
    true surface it is the chord sag, which the tessellator keeps small; a surface that
    only passes through the corners (a torus through the two end rings of a long rod
    strip) bulges away from the edge between them."""
    T = tm.V[tm.F[faces]]
    M = (T + np.roll(T, -1, axis=1)) / 2
    d, _ = residual(kind, prm, M.reshape(-1, 3), np.zeros((3 * len(faces), 3)))
    d = np.where(np.isfinite(d), d, np.inf)
    return d.reshape(-1, 3).max(axis=1)


def _holds(tm: TriMesh, kind, prm, faces, tol, sag_max, cos_min) -> bool:
    if kind == "freeform" or not prm:
        return False
    if not (tm.vert_residual(kind, prm, faces) <= tol).all():
        return False
    if not tm.normal_ok(kind, prm, faces, cos_min).all():
        return False
    return kind == "plane" or bool((_mid_sag(tm, kind, prm, faces) <= sag_max).all())


def _pieces(tm: TriMesh, faces: np.ndarray):
    if len(faces) == 0:
        return []
    sub = tm.Gall[faces][:, faces]
    _, comp = connected_components(sub, directed=False)
    return [faces[comp == c] for c in np.unique(comp)]


def _fresh_fit(tm: TriMesh, faces, tol, sag_max, cos_min):
    """The simplest surface that holds these triangles on its own (corners, normals and
    edge midpoints), or None."""
    P = tm.verts(faces)
    if len(faces) < 3 or len(P) < 4:  # a lone triangle fits anything: never a face of its own
        return None
    prm = fit_plane(P, tm.FN[faces])
    if _holds(tm, "plane", prm, faces, tol, sag_max, cos_min):
        return "plane", prm
    found = []
    for kind, guess in _hypotheses(tm, faces):
        if not _valid(kind, guess, tm.diag):
            continue
        new = _refine(kind, guess, P, tol)
        if _valid(kind, new, tm.diag) and _holds(tm, kind, new, faces, tol, sag_max, cos_min):
            found.append((kind, new))
    rank = {"cylinder": 0, "cone": 1, "sphere": 2, "torus": 3}
    found.sort(key=lambda kp: rank[kp[0]])
    return found[0] if found else None


def _split_mixed(tm: TriMesh, regions, label, tol: float, cos_min: float, log=None):
    """Regions that mix several true faces. A thin rod meshed with one row of long
    slivers has its vertices on two end rings only, so any surface through both rings
    fits them, and the rod's strip plus a few triangles of the bend or cap next to it
    come out as one region on a made-up torus. Its long edges give it away: their
    midpoints lie on the rod's cylinder but far off the torus. Triangles whose edges
    leave the surface by more than the mesh's own chord sag move to a neighbour whose
    surface holds them, or to a surface fitted on them alone; what is left of the region
    joins a neighbour that holds it, or keeps its own (simplest) surface."""
    regions = list(regions)
    label = label.copy()
    curved = [i for i, (k, _, f) in enumerate(regions) if k in CURVED_KINDS and len(f) >= 2]
    if not curved:
        return regions, label
    sags = {i: _mid_sag(tm, regions[i][0], regions[i][1], regions[i][2]) for i in curved}
    allsag = np.concatenate(list(sags.values()))
    allsag = allsag[np.isfinite(allsag)]
    if not len(allsag):
        return regions, label
    sag_max = max(SPLIT_SAG_TOL * tol, SPLIT_SAG_DELTA * float(np.percentile(allsag, 99)))
    n_split = 0
    for i in curved:
        kind, prm, faces = regions[i]
        if len(faces) != len(sags[i]):  # changed by an earlier split this pass
            continue
        bad = sags[i] > sag_max
        if not bad.any():
            continue
        B, G = faces[bad], faces[~bad]
        moves = []  # (faces, target region index | (kind, prm) for a new region)
        ok = True
        for piece in _pieces(tm, B):
            # the new surface must explain the piece clearly better, not just pass
            lim = min(sag_max, 0.5 * float(sags[i][np.isin(faces, piece)].max()))
            nb = np.setdiff1d(tm.neighbours(piece), faces)
            cand = [int(r) for r in np.unique(label[nb]) if r >= 0 and r != i]
            cand.sort(key=lambda r: -len(regions[r][2]))
            home = next((r for r in cand if _holds(tm, regions[r][0], regions[r][1], piece, tol, lim, cos_min)),
                        None)
            if home is None:
                home = _fresh_fit(tm, piece, tol, lim, cos_min)
            if home is None:
                ok = False
                break
            moves.append((piece, home))
        if not ok:
            continue
        keep = []
        for piece in _pieces(tm, G):
            nb = np.setdiff1d(tm.neighbours(piece), faces)
            cand = [int(r) for r in np.unique(label[nb]) if r >= 0 and r != i]
            cand.sort(key=lambda r: -len(regions[r][2]))
            home = next((r for r in cand if _holds(tm, regions[r][0], regions[r][1], piece, tol, sag_max, cos_min)),
                        None)
            if home is None:
                keep.append(piece)
            else:
                moves.append((piece, home))
        for piece, home in moves:
            if isinstance(home, tuple):
                label[piece] = len(regions)
                regions.append((home[0], home[1], piece))
            else:
                k2, p2, f2 = regions[home]
                regions[home] = (k2, p2, np.r_[f2, piece])
                label[piece] = home
        if keep:
            rest = np.concatenate(keep)
            k1, p1 = _simplest(tm, kind, prm, rest, tol, cos_min)
            if not _holds(tm, k1, p1, rest, tol, sag_max, cos_min):
                k1, p1 = kind, prm
            regions[i] = (k1, p1, rest)
        else:
            regions[i] = (kind, prm, faces[:0])
        n_split += 1
        if log:
            log(f"mixed region {i} ({kind}, {len(faces)} triangles): {int(bad.sum())} off its surface moved")
    if not n_split:
        return regions, label
    if log:
        log(f"{n_split} regions mixing several faces split (edge sag limit {sag_max:.3g})")
    return _compact(regions, label)


SMOOTH_BORDERS = True
TOOTH_TIP_DEG = 120.0  # a tooth whose tip is sharper than this makes the border zig-zag
ISLAND_MAX = 200  # pieces of a region cut off from its main part, up to this many triangles
FLAT_CONE_DEG = 75.0  # cones flatter than this keep their borders
TINY_MAX = 3  # exact regions this small inside a freeform face are specks of it
TINY_AREA = 0.02  # ... if their area is below this share of that freeform face


def _across(tm: TriMesh, G) -> np.ndarray:
    """(n,3): the triangle across each edge (edge j runs from corner j to j+1) in graph G, or -1."""
    coo = G.tocoo()
    a, b = coo.row.astype(np.int64), coo.col.astype(np.int64)
    out = np.full((len(tm.F), 3), -1, np.int64)
    Fa, Fb = tm.F[a], tm.F[b]
    for j in range(3):
        p, q = Fa[:, j], Fa[:, (j + 1) % 3]
        hit = ((Fb == p[:, None]).any(1)) & ((Fb == q[:, None]).any(1))
        out[a[hit], j] = b[hit]
    return out


def _sag_limit(tm: TriMesh, regions, tol: float) -> float:
    curved = [(k, p, f) for k, p, f in regions if k in CURVED_KINDS and len(f) >= 2]
    if not curved:
        return SPLIT_SAG_TOL * tol
    s = np.concatenate([_mid_sag(tm, k, p, f) for k, p, f in curved])
    s = s[np.isfinite(s)]
    return max(SPLIT_SAG_TOL * tol, SPLIT_SAG_DELTA * float(np.percentile(s, 99))) if len(s) else SPLIT_SAG_TOL * tol


def _takes(tm: TriMesh, regions, src, dst, faces, tol, sag_max, cos_min) -> np.ndarray:
    """Per triangle: may it move from region src to region dst? An exact face takes it if
    its surface holds the corners no worse than the triangle's own surface does (or within
    tol), the normals agree and edge midpoints stay within the mesh's chord sag; both fitting
    is the usual case along a tangent border. A freeform face takes nothing (it can only give
    triangles to an exact face that holds them), and nearly flat cones keep their borders."""
    kd, pd, _ = regions[dst]
    ks, ps, _ = regions[src]
    if kd == "freeform" or not pd:
        # a freeform face takes nothing: its spline patch is fitted to exactly the triangles
        # it has, earlier steps cut wrapping surfaces so each piece is one patch, and even
        # a few triangles more along its border can leave the patch invalid
        return np.zeros(len(faces), bool)
    # a nearly flat cone (half angle near 90 degrees) is an ill-conditioned fit, usually a
    # made-up tile of some other surface: its border stays where it is
    for k_, p_ in ((ks, ps), (kd, pd)):
        if k_ == "cone" and abs(float(p_["half_angle"])) > np.radians(FLAT_CONE_DEG):
            return np.zeros(len(faces), bool)
    rd = tm.vert_residual(kd, pd, faces)
    if ks == "freeform" or not ps:
        lim = np.full(len(faces), tol)
    else:
        lim = np.maximum(tol, tm.vert_residual(ks, ps, faces))
    ok = (rd <= lim) & tm.normal_ok(kd, pd, faces, cos_min)
    if kd != "plane" and ok.any():
        ok[ok] = _mid_sag(tm, kd, pd, faces[ok]) <= sag_max
    return ok


def _corner_angles(tm: TriMesh) -> np.ndarray:
    """(n,3) the angle in degrees at each corner of each triangle."""
    V, F = tm.V, tm.F
    e0, e1 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 1]]

    def ang(u, v):
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-300)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    return np.c_[ang(e0, e1), ang(-e0, e2), ang(-e1, -e2)]


def _teeth(acr, lab, corner):
    """Triangles with two smooth edges on one other region and a tip sharper than
    TOOTH_TIP_DEG between them, or all three edges on it: (triangles, region they bite into)."""
    L = np.where(acr >= 0, lab[np.maximum(acr, 0)], -1)  # region across each smooth edge
    other = (L >= 0) & (L != lab[:, None])
    ts, bs = [], []
    for j in range(3):
        bj = L[:, j]
        nxt = other[:, j] & (L[:, (j + 1) % 3] == bj)
        prv = L[:, (j + 2) % 3] == bj
        t2 = np.flatnonzero(nxt & ~prv)  # edges j and j+1: the tip is corner j+1
        t2 = t2[corner[t2, (j + 1) % 3] < TOOTH_TIP_DEG]
        ts.append(t2)
        bs.append(bj[t2])
        if j == 0:
            t3 = np.flatnonzero(nxt & prv)
            ts.append(t3)
            bs.append(bj[t3])
    return np.concatenate(ts), np.concatenate(bs)


def _vertex_fans(tm: TriMesh):
    """Triangles around each vertex (CSR: ptr, tri, corner) and vertices on the mesh's open
    boundary (their fan is not closed, so their angles say nothing)."""
    F = tm.F
    nv = len(tm.V)
    v = F.ravel()
    o = np.argsort(v, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(v, minlength=nv))]
    tri, cor = o // 3, o % 3
    e = np.sort(np.c_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]].reshape(-1, 2), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    open_v = np.zeros(nv, bool)
    open_v[u[c == 1].ravel()] = True
    return ptr, tri, cor, open_v


def _spikes(tm, fans, acr, acr_all, lab, corner):
    """Border vertices where exactly two faces meet and one of them comes to a point sharper
    than TOOTH_TIP_DEG (share of the full angle around the vertex): a notch or spike in what
    should be one smooth border curve. -> [(vertex, region with the spike, other region)]"""
    ptr, tri, cor, open_v = fans
    nv = len(ptr) - 1
    vv = np.repeat(np.arange(nv), np.diff(ptr))
    lt = lab[tri]
    ang = corner[tri, cor]
    R1 = int(lab.max()) + 2
    key = vv * R1 + lt
    uk, inv = np.unique(key, return_inverse=True)
    asum = np.bincount(inv, weights=ang)
    kv = uk // R1
    nreg = np.bincount(kv, minlength=nv)
    total = np.bincount(vv, weights=ang, minlength=nv)
    two = np.flatnonzero((nreg == 2) & ~open_v & (total > 0))
    if not len(two):
        return []
    first = np.searchsorted(kv, two)
    a1, a2 = asum[first], asum[first + 1]
    r1, r2 = uk[first] % R1, uk[first + 1] % R1
    small = np.minimum(a1, a2) / total[two] * 360.0
    hit = small < TOOTH_TIP_DEG
    out = []
    for v, s1, s2, x1, x2, sm in zip(two[hit], r1[hit], r2[hit], a1[hit], a2[hit], small[hit]):
        spike, other = (s1, s2) if x1 <= x2 else (s2, s1)
        out.append((sm, int(v), int(spike), int(other)))
    out.sort()
    return [(v, a, b) for _, v, a, b in out]


def _border_delta(S, a, b, lab, acr_all, elen, mark) -> float:
    """Change in total border length if triangles S move from region a to region b."""
    mark[S] = True
    u = acr_all[S]
    ok = (u >= 0) & ~mark[np.maximum(u, 0)]
    lu = lab[np.maximum(u, 0)]
    before = ok & (lu != a)
    after = ok & (lu != b)
    mark[S] = False
    return float((elen[S] * (after.astype(float) - before)).sum())


def _smooth_borders(tm: TriMesh, regions, label, tol: float, cos_min: float, log=None):
    """A CAD face is bounded by smooth curves, never a saw edge. Where two faces meet
    tangentially (a fillet rolling into a plane, a torus into a cylinder) the triangles along
    the border fit both surfaces within the vertex tolerance, the greedy growth hands them out
    in turn, and the border zig-zags. A triangle with two smooth edges on one other region
    and a sharp tip between them (a tooth) moves to that region when its surface takes it;
    repeated, the border settles on a straight row of edges. A tooth spread over several
    triangles shows as a border vertex where one face comes to a sharp point: its triangles
    there move the same way. Every move must shorten the border, so nothing can cycle. Small
    pieces of a region cut off from its main part move to the neighbour they share most border
    with, and specks of an exact face inside a freeform one join it. Two curved faces that one
    surface fits together (made-up tiles of it) keep their border: it is not a real one."""
    regions = list(regions)
    lab = label.copy()
    if len(tm.F) < 4 or not regions:
        return regions, label
    acr = _across(tm, tm.G)  # smooth edges only: a crease is a real border
    acr_all = _across(tm, tm.Gall)
    fans = _vertex_fans(tm)
    Vt = tm.V[tm.F]
    elen = np.linalg.norm(Vt[:, [1, 2, 0]] - Vt, axis=2)  # edge j: corner j to j+1
    mark = np.zeros(len(tm.F), bool)
    eps = 1e-9 * tm.diag
    sag_max = _sag_limit(tm, regions, tol)
    corner = _corner_angles(tm)
    R1 = len(regions) + 1
    moved_teeth = moved_isl = 0
    tiles = {}

    def are_tiles(a, b):
        """Two curved faces that one surface fits together are made-up tiles of it (a torus
        covered by cones and cylinders): their border is not a real one, and moving
        triangles across it only unsettles both. Judged once, on the faces as they came."""
        key = (min(a, b), max(a, b))
        if key not in tiles:
            ka, kb = regions[a][0], regions[b][0]
            if ka not in CURVED_KINDS or kb not in CURVED_KINDS:
                tiles[key] = False
            else:
                both = np.flatnonzero((label == a) | (label == b))
                tiles[key] = len(both) <= 20000 and _fresh_fit(tm, both, tol, sag_max, cos_min) is not None
        return tiles[key]

    for _outer in range(3):
        changed = False
        for _ in range(30):
            t, b = _teeth(acr, lab, corner)
            ok = np.zeros(len(t), bool)
            key = lab[t] * R1 + b
            for k in np.unique(key):
                m = key == k
                if are_tiles(int(k // R1), int(k % R1)):
                    continue
                ok[m] = _takes(tm, regions, int(k // R1), int(k % R1), t[m], tol, sag_max, cos_min)
            t, b = t[ok], b[ok]
            o = np.argsort(t, kind="stable")
            moves = 0
            for tt, bb in zip(t[o].tolist(), b[o].tolist()):
                nb = acr[tt][acr[tt] >= 0]
                # still a tooth into bb after the moves before it this round?
                if lab[tt] == bb or int((lab[nb] == bb).sum()) < 2:
                    continue
                # only moves that shorten the border: the process cannot cycle
                if _border_delta(np.array([tt]), int(lab[tt]), bb, lab, acr_all, elen, mark) >= -eps:
                    continue
                lab[tt] = bb
                moves += 1
            # spikes and notches at border vertices (a tooth spread over several triangles)
            ptr, tri, _, _ = fans
            for v, a, b in _spikes(tm, fans, acr, acr_all, lab, corner):
                fan = tri[ptr[v]:ptr[v + 1]]
                ls = set(lab[fan].tolist())
                if ls != {a, b}:
                    continue  # changed by an earlier move this round
                sp = fan[lab[fan] == a]
                # the spike must meet the other face across smooth edges only (a crease is a real border)
                nb_all = acr_all[sp]
                nb_s = acr[sp]
                cross = (nb_all >= 0) & (lab[np.maximum(nb_all, 0)] == b)
                if (cross & (nb_s != nb_all)).any():
                    continue
                if _border_delta(sp, a, b, lab, acr_all, elen, mark) >= -eps:
                    continue
                if are_tiles(a, b):
                    continue
                if not _takes(tm, regions, a, b, sp, tol, sag_max, cos_min).all():
                    continue
                lab[sp] = b
                moves += len(sp)
            if not moves:
                break
            moved_teeth += moves
            changed = True
        # small pieces cut off from their region's main part
        order = np.argsort(lab, kind="stable")
        bnd = np.searchsorted(lab[order], np.arange(len(regions) + 1))
        for i in range(len(regions)):
            faces = order[bnd[i]:bnd[i + 1]]
            if len(faces) < 2:
                continue
            pcs = _pieces(tm, faces)
            if len(pcs) < 2:
                continue
            pcs.sort(key=len, reverse=True)
            for p in pcs[1:]:
                if len(p) > ISLAND_MAX or len(p) > 0.5 * len(pcs[0]):
                    continue
                nl = lab[acr[p][acr[p] >= 0]]
                nl = nl[(nl != i) & (nl >= 0)]
                if not len(nl):
                    continue
                labs, cnt = np.unique(nl, return_counts=True)
                for dst in labs[np.argsort(-cnt, kind="stable")].tolist():
                    if _takes(tm, regions, i, dst, p, tol, sag_max, cos_min).all():
                        lab[p] = dst
                        moved_isl += len(p)
                        changed = True
                        break
        # one to three triangles always fit a plane (or some surface): inside a freeform
        # face, meeting it smoothly on most of their border, they are a speck of it
        order = np.argsort(lab, kind="stable")
        bnd = np.searchsorted(lab[order], np.arange(len(regions) + 1))
        for i, (k, _, _) in enumerate(regions):
            faces = order[bnd[i]:bnd[i + 1]]
            if k == "freeform" or not 1 <= len(faces) <= TINY_MAX:
                continue
            u = acr_all[faces]
            u = u[u >= 0]
            u = u[lab[u] != i]
            if not len(u):
                continue
            labs, cnt = np.unique(lab[u], return_counts=True)
            j = int(labs[cnt.argmax()])
            if regions[j][0] != "freeform" or cnt.max() < 0.7 * len(u):
                continue
            # a flat face of a coarse mesh is often two big triangles: only a speck small
            # beside the freeform face around it goes
            if tm.A[faces].sum() > TINY_AREA * tm.A[lab == j].sum():
                continue
            us = acr[faces]
            if int(((us >= 0) & (lab[np.maximum(us, 0)] == j)).sum()) < cnt.max():
                continue  # some of that border is a crease: a real small face
            lab[faces] = j
            moved_isl += len(faces)
            changed = True
        if not changed:
            break
    if not (moved_teeth or moved_isl):
        return regions, label
    order = np.argsort(lab, kind="stable")
    bnd = np.searchsorted(lab[order], np.arange(len(regions) + 1))
    regions = [(k, p, order[bnd[i]:bnd[i + 1]]) for i, (k, p, _) in enumerate(regions)]
    if log:
        log(f"saw-tooth borders straightened: {moved_teeth} triangles moved, {moved_isl} in cut-off pieces")
    return _compact(regions, lab)


def _merge_same_any(tm: TriMesh, regions, label, tol: float, log=None):
    """Adjacent faces on one curved surface are one face, whatever the edge between them. A coarse
    rod meets its neighbouring facets at more than the smooth-edge angle, so the early merge
    (smooth edges only) leaves its pieces apart, and later passes can leave two pieces of
    one cylinder side by side. The union is refitted and kept only if every corner still
    lies on the surface; two sides of a folded sheet (normals opposite) never merge."""
    coo = tm.Gall.tocoo()
    la, lb = label[coo.row], label[coo.col]
    m = (la != lb) & (la >= 0) & (lb >= 0)
    if not m.any():
        return regions, label
    pairs = np.unique(np.sort(np.c_[la[m], lb[m]], axis=1), axis=0)
    parent = list(range(len(regions)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def side(kind, prm, faces):
        if kind == "plane":
            n = np.broadcast_to(_unit(prm["normal"]), (len(faces), 3))
        else:
            n = _surface_normals(kind, prm, tm.C[faces])
        return float(((tm.FN[faces] * n).sum(1) * tm.A[faces]).sum())

    for i, j in pairs.tolist():
        ki, pi, fi = regions[i]
        kj, pj, fj = regions[j]
        # coplanar planes stay apart: CAD imprints faces on one plane (a caliper's
        # scale marks), and joining one into the big face around them breaks them all
        if ki not in CURVED_KINDS or not _same_surface(ki, pi, kj, pj, tol, tm.diag):
            continue
        if side(ki, pi, fi) * side(ki, pi, fj) <= 0:
            continue
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups = {}
    for i in range(len(regions)):
        groups.setdefault(find(i), []).append(i)
    out = []
    n_merged = 0
    for root in sorted(groups):
        members = groups[root]
        if len(members) == 1:
            out.append(regions[root])
            continue
        faces = np.concatenate([regions[i][2] for i in members])
        big = max(members, key=lambda i: len(regions[i][2]))
        kind, prm, _ = regions[big]
        new = fit_plane(tm.verts(faces), tm.FN[faces]) if kind == "plane" else \
            refine_geometric(kind, prm, tm.verts(faces), tol / 3)
        best = None
        for cand in (new, prm):
            if _valid(kind, cand, tm.diag) and (tm.vert_residual(kind, cand, faces) <= tol).all():
                best = cand
                break
        if best is None:  # not one surface after all: keep the pieces
            out.extend(regions[i] for i in members)
            continue
        out.append((kind, best, faces))
        n_merged += len(members) - 1
    if not n_merged:
        return regions, label
    lab = np.full(len(label), -1, np.int64)
    for k, (_, _, f) in enumerate(out):
        lab[f] = k
    if log:
        log(f"{n_merged} faces joined to a neighbour on the same surface")
    return out, lab


JOIN_SMALL = 50  # exact pieces up to this many triangles may join a bigger exact neighbour
JOIN_PLANE_TORUS_EDGES = 0  # plane pieces sharing at most this many edges with a torus join it (0: off)


def _sag_bound(kind, prm, L, P=None):
    """The largest chord sag an edge of length L can have on this surface: that of a circle
    with the surface's tightest curvature radius."""
    if kind == "plane":
        return np.zeros_like(L)
    if kind in ("cylinder", "sphere"):
        r = abs(float(prm["radius"]))
    elif kind == "torus":
        R_, r_ = abs(float(prm["R"])), abs(float(prm["r"]))
        r = min(r_, R_ - r_) if R_ > r_ else r_
    elif kind == "cone":
        a = _unit(prm["axis"])
        v = P - prm["apex"]
        rho = np.linalg.norm(v - np.outer(v @ a, a), axis=1)
        r = float(rho.min()) / max(np.cos(float(prm["half_angle"])), 1e-6)
    else:
        return np.full_like(L, np.inf)
    r = max(r, 1e-300)
    h = np.minimum(L / 2, r)
    return r - np.sqrt(np.maximum(r * r - h * h, 0.0))


def _lies_on(tm: TriMesh, kind, prm, faces, tol) -> bool:
    """Do these triangles lie on this surface: every corner within tol, every edge midpoint
    no further inside than a chord of the surface can sag, facets roughly along it (a
    coarse tube's facets lean up to ~45 degrees off its normal)?"""
    if not (tm.vert_residual(kind, prm, faces) <= tol).all():
        return False
    _, c = residual(kind, prm, tm.C[faces], tm.FN[faces])
    if (c < 0.5).any():
        return False
    T = tm.V[tm.F[faces]]
    M = ((T + np.roll(T, -1, axis=1)) / 2).reshape(-1, 3)
    L = np.linalg.norm(T - np.roll(T, -1, axis=1), axis=2).reshape(-1)
    d, _ = residual(kind, prm, M, np.zeros_like(M))
    return bool((d <= _sag_bound(kind, prm, L, T.reshape(-1, 3)) + tol).all())


def _join_on_surface(tm: TriMesh, regions, label, tol: float, log=None):
    """A piece whose every corner lies on a neighbouring exact face's surface is part of
    that face. A coarse 0.8 mm tube round a cup's rim is meshed with facets leaning 40
    degrees off the surface: they fail the growth test and come out as dozens of small
    freeform (or cylinder, cone) pieces next to the torus that holds their corners exactly.
    Freeform pieces of any size and curved pieces up to JOIN_SMALL triangles (never planes) join the
    neighbour they share most border with, if they lie on its surface and face the same way."""
    regions = list(regions)
    lab = label.copy()
    acr_all = _across(tm, tm.Gall)
    joined = 0

    def side(kind, prm, faces):
        n = np.broadcast_to(_unit(prm["normal"]), (len(faces), 3)) if kind == "plane" else             _surface_normals(kind, prm, tm.C[faces])
        return float(((tm.FN[faces] * n).sum(1) * tm.A[faces]).sum())

    for _ in range(4):
        order = np.argsort(lab, kind="stable")
        bnd = np.searchsorted(lab[order], np.arange(len(regions) + 1))
        faces_of = [order[bnd[i]:bnd[i + 1]] for i in range(len(regions))]
        moved = 0
        for i in sorted(range(len(regions)), key=lambda i: len(faces_of[i])):
            k, _, f = regions[i][0], None, faces_of[i]
            if not len(f) or len(np.unique(tm.F[f])) < 4:
                continue
            if k == "plane":
                # a flat whose corners lie on a curved face is a real flat (a D-shaft's
                # flat, a chord cut): its midpoints sag exactly like a chord of that face.
                # Learned from the benchmark's true faces (.cache/r30): on a torus, a plane
                # piece sharing only a few edges with it is a facet of a coarse tube instead
                # (231 such pieces vs 19 real flats, which share 20-100 edges)
                if JOIN_PLANE_TORUS_EDGES <= 0:
                    continue
            elif k != "freeform" and len(f) > JOIN_SMALL:
                continue
            u = acr_all[f]
            u = u[u >= 0]
            nl = lab[u]
            nl = nl[nl != i]
            if not len(nl):
                continue
            labs, cnt = np.unique(nl, return_counts=True)
            shared = dict(zip(labs.tolist(), cnt.tolist()))
            for j in labs[np.argsort(-cnt, kind="stable")].tolist():
                kj, pj, _ = regions[j]
                fj = faces_of[j]
                if kj == "freeform" or not pj or len(fj) < len(f):
                    continue
                if k == "plane" and (kj != "torus" or shared[j] > JOIN_PLANE_TORUS_EDGES):
                    continue
                if not _lies_on(tm, kj, pj, f, tol):
                    continue
                if side(kj, pj, f) * side(kj, pj, fj) <= 0:
                    continue
                lab[f] = j
                faces_of[j] = np.r_[fj, f]
                faces_of[i] = f[:0]
                moved += 1
                break
        if not moved:
            break
        joined += moved
    if not joined:
        return regions, label
    order = np.argsort(lab, kind="stable")
    bnd = np.searchsorted(lab[order], np.arange(len(regions) + 1))
    regions = [(k, p, order[bnd[i]:bnd[i + 1]]) for i, (k, p, _) in enumerate(regions)]
    if log:
        log(f"{joined} pieces joined the exact face whose surface holds them")
    return _compact(regions, lab)


ROD_DECOMPOSE = True


def _circle3(P):
    """A circle in space through points P: (plane normal, centre, radius, largest miss)."""
    c0 = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c0, full_matrices=False)
    e1, e2, n = vt[0], vt[1], vt[2]
    x, y = (P - c0) @ e1, (P - c0) @ e2
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    c = c0 + cx * e1 + cy * e2
    w = P - c
    h = w @ n
    miss = np.maximum(np.abs(h), np.abs(np.linalg.norm(w - np.outer(h, n), axis=1) - r))
    return n, c, r, float(miss.max())


def _band_surface(ring_a, ring_b, tol):
    """The surface between two neighbouring rings of a rod: a cylinder if they are parallel,
    else the torus whose axis is where their planes meet (a torus's meridian circles all lie
    in planes through its axis). None if the rings fit neither."""
    (na, ca, ra), (nb, cb, rb) = ring_a, ring_b
    if abs(ra - rb) > 10 * tol:
        return None
    r = 0.5 * (ra + rb)
    cross = np.cross(na, nb)
    s = float(np.linalg.norm(cross))
    if s < 1e-4:  # parallel rings: a straight run
        d = cb - ca
        if np.linalg.norm(d - (d @ na) * na) > 10 * tol:
            return None
        return "cylinder", {"axis": _unit(na), "point": ca.copy(), "radius": r}
    a = cross / s
    M = np.vstack([na, nb, a])
    p0 = np.linalg.solve(M, np.array([na @ ca, nb @ cb, a @ ca]))
    C = p0 + ((ca - p0) @ a) * a
    R = float(np.linalg.norm(ca - C))
    if abs(float(np.linalg.norm(cb - C)) - R) > 10 * tol or abs(float((cb - C) @ a)) > 10 * tol or R <= r:
        return None
    return "torus", {"center": C, "axis": a, "R": R, "r": r}


def _rings(tm: TriMesh, faces, tol: float):
    """Rings of a CAD-meshed tube: loops of vertices on one circle all the way round. For each
    vertex, circles through it and two of its neighbours are tried; one that a full loop of
    the group's vertices lies on (within tol) is a ring. -> (ring of each vertex or -1, rings
    as (normal, centre, radius))."""
    from itertools import combinations
    vid = np.unique(tm.F[faces])
    if len(vid) > 6000:
        return None, []
    P = tm.V[vid]
    loc = {int(v): i for i, v in enumerate(vid)}
    E = np.unique(np.sort(tm.F[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    nbr = [[] for _ in vid]
    for a, b in E.tolist():
        nbr[loc[a]].append(loc[b])
        nbr[loc[b]].append(loc[a])
    ring = np.full(len(vid), -1, np.int64)
    rings = []
    span = float(np.ptp(P, axis=0).max())
    for i in range(len(vid)):
        if ring[i] >= 0:
            continue
        best = None
        for u, w in combinations(nbr[i], 2):
            A, B, C = P[u], P[i], P[w]
            ab, ac = B - A, C - A
            nv = np.cross(ab, ac)
            nn = float(np.linalg.norm(nv))
            if nn < 1e-12 * span * span:
                continue
            n = nv / nn
            # circumcentre of A, B, C
            c = A + (np.cross(nv, ab) * (ac @ ac) + np.cross(ac, nv) * (ab @ ab)) / (2 * nn * nn)
            r = float(np.linalg.norm(A - c))
            if r > span or r < 3 * tol:
                continue
            w_ = P - c
            h = w_ @ n
            on = (np.abs(h) <= tol) & (np.abs(np.linalg.norm(w_ - np.outer(h, n), axis=1) - r) <= tol) & (ring < 0)
            k = int(on.sum())
            if k < 6 or (best is not None and k <= best[0]):
                continue
            e1, e2 = _perp2(n)
            ang = np.sort(np.arctan2(w_[on] @ e2, w_[on] @ e1))
            if max(float(np.diff(ang).max()), float(ang[0] + 2 * np.pi - ang[-1])) > np.pi / 2:
                continue  # not all the way round
            best = (k, on, n, c, r)
        if best is not None:
            _, on, n, c, r = best
            n2, c2, r2, miss = _circle3(P[on])
            if miss <= tol:
                n, c, r = n2, c2, r2
            ring[on] = len(rings)
            rings.append((n, c, r))
    out = np.full(len(tm.V), -1, np.int64)
    out[vid] = ring
    return out, rings


def _rod_decompose(tm: TriMesh, regions, label, tol: float, cos_min: float, log=None):
    """A bent rod in CAD is straight cylinders joined by torus elbows. Meshed coarsely, the
    cylinder grown from a straight run takes the first rings of the elbow next to it, and
    what is left of the elbow ends up freeform or as a made-up surface. A CAD mesh of a rod
    has its vertices on rings round the tube: between two neighbouring rings there is a
    cylinder band (parallel rings) or a torus band (rings tilted about one axis). Smooth
    groups of curved and freeform faces are searched for such rings, rings joined by
    triangles give bands, bands on one surface merge, and each becomes an exact face.
    Triangles off the rings (a rounded end cap) keep their face."""
    from collections import defaultdict
    regions = list(regions)
    lab = label.copy()
    R0 = len(regions)
    soft = np.array([k in CURVED_KINDS or k == "freeform" for k, _, _ in regions], bool)
    if not soft.any():
        return regions, label
    coo = tm.G.tocoo()
    la, lb = lab[coo.row], lab[coo.col]
    m = (la != lb) & soft[la] & soft[lb]
    parent = list(range(R0))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in set(zip(la[m].tolist(), lb[m].tolist())):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for i in range(R0):
        if soft[i] and len(regions[i][2]):
            groups[find(i)].append(i)
    added = rods = 0
    for members in groups.values():
        if len(members) < 2 and regions[members[0]][0] != "freeform":
            continue  # one exact face: nothing to cut
        faces = np.concatenate([regions[i][2] for i in members])
        if len(faces) < 24:
            continue
        ring_of, rings = _rings(tm, faces, tol)
        if len(rings) < 2:
            continue
        rf = ring_of[tm.F[faces]]
        okr = (rf >= 0).all(axis=1)
        a_ = np.where(okr, rf.min(axis=1), -1)
        b_ = np.where(okr, rf.max(axis=1), -1)
        band = okr & (a_ != b_)
        # judged over the faces with any band triangles: a rounded end cap lies off the
        # rings and keeps its face, and two caps can outweigh a thin rod's few slivers
        owner = lab[faces]
        has_band = np.isin(owner, np.unique(owner[band]))
        if band.sum() < 0.5 * has_band.sum():
            continue
        # bands: pairs of rings joined by triangles, each with one surface of its own (a band
        # that fits neither, between the smaller rings of a rounded end cap, keeps its face)
        surf = {}
        for pa, pb in set(zip(a_[band].tolist(), b_[band].tolist())):
            fb = faces[band & (a_ == pa) & (b_ == pb)]
            s_ = _band_surface(rings[pa], rings[pb], tol)
            if s_ is None or not (tm.vert_residual(s_[0], s_[1], fb) <= tol).all() \
                    or not tm.normal_ok(s_[0], s_[1], fb, cos_min).all():
                continue
            surf[(pa, pb)] = s_ + (fb,)
        if not surf:
            continue
        # bands sharing a ring and lying on one surface are one face
        keys = list(surf)
        par = {k: k for k in keys}

        def fk(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for i_, k1 in enumerate(keys):
            for k2 in keys[i_ + 1:]:
                if set(k1) & set(k2) and _same_surface(surf[k1][0], surf[k1][1], surf[k2][0], surf[k2][1], tol, tm.diag):
                    r1, r2 = fk(k1), fk(k2)
                    if r1 != r2:
                        par[r2] = r1
        runs = defaultdict(list)
        for k in keys:
            runs[fk(k)].append(k)
        new = []
        for root, ks in runs.items():
            kind, prm = surf[root][0], surf[root][1]
            fb = np.concatenate([surf[k][2] for k in ks])
            q = refine_geometric(kind, prm, tm.verts(fb), tol / 3)
            if _valid(kind, q, tm.diag) and (tm.vert_residual(kind, q, fb) <= tol).all():
                prm = q
            new.append((kind, prm, fb))
        old_ids = np.unique(lab[np.concatenate([fb for _, _, fb in new])])
        # a coil spring gives a slightly different torus per band, so its bands never merge:
        # not a rod. A rod's elbow bands merge into one torus each.
        if len(new) > 0.6 * len(surf) and len(new) > len(old_ids):
            continue
        # each surface on its own: worth taking if its triangles now sit in a freeform face,
        # in several faces or in a face of another type (a bead torus inside a cup group
        # counts even though the cup walls make most of the group); already one face of this
        # type: nothing gained
        keep = []
        for kind, prm, fb in new:
            olds = np.unique(lab[fb])
            if len(olds) == 1 and regions[olds[0]][0] == kind:
                continue
            keep.append((kind, prm, fb))
        # an exact face is never cut in part: a surface that takes some of an exact face's
        # triangles is dropped unless the kept surfaces take all of it (the fragments left
        # over were what broke faces on real parts); freeform faces may give part of theirs
        while keep:
            taken = np.zeros(len(tm.F), bool)
            for _, _, fb in keep:
                taken[fb] = True
            whole = {}
            for rid in np.unique(lab[taken]).tolist():
                k0, _, f0 = regions[rid]
                whole[rid] = k0 == "freeform" or bool(taken[f0].all())
            ok_ = [s_ for s_ in keep if all(whole[r_] for r_ in np.unique(lab[s_[2]]).tolist())]
            if len(ok_) == len(keep):
                break
            keep = ok_
        if not keep:
            continue
        for kind, prm, fb in keep:
            lab[fb] = len(regions)
            regions.append((kind, prm, fb))
            added += 1
        rods += 1
    if not added:
        return regions, label
    order = np.argsort(lab, kind="stable")
    bnd = np.searchsorted(lab[order], np.arange(len(regions) + 1))
    regions = [(k, p, order[bnd[i]:bnd[i + 1]]) for i, (k, p, _) in enumerate(regions)]
    if log:
        log(f"{rods} rods cut into straight cylinders and torus elbows ({added} faces)")
    return _compact(regions, lab)


FILLET_RAILS = True


RAIL_SPAN_DEG = 50.0  # most a fillet facet may turn round the axis
RAIL_FINE_DEG = 10.0  # a mesh whose cylinders turn less than this per facet is fine: no search


def _span_deg(tm: TriMesh, prm, faces) -> np.ndarray:
    """Per triangle: how far it turns round the cylinder's axis (degrees)."""
    a = _unit(prm["axis"])
    e1, e2 = _perp2(a)
    T = tm.V[tm.F[faces]] - prm["point"]
    th = np.sort(np.arctan2(T @ e2, T @ e1), axis=1)
    gaps = np.c_[th[:, 1] - th[:, 0], th[:, 2] - th[:, 1], th[:, 0] + 2 * np.pi - th[:, 2]]
    return np.degrees(2 * np.pi - gaps.max(axis=1))


def _rail_span_limit(tm: TriMesh, regions) -> float:
    """How far a fillet facet may turn round its axis in this mesh: 1.5 x what the cylinders
    found so far show (95th percentile), at most RAIL_SPAN_DEG. A fine mesh's cylinders turn
    a few degrees per facet, so an octagonal pin (flat faces an eighth of a turn each) is no
    fillet there; a coarse mesh's turn 30 or 40."""
    sp = [_span_deg(tm, p, f) for k, p, f in regions if k == "cylinder" and len(f) >= 4]
    if not sp:
        return RAIL_SPAN_DEG
    return float(min(RAIL_SPAN_DEG, max(10.0, 1.5 * np.percentile(np.concatenate(sp), 95))))


def _span_ok(tm: TriMesh, prm, faces, limit: float) -> np.ndarray:
    """Per triangle: does it turn at most `limit` degrees round the cylinder's axis? A fillet's
    facets each cover a small part of the arc; the faces of a coarse box lie on the cylinder
    through three of its edges too (any three corners of a rectangle are on one circle),
    but each covers a quarter turn of it."""
    return _span_deg(tm, prm, faces) <= limit


RAIL_TANGENT_DEG = 3.0  # a fillet's flat neighbour is tangent to it along the shared rail


def _rail_borders(tm: TriMesh, prm, piece) -> tuple[int, int]:
    """A fillet runs smoothly into the faces beside it: across each long border edge (along
    the axis) the neighbour facet is tangent to the cylinder there, while the strip's own
    facet, a chord, is turned from it. A run of flat facets that merely lies on a circle (the
    sides of an octagon, a chamfer) has neighbours that are further chords or a sharp edge,
    and a strip cut out of a flat face has a neighbour in the same plane as its own facet.
    -> (tangent borders, borders splitting a flat face)"""
    a = _unit(prm["axis"])
    P = np.asarray(prm["point"], float)
    inside = np.zeros(len(tm.F), bool)
    inside[piece] = True
    coo = tm.Gall[piece].tocoo()
    t, nb = piece[coo.row], coo.col
    m = ~inside[nb]
    t, nb = t[m], nb[m]
    if not len(t):
        return 0, 0
    Ft, Fn = tm.F[t], tm.F[nb]
    sh = (Ft[:, :, None] == Fn[:, None, :]).any(axis=2)  # which of t's corners are shared
    m = sh.sum(axis=1) == 2
    t, nb, Ft, sh = t[m], nb[m], Ft[m], sh[m]
    if not len(t):
        return 0, 0
    ij = np.argsort(~sh, axis=1, kind="stable")[:, :2]
    A_, B_ = tm.V[Ft[np.arange(len(t)), ij[:, 0]]], tm.V[Ft[np.arange(len(t)), ij[:, 1]]]
    E = B_ - A_
    along = np.abs((E / np.linalg.norm(E, axis=1, keepdims=True)) @ a) > 0.99
    t, nb, A_, B_ = t[along], nb[along], A_[along], B_[along]
    if not len(t):
        return 0, 0
    M = 0.5 * (A_ + B_) - P
    R = M - np.outer(M @ a, a)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    cos_flat = np.cos(np.radians(0.3))
    coplanar = np.abs(np.einsum("ij,ij->i", tm.FN[t], tm.FN[nb])) >= cos_flat
    tangent = (np.abs(np.einsum("ij,ij->i", tm.FN[nb], R)) >= np.cos(np.radians(RAIL_TANGENT_DEG))) & ~coplanar
    return int(tangent.sum()), int(coplanar.sum())


def _rail_piece_ok(tm: TriMesh, prm, piece) -> bool:
    tan, flat = _rail_borders(tm, prm, piece)
    return tan >= 1 and flat == 0


def _fillet_rails(tm: TriMesh, free: np.ndarray, tol: float, cos_min: float, regions=()):
    """Coarse fillets the seeds missed. A fillet meshed with a few facets across is long thin
    strips, and a seed's neighbourhood on one mixes fillet, flat and corner triangles, so
    the usual guesses miss it. Its cylinder is found from the strip itself, two ways: its
    long edges along the fillet are parallel lines, and the cross product of two
    neighbouring facets' normals is the axis however it is triangulated. Seen along the
    axis the strip's vertices lie on one circle: circle and axis are the cylinder, exactly.
    -> [(cylinder params, triangles)]"""
    # A mesher uses one angle for every curved face, so in a finely meshed part the fillets
    # are fine too and the seeds found them; what is left there is bands of tori and cones,
    # which lie on some cylinder as well (on the benchmark's real parts strips found there
    # broke faces). Only a coarse mesh - its cylinders turning RAIL_FINE_DEG or more per
    # facet - gets the search.
    sp = [_span_deg(tm, p, f) for k, p, f in regions if k == "cylinder" and len(f) >= 4]
    if len(sp) >= 3 and float(np.median(np.concatenate(sp))) < RAIL_FINE_DEG:
        return []
    free = free.copy()
    span_lim = _rail_span_limit(tm, regions)
    lab = np.full(len(tm.F), -1, np.int64)
    for i, (_, _, f) in enumerate(regions):
        lab[f] = i
    ctx = (lab, regions)
    out = _fillet_rails_edges(tm, free, tol, cos_min, span_lim, ctx)
    for _, faces in out:
        free[faces] = False
    return out + _fillet_rails_normals(tm, free, tol, cos_min, span_lim, ctx)




def _rail_flood(tm: TriMesh, prm, seeds: np.ndarray, allowed: np.ndarray, tol: float, cos_min: float,
                span_lim: float) -> np.ndarray:
    """Triangles reached from `seeds` across any edge, through allowed triangles that hold on
    the cylinder (corners within tol, normals the same way, each turning at most span_lim
    round the axis). Grown from the strip itself, so the cost follows the strip's size, not
    the mesh's."""
    def ok(ts):
        ts = ts[allowed[ts]]
        if not len(ts):
            return ts
        m = (tm.vert_residual("cylinder", prm, ts) <= tol) & tm.normal_ok("cylinder", prm, ts, cos_min)
        ts = ts[m]
        return ts[_span_ok(tm, prm, ts, span_lim)] if len(ts) else ts

    seen = set(np.unique(seeds).tolist())
    front = ok(np.unique(seeds))
    got = [front]
    while len(front):
        nb = np.unique(tm.Gall[front].indices)
        nb = np.array([t for t in nb.tolist() if t not in seen], np.int64)
        if not len(nb):
            break
        seen.update(nb.tolist())
        front = ok(nb)
        got.append(front)
    return np.unique(np.concatenate(got)) if got else np.zeros(0, np.int64)


def _rail_edges(tm: TriMesh, free: np.ndarray, hi_deg: float):
    """Candidate rails: every edge of an unassigned triangle that meets a neighbour (any
    neighbour) at a crease, normals 0.3 to hi_deg degrees apart, or is flat-joined to one
    that does. A coarse fillet's facets all touch such creases; a flat face's inner
    triangles touch none and its sharp borders are left out, so a big model's planes cost
    next to nothing here. -> (vertex pairs, the creased
    unassigned triangles on each edge as an (m, 2) array, -1 for none)"""
    nV = int(len(tm.V)) + 1
    E = np.sort(tm.F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1).astype(np.int64)
    T = np.repeat(np.arange(len(tm.F)), 3)
    key = E[:, 0] * nV + E[:, 1]
    o = np.argsort(key, kind="stable")
    key, E, T = key[o], E[o], T[o]
    pair = np.flatnonzero(key[1:] == key[:-1])
    t1, t2 = T[pair], T[pair + 1]
    c = np.clip(np.einsum("ij,ij->i", tm.FN[t1], tm.FN[t2]), -1.0, 1.0)
    ang = np.degrees(np.arccos(c))
    cr = (ang > 0.3) & (ang <= hi_deg)
    creased = np.zeros(len(tm.F), bool)
    creased[t1[cr]] = True
    creased[t2[cr]] = True
    # and their flat neighbours: a gently curved strip has long edges inside flat pairs of
    # triangles too
    fl = ang <= 0.3
    grow = creased.copy()
    grow[t1[fl & creased[t2]]] = True
    grow[t2[fl & creased[t1]]] = True
    creased = grow & free
    m = creased[T]
    if not m.any():
        return np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64)
    key, E, T = key[m], E[m], T[m]
    first = np.r_[True, key[1:] != key[:-1]]
    grp = np.cumsum(first) - 1
    TT = np.full((int(first.sum()), 2), -1, np.int64)
    TT[grp[first], 0] = T[first]
    second = ~first
    TT[grp[second], 1] = T[second]
    return E[first], TT


def _circles_through(Q: np.ndarray, tol: float, rmin: float, rmax: float, k: int = 6):
    """Circles through three near points of Q (2D) that at least three of the points lie on.
    -> [(points on it, centre, radius)], best supported first."""
    from itertools import combinations

    from scipy.spatial import cKDTree
    out = []
    if len(Q) < 3:
        return out
    kt = cKDTree(Q)
    tried = set()
    for a in range(len(Q)):
        near = kt.query(Q[a], k=min(k, len(Q)))[1]
        for b, c in combinations([int(x) for x in np.atleast_1d(near) if x != a], 2):
            key = tuple(sorted((a, b, c)))
            if key in tried:
                continue
            tried.add(key)
            A2, B2, C2 = Q[a], Q[b], Q[c]
            den = 2 * ((B2[0] - A2[0]) * (C2[1] - A2[1]) - (B2[1] - A2[1]) * (C2[0] - A2[0]))
            if abs(den) < 1e-18:
                continue
            b2, c2 = B2 - A2, C2 - A2
            ux = ((C2[1] - A2[1]) * (b2 @ b2) - (B2[1] - A2[1]) * (c2 @ c2)) / den
            uy = ((B2[0] - A2[0]) * (c2 @ c2) - (C2[0] - A2[0]) * (b2 @ b2)) / den
            ctr = A2 + np.array([ux, uy])
            r = float(np.hypot(ux, uy))
            if r < rmin or r > rmax:
                continue
            on = np.abs(np.linalg.norm(Q - ctr, axis=1) - r) <= tol
            if on.sum() >= 3:
                out.append((on, ctr, r))
    out.sort(key=lambda c: -int(c[0].sum()))
    return out


def _rail_accept(tm: TriMesh, prm, piece, e1, e2, tol: float, ctx=None):
    """A strip found on a candidate cylinder: kept if its vertices sit at three or more places
    round the circle, it is not flat, it runs smoothly into its neighbours, and no curved
    face beside it already holds it (then it is the rest of that face, left for the joining
    steps: a band of a finely meshed torus or cone lies on some cylinder too). -> refitted
    params or None."""
    if len(piece) < 2:
        return None
    W = tm.verts(piece)
    if len(np.unique(np.round(np.c_[W @ e1, W @ e2] / (2 * tol)).astype(np.int64), axis=0)) < 3:
        return None
    if _planar(tm, piece, tol):
        return None
    q = refine_geometric("cylinder", prm, W, tol / 3)
    p_ = q if _valid("cylinder", q, tm.diag) and (tm.vert_residual("cylinder", q, piece) <= tol).all() else prm
    if not _rail_piece_ok(tm, p_, piece):
        return None
    if ctx is not None:
        lab, regions = ctx
        nb = np.unique(lab[tm.Gall[piece].indices])
        for q in nb[nb >= 0].tolist():
            kq, pq, _ = regions[q]
            if kq in CURVED_KINDS and pq and float((tm.vert_residual(kq, pq, piece) <= tol).mean()) >= 0.9:
                return None
    return p_


def _fillet_rails_edges(tm: TriMesh, free: np.ndarray, tol: float, cos_min: float, span_lim: float = 50.0,
                        ctx=None):
    """Fillet cylinders from parallel long edges ("rails") of the creased unassigned triangles:
    seen along their direction, three or more rails on one circle give the cylinder."""
    from scipy.spatial import cKDTree
    E, TT = _rail_edges(tm, free, span_lim + 1.0)
    if len(E) < 3:
        return []
    P0, P1 = tm.V[E[:, 0]], tm.V[E[:, 1]]
    L = np.linalg.norm(P1 - P0, axis=1)
    keep = L > 20 * tol
    E, TT, P0, P1, L = E[keep], TT[keep], P0[keep], P1[keep], L[keep]
    if len(E) < 3:
        return []
    D = (P1 - P0) / L[:, None]
    big = np.argmax(np.abs(D), axis=1)
    D = D * np.sign(D[np.arange(len(D)), big])[:, None]  # one sign per direction
    tree = cKDTree(D)
    seen = np.zeros(len(D), bool)
    out = []
    allowed = free.copy()
    for i in np.argsort(-L):
        if seen[i]:
            continue
        grp = np.array(tree.query_ball_point(D[i], 2e-4), int)
        seen[grp] = True
        if len(grp) < 3:
            continue
        d = _unit(D[grp].mean(axis=0))
        e1, e2 = _perp2(d)
        M = 0.5 * (P0[grp] + P1[grp])
        Q = np.c_[M @ e1, M @ e2]  # each rail as a point seen along d
        # the same line may appear as several edges: one point per line
        cell = np.round(Q / (2 * tol)).astype(np.int64)
        _, first, inv = np.unique(cell, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).ravel()
        Qu = Q[first]
        for on, ctr, r in _circles_through(Qu, tol, 3 * tol, 0.25 * tm.diag):
            prm = {"axis": d, "point": ctr[0] * e1 + ctr[1] * e2, "radius": r}
            seeds = TT[grp[on[inv]]].ravel()
            seeds = seeds[seeds >= 0]
            seeds = seeds[allowed[seeds]]
            if len(seeds) < 2:
                continue
            hold = _rail_flood(tm, prm, seeds, allowed, tol, cos_min, span_lim)
            if len(hold) < 2:
                continue
            for piece in _pieces(tm, hold):
                p_ = _rail_accept(tm, prm, piece, e1, e2, tol, ctx)
                if p_ is None:
                    continue
                allowed[piece] = False
                out.append((p_, piece))
    return out


def _fillet_rails_normals(tm: TriMesh, free: np.ndarray, tol: float, cos_min: float, span_lim: float = 50.0,
                          ctx=None):
    """Coarse fillets: a fillet meshed with a few facets across is long thin strips. A seed's
    neighbourhood on such a strip mixes fillet, flat and corner triangles, so the usual
    guesses miss it. But two neighbouring facets of one cylinder both have normals square
    to its axis, so the cross product of their normals is the axis direction, however the
    strip is triangulated; seen along that direction the strip's vertices lie on one circle,
    and the circle and the direction are the cylinder, exactly. Axis candidates come from
    pairs of unassigned neighbouring facets meeting at a gentle angle; near-parallel ones
    are grouped, circles through three of their vertices seen end-on are tried. ->
    [(cylinder params, triangles)] for strips that hold on such a cylinder with vertices
    at three or more places round it."""
    from scipy.spatial import cKDTree
    if free.sum() < 4:
        return []
    coo = tm.G.tocoo()
    a_, b_ = coo.row, coo.col
    m = free[a_] & free[b_] & (a_ < b_)
    a_, b_ = a_[m], b_[m]
    X = np.cross(tm.FN[a_], tm.FN[b_])
    s = np.linalg.norm(X, axis=1)
    m = (s > np.sin(np.radians(1.0))) & (s < np.sin(np.radians(45.0)))
    a_, b_, X, s = a_[m], b_[m], X[m], s[m]
    if len(X) < 2:
        return []
    D = X / s[:, None]
    big = np.argmax(np.abs(D), axis=1)
    D = D * np.sign(D[np.arange(len(D)), big])[:, None]  # one sign per direction
    tree = cKDTree(D)
    seen = np.zeros(len(D), bool)
    out = []
    allowed = free.copy()
    for i in range(len(D)):
        if seen[i]:
            continue
        grp = np.array(tree.query_ball_point(D[i], 5e-3), int)
        grp = grp[~seen[grp]]
        seen[grp] = True
        if len(grp) < 2:
            continue
        d = _unit(D[grp].mean(axis=0))
        e1, e2 = _perp2(d)
        tris = np.unique(np.r_[a_[grp], b_[grp]])
        tris = tris[allowed[tris]]
        if len(tris) < 2:
            continue
        W = tm.verts(tris)
        Q = np.c_[W @ e1, W @ e2]
        _, first = np.unique(np.round(Q / (2 * tol)).astype(np.int64), axis=0, return_index=True)
        Q = Q[np.sort(first)]
        # the best-supported few, in turn: a chance circle through vertices of other
        # features finds no strip and the next one is tried
        for _, ctr, rr in _circles_through(Q, tol, 3 * tol, 0.25 * tm.diag)[:5]:
            prm = {"axis": d, "point": ctr[0] * e1 + ctr[1] * e2, "radius": rr}
            hold = _rail_flood(tm, prm, tris[allowed[tris]], allowed, tol, cos_min, span_lim)
            got = False
            for piece in _pieces(tm, hold):
                p_ = _rail_accept(tm, prm, piece, e1, e2, tol, ctx)
                if p_ is None:
                    continue
                allowed[piece] = False
                out.append((p_, piece))
                got = True
            if got:
                break
    return out


JOIN_BENDS = True


def _bend_guess(tm: TriMesh, pi, fi, pj, fj, tol: float):
    """The torus two neighbouring cylinder pieces of one bend lie on. A short band of a bend
    fits a cylinder whose axis is nearly the bend's centre line there, so the circle round
    the middle of each piece is nearly one of the torus's meridian circles, and two meridian
    circles give the torus (their planes meet in its axis). -> torus params or None."""
    rings = []
    for p, f in ((pi, fi), (pj, fj)):
        a = _unit(p["axis"])
        q = np.asarray(p["point"], float)
        t = (tm.verts(f) - q) @ a
        rings.append((a, q + 0.5 * (t.min() + t.max()) * a, float(p["radius"])))
    g = _band_surface(rings[0], rings[1], 3 * tol)
    return g[1] if g is not None and g[0] == "torus" else None


def _join_bends(tm: TriMesh, regions, label, tol: float, cos_min: float, log=None):
    """A bent tube or wire comes out as a row of short cylinder pieces round each bend: a
    short band of a torus fits a cylinder to the micron, so the bend is cut wherever the
    growing cylinder stopped holding it. CAD makes each bend one torus face (the
    benchmark's true faces: 217 such joins on real parts, 575 on synthetic rods, and only
    a handful of cylinder pairs that one surface fits are two true faces). Neighbouring
    cylinder (and torus) pieces of one radius that one torus holds - every corner within
    tolerance, normals the same way, edge midpoints no deeper than the mesh's own chord
    sag - become one torus face."""
    regions = list(regions)
    R = len(regions)
    cur = np.zeros(R, bool)
    for i, (k, p, f) in enumerate(regions):
        cur[i] = k in ("cylinder", "torus") and bool(p) and len(f) >= 3
    if cur.sum() < 2:
        return regions, label
    coo = tm.G.tocoo()
    la, lb = label[coo.row], label[coo.col]
    m = (la < lb) & (la >= 0)
    nbrs = [set() for _ in range(R)]
    for a, b in np.unique(np.c_[la[m], lb[m]], axis=0):
        if cur[a] and cur[b]:
            nbrs[a].add(int(b))
            nbrs[b].add(int(a))
    if not any(nbrs):
        return regions, label
    sag_max = _sag_limit(tm, regions, tol)

    def holds(prm, f):
        return bool((tm.vert_residual("torus", prm, f) <= tol).all() and tm.normal_ok("torus", prm, f, cos_min).all())

    def radius(i):
        p = regions[i][1]
        return float(p["radius"] if regions[i][0] == "cylinder" else p["r"])

    def refit(prm, members):
        f = np.concatenate([regions[x][2] for x in members])
        try:
            q = _refine("torus", prm, tm.verts(f), tol)
        except Exception:
            return prm
        return q if _valid("torus", q, tm.diag) and holds(q, f) else prm

    def grow(seed_members, prm, done):
        members = list(seed_members)
        inside = set(members)
        tried = set()
        while True:
            wave = []
            for x in list(members):
                for j in nbrs[x]:
                    if j in inside or done[j]:
                        continue
                    fj = regions[j][2]
                    # a piece of a few triangles has no reliable radius of its own: only
                    # the surface test counts for it
                    if len(fj) >= 50 and abs(radius(j) - prm["r"]) > 3 * tol:
                        continue
                    if holds(prm, fj):
                        wave.append(j)
                        inside.add(j)
                    elif j not in tried and float(tm.vert_residual("torus", prm, fj).max()) <= 3 * tol:
                        # nearly held: the torus fitted on all of them may hold it
                        tried.add(j)
                        q = refit(prm, members + [j])
                        if q is not prm and holds(q, fj):
                            prm = q
                            wave.append(j)
                            inside.add(j)
            if not wave:
                return members, prm
            members += wave
            prm = refit(prm, members)

    # every seed's best group first, then the biggest groups win: grown greedily, one band
    # of a bend paired with the short straight run beside it (two short pieces always lie
    # on some torus) would take that band before the bend's own group is seen
    none_done = np.zeros(R, bool)
    cands = []
    covered = set()  # pieces already in a candidate: a start inside one gives that group again
    for i in np.flatnonzero(cur).tolist():
        if not nbrs[i]:
            continue
        ki, pi, fi = regions[i]
        starts = []
        if ki == "torus":
            if i not in covered:
                starts.append(([i], dict(pi)))
        else:
            for j in nbrs[i]:
                if (j < i and regions[j][0] == "cylinder") or (i in covered and j in covered):
                    continue  # tried from j, or inside a group found already
                if regions[j][0] != "cylinder" or abs(radius(j) - radius(i)) > 3 * tol:
                    continue
                g = _bend_guess(tm, pi, fi, regions[j][1], regions[j][2], tol)
                if g is None or not _valid("torus", g, tm.diag):
                    continue
                g = refit(g, [i, j])
                if holds(g, np.r_[fi, regions[j][2]]):
                    starts.append(([i, j], g))
        for mem, g in starts:
            members, prm = grow(mem, g, none_done)
            if len(members) >= 2:
                cands.append((int(sum(len(regions[x][2]) for x in members)), members, prm))
                covered.update(members)
    cands.sort(key=lambda c: (-c[0], min(c[1])))
    faces_of = [np.asarray(r[2]) for r in regions]
    lab = label.copy()
    done = np.zeros(R, bool)
    groups = []
    for _, members, prm in cands:
        if done[members].any():
            continue
        f = np.concatenate([faces_of[x] for x in members])
        kind = "torus"
        # two pieces of one straight run also lie on a torus a hundred times the tube's
        # size (any slight tilt between their fitted axes makes one): the simplest
        # surface that holds them all wins
        big = max(members, key=lambda x: len(faces_of[x]))
        if regions[big][0] == "cylinder":
            try:
                qc = _refine("cylinder", regions[big][1], tm.verts(f), tol)
                if _valid("cylinder", qc, tm.diag) and _holds(tm, "cylinder", qc, f, tol, sag_max, cos_min):
                    kind, prm = "cylinder", qc
            except Exception:
                pass
        if kind == "torus" and not _holds(tm, "torus", prm, f, tol, sag_max, cos_min):
            continue
        if kind == "torus" and not prm["R"] > prm["r"]:
            continue  # a bend or a fillet round an edge is a ring torus; a dome is not a bend
        members = list(members)
        if kind == "torus":
            # the end band of a bend often took a triangle or two of the straight run
            # beside it, so the torus does not hold it whole: its triangles on the torus
            # join the bend, the others the neighbour whose surface they lie on
            for x in list(members):
                for j in sorted(nbrs[x]):
                    if done[j] or j in members:
                        continue
                    fj = faces_of[j]
                    if not len(fj):
                        continue
                    ok = (tm.vert_residual("torus", prm, fj) <= tol) & tm.normal_ok("torus", prm, fj, cos_min)
                    if ok.mean() < 0.8:
                        continue
                    rest = fj[~ok]
                    dest = None
                    if len(rest):
                        nb = tm.Gall[rest].indices
                        for q in np.unique(lab[nb]).tolist():
                            if q < 0 or q == j or q in members or done[q]:
                                continue
                            kq, pq, _ = regions[q]
                            if kq == "freeform" or not pq:
                                continue
                            if (tm.vert_residual(kq, pq, rest) <= tol).all() and tm.normal_ok(kq, pq, rest, cos_min).all():
                                dest = q
                                break
                        if dest is None:
                            continue
                        faces_of[dest] = np.r_[faces_of[dest], rest]
                        lab[rest] = dest
                    faces_of[j] = fj[ok]
                    members.append(j)
            f = np.concatenate([faces_of[x] for x in members])
        done[members] = True
        lab[f] = members[0]
        groups.append((members, kind, prm))
    if not groups:
        return regions, label
    for i in range(R):
        regions[i] = (regions[i][0], regions[i][1], faces_of[i])
    for members, kind, prm in groups:
        keep = members[0]
        f = np.concatenate([faces_of[x] for x in members])
        regions[keep] = (kind, prm, f)
        for x in members[1:]:
            regions[x] = (regions[x][0], regions[x][1], faces_of[x][:0])
    if log:
        nt = sum(1 for _, k, _ in groups if k == "torus")
        log(f"{sum(len(m) for m, _, _ in groups)} pieces joined: {nt} bends into one torus each, "
            f"{len(groups) - nt} straight runs into one cylinder each")
    return _compact(regions, lab)
