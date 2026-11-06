"""Identify surface regions in an oriented point cloud.

Two passes:

1. **Smooth patches**: connect neighbouring points whose normals differ by less
   than ``angle_deg``. Connected components split the cloud at sharp edges.
2. **Primitive extraction** inside each patch (RANSAC with normals, in the
   spirit of Schnabel et al. 2007): repeatedly find the plane / cylinder / cone /
   sphere that explains the largest connected set of points, refine it with least
   squares and peel it off. Whatever is left is a *freeform* region.

The primitive found here only seeds a coordinate frame; the actual surface
equation is discovered afterwards by symbolic regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

PENALTY = {"plane": 1.0, "cylinder": 0.98, "sphere": 0.97, "cone": 0.94, "torus": 0.9}
ALL_KINDS = ("plane", "cylinder", "sphere", "cone", "torus")


@dataclass
class SegParams:
    angle_deg: float = 30.0      # sharp-edge threshold between neighbouring normals
    tol: float = 0.0             # inlier distance, fraction of bbox diagonal (0 = estimate from the data)
    sag: float = 0.0             # mesh facet sag in model units (from the triangles, 0 = unknown)
    normal_deg: float = 20.0     # inlier normal deviation
    min_frac: float = 0.0015     # smallest region, fraction of all points
    quality: float = 0.2         # max 60th-percentile inlier residual, fraction of the tolerance
    kinds: tuple = ALL_KINDS
    knn: int = 10
    seed: int = 0


@dataclass
class Region:
    id: int
    idx: np.ndarray
    prim: str
    params: dict = field(default_factory=dict)
    rms: float = float("nan")
    exact: bool = False  # taken exactly from the mesh (coarse-to-fine planes): no search needed


# ------------------------------------------------------------------ helpers

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


def _perp(a: np.ndarray) -> np.ndarray:
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    return _unit(np.cross(a, t))


def residual(kind: str, prm: dict, P: np.ndarray, N: np.ndarray):
    """Distance to the primitive and |cos| between point normal and surface normal."""
    if kind == "plane":
        n = prm["normal"]
        return np.abs((P - prm["point"]) @ n), np.abs(N @ n)
    if kind == "sphere":
        v = P - prm["center"]
        r = np.linalg.norm(v, axis=1)
        return np.abs(r - prm["radius"]), np.abs(np.einsum("ij,ij->i", N, v) / np.maximum(r, 1e-300))
    if kind == "cylinder":
        a = prm["axis"]
        v = P - prm["point"]
        rad = v - np.outer(v @ a, a)
        rho = np.linalg.norm(rad, axis=1)
        return np.abs(rho - prm["radius"]), np.abs(np.einsum("ij,ij->i", N, rad) / np.maximum(rho, 1e-300))
    if kind == "cone":
        a, t = prm["axis"], prm["half_angle"]
        v = P - prm["apex"]
        h = v @ a
        rad = v - np.outer(h, a)
        rho = np.maximum(np.linalg.norm(rad, axis=1), 1e-300)
        d = np.abs(rho * np.cos(t) - h * np.sin(t))
        d = np.where(h > 0, d, np.inf)
        nexp = rad / rho[:, None] * np.cos(t) - np.outer(np.full(len(P), np.sin(t)), a)
        return d, np.abs(np.einsum("ij,ij->i", N, nexp))
    if kind == "torus":
        a, R, r = prm["axis"], prm["R"], prm["r"]
        v = P - prm["center"]
        h = v @ a
        rad = v - np.outer(h, a)
        rho = np.maximum(np.linalg.norm(rad, axis=1), 1e-300)
        dr = rho - R
        core = np.maximum(np.hypot(dr, h), 1e-300)
        nexp = (rad / rho[:, None] * dr[:, None] + np.outer(h, a)) / core[:, None]
        return np.abs(core - r), np.abs(np.einsum("ij,ij->i", N, nexp))
    raise ValueError(kind)


# --------------------------------------------------- minimal-sample builders

def plane_from(p, n):
    return {"point": p.copy(), "normal": _unit(n)}


def _closest_between_lines(p1, d1, p2, d2):
    w0 = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if den < 1e-9:
        return None
    t = (b * e - c * d) / den
    s = (a * e - b * d) / den
    return p1 + t * d1, p2 + s * d2


def sphere_from(p1, n1, p2, n2):
    r = _closest_between_lines(p1, n1, p2, n2)
    if r is None:
        return None
    c1, c2 = r
    c = 0.5 * (c1 + c2)
    r1, r2 = np.linalg.norm(p1 - c), np.linalg.norm(p2 - c)
    if np.linalg.norm(c1 - c2) > 0.05 * (r1 + r2) or abs(r1 - r2) > 0.05 * (r1 + r2):
        return None
    return {"center": c, "radius": 0.5 * (r1 + r2)}


def cylinder_from(p1, n1, p2, n2):
    a = np.cross(n1, n2)
    if np.linalg.norm(a) < 0.05:
        return None
    a = _unit(a)
    q1, q2 = p1 - (p1 @ a) * a, p2 - (p2 @ a) * a
    m1, m2 = n1 - (n1 @ a) * a, n2 - (n2 @ a) * a
    r = _closest_between_lines(q1, m1, q2, m2)
    if r is None:
        return None
    c = 0.5 * (r[0] + r[1])
    r1, r2 = np.linalg.norm(q1 - c), np.linalg.norm(q2 - c)
    if abs(r1 - r2) > 0.05 * (r1 + r2):
        return None
    return {"point": c, "axis": a, "radius": 0.5 * (r1 + r2)}


def cone_from(ps, ns):
    A = np.asarray(ns)
    if abs(np.linalg.det(A)) < 1e-3:
        return None
    apex = np.linalg.solve(A, np.einsum("ij,ij->i", A, np.asarray(ps)))
    u = _unit(np.asarray(ps) - apex)
    axis = np.cross(u[1] - u[0], u[2] - u[0])
    if np.linalg.norm(axis) < 1e-6:
        return None
    axis = _unit(axis)
    if axis @ u[0] < 0:
        axis = -axis
    cos_t = float(np.clip(np.mean(u @ axis), -1, 1))
    t = np.arccos(cos_t)
    if not (np.radians(4) < t < np.radians(86)):
        return None
    return {"apex": apex, "axis": axis, "half_angle": float(t)}


# --------------------------------------------------------- least squares fits

def fit_plane(P, N=None):
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    if N is not None and (N @ n).mean() < 0:
        n = -n
    return {"point": c, "normal": n}


def fit_sphere(P, N=None):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P * P).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r2 = sol[3] + c @ c
    if r2 <= 0:
        return None
    return {"center": c, "radius": float(np.sqrt(r2))}


def _fit_circle_2d(Q):
    A = np.c_[2 * Q, np.ones(len(Q))]
    b = (Q * Q).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r2 = sol[2] + c @ c
    return c, float(np.sqrt(max(r2, 0.0)))


def fit_cylinder(P, N, prev=None):
    w, V = np.linalg.eigh(N.T @ N / len(N))
    a = V[:, 0]
    if prev is not None and w[1] < 0.02:  # normals span too little arc to trust
        a = prev["axis"]
    e1 = _perp(a)
    e2 = np.cross(a, e1)
    Q = np.c_[P @ e1, P @ e2]
    c2, r = _fit_circle_2d(Q)
    point = c2[0] * e1 + c2[1] * e2
    return {"point": point, "axis": a, "radius": r}


def fit_cone(P, N, prev=None):
    M = np.einsum("ij,ik->jk", N, N)
    b = np.einsum("ij,ik,ik->j", N, N, P)
    try:
        apex = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return prev
    u = _unit(P - apex)
    uc = u - u.mean(0)
    _, V = np.linalg.eigh(uc.T @ uc)
    axis = V[:, 0]
    if axis @ u.mean(0) < 0:
        axis = -axis
    t = float(np.arccos(np.clip(np.mean(u @ axis), -1, 1)))
    if not (np.radians(2) < t < np.radians(88)):
        return prev
    return {"apex": apex, "axis": axis, "half_angle": t}


def _meridian(P, N, a, q):
    """Coordinates in the half-plane through the axis: (rho, z) and 2D normals."""
    v = P - q
    z = v @ a
    rad = v - np.outer(z, a)
    rho = np.linalg.norm(rad, axis=1)
    rh = rad / np.maximum(rho, 1e-300)[:, None]
    return rho, z, np.einsum("ij,ij->i", N, rh), N @ a


def _axis_from_normals(P, N):
    """Revolution axis from normal lines (they all meet the axis), or None."""
    c0 = P.mean(axis=0)
    scale = float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0
    Q = (P - c0) / scale
    V = np.c_[np.cross(Q, N), N]
    w, E = np.linalg.eigh(V.T @ V / len(V))
    c, cb = E[:3, 0], E[3:, 0]
    nc = float(np.linalg.norm(c))
    if nc < 1e-6 or w[1] < 1e-6 or w[0] / w[1] > 0.1 or abs(float(c @ cb)) / nc ** 2 > 0.05:
        return None
    return c / nc, c0 + np.cross(c, cb) / nc ** 2 * scale


def torus_from(P, N, s, j, nb):
    """Torus candidate: local axis from the neighbourhood, profile circle from two oriented points."""
    ax = _axis_from_normals(P[nb], N[nb])
    if ax is None:
        return None
    a, q = ax
    rho, z, nr, nz = _meridian(P[[s, j]], N[[s, j]], a, q)
    d1, d2 = np.array([nr[0], nz[0]]), np.array([nr[1], nz[1]])
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-3:
        return None
    t1, t2 = np.linalg.solve(A, np.array([rho[1] - rho[0], z[1] - z[0]]))
    c1 = np.array([rho[0], z[0]]) + t1 * d1
    r1, r2 = abs(t1) * np.linalg.norm(d1), abs(t2) * np.linalg.norm(d2)
    if abs(r1 - r2) > 0.05 * (r1 + r2) or c1[0] <= 0:
        return None
    return {"center": q + c1[1] * a, "axis": a, "R": float(c1[0]), "r": float(0.5 * (r1 + r2))}


def fit_torus(P, N, prev=None):
    ax = _axis_from_normals(P, N)
    a, q = ax if ax is not None else (prev["axis"], prev["center"]) if prev else (None, None)
    if a is None:
        return prev
    if prev is not None and a @ prev["axis"] < 0:
        a = -a
    rho, z, _, _ = _meridian(P, N, a, q)
    (cr, cz), r = _fit_circle_2d(np.c_[rho, z])
    if not (cr > 0 and r > 0):
        return prev
    return {"center": q + cz * a, "axis": a, "R": float(cr), "r": float(r)}


def refine_geometric(kind: str, prm: dict, P: np.ndarray, f_scale: float, max_points: int = 5000) -> dict:
    """Robust geometric least squares on true point-to-surface distances.

    The algebraic fits above are fast but slightly biased; an axis that is off by
    a fraction of a degree turns a cone's exact r = a*h + b into a hyperbola when
    symbolic regression measures it. Minimal parameterisations keep the problem
    well posed; soft-L1 loss ignores tangent spill-over at region borders.
    """
    from scipy.optimize import least_squares

    if kind not in ("plane", "sphere", "cylinder", "cone", "torus") or len(P) < 20:
        return prm
    if len(P) > max_points:
        P = P[np.random.default_rng(0).choice(len(P), max_points, replace=False)]
    opts = dict(loss="soft_l1", f_scale=max(f_scale, 1e-12), method="trf", max_nfev=200)
    try:
        if kind == "plane":
            n0, q0 = prm["normal"], prm["point"]
            e1 = _perp(n0)
            e2 = np.cross(n0, e1)

            def unpack(x):
                return _unit(n0 + x[0] * e1 + x[1] * e2), q0 + x[2] * n0

            def f(x):
                n, q = unpack(x)
                return (P - q) @ n

            n, q = unpack(least_squares(f, np.zeros(3), **opts).x)
            return {"point": q, "normal": n}
        if kind == "sphere":
            def f(x):
                return np.linalg.norm(P - x[:3], axis=1) - x[3]
            x = least_squares(f, np.r_[prm["center"], prm["radius"]], **opts).x
            return {"center": x[:3], "radius": float(abs(x[3]))}
        a0 = prm["axis"]
        e1 = _perp(a0)
        e2 = np.cross(a0, e1)
        if kind == "cylinder":
            p0 = prm["point"]

            def unpack(x):
                return p0 + x[0] * e1 + x[1] * e2, _unit(a0 + x[2] * e1 + x[3] * e2), x[4]

            def f(x):
                p, a, r = unpack(x)
                v = P - p
                return np.linalg.norm(v - np.outer(v @ a, a), axis=1) - r

            p, a, r = unpack(least_squares(f, [0, 0, 0, 0, prm["radius"]], **opts).x)
            return {"point": p, "axis": a, "radius": float(abs(r))}
        if kind == "torus":
            c0 = prm["center"]

            def unpack(x):
                return (c0 + x[0] * e1 + x[1] * e2 + x[2] * a0, _unit(a0 + x[3] * e1 + x[4] * e2),
                        x[5], x[6])

            def f(x):
                c, a, R, r = unpack(x)
                v = P - c
                h = v @ a
                rho = np.linalg.norm(v - np.outer(h, a), axis=1)
                return np.hypot(rho - R, h) - r

            c, a, R, r = unpack(least_squares(f, [0, 0, 0, 0, 0, prm["R"], prm["r"]], **opts).x)
            if not (R > 0 and r > 0):
                return prm
            return {"center": c, "axis": a, "R": float(R), "r": float(abs(r))}
        apex0, t0 = prm["apex"], prm["half_angle"]

        def unpack(x):
            return apex0 + x[:3], _unit(a0 + x[3] * e1 + x[4] * e2), t0 + x[5]

        def f(x):
            apex, a, t = unpack(x)
            v = P - apex
            h = v @ a
            rho = np.linalg.norm(v - np.outer(h, a), axis=1)
            return rho * np.cos(t) - h * np.sin(t)

        apex, a, t = unpack(least_squares(f, np.zeros(6), **opts).x)
        if not (0 < t < np.pi / 2):
            return prm
        return {"apex": apex, "axis": a, "half_angle": float(t)}
    except Exception:
        return prm


REFIT = {
    "plane": lambda P, N, prev: fit_plane(P, N),
    "sphere": lambda P, N, prev: fit_sphere(P, N),
    "cylinder": fit_cylinder,
    "cone": fit_cone,
    "torus": fit_torus,
}


# --------------------------------------------------------------- segmenter

class Segmenter:
    def __init__(self, P: np.ndarray, N: np.ndarray, params: SegParams, progress=None, groups=None):
        self.P, self.N, self.prm = P, N, params
        self.n = len(P)
        self.rng = np.random.default_rng(params.seed)
        self.progress = progress or (lambda *a, **k: None)
        lo, hi = P.min(0), P.max(0)
        self.diag = float(np.linalg.norm(hi - lo))
        self.eps = 0.0  # set below once neighbourhoods are known
        self.cos_a = np.cos(np.radians(params.normal_deg))
        self.min_pts = max(30, int(params.min_frac * self.n))

        self.tree = cKDTree(P)
        k = params.knn
        d, idx = self.tree.query(P, k=k + 1)
        self.knn_d, self.knn_idx = d[:, 1:], idx[:, 1:]
        self.spacing = float(np.median(self.knn_d[:, 0]))
        # Separate parts of an assembly (mesh bodies) never share a region: drop
        # neighbours that belong to another body.
        self.groups = None if groups is None else np.asarray(groups)
        if self.groups is not None:
            cross = self.groups[self.knn_idx] != self.groups[:, None]
            self.knn_idx = np.where(cross, np.arange(self.n)[:, None], self.knn_idx)
            self.knn_d = np.where(cross, np.inf, self.knn_d)
            self.group_size = np.bincount(self.groups)[self.groups]
        self.noise = self._roughness()
        if params.tol > 0:
            self.eps = params.tol * self.diag
        else:
            # Tolerance from what the data supports, not from the part's size: a
            # fixed 0.5 % of a 130 mm housing is 19 % of a 3.5 mm hole's radius.
            self.eps = float(np.clip(max(4 * self.noise, 3 * params.sag, 0.0015 * self.diag),
                                     0.0005 * self.diag, 0.01 * self.diag))
        keep = (self.knn_d < 4 * self.spacing).ravel()
        self.rows = np.repeat(np.arange(self.n), k)[keep]
        self.cols = self.knn_idx.ravel()[keep]
        self._row_ptr = None

    def _roughness(self) -> float:
        """Typical out-of-plane scatter of a point's neighbourhood (scan noise level)."""
        sel = np.arange(self.n) if self.n <= 20000 else self.rng.choice(self.n, 20000, replace=False)
        nb = self.P[np.c_[sel, self.knn_idx[sel]]]
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb) / nb.shape[1]
        lam = np.linalg.eigvalsh(cov)[:, 0]
        return float(np.median(np.sqrt(np.maximum(lam, 0))))

    # graph utilities -------------------------------------------------------
    def _components(self, sel: np.ndarray, edge_ok: np.ndarray | None = None):
        # Only touch the edges leaving `sel` (rows are sorted, so each point's edges
        # are one slice): this runs thousands of times on small patches.
        if len(sel) == 0:
            return np.zeros(0, np.int64)
        if self._row_ptr is None:
            self._row_ptr = np.searchsorted(self.rows, np.arange(self.n + 1))
            self._local = np.full(self.n, -1, np.int64)
        ptr = self._row_ptr
        lo, cnt = ptr[sel], ptr[sel + 1] - ptr[sel]
        total = int(cnt.sum())
        e = np.repeat(lo - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(total)
        if edge_ok is not None:
            e = e[edge_ok[e]]
        local = self._local
        local[sel] = np.arange(len(sel))
        a, b = local[self.rows[e]], local[self.cols[e]]
        ok = b >= 0
        local[sel] = -1
        A = csr_matrix((np.ones(int(ok.sum())), (a[ok], b[ok])), shape=(len(sel), len(sel)))
        _, lab = connected_components(A, directed=False)
        return lab

    def _largest_component(self, sel: np.ndarray) -> np.ndarray:
        if len(sel) == 0:
            return sel
        lab = self._components(sel)
        return sel[lab == np.bincount(lab).argmax()]

    # pass 1 ----------------------------------------------------------------
    def smooth_patches(self) -> list[np.ndarray]:
        cos_s = np.cos(np.radians(self.prm.angle_deg))
        smooth = np.einsum("ij,ij->i", self.N[self.rows], self.N[self.cols]) > cos_s
        self._smooth_edges = smooth
        lab = self._components(np.arange(self.n), smooth)
        order = np.argsort(-np.bincount(lab))
        return [np.where(lab == c)[0] for c in order]

    # pass 2 ----------------------------------------------------------------
    def _candidates(self, rem: np.ndarray, n_hyp: int):
        P, N, rng = self.P, self.N, self.rng
        kinds = self.prm.kinds
        cands = []
        if "plane" in kinds:
            for s in rng.choice(rem, size=min(n_hyp // 2, len(rem)), replace=False):
                cands.append(("plane", plane_from(P[s], N[s])))
        if not any(k in kinds for k in ("cylinder", "sphere", "cone")):
            return cands
        in_rem = np.zeros(self.n, bool)
        in_rem[rem] = True
        extent = float(np.linalg.norm(P[rem].max(0) - P[rem].min(0)))
        radius = max(min(0.15 * self.diag, 0.6 * extent), 6 * self.spacing)
        seeds = rng.choice(rem, size=min(n_hyp, len(rem)), replace=False)
        # half the seeds look for a partner nearby: more likely on the same surface
        radii = np.where(rng.random(len(seeds)) < 0.5, radius, max(radius / 3, 5 * self.spacing))
        for s, nb in zip(seeds, self.tree.query_ball_point(P[seeds], r=radii)):
            nb = np.asarray(nb, dtype=np.int64)
            nb = nb[in_rem[nb]]
            far = nb[np.linalg.norm(P[nb] - P[s], axis=1) > 2.5 * self.spacing]
            if len(far) < 2:
                continue
            j, k = rng.choice(far, size=2, replace=False)
            if "cylinder" in kinds:
                c = cylinder_from(P[s], N[s], P[j], N[j])
                if c and 0.002 * self.diag < c["radius"] < 3 * self.diag:
                    cands.append(("cylinder", c))
            if "sphere" in kinds:
                c = sphere_from(P[s], N[s], P[j], N[j])
                if c and 0.002 * self.diag < c["radius"] < 3 * self.diag:
                    cands.append(("sphere", c))
            if "torus" in kinds and len(nb) >= 30:
                sub_nb = nb if len(nb) <= 400 else rng.choice(nb, 400, replace=False)
                c = torus_from(P, N, s, j, sub_nb)
                if c and 0.004 * self.diag < c["r"] < 0.5 * self.diag and c["R"] < 3 * self.diag:
                    cands.append(("torus", c))
            if "cone" in kinds:
                c = cone_from([P[s], P[j], P[k]], [N[s], N[j], N[k]])
                if c:
                    cands.append(("cone", c))
        return cands

    def _inliers(self, kind, prm, idx):
        d, c = residual(kind, prm, self.P[idx], self.N[idx])
        return idx[(d < self.eps) & (c > self.cos_a)]

    def _best_primitive(self, rem: np.ndarray):
        cands = self._candidates(rem, n_hyp=200)
        if not cands:
            return None
        sub = rem if len(rem) <= 3000 else self.rng.choice(rem, 3000, replace=False)
        qcut = self.prm.quality * self.eps
        scores = np.array([len(self._inliers(k, p, sub)) * PENALTY[k] for k, p in cands])
        # Evaluate the best few candidates of every kind (refit + quality check).
        # Ranking on raw support alone lets one kind crowd out the rest: on a
        # rounded box the top slots all go to cones spanning several fillets.
        kinds_of = np.array([k for k, _ in cands])
        shortlist = []
        for k in np.unique(kinds_of):
            idx = np.where(kinds_of == k)[0]
            shortlist += list(idx[np.argsort(-scores[idx])][:4])
        best = None
        for ci in sorted(shortlist, key=lambda i: -scores[i]):
            if scores[ci] <= 0:
                continue
            kind, prm = cands[ci]
            comp = self._largest_component(self._inliers(kind, prm, rem))
            for _ in range(2):  # refine on the support set, then re-collect
                if len(comp) < 10:
                    break
                new = REFIT[kind](self.P[comp], self.N[comp], prm)
                if new is None:
                    break
                prm = new
                comp = self._largest_component(self._inliers(kind, prm, rem))
            if len(comp) < 10:
                continue
            # A true primitive leaves most residuals near zero; a freeform surface
            # forced onto a primitive fills the tolerance band evenly. Judge the
            # bulk (60th percentile), not the rms: where surfaces blend tangentially
            # (fillets, rounded corners) the inlier set also catches a strip of the
            # neighbour that sits inside the band, which must not veto a real primitive.
            d, _ = residual(kind, prm, self.P[comp], self.N[comp])
            # Real primitive: a tight core (40th percentile) and possibly a strip of a
            # tangent neighbour inside the band (60th percentile can rise). A freeform
            # surface forced onto a primitive has no tight core.
            q40, q60 = np.percentile(d, [40, 60])
            # scan noise alone puts |N(0, s)|'s 40th / 60th percentiles at 0.52s / 0.84s
            core_cut = max(0.75 * qcut, 1.2 * self.noise)
            if q40 > core_cut or q60 > max(2 * qcut, 2.5 * self.noise):
                continue
            # Support first, simplicity second, but a much tighter fit beats a simpler
            # shape: a narrow cone band is also a big sphere within tolerance, yet the
            # cone fits it ~20x better.
            score = len(comp) * PENALTY[kind] * (1.0 - 0.5 * min(1.0, q40 / max(core_cut, 1e-12)))
            if best is None or score > best[0]:
                best = (score, kind, prm, comp)
        return best

    def _extract(self, rem: np.ndarray, labels: np.ndarray, regions: list[Region], min_pts: int) -> np.ndarray:
        """Peel primitives off `rem` until none passes; returns what is left."""
        misses = 0
        for _ in range(80):
            if len(rem) < min_pts:
                break
            found = self._best_primitive(rem)
            if found is None or len(found[3]) < min_pts:
                # RANSAC is random: one unlucky batch of candidates must not end the
                # search while small features (fillets, corners) are still left
                misses += 1
                if misses >= 3:
                    break
                continue
            misses = 0
            _, kind, prm, comp = found
            prm = refine_geometric(kind, prm, self.P[comp], self.eps / 3, max_points=3000)
            radius = prm.get("radius", prm.get("r"))
            if radius is not None and not (0.004 * self.diag < radius < 3 * self.diag):
                break  # degenerate (needle-thin or effectively flat)
            comp = self._largest_component(self._inliers(kind, prm, rem))
            if len(comp) < min_pts:
                break
            labels[comp] = len(regions)
            regions.append(Region(len(regions), comp, kind, prm))
            rem = np.setdiff1d(rem, comp, assume_unique=True)
        return rem

    def _split_two_sided(self, labels: np.ndarray, regions: list[Region]):
        """Split thin wrapped freeform sheets (a blade's skin) into their two sides.

        Such a region is neither a height field nor a surface of revolution, and no
        single frame describes it, but its normals form two opposing families
        across the thickness direction. Each side is then a clean height field
        over that direction, which is how CAD models a blade.
        """
        out = list(regions)
        for r in regions:
            if r.prim != "freeform" or len(r.idx) < 4 * self.min_pts:
                continue
            P, N = self.P[r.idx], self.N[r.idx]
            m = N.mean(axis=0)
            if np.linalg.norm(m) > 0.5 and (N @ (m / np.linalg.norm(m))).min() > np.cos(np.radians(80)):
                continue  # already a height field
            if _axis_from_normals(P, N) is not None:
                continue  # surface of revolution: one frame works
            c = P - P.mean(axis=0)
            radial = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
            if np.percentile(np.einsum("ij,ij->i", radial, N), 5) > 0.2:
                continue  # star-shaped about its centre (blob, ellipsoid): spherical frame works
            _, V = np.linalg.eigh(c.T @ c)
            d = V[:, 0]  # thinnest direction
            s = N @ d
            # The two skins meet only along a narrow band where the normal turns
            # through the thickness direction (a leading edge). Cut that band out and
            # split by connectivity, which, unlike the sign of n.d, is not fooled by
            # a concave (cambered) skin whose normals tilt the "wrong" way.
            band = np.abs(s) < 0.3
            core = r.idx[~band]
            if len(core) < 2 * self.min_pts:
                continue
            lab = self._components(core, self._smooth_edges)
            sizes = np.bincount(lab)
            order = np.argsort(-sizes)
            if len(sizes) < 2 or sizes[order[1]] < 0.2 * len(r.idx):
                continue
            a_idx, b_idx = core[lab == order[0]], core[lab == order[1]]
            # every other point (the band, crumbs) joins the nearer skin
            rest = np.setdiff1d(r.idx, np.r_[a_idx, b_idx])
            if len(rest):
                da = cKDTree(self.P[a_idx]).query(self.P[rest])[0]
                db = cKDTree(self.P[b_idx]).query(self.P[rest])[0]
                a_idx = np.r_[a_idx, rest[da <= db]]
                b_idx = np.r_[b_idx, rest[da > db]]
            na = d if (self.N[a_idx] @ d).mean() >= 0 else -d
            new_id = len(out)
            labels[b_idx] = new_id
            out.append(Region(new_id, np.sort(b_idx), "freeform", {"normal": -na}))
            r.idx = np.sort(a_idx)
            r.params = {"normal": na}
        return labels, out

    def _absorb_by_distance(self, rem: np.ndarray, labels: np.ndarray, regions: list[Region], first: int,
                            passes: int = 6) -> np.ndarray:
        """Leftover points that lie on a neighbouring primitive join it.

        With neighbourhood normals (scans) points near a crease get a blended
        normal that matches neither face, so they fail the normal test and would
        survive as thin freeform strips along every edge. Here only distance
        counts, and only primitives the point already touches are considered.
        """
        ids = {r.id: r for r in regions[first:]}
        for _ in range(passes):
            if len(rem) == 0:
                break
            nb = labels[self.knn_idx[rem]]
            best_d = np.full(len(rem), np.inf)
            best_r = np.full(len(rem), -1)
            for rid, r in ids.items():
                touch = (nb == rid).any(axis=1)
                if not touch.any():
                    continue
                d, _ = residual(r.prim, r.params, self.P[rem[touch]], self.N[rem[touch]])
                sel = np.where(touch)[0]
                better = d < best_d[sel]
                best_d[sel[better]] = d[better]
                best_r[sel[better]] = rid
            take = best_d <= self.eps
            if not take.any():
                break
            for rid in np.unique(best_r[take]):
                pts = rem[take & (best_r == rid)]
                labels[pts] = rid
                ids[rid].idx = np.union1d(ids[rid].idx, pts)
            rem = rem[~take]
        return rem

    def _is_carved_revolution(self, patch: np.ndarray, found: list[Region]) -> bool:
        curved = [r for r in found if r.prim in ("cylinder", "cone", "sphere", "torus")]
        if len(curved) < 5 or any(r.prim == "plane" for r in found):
            return False
        ax = _axis_from_normals(self.P[patch], self.N[patch])
        if ax is None:
            return False
        a, q = ax

        def off_axis(p):
            v = p - q
            return float(np.linalg.norm(v - (v @ a) * a))

        for r in curved:
            prm = r.params
            if r.prim == "sphere":
                if off_axis(prm["center"]) > 2 * self.eps:
                    return False
                continue
            axis = prm["axis"]
            anchor = prm.get("point", prm.get("apex", prm.get("center")))
            if abs(float(axis @ a)) < np.cos(np.radians(2)) or off_axis(anchor) > 2 * self.eps:
                return False
        return True

    def _second_chance(self, rem: np.ndarray, labels: np.ndarray, regions: list[Region], first: int) -> np.ndarray:
        """Find small features squeezed between tangent neighbours (rounded-box corners).

        Each primitive found first also took a strip of its tangent neighbours. Release
        the points it explains poorly, search the released points plus the leftovers
        again with a lower size floor, and give unclaimed points back to their owner.
        """
        qcut = self.prm.quality * self.eps
        released, owner = [], []
        for r in regions[first:]:
            d, _ = residual(r.prim, r.params, self.P[r.idx], self.N[r.idx])
            bad = d > qcut
            if bad.any():
                released.append(r.idx[bad])
                owner.append(np.full(int(bad.sum()), r.id))
                r.idx = r.idx[~bad]
                labels[released[-1]] = -1
        if not released:
            return rem
        rel = np.concatenate(released)
        own = np.concatenate(owner)
        # Spill strips usually belong to another primitive already found (a plane's
        # strip is really the fillet next to it): hand those over first, and only
        # search again among points that no existing primitive explains.
        best_d = np.full(len(rel), np.inf)
        best_r = np.full(len(rel), -1)
        for r in regions[first:]:
            d, c = residual(r.prim, r.params, self.P[rel], self.N[rel])
            ok = (d <= qcut) & (c > self.cos_a) & (d < best_d)
            best_d[ok], best_r[ok] = d[ok], r.id
        took = best_r >= 0
        for r in regions[first:]:
            mine = rel[best_r == r.id]
            if len(mine):
                r.idx = np.union1d(r.idx, mine)
                labels[mine] = r.id
        rel, own = rel[~took], own[~took]
        if len(rel) == 0:
            return rem
        pool = np.union1d(rem, rel)
        left = self._extract(pool, labels, regions, max(30, self.min_pts // 2))
        back = np.isin(rel, left)
        labels[rel[back]] = own[back]
        for r in regions[first:]:
            mine = rel[back & (own == r.id)]
            if len(mine):
                r.idx = np.union1d(r.idx, mine)
        return np.setdiff1d(left, rel, assume_unique=False)

    def run(self) -> tuple[list[Region], np.ndarray]:
        patches = self.smooth_patches()
        labels = np.full(self.n, -1, np.int64)
        regions: list[Region] = []
        global_min = self.min_pts
        big = [p for p in patches if len(p) >= 30]
        for pi, patch in enumerate(big):
            if pi % 25 == 0 or len(patch) > 2000:
                self.progress(pi / max(1, len(big)), f"Fitting primitives in patch {pi + 1}/{len(big)}")
            # small parts get a size floor relative to themselves, not to the whole model
            body = int(self.group_size[patch[0]]) if self.groups is not None else self.n
            self.min_pts = max(30, min(global_min, int(0.1 * body)))
            if len(patch) < self.min_pts:
                continue
            first = len(regions)
            rem = self._extract(patch, labels, regions, self.min_pts)
            if len(regions) > first:
                rem = self._second_chance(rem, labels, regions, first)
            if len(regions) > first and len(rem):
                rem = self._absorb_by_distance(rem, labels, regions, first)
            if self._is_carved_revolution(patch, regions[first:]):
                # A smooth turned profile (bottle, ogive) is matched band by band by
                # tangent cones / spheres / tori, each within tolerance. Keep it as one
                # surface of revolution and let symbolic regression find r = f(h).
                for r in regions[first:]:
                    labels[r.idx] = -1
                del regions[first:]
                rem = patch
            # Primitives that only nibble at a large freeform patch are local
            # approximations (any small piece of a smooth surface looks like a cone),
            # not CAD features: keep the whole patch freeform.
            if len(regions) > first and len(rem) >= self.min_pts and len(patch) - len(rem) < 0.5 * len(patch):
                for r in regions[first:]:
                    labels[r.idx] = -1
                del regions[first:]
                rem = patch
            # leftovers that are still large and connected become freeform regions
            if len(rem) >= self.min_pts:
                lab = self._components(rem, self._smooth_edges)
                for c in range(lab.max() + 1):
                    part = rem[lab == c]
                    if len(part) >= self.min_pts:
                        labels[part] = len(regions)
                        regions.append(Region(len(regions), part, "freeform"))
        self.min_pts = global_min
        self._regions_ref = regions
        labels = self._fill_unassigned(labels, len(regions))
        labels, regions = self._split_two_sided(labels, regions)
        self._regions_ref = regions
        self.progress(0.95, "Refining region boundaries")
        labels = self._refine_boundaries(labels, regions)
        # fragments that boundary refinement shrank away are dissolved into neighbours
        sizes = np.bincount(labels, minlength=len(regions))
        floor = np.full(len(regions), max(30, self.min_pts // 2))
        if self.groups is not None:
            # relative to the region's own part: a small part's faces are small too
            for r in regions:
                if len(r.idx):
                    body = int(self.group_size[r.idx[0]])
                    floor[r.id] = max(15, min(floor[r.id], int(0.05 * body)))
                    if sizes[r.id] >= 0.9 * body:  # the whole part: never dissolve
                        floor[r.id] = 0
        tiny = np.where(sizes < floor)[0]
        if len(tiny) and len(tiny) < len(regions):
            labels[np.isin(labels, tiny)] = -1
            labels = self._fill_unassigned(labels, len(regions))
        return self._finalize(labels, regions)

    def _refine_boundaries(self, labels: np.ndarray, regions: list[Region], passes: int = 10) -> np.ndarray:
        """Move boundary points to whichever neighbouring surface explains them best.

        Matters where surfaces meet tangentially (fillets, a capsule's cylinder and
        hemispheres): RANSAC lets the first primitive found spill over the tangent
        line by up to the tolerance; this walks the boundary back ring by ring.
        """
        prm = {r.id: r.params for r in regions}
        prim = {r.id: r.prim for r in regions}
        freeform_cost = 0.9
        for _ in range(passes):
            nb = labels[self.knn_idx]
            bidx = np.where((nb != labels[:, None]).any(axis=1))[0]
            if len(bidx) == 0:
                break
            cand = np.c_[labels[bidx], nb[bidx]]
            best_cost = np.full(len(bidx), np.inf)
            best_lab = labels[bidx].copy()
            for c in np.unique(cand):
                m = (cand == c).any(axis=1)
                pts = bidx[m]
                if prim[c] == "freeform":
                    cost = np.full(len(pts), freeform_cost)
                else:
                    d, cs = residual(prim[c], prm[c], self.P[pts], self.N[pts])
                    cost = d / self.eps + (1 - cs) / max(1 - self.cos_a, 1e-6)
                sel = np.where(m)[0]
                better = cost < best_cost[sel]
                best_cost[sel[better]] = cost[better]
                best_lab[sel[better]] = c
            changed = best_lab != labels[bidx]
            if not changed.any():
                break
            labels = labels.copy()
            labels[bidx[changed]] = best_lab[changed]
            for c in np.unique(np.r_[best_lab[changed], cand[changed, 0]]):
                if prim[c] != "freeform":
                    idx = np.where(labels == c)[0]
                    if len(idx) >= 10:
                        prm[c] = REFIT[prim[c]](self.P[idx], self.N[idx], prm[c]) or prm[c]
        for r in regions:
            r.params = prm[r.id]
        return labels

    def _fill_unassigned(self, labels: np.ndarray, n_regions: int) -> np.ndarray:
        if n_regions == 0:
            return np.zeros(self.n, np.int64)
        for _ in range(30):
            un = np.where(labels < 0)[0]
            if len(un) == 0:
                return labels
            nb = labels[self.knn_idx[un]]
            changed = False
            for i, row in zip(un, nb):
                row = row[row >= 0]
                if len(row):
                    labels[i] = np.bincount(row).argmax()
                    changed = True
            if not changed:
                break
        un = labels < 0
        if un.any() and self.groups is not None:
            # parts too small to segment: one region per part (never borrow another part's)
            for g in np.unique(self.groups[un]):
                pts = np.where(un & (self.groups == g))[0]
                if len(pts) >= 3:
                    labels[pts] = self._new_small_region(pts)
            un = labels < 0
        if un.any():
            lab_idx = np.where(~un)[0]
            _, j = cKDTree(self.P[lab_idx]).query(self.P[un])
            labels[un] = labels[lab_idx[j]]
        return labels

    def _new_small_region(self, pts: np.ndarray) -> int:
        rid = len(self._regions_ref)
        self._regions_ref.append(Region(rid, pts, "freeform"))
        return rid

    def _finalize(self, labels, regions):
        # refit each primitive on its final membership, renumber by size
        sizes = np.bincount(labels, minlength=len(regions))
        order = [r for r in np.argsort(-sizes) if sizes[r] > 0]
        remap = np.full(len(regions), -1)
        remap[order] = np.arange(len(order))
        labels = remap[labels]
        out = []
        for new_id, old in enumerate(order):
            r = regions[old]
            idx = np.where(labels == new_id)[0]
            prm = r.params
            if r.prim != "freeform":
                prm = REFIT[r.prim](self.P[idx], self.N[idx], prm) or prm
                prm = refine_geometric(r.prim, prm, self.P[idx], self.eps / 3)
                d, _ = residual(r.prim, prm, self.P[idx], self.N[idx])
                rms = float(np.sqrt(np.mean(np.minimum(d, 10 * self.eps) ** 2)))
            else:
                rms = float("nan")
            out.append(Region(new_id, idx, r.prim, prm, rms))
        return out, labels


def face_labels(n_faces: int, fidx: np.ndarray, labels: np.ndarray, face_centroids: np.ndarray,
                points: np.ndarray) -> np.ndarray:
    """Transfer point labels to mesh faces (majority vote, nearest point for empty faces)."""
    R = int(labels.max()) + 1
    key = fidx * R + labels
    uk, cnt = np.unique(key, return_counts=True)
    f, lab = uk // R, uk % R
    order = np.lexsort((-cnt, f))  # per face, highest count first
    f, lab = f[order], lab[order]
    first = np.r_[True, f[1:] != f[:-1]]
    out = np.full(n_faces, -1, np.int64)
    out[f[first]] = lab[first]
    empty = out < 0
    if empty.any():
        _, j = cKDTree(points).query(face_centroids[empty])
        out[empty] = labels[j]
    return out


def clean_face_islands(face_adjacency: np.ndarray, labels: np.ndarray, max_passes: int = 5) -> np.ndarray:
    """Keep one connected patch of faces per region; merge stray islands into their surroundings.

    Every region becomes exactly one CAD face, so a detached sliver would
    otherwise turn into a separate tiny face.
    """
    labels = labels.copy()
    n = len(labels)
    a, b = face_adjacency[:, 0], face_adjacency[:, 1]
    for _ in range(max_passes):
        same = labels[a] == labels[b]
        A = csr_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(n, n))
        _, comp = connected_components(A, directed=False)
        size = np.bincount(comp)
        # largest component per label
        order = np.lexsort((-size[comp], labels))
        first = np.r_[True, labels[order][1:] != labels[order][:-1]]
        keep_comp = np.zeros(len(size), bool)
        keep_comp[comp[order[first]]] = True
        stray = ~keep_comp[comp]
        if not stray.any():
            break
        # relabel each stray component to its most common neighbouring label
        diff = ~same
        ea, eb = a[diff], b[diff]
        pairs = np.r_[np.c_[comp[ea], labels[eb]], np.c_[comp[eb], labels[ea]]]
        pairs = pairs[~keep_comp[pairs[:, 0]]]
        if len(pairs) == 0:
            break
        uk, cnt = np.unique(pairs, axis=0, return_counts=True)
        best = {}
        for (c, lab), k in zip(uk, cnt):
            if k > best.get(c, (None, -1))[1]:
                best[c] = (lab, k)
        new_label = labels.copy()
        for c, (lab, _) in best.items():
            new_label[comp == c] = lab
        labels = new_label
    return labels


def fit_face_labels(mesh, labels: np.ndarray, regions: list[Region], eps: float, passes: int = 3,
                    fixed: np.ndarray | None = None) -> np.ndarray:
    """Re-decide labels of border triangles by geometry instead of sample votes.

    A coarse CAD triangle can carry only a handful of samples; when it bridges a
    narrow gap (a hole 1 mm from a fillet) a mislabel merges two boundary loops.
    Each border face goes to whichever candidate region's surface contains its
    corners and agrees with its normal (freeform regions get a neutral cost).
    """
    labels = labels.copy()
    V, F = mesh.vertices, mesh.faces
    fn = mesh.face_normals
    adj = mesh.face_adjacency
    prim = {r.id: r.prim for r in regions}
    prm = {r.id: r.params for r in regions}
    cos_a = np.cos(np.radians(20))
    n_lab = int(max(labels.max(), max(prim, default=0))) + 2
    for _ in range(passes):
        a, b = adj[:, 0], adj[:, 1]
        diff = labels[a] != labels[b]
        if not diff.any():
            break
        border = np.unique(np.r_[a[diff], b[diff]])
        pairs = np.r_[np.c_[a[diff], labels[b[diff]]], np.c_[b[diff], labels[a[diff]]],
                      np.c_[border, labels[border]]]
        if fixed is not None:  # faces that keep their label (exact planes) are never re-decided
            pairs = pairs[~fixed[pairs[:, 0]]]
        pairs = pairs[pairs[:, 1] >= 0]
        code = np.unique(pairs[:, 0] * n_lab + pairs[:, 1])
        cand = np.c_[code // n_lab, code % n_lab]
        best_cost = np.full(len(labels), np.inf)
        best_lab = labels.copy()
        cand = cand[np.lexsort((cand[:, 0], cand[:, 1]))]  # grouped by region, ascending
        rids, starts = np.unique(cand[:, 1], return_index=True)
        for rid, s0, e0 in zip(rids, starts, np.r_[starts[1:], len(cand)]):
            faces = cand[s0:e0, 0]
            if prim.get(rid, "freeform") == "freeform":
                cost = np.full(len(faces), 1.0)
            else:
                pts = V[F[faces]].reshape(-1, 3)
                nn = np.repeat(fn[faces], 3, axis=0)
                d, c = residual(prim[rid], prm[rid], pts, nn)
                d = d.reshape(-1, 3).max(axis=1)
                c = c.reshape(-1, 3).min(axis=1)
                cost = d / max(eps, 1e-12) + (1 - c) / max(1 - cos_a, 1e-6)
            better = cost < best_cost[faces]
            best_cost[faces[better]] = cost[better]
            best_lab[faces[better]] = rid
        if np.array_equal(best_lab, labels):
            break
        labels = best_lab
    return labels


def smooth_face_labels(face_adjacency: np.ndarray, labels: np.ndarray, iters: int = 4) -> np.ndarray:
    """Majority filter along region borders for dense meshes (scans).

    With far more triangles than samples, per-face labels zigzag along borders and
    three-region junctions alternate, which would split edges into many tiny
    pieces. A face switches to a label held by at least two of its three
    neighbours. (Not used on coarse CAD meshes, where a chamfer can be one
    triangle wide.)
    """
    labels = labels.copy()
    n = len(labels)
    a, b = face_adjacency[:, 0], face_adjacency[:, 1]
    for _ in range(iters):
        diff = labels[a] != labels[b]
        if not diff.any():
            break
        # votes[f] = labels of neighbours; collect per face via the edge list
        src = np.r_[a, b]
        nb_lab = np.r_[labels[b], labels[a]]
        order = np.lexsort((nb_lab, src))
        src, nb_lab = src[order], nb_lab[order]
        key_change = np.r_[True, (src[1:] != src[:-1]) | (nb_lab[1:] != nb_lab[:-1])]
        starts = np.flatnonzero(key_change)
        counts = np.diff(np.r_[starts, len(src)])
        f, lab = src[starts], nb_lab[starts]
        strong = (counts >= 2) & (lab != labels[f])
        new = labels.copy()
        new[f[strong]] = lab[strong]
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def region_adjacency(labels: np.ndarray, seg: Segmenter) -> list[tuple[int, int]]:
    a, b = labels[seg.rows], labels[seg.cols]
    m = a != b
    pairs = np.unique(np.sort(np.c_[a[m], b[m]], axis=1), axis=0)
    return [tuple(map(int, p)) for p in pairs]
