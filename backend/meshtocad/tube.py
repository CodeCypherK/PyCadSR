"""Tubes: surfaces at a constant distance from a spine curve (a coil spring, a bent
wire or cable). No analytic primitive fits one, and no height field holds it, but it
is one smooth CAD face: a circle swept along the spine.

`fit_tube` finds the radius and spine from a mesh patch's vertices; `TubeSurface`
projects points onto the tube and gives the (spine, angle) coordinates the STEP
writer uses for its B-spline surface and seam.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.interpolate import splev, splprep
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree


def _vertex_normals(V, F, FN, A):
    VN = np.zeros_like(V)
    np.add.at(VN, F.ravel(), np.repeat(FN * A[:, None], 3, axis=0))
    return VN / np.maximum(np.linalg.norm(VN, axis=1, keepdims=True), 1e-300)


def _longest_path(Cq, link):
    """Spine samples in order: minimum spanning tree over nearby samples, its longest path.
    None if the samples do not form one chain."""
    n = len(Cq)
    pairs = cKDTree(Cq).query_pairs(link, output_type="ndarray")
    if len(pairs) < n - 1:
        return None
    w = np.linalg.norm(Cq[pairs[:, 0]] - Cq[pairs[:, 1]], axis=1) + 1e-12
    G = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    ncomp, comp = connected_components(G, directed=False)
    if ncomp > 1 and np.bincount(comp).max() < 0.98 * n:
        return None
    T = minimum_spanning_tree(G)
    T = (T + T.T).tocsr()

    def far(src):
        o, pred = breadth_first_order(T, src, directed=False)
        dist = np.zeros(n)
        for v in o[1:]:
            dist[v] = dist[pred[v]] + T[v, pred[v]]
        return int(o[np.argmax(dist[o])]), pred, dist

    a, _, _ = far(int(np.argmax(np.bincount(comp)[comp])))
    b, pred, dist = far(a)
    path = [b]
    while path[-1] != a:
        path.append(int(pred[path[-1]]))
    return np.array(path[::-1]), float(dist[b])


def _spine_from(V, VN, r):
    """Spine spline for radius r (or None), and the residual |dist - r| of every vertex."""
    Q = V - r * VN
    cell = 0.5 * r
    _, inv = np.unique(np.floor(Q / cell).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    Cq = np.stack([np.bincount(inv, weights=Q[:, i]) for i in range(3)], axis=1) / cnt[:, None]
    got = _longest_path(Cq, 3.5 * cell)
    if got is None:
        return None
    path, length = got
    if length < 8 * r:
        return None  # a blob, not a tube
    P_ = Cq[path]
    s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P_, axis=0), axis=1))]
    _, nearest = cKDTree(P_).query(Q)
    u = s_[nearest]
    nb = max(8, int(s_[-1] / (0.5 * r)))
    idx = np.clip(np.searchsorted(np.linspace(0, s_[-1], nb + 1), u) - 1, 0, nb - 1)
    keep = np.bincount(idx, minlength=nb) > 0
    C = np.stack([np.bincount(idx, weights=Q[:, i], minlength=nb) for i in range(3)], axis=1)[keep]
    C /= np.bincount(idx, minlength=nb)[keep][:, None]
    if len(C) < 6:
        return None
    # a little beyond both ends, along the end tangents: the end rings project inside
    t0, t1 = C[1] - C[0], C[-1] - C[-2]
    t0 /= max(np.linalg.norm(t0), 1e-300)
    t1 /= max(np.linalg.norm(t1), 1e-300)
    C = np.vstack([C[0] - 2 * r * t0, C[0] - r * t0, C, C[-1] + r * t1, C[-1] + 2 * r * t1])
    try:
        tck, _ = splprep(C.T, s=len(C) * (0.02 * r) ** 2, k=3)
    except Exception:
        return None
    return tck


class TubeSurface:
    """A circle of radius r swept along a spine spline, with a rotation-minimising
    frame along it: coordinates (t, phi), t the spine parameter in [0, 1]."""

    def __init__(self, frame, r, tck, n=None):
        self.frame = frame
        self.r = float(r)
        self.tck = (np.asarray(tck[0], float), [np.asarray(c, float) for c in tck[1]], int(tck[2]))
        self.symbols = sp.symbols(frame.var_names) if frame is not None else ()
        self.expr = sp.Symbol("tube")
        L = self._length()
        n = n or int(np.clip(L / (0.05 * self.r), 2000, 200000))
        self.t = np.linspace(0.0, 1.0, n)
        self.S = np.array(splev(self.t, self.tck)).T
        T = np.array(splev(self.t, self.tck, der=1)).T
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-300)
        self.T = T
        # rotation-minimising frame (double reflection)
        N = np.zeros_like(T)
        a = np.cross(T[0], [1.0, 0, 0])
        if np.linalg.norm(a) < 0.1:
            a = np.cross(T[0], [0, 1.0, 0])
        N[0] = a / np.linalg.norm(a)
        for i in range(n - 1):
            v1 = self.S[i + 1] - self.S[i]
            c1 = v1 @ v1
            if c1 < 1e-300:
                N[i + 1] = N[i]
                continue
            rL = N[i] - (2 / c1) * (v1 @ N[i]) * v1
            tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
            v2 = T[i + 1] - tL
            c2 = v2 @ v2
            N[i + 1] = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-300 else rL
            N[i + 1] -= (N[i + 1] @ T[i + 1]) * T[i + 1]
            N[i + 1] /= max(np.linalg.norm(N[i + 1]), 1e-300)
        self.N = N
        self.B = np.cross(T, N)
        self._kd = cKDTree(self.S)

    def _length(self):
        P = np.array(splev(np.linspace(0, 1, 4000), self.tck)).T
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    def to_json(self):
        t, c, k = self.tck
        return {"tube_r": self.r, "tube_t": t.tolist(), "tube_c": [ci.tolist() for ci in c], "tube_k": k}

    @classmethod
    def from_params(cls, frame, p):
        return cls(frame, p["tube_r"], (p["tube_t"], p["tube_c"], p["tube_k"]))

    def _foot(self, P):
        """Spine parameter of the nearest spine point, refined on the neighbouring span."""
        P = np.atleast_2d(P)
        _, j = self._kd.query(P)
        j = np.clip(j, 1, len(self.t) - 2)
        best_t = self.t[j].copy()
        for side in (-1, 1):
            a, b = self.S[j], self.S[j + side]
            d = b - a
            w = np.clip(np.einsum("ij,ij->i", P - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-300), 0, 1)
            q = a + w[:, None] * d
            tt = self.t[j] + side * w * (self.t[1] - self.t[0])
            better = np.linalg.norm(P - q, axis=1) < np.linalg.norm(P - np.array(splev(best_t, self.tck)).T, axis=1)
            best_t = np.where(better, tt, best_t)
        return best_t

    def _frame_at(self, t):
        x = np.clip(t, 0, 1) * (len(self.t) - 1)
        i = np.clip(np.floor(x).astype(int), 0, len(self.t) - 2)
        w = (x - i)[:, None]
        N = (1 - w) * self.N[i] + w * self.N[i + 1]
        T = np.array(splev(t, self.tck, der=1)).T
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-300)
        N -= np.einsum("ij,ij->i", N, T)[:, None] * T
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
        return np.array(splev(t, self.tck)).T, T, N, np.cross(T, N)

    def angles(self, P):
        """(t, phi): spine parameter and angle round the tube (0 along the frame's N)."""
        t = self._foot(P)
        C, T, N, B = self._frame_at(t)
        d = np.atleast_2d(P) - C
        return t, np.arctan2(np.einsum("ij,ij->i", d, B), np.einsum("ij,ij->i", d, N))

    def point(self, t, ph):
        t, ph = np.atleast_1d(np.asarray(t, float)), np.atleast_1d(np.asarray(ph, float))
        C, T, N, B = self._frame_at(t)
        return C + self.r * (np.cos(ph)[:, None] * N + np.sin(ph)[:, None] * B)

    def project_points(self, P):
        t = self._foot(P)
        C = np.array(splev(t, self.tck)).T
        d = np.atleast_2d(P) - C
        dn = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)
        return C + d / dn * self.r

    def normals(self, P):
        t = self._foot(P)
        d = np.atleast_2d(P) - np.array(splev(t, self.tck)).T
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)

    def residual(self, P):
        t = self._foot(P)
        return np.linalg.norm(np.atleast_2d(P) - np.array(splev(t, self.tck)).T, axis=1) - self.r

    seam = None  # (t_a, phi_a, t_b, phi_b): the seam runs between two boundary vertices

    def set_seam(self, pa, pb):
        (ta, tb), (fa, fb) = self.angles(np.vstack([pa, pb]))
        self.seam = (float(ta), float(fa), float(tb), float(fa + np.angle(np.exp(1j * (fb - fa)))))

    def seam_phi(self, t):
        """Tube angle of the seam along the spine (linear between its two ends): the
        STEP surface's closing line, so the seam edge is an exact iso-line of it."""
        t = np.asarray(t, float)
        if self.seam is None:
            return np.zeros_like(t)
        ta, fa, tb, fb = self.seam
        if abs(tb - ta) < 1e-12:
            return np.full_like(t, fa)
        return fa + (t - ta) / (tb - ta) * (fb - fa)

    def seam_points(self, pa, pb, P_face=None, n=32):
        """The seam from pa to pb along the spine."""
        if self.seam is None:
            self.set_seam(pa, pb)
        ta, _, tb, _ = self.seam
        da = np.linalg.norm(pa - self.point(ta, self.seam_phi(ta))[0])
        db = np.linalg.norm(pa - self.point(tb, self.seam_phi(tb))[0])
        if db < da:
            ta, tb = tb, ta
        t = np.linspace(ta, tb, n)
        pts = self.point(t, self.seam_phi(t))
        pts[0], pts[-1] = pa, pb
        return pts

    def __call__(self, X):  # frame-based callers (previews): not a function of the frame
        return np.full(len(X), np.nan)

    def preview(self, P, nv: int = 16):
        """A triangle mesh of the tube over the spine span of points P (vertices, faces)."""
        t, _ = self.angles(P)
        lo, hi = float(t.min()), float(t.max())
        nu = int(np.clip(self._length() * (hi - lo) / (0.5 * self.r), 8, 4000))
        A = np.linspace(lo, hi, nu)
        Bv = np.linspace(0.0, 2 * np.pi, nv, endpoint=False)
        TT, VV = np.meshgrid(A, Bv, indexing="ij")
        V = self.point(TT.ravel(), self.seam_phi(TT.ravel()) + VV.ravel())
        i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv), indexing="ij")
        a = i * nv + j
        b = i * nv + (j + 1) % nv
        c, d = a + nv, b + nv
        F = np.r_[np.c_[a.ravel(), b.ravel(), d.ravel()], np.c_[a.ravel(), d.ravel(), c.ravel()]]
        return V, F

    def latex(self) -> str:
        return rf"\text{{tube, wire radius }} {self.r:.6g}"

    def local_world(self) -> tuple[str, str]:
        return (f"tube: circle of radius {self.r:.6g} swept along a spine spline",
                rf"\text{{tube: radius }} {self.r:.6g} \text{{ swept along a spine}}")


def fit_tube(V, F, FN, A, tol):
    """A tube through these faces' vertices: dict(r, tck, rms, p99) or None."""
    vid = np.unique(F)
    remap = np.full(len(V), -1)
    remap[vid] = np.arange(len(vid))
    Fl = remap[F]
    P = V[vid]
    VN = _vertex_normals(P, Fl, FN, A)
    # radius from the curvature across the tube: |n_i - n_j| / |p_i - p_j| peaks at 1/r
    E = np.unique(np.sort(Fl[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    dn = np.linalg.norm(VN[E[:, 0]] - VN[E[:, 1]], axis=1)
    dp = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    k = dn / np.maximum(dp, 1e-300)
    kv = np.zeros(len(P))
    np.maximum.at(kv, E[:, 0], k)
    np.maximum.at(kv, E[:, 1], k)
    if not (kv > 0).any():
        return None
    r0 = 1.0 / float(np.median(kv[kv > 0]))
    best = None
    for r in r0 * np.linspace(0.9, 1.15, 11):
        tck = _spine_from(P, VN, r)
        if tck is None:
            continue
        tube = TubeSurface(None, r, tck, n=4000)
        # radius refit to the spine: the mean distance
        d = tube.residual(P) + r
        r2 = float(np.median(d))
        res = d - r2
        rms = float(np.sqrt(np.mean(res ** 2)))
        if best is None or rms < best["rms"]:
            best = {"r": r2, "tck": tck, "rms": rms, "p99": float(np.quantile(np.abs(res), 0.99))}
    return best
