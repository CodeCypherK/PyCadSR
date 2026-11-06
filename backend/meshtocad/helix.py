"""Helical sweeps: a profile curve swept along a helix (a thread flank). No analytic
primitive fits one, and over several turns no height field holds it, but it is one
exact surface: around axis a through c, with lead L (advance per turn),

    P(s, x) = c + (q(x) + L s / 2 pi) a + rho(x) (cos s e1 + sin s e2)

where s is the angle round the axis (unwrapped over the turns) and (rho, q) the
profile in the half-plane at s = 0: the distance from the axis and the height. The
profile is a low-degree polynomial in whichever of rho and q it spans more of.

`fit_helix` finds axis, lead and profile from a region's exact vertices;
`HelixSurface` projects points onto it and gives the (s, x) coordinates the STEP
writer samples its B-spline surface on.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .segmentation import _perp


class HelixSurface:
    def __init__(self, frame, axis, point, lead, by, coef, x0, xs, q0, s_rng, x_rng, spline=None):
        self.frame = frame
        a = np.asarray(axis, float)
        self.a = a / np.linalg.norm(a)
        self.c = np.asarray(point, float)
        self.L = float(lead)
        self.e1 = _perp(self.a)
        self.e2 = np.cross(self.a, self.e1)
        # "rho": q = g(rho); "q": rho = g(q); "spline": (rho, q) = a cubic B-spline of the
        # profile's arc length, for a profile that is neither (a round groove)
        self.by = by
        self.coef = np.asarray(coef, float)
        self.x0, self.xs, self.q0 = float(x0), float(xs), float(q0)
        self.s_rng, self.x_rng = tuple(map(float, s_rng)), tuple(map(float, x_rng))
        self.spline = None
        self.n_profile = 9  # profile samples the STEP writer takes across the face
        if by == "spline":
            from scipy.interpolate import BSpline
            from scipy.spatial import cKDTree
            kn, crho, cq = (np.asarray(v, float) for v in spline)
            self.spline = (kn, crho, cq)
            self._spl = BSpline(kn, np.c_[crho, cq], 3, extrapolate=True)
            self._d1, self._d2 = self._spl.derivative(1), self._spl.derivative(2)
            lo, hi = float(kn[0]), float(kn[-1])
            m = 0.1 * (hi - lo)
            self._xd = np.linspace(lo - m, hi + m, 4097)
            self._kd = cKDTree(self._sp(self._xd))
            self.n_profile = int(np.clip(2 * (len(kn) - 7) + 1, 9, 81))
        self.symbols = sp.symbols(frame.var_names) if frame is not None else ()
        self.expr = sp.Symbol("helix")
        self.rms = 0.0

    # ---------------------------------------------------------------- profile
    def g(self, x, der=0):
        """The profile's other coordinate (and its derivative) at x."""
        t = (np.asarray(x, float) - self.x0) / self.xs
        c = self.coef
        for _ in range(der):
            c = np.polyder(c)
        return np.polyval(c, t) / self.xs ** der

    def _sp(self, x, der=0):
        """The spline profile (or a derivative) at x, carried on straight along its end
        tangents beyond its ends: the STEP writer samples a margin past the face, and a
        cubic end piece extrapolated round a tight bend flares out by millimetres."""
        x = np.asarray(x, float)
        lo, hi = float(self.spline[0][0]), float(self.spline[0][-1])
        xc = np.clip(x, lo, hi)
        if der == 0:
            v = self._spl(xc)
            d = (x - xc)[..., None]
            return v + d * self._d1(xc) if np.any(d) else v
        if der == 1:
            return self._d1(xc)
        v = self._d2(xc)
        return np.where(((x < lo) | (x > hi))[..., None], 0.0, v)

    def profile(self, x):
        """(rho, q) of the profile at x."""
        x = np.asarray(x, float)
        if getattr(self, "spline", None) is not None:
            v = self._sp(x)
            return v[..., 0], v[..., 1]
        return (x, self.g(x)) if self.by == "rho" else (self.g(x), x)

    # ---------------------------------------------------------------- coordinates
    def _raw(self, P):
        v = np.atleast_2d(P) - self.c
        h = v @ self.a
        rho = np.linalg.norm(v - np.outer(h, self.a), axis=1)
        th = np.arctan2(v @ self.e2, v @ self.e1)
        q = h - self.L * th / (2 * np.pi)
        k = np.round((q - self.q0) / self.L)
        return th + 2 * np.pi * k, rho, q - self.L * k

    def coords(self, P):
        """(s, x): angle round the axis over the turns, and the profile parameter."""
        s, rho, q = self._raw(P)
        return s, self._nearest_x(rho, q)

    def _nearest_x(self, rho, q):
        """Profile parameter of the nearest profile point to (rho, q) (Newton steps)."""
        if getattr(self, "spline", None) is not None:
            X = np.c_[np.atleast_1d(rho), np.atleast_1d(q)]
            d0, j = self._kd.query(X)
            x0 = x = self._xd[j]
            step = self._xd[1] - self._xd[0]
            for _ in range(5):
                d = self._sp(x) - X
                d1, d2 = self._sp(x, 1), self._sp(x, 2)
                f = np.einsum("ij,ij->i", d, d1)
                fp = np.einsum("ij,ij->i", d1, d1) + np.einsum("ij,ij->i", d, d2)
                # Newton from the nearest sample, never more than a sample spacing away
                x = np.clip(x - f / np.where(fp > 1e-300, fp, 1e-300), x0 - step, x0 + step)
            worse = np.linalg.norm(self._sp(x) - X, axis=1) > d0
            return np.where(worse, x0, x)
        x = np.array(rho if self.by == "rho" else q, float)
        for _ in range(4):
            r_, q_ = self.profile(x)
            d1 = self.g(x, 1)
            dr, dq = (np.ones_like(x), d1) if self.by == "rho" else (d1, np.ones_like(x))
            d2 = self.g(x, 2)
            ddr, ddq = (np.zeros_like(x), d2) if self.by == "rho" else (d2, np.zeros_like(x))
            f = (r_ - rho) * dr + (q_ - q) * dq
            fp = dr * dr + dq * dq + (r_ - rho) * ddr + (q_ - q) * ddq
            x = x - f / np.where(np.abs(fp) > 1e-300, fp, 1e-300)
        return x

    def point(self, s, x):
        s, x = np.atleast_1d(np.asarray(s, float)), np.atleast_1d(np.asarray(x, float))
        rho, q = self.profile(x)
        return (self.c + np.outer(q + self.L * s / (2 * np.pi), self.a)
                + rho[:, None] * (np.cos(s)[:, None] * self.e1 + np.sin(s)[:, None] * self.e2))

    def project_points(self, P):
        s, x = self.coords(P)
        return self.point(s, x)

    def normals(self, P):
        s, x = self.coords(P)
        d = 1e-6
        ds = self.point(s + d, x) - self.point(s - d, x)
        dx = self.point(s, x + d * self.xs) - self.point(s, x - d * self.xs)
        n = np.cross(ds, dx)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    def residual(self, P):
        return np.linalg.norm(np.atleast_2d(P) - self.project_points(P), axis=1)

    def __call__(self, X):  # frame-based callers: not a function of the frame
        return np.full(len(X), np.nan)

    def preview(self, P, nv: int = 12):
        s, x = self.coords(P)
        lo, hi = float(s.min()), float(s.max())
        nu = int(np.clip((hi - lo) / (2 * np.pi) * 48, 8, 4000))
        S, X = np.meshgrid(np.linspace(lo, hi, nu), np.linspace(float(x.min()), float(x.max()), nv), indexing="ij")
        V = self.point(S.ravel(), X.ravel())
        i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
        a = i * nv + j
        b, c_, d = a + 1, a + nv, a + nv + 1
        F = np.r_[np.c_[a.ravel(), c_.ravel(), d.ravel()], np.c_[a.ravel(), d.ravel(), b.ravel()]]
        return V, F

    def to_json(self):
        out = {"helix_axis": self.a.tolist(), "helix_point": self.c.tolist(), "helix_lead": self.L,
               "profile_by": self.by, "profile_coef": self.coef.tolist(), "profile_x0": self.x0,
               "profile_xs": self.xs, "helix_q0": self.q0, "helix_s_rng": list(self.s_rng),
               "helix_x_rng": list(self.x_rng)}
        if getattr(self, "spline", None) is not None:
            out["profile_spline"] = [v.tolist() for v in self.spline]
        return out

    @classmethod
    def from_params(cls, frame, p):
        """The surface back from `to_json` (a region found as a helical face when segmenting)."""
        hx = cls(frame, p["helix_axis"], p["helix_point"], p["helix_lead"], p["profile_by"], p["profile_coef"],
                 p["profile_x0"], p["profile_xs"], p["helix_q0"], p["helix_s_rng"], p["helix_x_rng"],
                 spline=p.get("profile_spline"))
        hx.rms = float(p.get("helix_rms", 0.0))
        return hx

    def latex(self) -> str:
        return rf"\text{{helical sweep, lead }} {self.L:.6g}"

    def local_world(self) -> tuple[str, str]:
        hand = "right" if self.L > 0 else "left"
        prof = "spline profile" if getattr(self, "spline", None) is not None else f"profile of degree {len(self.coef) - 1}"
        return (f"helical sweep: lead {abs(self.L):.6g} ({hand}-handed), {prof}",
                rf"\text{{helical sweep, lead }} {abs(self.L):.6g}")


def _coords(P, a, c, L):
    e1 = _perp(a)
    e2 = np.cross(a, e1)
    v = P - c
    h = v @ a
    rho = np.linalg.norm(v - np.outer(h, a), axis=1)
    q = h - L * np.arctan2(v @ e2, v @ e1) / (2 * np.pi)
    q0 = float(np.median(q))
    return rho, q - L * np.round((q - q0) / L), q0


def _profile_fit(rho, q, deg):
    """Polynomial profile in the better-conditioned variable: (by, coef, x0, xs, normal residuals)."""
    if np.ptp(rho) >= np.ptp(q):
        by, x, y = "rho", rho, q
    else:
        by, x, y = "q", q, rho
    x0, xs = float(x.mean()), max(float(np.ptp(x)) / 2, 1e-300)
    t = (x - x0) / xs
    coef = np.polyfit(t, y, deg)
    sl = np.polyval(np.polyder(coef), t) / xs
    return by, coef, x0, xs, (y - np.polyval(coef, t)) / np.sqrt(1 + sl * sl)


def fit_helix(P, N, frame, tol, max_points: int = 3000):
    """The helical sweep through a region's points (P, with normals N): a HelixSurface
    with its rms set, or None when the points do not lie on one."""
    from scipy.optimize import least_squares

    from .surfaces import helix_axis

    P, N = np.asarray(P, float), np.asarray(N, float)
    if len(P) < 30:
        return None
    scale = float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1).mean())) or 1.0
    # a fine thread advances little per radian against the region's size (M6 over a
    # 6 mm nut: 0.048); the fit below decides whether it really is a helix
    h = helix_axis(P, N, scale, min_pitch=0.01)
    if h is None:
        return None
    a0, c0, ppr = h
    L0 = 2 * np.pi * ppr
    sub = P if len(P) <= max_points else P[np.random.default_rng(0).choice(len(P), max_points, replace=False)]
    u = _perp(a0)
    w = np.cross(a0, u)

    def unpack(x):
        a = a0 + x[0] * u + x[1] * w
        return a / np.linalg.norm(a), c0 + scale * (x[2] * u + x[3] * w), L0 * (1 + x[4])

    def resid(x):
        a, c, L = unpack(x)
        rho, q, _ = _coords(sub, a, c, L)
        return _profile_fit(rho, q, 3)[4]

    try:
        sol = least_squares(resid, np.zeros(5), method="lm", xtol=1e-14, ftol=1e-14, max_nfev=2000)
    except Exception:
        return None
    a, c, L = unpack(sol.x)
    rho, q, q0 = _coords(P, a, c, L)
    # the lowest profile degree that fits as well as any
    fits = {d: _profile_fit(rho, q, d) for d in (1, 2, 3, 5) if len(P) > 3 * (d + 1)}
    if not fits:
        return None
    best = min(float(np.sqrt(np.mean(f[4] ** 2))) for f in fits.values())
    deg = min(d for d, f in fits.items() if np.sqrt(np.mean(f[4] ** 2)) <= max(0.02 * tol, 1.2 * best))
    by, coef, x0, xs, res = fits[deg]
    if rho.min() <= 0 or rho.max() > 2 * rho.min():
        # a thread flank runs from root to crest, well within twice its root radius; a
        # strip far off any axis that a loose helix happens to follow is no thread
        return None
    span_h = float(np.ptp((P - c) @ a))
    if abs(L) > 10 * max(span_h, 1e-300):
        # a straight extrusion fits a helix of enormous lead; a thread advances by its
        # lead within a few times its own length
        return None
    if np.ptp(q) > 0.5 * abs(L):
        # a flank spans a fraction of the lead along the axis; a profile across a whole
        # lead is a surface of revolution (a cylinder fits a helix of any lead), and
        # swept along the helix it would cover itself
        return None
    x = rho if by == "rho" else q
    th = np.arctan2((P - c) @ np.cross(a, _perp(a)), (P - c) @ _perp(a))
    hh = (P - c) @ a
    s = th + 2 * np.pi * np.round((hh - L * th / (2 * np.pi) - q0) / L)
    hx = HelixSurface(frame, a, c, L, by, coef, x0, xs, q0, (s.min(), s.max()), (x.min(), x.max()))
    hx.rms = float(np.sqrt(np.mean(hx.residual(P) ** 2)))
    return hx


# ------------------------------------------------------------ helical faces found whole
def _thick_at(X, qi, nn):
    """Signed distance of the 2-D points X[qi] from the line through their neighbours
    X[nn] (one row of nn per point)."""
    Q = X[nn]
    m = Q.mean(axis=1)
    D = Q - m[:, None, :]
    cxx, cyy = (D[:, :, 0] ** 2).sum(1), (D[:, :, 1] ** 2).sum(1)
    cxy = (D[:, :, 0] * D[:, :, 1]).sum(1)
    ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)  # principal direction of the neighbours
    d = X[qi] - m
    return d[:, 1] * np.cos(ang) - d[:, 0] * np.sin(ang)


def _thickness(X, nn):
    return _thick_at(X, np.arange(len(X)), nn)


def _lead_scan(P, a, c, L0):
    """The lead around L0 that maps P onto the thinnest profile: the fewest grid cells
    occupied in the (rho, q) half-plane, coarse to fine (the normal-line estimate of
    the lead is only good to a few percent)."""
    e1 = _perp(a)
    e2 = np.cross(a, e1)
    v = P - c
    hh = v @ a
    rho = np.linalg.norm(v - np.outer(hh, a), axis=1)
    th = np.arctan2(v @ e2, v @ e1) / (2 * np.pi)
    ext = max(float(np.ptp(rho)), 1e-300)

    def occupied(L, cell):
        q = np.mod(hh - L * th, abs(L))
        key = np.floor(rho / cell).astype(np.int64) * 10_000_019 + np.floor(q / cell).astype(np.int64)
        return len(np.unique(key))

    best, span = L0, 0.15 * abs(L0)
    for div in (40.0, 200.0, 1000.0, 5000.0):
        Ls = best + np.linspace(-span, span, 121)
        Ls = Ls[np.abs(Ls) > 1e-3 * abs(L0)]
        best = float(Ls[int(np.argmin([occupied(x, ext / div) for x in Ls]))])
        span = 4 * span / 120
    return best


def screw_motion(P, N, tol, k: int = 9):
    """The screw motion (axis a through c, lead L) a smooth area is invariant under, or None.

    A helical sweep maps onto its profile curve in the (rho, q) half-plane, every turn on
    top of the others, so its vertices form a thin curve there. The first guess comes
    from the normal lines (a linear line complex); it is refined by making that curve
    as thin as possible: each vertex's distance from the line through its nearest
    profile neighbours, which come from other turns. Returns (a, c, L, thickness)."""
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree

    from .surfaces import helix_axis

    P, N = np.asarray(P, float), np.asarray(N, float)
    if len(P) < 60:
        return None
    scale = float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1).mean())) or 1.0
    h = helix_axis(P, N, scale, min_pitch=0.01)
    if h is None:
        return None
    a, c, ppr = h
    L = _lead_scan(P, a, c, 2 * np.pi * ppr)
    rng = np.random.default_rng(0)
    qi = np.arange(len(P)) if len(P) <= 6000 else rng.choice(len(P), 6000, replace=False)
    for _ in range(4):
        u = _perp(a)
        w = np.cross(a, u)
        a0, c0, L0 = a, c, L
        rho, q, _ = _coords(P, a0, c0, L0)
        X = np.c_[rho, q]
        _, nn = cKDTree(X).query(X[qi], k=k + 1)
        nn = nn[:, 1:]
        # points where two profile curves meet (a crest's corner) are never thin: left out
        th = np.abs(_thick_at(X, qi, nn))
        inl = th <= max(5 * float(np.median(th)), 0.02 * tol)
        qin, nin = qi[inl], nn[inl]
        e1 = _perp(a0)
        e2 = np.cross(a0, e1)
        v = P - c0
        th0 = np.arctan2(v @ e2, v @ e1)
        turn = np.round(((v @ a0) - L0 * th0 / (2 * np.pi) - q) / L0)  # each point's turn, kept fixed

        def unpack(x, a0=a0, c0=c0, L0=L0, u=u, w=w):
            aa = a0 + x[0] * u + x[1] * w
            return aa / np.linalg.norm(aa), c0 + scale * (x[2] * u + x[3] * w), L0 * (1 + x[4])

        def resid(x, th0=th0, turn=turn, qin=qin, nin=nin):
            aa, cc, LL = unpack(x)
            f1 = _perp(aa)
            f2 = np.cross(aa, f1)
            vv = P - cc
            hh = vv @ aa
            rr = np.linalg.norm(vv - np.outer(hh, aa), axis=1)
            # each point's angle stays continuous with the first guess (no jump at +-pi)
            tt = th0 + np.angle(np.exp(1j * (np.arctan2(vv @ f2, vv @ f1) - th0)))
            qq = hh - LL * tt / (2 * np.pi) - LL * turn
            return _thick_at(np.c_[rr, qq], qin, nin)

        try:
            sol = least_squares(resid, np.zeros(5), method="lm", xtol=1e-15, ftol=1e-15, max_nfev=200)
        except Exception:
            return None
        a, c, L = unpack(sol.x)
        if np.max(np.abs(sol.x)) < 1e-10:
            break
    rho, q, _ = _coords(P, a, c, L)
    X = np.c_[rho, q]
    _, nn = cKDTree(X).query(X, k=k + 1)
    return a, c, L, np.abs(_thickness(X, nn[:, 1:]))


def _order_profile(X, h):
    """Arc-length parameter of 2-D points along one open thin curve, or None when the
    points form no single open curve (a loop, a branch, a gap)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
    from scipy.spatial import cKDTree

    cells = np.floor((X - X.min(axis=0)) / h).astype(np.int64)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    cnt = np.bincount(inv)
    Cn = np.c_[np.bincount(inv, X[:, 0]), np.bincount(inv, X[:, 1])] / cnt[:, None]
    n = len(Cn)
    if n < 6:
        return None
    k = min(8, n - 1)
    d, j = cKDTree(Cn).query(Cn, k=k + 1)
    G = csr_matrix((d[:, 1:].ravel() + 1e-300, (np.repeat(np.arange(n), k), j[:, 1:].ravel())), shape=(n, n))
    T = minimum_spanning_tree(G)
    T = T + T.T
    if connected_components(T, directed=False)[0] != 1:
        return None
    a = int(np.argmax(dijkstra(T, indices=0)))
    da, pred = dijkstra(T, indices=a, return_predecessors=True)
    b = int(np.argmax(da))
    path = [b]
    while path[-1] != a and len(path) <= n:
        path.append(int(pred[path[-1]]))
    poly = Cn[path[::-1]]
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    length = float(seg.sum())
    if length <= 0 or np.linalg.norm(poly[0] - poly[-1]) < 0.05 * length:
        return None  # a closed profile (a coil's round section): not this kind of face
    if np.max(seg) > 0.35 * length:
        return None  # a gap: two curves (a mesh samples a profile at a few dozen rows only)
    # points onto the polyline (densified), by arc length
    cum = np.r_[0.0, np.cumsum(seg)]
    m = int(np.clip(length / (0.1 * h), 200, 200000))
    sd = np.linspace(0.0, length, m)
    i = np.clip(np.searchsorted(cum, sd, side="right") - 1, 0, len(seg) - 1)
    f = (sd - cum[i]) / np.maximum(seg[i], 1e-300)
    D = poly[i] + f[:, None] * (poly[i + 1] - poly[i])
    dist, jj = cKDTree(D).query(X)
    if np.max(dist) > 3 * h:
        return None  # a branch off the main curve
    # continuous arc length: slide along the local direction of the polyline
    tg = D[np.minimum(jj + 1, m - 1)] - D[np.maximum(jj - 1, 0)]
    tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-300)
    return sd[jj] + np.einsum("ij,ij->i", X - D[jj], tg)


def _profile_spline(X, t, tol):
    """Cubic B-spline profile (rho, q)(x), x ~ arc length, through 2-D points X ordered
    by t: (knots, coefficients (n, 2)). A mesh of a sweep repeats the same few profile
    positions (a few dozen rows) on every turn, so the fit goes through one point per
    distinct position. Rows close together (round a tight bend) can come out of the
    rough ordering swapped, and a spline through them in that order loops: a smooth
    first fit only sets the order (foot points along it), the second one is accurate."""
    from scipy.interpolate import BSpline, splprep
    from scipy.spatial import cKDTree

    key = np.round(X / max(0.05 * tol, 1e-300)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    cnt = np.bincount(inv).astype(float)
    R = np.c_[np.bincount(inv, X[:, 0]), np.bincount(inv, X[:, 1])] / cnt[:, None]
    u = np.bincount(inv, t) / cnt
    if len(R) < 4:
        return None
    o = np.argsort(u, kind="stable")
    R, u, w = R[o], u[o], np.sqrt(cnt[o])

    def fit(R, u, w, frac):
        uu = (u - u[0]) / max(float(u[-1] - u[0]), 1e-300)
        k = np.r_[True, np.diff(uu) > 1e-9]
        tck, _ = splprep([R[k, 0], R[k, 1]], u=uu[k], w=w[k], k=3, s=float((w[k] ** 2).sum()) * (frac * tol) ** 2)
        kn, (cr, cq), _ = tck
        cf = np.c_[np.asarray(cr)[:len(kn) - 4], np.asarray(cq)[:len(kn) - 4]]
        if not np.isfinite(cf).all():
            raise ValueError("spline")
        return BSpline(np.asarray(kn), cf, 3)

    def foot(spl, Q):
        xs = np.linspace(0.0, 1.0, 4001)
        _, j = cKDTree(spl(xs)).query(Q)
        x = xs[j]
        lo_, hi_ = xs[np.maximum(j - 1, 0)], xs[np.minimum(j + 1, 4000)]
        d1, d2 = spl.derivative(1), spl.derivative(2)
        for _ in range(5):
            d = spl(x) - Q
            g1, g2 = d1(x), d2(x)
            fp = (g1 * g1).sum(1) + (d * g2).sum(1)
            x = np.clip(x - (d * g1).sum(1) / np.where(fp > 1e-300, fp, 1e-300), lo_, hi_)
        return x, np.linalg.norm(spl(x) - Q, axis=1)

    try:
        spl = fit(R, u, w, 0.3)
    except Exception:
        return None
    xs = np.linspace(0.0, 1.0, 4001)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(spl(xs), axis=0), axis=1))]
    x, _ = foot(spl, R)
    u2 = np.interp(x, xs, arc)
    o = np.argsort(u2, kind="stable")
    R, u2, w = R[o], u2[o], w[o]
    # one point per position along the curve: the same row on other turns lands a hair
    # to either side of it, and a spline through those in turn zig-zags into loops
    g = np.r_[0, np.cumsum(np.diff(u2) > tol)]
    ww = w ** 2
    sw = np.bincount(g, ww)
    R = np.c_[np.bincount(g, ww * R[:, 0]), np.bincount(g, ww * R[:, 1])] / sw[:, None]
    u2 = np.bincount(g, ww * u2) / sw
    w = np.sqrt(sw)
    if len(R) < 4:
        return None
    length = float(arc[-1])
    best = None
    for frac in (0.1, 0.03, 0.01):
        try:
            spl2 = fit(R, u2, w, frac)
        except Exception:
            continue
        _, res = foot(spl2, R)
        if best is None or res.max() < best[1]:
            best = (spl2, float(res.max()))
        if res.max() <= 0.3 * tol:
            break
    if best is None or length <= 0:
        return None
    spl2 = best[0]
    return np.asarray(spl2.t) * length, np.asarray(spl2.c)


def fit_helix_fixed(P, a, c, L, tol, frame=None):
    """A helical sweep with a given screw motion through P, its profile a cubic B-spline
    of arc length (any open curve: a round groove, a crest). None when the points form
    no single open profile curve."""
    P = np.asarray(P, float)
    if len(P) < 30:
        return None
    rho, q, q0 = _coords(P, a, c, L)
    X = np.c_[rho, q]
    ext = float(np.ptp(X, axis=0).max())
    if ext <= 0 or np.ptp(q) > 0.5 * abs(L):
        # a profile across half a lead or more would cover itself when swept
        return None
    t = None
    for div in (300.0, 120.0, 50.0):
        t = _order_profile(X, ext / div)
        if t is not None:
            break
    if t is None:
        return None
    th = np.arctan2((P - c) @ np.cross(a, _perp(a)), (P - c) @ _perp(a))
    hh = (P - c) @ a
    s = th + 2 * np.pi * np.round((hh - L * th / (2 * np.pi) - q0) / L)

    def build(sel):
        fit = _profile_spline(X[sel], t[sel], tol)
        if fit is None:
            return None
        kn, cf = fit
        lo, hi = float(kn[0]), float(kn[-1])
        return HelixSurface(frame, a, c, L, "spline", [0.0], 0.5 * (lo + hi), max(0.5 * (hi - lo), 1e-300), q0,
                            (s.min(), s.max()), (lo, hi), spline=(kn, cf[:, 0], cf[:, 1]))

    sel = np.ones(len(X), bool)
    hx = build(sel)
    if hx is None:
        return None
    dev, xg = _profile_gaps(hx, X, tol)
    if len(dev) and not np.all(dev <= 0):
        # the curve loops out across one gap to reach a few stray rows at an end (a
        # neighbouring face's corner): fit without the rows beyond it; the caller drops
        # the triangles that then miss
        i = int(np.argmax(dev))
        xx = hx._nearest_x(X[:, 0], X[:, 1])
        beyond = xx > xg[i] if i >= len(dev) / 2 else xx < xg[i]
        if beyond.sum() > 0.1 * len(X):
            return None
        sel = ~beyond
        hx = build(sel)
        if hx is None:
            return None
        dev, _ = _profile_gaps(hx, X[sel], tol)
        if len(dev) and not np.all(dev <= 0):
            return None
    if _profile_sharp(hx):
        return None
    r = hx.residual(P)
    hx.rms = float(np.sqrt(np.mean(r ** 2)))
    hx.max_res = float(np.max(r))
    return hx


def _profile_sharp(hx, max_turn_deg: float = 60.0, window: float = 0.02) -> bool:
    """A corner in the profile: the curve turns more than max_turn_deg within a stretch of
    `window` of its length. Two faces meeting at a crest edge are two faces; swept as
    one, the profile folds back on itself there and the surface overlaps itself. A
    round groove's tight bend turns gradually and passes."""
    lo, hi = hx.x_rng
    xs = np.linspace(lo, hi, 2001)
    d = hx._sp(xs, 1)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    seg = np.linalg.norm(np.diff(hx._sp(xs), axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    if arc[-1] <= 0:
        return False
    j = np.searchsorted(arc, arc + window * arc[-1])
    j = np.minimum(j, len(xs) - 1)
    return bool(np.max(np.abs(ang[j] - ang)) > np.radians(max_turn_deg))


def _profile_gaps(hx, X, tol):
    """How far the curve strays, between each two neighbouring profile positions (rows of
    mesh vertices), beyond what a smooth bend allows (positive = too far), and the gaps'
    mid parameters. A smooth profile bends a little between rows; a spline looping out
    to reach a stray row (a crest's corner) strays by about the gap itself."""
    x = np.sort(hx._nearest_x(X[:, 0], X[:, 1]))
    span = float(x[-1] - x[0]) if len(x) else 0.0
    x = x[np.r_[True, np.diff(x) > 1e-4 * max(span, 1e-300)]] if len(x) else x
    if len(x) < 2:
        return np.zeros(0), np.zeros(0)
    x0, x1 = x[:-1], x[1:]
    t = np.linspace(0.0, 1.0, 9)[1:-1]
    xs = x0[:, None] + t[None, :] * (x1 - x0)[:, None]
    A = hx._sp(x0)
    B = hx._sp(x1)
    S = hx._sp(xs.ravel()).reshape(len(x0), len(t), 2)
    ab = B - A
    L2 = np.maximum((ab * ab).sum(1), 1e-300)
    w = np.clip(((S - A[:, None, :]) * ab[:, None, :]).sum(-1) / L2[:, None], 0.0, 1.0)
    dev = np.linalg.norm(S - (A[:, None, :] + w[..., None] * ab[:, None, :]), axis=-1).max(axis=1)
    return dev - (0.15 * np.sqrt(L2) + 0.5 * tol), 0.5 * (x0 + x1)
