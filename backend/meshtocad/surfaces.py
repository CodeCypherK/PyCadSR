"""Coordinate frames, equation evaluation and surface reconstruction.

Each region gets a local frame in which its surface is a *function* that symbolic
regression can discover:

    height       w = f(u, v)            planes, gentle freeform patches
    cylindrical  r = f(theta, h)        cylinders (r=R), cones (r=a*h+b), revolved shapes
    spherical    r = f(theta, lat)      spheres (r=R), blobs

All quantities fed to PySR are normalised by the region scale ``s`` so constants
stay near 1. ``to_param``/``from_param`` convert between world points and
(X, y) samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .segmentation import Region, _perp, _unit, fit_cylinder

VAR_NAMES = {
    "height": ["u", "v"],
    "cylindrical": ["theta", "h"],
    "spherical": ["theta", "lat"],
}


@dataclass
class Frame:
    kind: str
    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    e3: np.ndarray
    scale: float

    @property
    def var_names(self) -> list[str]:
        return VAR_NAMES[self.kind]

    @property
    def R(self) -> np.ndarray:
        return np.vstack([self.e1, self.e2, self.e3])

    def local(self, P: np.ndarray) -> np.ndarray:
        return (P - self.origin) @ self.R.T

    def to_param(self, P: np.ndarray):
        L = self.local(P) / self.scale
        x, y, z = L[:, 0], L[:, 1], L[:, 2]
        if self.kind == "height":
            return np.c_[x, y], z
        if self.kind == "cylindrical":
            return np.c_[np.arctan2(y, x), z], np.hypot(x, y)
        r = np.linalg.norm(L, axis=1)
        lat = np.arcsin(np.clip(z / np.maximum(r, 1e-300), -1, 1))
        return np.c_[np.arctan2(y, x), lat], r

    def from_param(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        a, b = X[:, 0], X[:, 1]
        if self.kind == "height":
            L = np.c_[a, b, y]
        elif self.kind == "cylindrical":
            L = np.c_[y * np.cos(a), y * np.sin(a), b]
        else:
            L = np.c_[y * np.cos(b) * np.cos(a), y * np.cos(b) * np.sin(a), y * np.sin(b)]
        return self.origin + (L * self.scale) @ self.R

    def to_json(self) -> dict:
        return {"kind": self.kind, "origin": self.origin, "e1": self.e1, "e2": self.e2,
                "e3": self.e3, "scale": self.scale, "var_names": self.var_names}


# ----------------------------------------------------------- frame builders

def _mean_dir(D: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, float]:
    m = D.mean(axis=0)
    norm = float(np.linalg.norm(m))
    if norm < 0.25:
        return _perp(fallback), norm
    return m / norm, norm


def _height_frame(P, N, normal=None):
    c = P.mean(axis=0)
    if normal is None:
        normal = _unit(N.mean(axis=0))
        # A freeform patch's mean normal tilts with its shape (a sine wave leans one
        # way). If it is within a few degrees of a world axis, the designer almost
        # surely meant that axis; a 0.4 degree tilt is 0.2 mm across a 60 mm part.
        k = int(np.argmax(np.abs(normal)))
        if abs(normal[k]) > np.cos(np.radians(4)):
            normal = np.eye(3)[k] * np.sign(normal[k])
    Q = (P - c) - np.outer((P - c) @ normal, normal)
    axes = np.eye(3)
    if np.abs(axes @ normal).max() > np.cos(np.radians(15)):
        # Axis-aligned surface (typical for CAD): align u/v with world axes so the
        # discovered equation reads in the model's own directions.
        proj = axes - np.outer(axes @ normal, normal)
        k = int(np.argmax(np.linalg.norm(proj, axis=1)))
        e1 = _unit(proj[k])
    else:
        _, _, vt = np.linalg.svd(Q, full_matrices=False)
        e1 = _unit(vt[0] - (vt[0] @ normal) * normal)
    e2 = np.cross(normal, e1)
    s = float(np.sqrt(np.mean(np.sum(Q ** 2, axis=1)))) or 1.0
    # put the origin on the surface (median height) so the fit starts near w=0
    w = (P - c) @ normal
    origin = c + np.median(w) * normal
    return Frame("height", origin, e1, e2, normal, s)


def _cyl_frame(P, axis, point, radius, outward_hint=None):
    a = _unit(axis)
    c = P.mean(axis=0)
    origin = point + ((c - point) @ a) * a
    rad = _unit((P - origin) - np.outer((P - origin) @ a, a))
    e1, strength = _mean_dir(rad, a)
    if strength < 0.25:
        # Full turn: no natural direction. Use a deterministic one (a world axis
        # projected off the axis) so coaxial regions share the same seam angle.
        axes = np.eye(3)
        k = int(np.argmin(np.abs(axes @ a)))
        e1 = axes[k]
    e1 = _unit(e1 - (e1 @ a) * a)
    e2 = np.cross(a, e1)
    return Frame("cylindrical", origin, e1, e2, a, float(radius) or 1.0)


def _sph_frame(P, center, radius):
    rad = _unit(P - center)
    e1, strength = _mean_dir(rad, np.array([0, 0, 1.0]))
    if strength < 0.25:
        # Closed all round (a blob, an ellipsoid): no natural direction, so use the
        # world axes - pole on Z, theta from X - which is how such shapes are modelled.
        return Frame("spherical", center.copy(), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                     np.array([0, 0, 1.0]), float(radius) or 1.0)
    # e3: direction of least spread among the radial directions, orthogonal to e1
    rc = rad - np.outer(rad @ e1, e1)
    _, V = np.linalg.eigh(rc.T @ rc)
    e3 = None
    for k in range(3):
        cand = V[:, k] - (V[:, k] @ e1) * e1
        if np.linalg.norm(cand) > 0.5:
            e3 = _unit(cand)
            break
    if e3 is None:
        e3 = _perp(e1)
    e2 = np.cross(e3, e1)
    return Frame("spherical", center.copy(), e1, e2, e3, float(radius) or 1.0)


def frame_for_region(region: Region, P: np.ndarray, N: np.ndarray, kind: str = "auto") -> Frame:
    Pr, Nr = P[region.idx], N[region.idx]
    prm = region.params
    prim = region.prim
    rev = None
    if kind == "auto":
        if prim == "plane":
            kind = "height"
        elif prim in ("cylinder", "cone", "torus"):
            kind = "cylindrical"
        elif prim == "sphere":
            kind = "spherical"
        elif "normal" in prm:
            kind = "height"  # one side of a split two-sided sheet
        else:
            kind = _freeform_kind(Nr)
            if kind != "height":
                # surfaces of revolution (fillets around an axis, turned profiles): r = f(h)
                extent = float(np.linalg.norm(Pr.max(0) - Pr.min(0))) or 1.0
                rev = revolution_axis(Pr, Nr, extent)
                if rev is None:
                    # a thread: helical, single-valued as r = f(theta, h) around its axis
                    hx = helix_axis(Pr, Nr, extent)
                    if hx is not None:
                        rev = (hx[0], hx[1])
                if rev is not None:
                    kind = "cylindrical"
    elif kind == "cylindrical" and prim == "freeform":
        extent = float(np.linalg.norm(Pr.max(0) - Pr.min(0))) or 1.0
        rev = revolution_axis(Pr, Nr, extent)
    if rev is not None and kind == "cylindrical":
        a, q = rev
        v = Pr - q
        rad = np.linalg.norm(v - np.outer(v @ a, a), axis=1)
        return _cyl_frame(Pr, a, q, float(np.mean(rad)))

    if kind == "height":
        normal = None
        if prim == "plane" or (prim == "freeform" and "normal" in prm):
            normal = prm["normal"] if (Nr @ prm["normal"]).mean() >= 0 else -prm["normal"]
        return _height_frame(Pr, Nr, normal)
    if kind == "cylindrical":
        if prim == "cylinder":
            return _cyl_frame(Pr, prm["axis"], prm["point"], prm["radius"])
        if prim == "torus":
            return _cyl_frame(Pr, prm["axis"], prm["center"], prm["R"])
        if prim == "cone":
            a = prm["axis"]
            f = _cyl_frame(Pr, a, prm["apex"], 1.0)
            h0 = (f.origin - prm["apex"]) @ a
            f.scale = float(abs(h0) * np.tan(prm["half_angle"])) or 1.0
            return f
        cyl = fit_cylinder(Pr, Nr)
        if prim == "freeform":
            # A closed freeform tube (blade, duct): centre on the section centroid,
            # which is inside the section, so every ray from the axis meets the
            # surface once. A circle fit to a long thin section lands beside it.
            a = cyl["axis"]
            c = Pr.mean(axis=0)
            v = Pr - c
            rad = float(np.mean(np.linalg.norm(v - np.outer(v @ a, a), axis=1)))
            return _cyl_frame(Pr, a, c, rad)
        return _cyl_frame(Pr, cyl["axis"], cyl["point"], cyl["radius"])
    # spherical
    if prim == "sphere":
        return _sph_frame(Pr, prm["center"], prm["radius"])
    center = _normal_lines_center(Pr, Nr)
    radius = float(np.mean(np.linalg.norm(Pr - center, axis=1)))
    return _sph_frame(Pr, center, radius)


def helix_axis(P: np.ndarray, N: np.ndarray, scale: float, min_pitch: float = 0.05):
    """Axis of a helical surface (a thread), or None.

    The normal lines of a helical surface also lie in a linear line complex, but one
    with a pitch (the advance per radian); its axis is the helix axis. Around it a
    thread is single-valued as r = f(theta, h).
    """
    c0 = P.mean(axis=0)
    Q = (P - c0) / scale
    V = np.c_[np.cross(Q, N), N]
    w, E = np.linalg.eigh(V.T @ V / len(V))
    c, cb = E[:3, 0], E[3:, 0]
    nc = float(np.linalg.norm(c))
    if nc < 1e-6 or w[1] < 1e-5 or w[0] / w[1] > 0.05:
        return None
    pitch = float(c @ cb) / nc ** 2
    if abs(pitch) <= min_pitch:  # no pitch: a plain surface of revolution
        return None
    return c / nc, c0 + np.cross(c, cb) / nc ** 2 * scale, pitch * scale


def revolution_axis(P: np.ndarray, N: np.ndarray, scale: float):
    """Axis of a surface of revolution, or None.

    Every normal line of a surface of revolution meets its axis, so the normals'
    Plücker coordinates (n, p x n) lie in a linear line complex whose coordinates
    are the smallest eigenvector of a 6x6 matrix (Pottmann & Randrup 1998).
    Coordinates are centred and scaled first for conditioning.
    """
    c0 = P.mean(axis=0)
    Q = (P - c0) / scale
    V = np.c_[np.cross(Q, N), N]
    w, E = np.linalg.eigh(V.T @ V / len(V))
    c, cb = E[:3, 0], E[3:, 0]
    nc = float(np.linalg.norm(c))
    if nc < 1e-6:
        return None
    pitch = float(c @ cb) / nc ** 2
    a = c / nc
    point = np.cross(c, cb) / nc ** 2
    # residual: mean distance between normal lines and the axis, relative to size
    if w[1] < 1e-5 or w[0] / w[1] > 0.05 or abs(pitch) > 0.05:
        return None
    return a, c0 + point * scale


def _freeform_kind(Nr: np.ndarray) -> str:
    m = Nr.mean(axis=0)
    if np.linalg.norm(m) > 0.5 and (Nr @ _unit(m)).min() > np.cos(np.radians(80)):
        return "height"
    w, _ = np.linalg.eigh(Nr.T @ Nr / len(Nr))
    return "cylindrical" if w[0] < 0.05 else "spherical"


def _normal_lines_center(P, N):
    M = np.zeros((3, 3))
    b = np.zeros(3)
    for p, n in zip(P[:: max(1, len(P) // 4000)], N[:: max(1, len(P) // 4000)]):
        A = np.eye(3) - np.outer(n, n)
        M += A
        b += A @ p
    return np.linalg.lstsq(M, b, rcond=None)[0]


# ------------------------------------------------------- equation handling

class Equation:
    """A discovered expression y = f(x0, x1) in normalised frame coordinates."""

    def __init__(self, expr: sp.Expr, frame: Frame):
        self.frame = frame
        self.symbols = sp.symbols(frame.var_names)
        names = {str(s): s for s in self.symbols}
        if not isinstance(expr, sp.Basic):
            expr = sp.sympify(str(expr), locals=names)
        # PySR may build its own Symbol objects; rebind by name.
        expr = expr.subs({s: names[str(s)] for s in expr.free_symbols if str(s) in names})
        self.expr = expr
        self._f = sp.lambdify(self.symbols, expr, modules="numpy")

    def __getstate__(self):  # the compiled function is rebuilt on load (sessions can be saved)
        state = self.__dict__.copy()
        state.pop("_f", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._f = sp.lambdify(self.symbols, self.expr, modules="numpy")

    def __call__(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            y = self._f(X[:, 0], X[:, 1])
        y = np.asarray(y, dtype=float)
        if y.ndim == 0:
            y = np.full(len(X), float(y))
        return y

    def latex(self) -> str:
        return sp.latex(_round_floats(self.expr, 5))

    # world-unit version of the local equation, e.g. r = 12.5 for a cylinder
    def local_world(self) -> tuple[str, str]:
        s = self.frame.scale
        a, b = self.symbols
        k = self.frame.kind
        if k == "height":
            out = sp.Symbol("w")
            e = s * self.expr.subs({a: a / s, b: b / s}, simultaneous=True)
        elif k == "cylindrical":
            out = sp.Symbol("r")
            e = s * self.expr.subs({b: b / s})
        else:
            out = sp.Symbol("r")
            e = s * self.expr
        # Only cheap rewrites here: sympy's simplify/trigsimp can run for minutes on
        # the kind of trig/sqrt expressions PySR produces, and this runs every iteration.
        if sp.count_ops(e) < 30:
            e = sp.expand_mul(e)
        e = _round_floats(e, 6)
        return f"{out} = {e}", f"{sp.latex(out)} = {sp.latex(e)}"


def polish(eq: Equation, X: np.ndarray, y: np.ndarray, max_points: int = 6000) -> Equation:
    """Refine the numeric constants of a discovered expression on all region points.

    PySR optimises constants on its training subsample; a final least-squares
    pass over the whole region keeps the structure and tightens the numbers.
    """
    from scipy.optimize import least_squares

    floats = sorted(eq.expr.atoms(sp.Float), key=lambda f: float(f))
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if not floats or len(y) <= len(floats):
        return eq
    if len(y) > max_points:
        sel = np.random.default_rng(0).choice(len(y), max_points, replace=False)
        X, y = X[sel], y[sel]
    params = sp.symbols(f"_p0:{len(floats)}")
    template = eq.expr.xreplace(dict(zip(floats, params)))
    f = sp.lambdify([*eq.symbols, *params], template, modules="numpy")

    def resid(p):
        with np.errstate(all="ignore"):
            r = np.broadcast_to(np.asarray(f(X[:, 0], X[:, 1], *p), float), y.shape) - y
        return np.where(np.isfinite(r), r, 1e3)

    x0 = np.array([float(v) for v in floats])
    try:
        sol = least_squares(resid, x0, max_nfev=200 * len(x0))
    except Exception:
        return eq
    if not np.all(np.isfinite(sol.x)) or np.sum(resid(sol.x) ** 2) >= np.sum(resid(x0) ** 2):
        return eq
    return Equation(template.xreplace({p: sp.Float(v, 10) for p, v in zip(params, sol.x)}), eq.frame)


def _round_floats(expr: sp.Expr, digits: int) -> sp.Expr:
    """Round constants for display; float32 noise like 6.9e-33 becomes 0."""
    return expr.xreplace({f: (sp.Integer(0) if abs(float(f)) < 1e-9 else sp.Float(f, digits))
                          for f in expr.atoms(sp.Float)})


def classify(eq: Equation, rms_rel: float) -> dict:
    """Recognise analytic CAD primitives from the discovered expression."""
    expr = eq.expr
    free = expr.free_symbols
    k = eq.frame.kind
    s = eq.frame.scale
    a_sym, b_sym = eq.symbols
    try:
        if not free and expr.is_number:
            c = float(expr)
            if k == "height":
                return {"type": "plane", "offset": c * s}
            if k == "cylindrical":
                return {"type": "cylinder", "radius": c * s}
            return {"type": "sphere", "radius": c * s}
        if k == "cylindrical" and free == {b_sym}:
            try:
                poly = sp.Poly(expr, b_sym)
                if poly.degree() == 1:
                    slope, c0 = [float(x) for x in poly.all_coeffs()]
                    return {"type": "cone", "slope": slope, "radius0": c0 * s}
            except sp.PolynomialError:
                pass
            return {"type": "revolve"}
        if k == "height" and free <= {a_sym, b_sym}:
            poly = sp.Poly(expr, a_sym, b_sym)
            if poly.total_degree() == 1:
                ca = float(poly.coeff_monomial(a_sym))
                cb = float(poly.coeff_monomial(b_sym))
                c0 = float(poly.coeff_monomial(1))
                return {"type": "plane", "slope_u": ca, "slope_v": cb, "offset": c0 * s}
    except (sp.PolynomialError, TypeError, ValueError):
        pass
    return {"type": "freeform"}


class DataSurface:
    """Smoothing spline y = g(x0, x1) fitted to the region's own samples.

    Used only for CAD output when symbolic regression did not reach the
    tolerance: the exported surface still follows the data. Same interface as
    Equation where the exporter needs it.
    """

    def __init__(self, frame: Frame, X: np.ndarray, y: np.ndarray, noise: float, helpers=None):
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        # Helper points (on the mesh triangles between sparse vertices) shape the patch
        # between the vertices; it is still judged by the vertices alone.
        if helpers is not None and len(helpers[1]):
            Xh, yh = helpers
            okh = np.isfinite(Xh).all(axis=1) & np.isfinite(yh)
            Xh, yh = Xh[okh], yh[okh]
        else:
            Xh, yh = np.zeros((0, 2)), np.zeros(0)
        if len(y) > 40000:
            k = np.random.default_rng(0).choice(len(y), 40000, replace=False)
            X, y = X[k], y[k]
        if frame.kind in ("cylindrical", "spherical"):
            srt = np.sort(X[:, 0])
            if np.diff(np.r_[srt, srt[0] + 2 * np.pi]).max() < np.radians(20):
                # full turn: copy samples across +-pi so both sides of the seam agree
                lo, hi = X[:, 0] < -np.pi + 0.6, X[:, 0] > np.pi - 0.6
                X = np.r_[X, X[lo] + [2 * np.pi, 0], X[hi] - [2 * np.pi, 0]]
                y = np.r_[y, y[lo], y[hi]]
        self.frame = frame
        self.symbols = sp.symbols(frame.var_names)
        self.expr = sp.Symbol("data_fit")
        # Robust: interpolate the samples onto a regular parameter grid, then fit an
        # interpolating spline through the grid (scattered smoothing splines are
        # numerically fragile at tight tolerances and around creases).
        from scipy.interpolate import RectBivariateSpline, griddata

        a, b = X[:, 0], X[:, 1]
        if len(yh):
            # a helper only where no vertex is close: next to a vertex it would just
            # pull the patch onto the flat triangle, off the true surface
            from scipy.spatial import cKDTree
            span = max(np.ptp(a), np.ptp(b), 1e-12)
            dv, _ = cKDTree(X).query(Xh)
            far = dv > 0.02 * span
            Xh, yh = Xh[far], yh[far]
        ga_all = np.r_[a, Xh[:, 0]]
        gb_all = np.r_[b, Xh[:, 1]]
        gy_all = np.r_[y, yh]
        ea, eb = max(np.ptp(a), 1e-6), max(np.ptp(b), 1e-6)
        n = max(400, min(40000, len(y) // 2))
        na = int(np.clip(np.sqrt(n * ea / eb), 8, 240))
        nb = int(np.clip(n / na, 8, 240))
        # The grid is refined until the spline follows its own samples to the noise
        # level: sized by the sample count alone, a curved blend of 500 vertices got a
        # 20 x 20 grid and missed them by 0.5 mm, and its true edges then sat that far
        # off the patch.
        # A finer grid that still misses the samples by more than the tolerance (4x the
        # noise, rms) is no surface to trust between them (a wavy patch over a flat panel):
        # then the first grid stays.
        first = None
        for _ in range(4):
            ga, gb = np.linspace(a.min(), a.max(), na), np.linspace(b.min(), b.max(), nb)
            A, B = np.meshgrid(ga, gb, indexing="ij")
            Z = griddata((ga_all, gb_all), gy_all, (A, B), method="linear")
            hole = ~np.isfinite(Z)
            if hole.any():
                Z[hole] = griddata((ga_all, gb_all), gy_all, (A[hole], B[hole]), method="nearest")
            self._spl = RectBivariateSpline(ga, gb, Z, kx=3, ky=3, s=0)
            first = first or self._spl
            err = float(np.sqrt(np.mean((self._spl.ev(a, b) - y) ** 2))) if len(y) else 0.0
            if err <= noise:
                break
            if na >= 240 and nb >= 240:
                if err > 4 * noise:
                    self._spl = first
                break
            na, nb = min(240, 2 * na), min(240, 2 * nb)
        else:
            if err > 4 * noise:
                self._spl = first

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._spl.ev(X[:, 0], X[:, 1]), float)

    def latex(self) -> str:
        return r"\text{data-fitted spline}"

    def local_world(self) -> tuple[str, str]:
        return "(data-fitted B-spline; no equation within tolerance)", r"\text{data-fitted B-spline}"


def seed_guesses(region: Region, frame: Frame, X: np.ndarray, y: np.ndarray, unary: list[str]) -> list[str]:
    """Starting expressions for PySR taken from the segmentation's primitive hypothesis.

    PySR keeps a seed only if nothing simpler or more accurate beats it, so this
    does not decide the answer; it saves the search from rediscovering a line or
    a constant from scratch (and from stopping early on an odd near-miss).
    """
    a, b = frame.var_names
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < 10:
        return []
    g = lambda v: f"{float(v):.8g}"
    prim, k = region.prim, frame.kind
    if k == "height" and prim == "plane":
        return [g(np.median(y))]
    if k in ("cylindrical", "spherical") and prim in ("cylinder", "sphere"):
        return [g(np.median(y))]
    if k == "cylindrical" and prim == "cone":
        m, c = np.linalg.lstsq(np.c_[X[:, 1], np.ones(len(y))], y, rcond=None)[0]
        return [f"{g(m)} * {b} + {g(c)}"]
    if k == "cylindrical" and prim == "torus" and "sqrt" in unary:
        p = region.params
        s = frame.scale
        hc = float((p["center"] - frame.origin) @ frame.e3) / s
        sign = "-" if np.median(y) < p["R"] / s else "+"
        # Samples at the tube's extreme sit a hair outside it; one negative sqrt
        # argument makes the seed's loss infinite and PySR throws it away.
        rr = max((p["r"] / s) ** 2, float(np.max((X[:, 1] - hc) ** 2))) + 1e-7
        return [f"{g(p['R'] / s)} {sign} sqrt({g(rr)} - ({b} - {g(hc)}) * ({b} - {g(hc)}))"]
    return []


def classify_region(eq: Equation, rms: float, region: Region, diag: float) -> dict:
    """classify() plus primitives that need the region's geometry.

    A torus (fillet around an axis) is discovered as r = R - sqrt(r0^2 - (h - hc)^2)
    in its cylindrical frame. When the discovered equation reproduces the data that
    closely, the region is exported as an exact torus.
    """
    cls = classify(eq, rms / diag)
    if region.prim == "torus" and cls["type"] in ("revolve", "freeform", "cylinder") and rms <= 1e-3 * diag:
        p = region.params
        cls = {"type": "torus", "R": float(p["R"]), "r": float(p["r"]),
               "center": [float(x) for x in p["center"]], "axis": [float(x) for x in p["axis"]]}
    return cls


def torus_equation(region: Region, frame: Frame, P: np.ndarray) -> Equation:
    """Exact torus profile r(h) in the region's cylindrical frame (used when PySR was not run)."""
    p = region.params
    s = frame.scale
    hc = float((p["center"] - frame.origin) @ frame.e3) / s
    v = P - p["center"]
    rho = np.linalg.norm(v - np.outer(v @ p["axis"], p["axis"]), axis=1)
    sign = -1 if np.median(rho) < p["R"] else 1
    th, h = sp.symbols(frame.var_names)
    # cover samples a hair beyond the tube's extreme (same reason as the PySR seed)
    hs = frame.to_param(P)[0][:, 1]
    rr = max((p["r"] / s) ** 2, float(np.max((hs - hc) ** 2))) + 1e-7
    expr = sp.Float(p["R"] / s, 10) + sign * sp.sqrt(sp.Float(rr, 10) - (h - sp.Float(hc, 10)) ** 2)
    return Equation(expr, frame)


# --------------------------------------------------------- reconstruction

class ExactTorus:
    """A torus found exactly from the mesh (a fillet around an axis).

    Its equation in a cylindrical frame, r = R +- sqrt(r0^2 - (h - hc)^2), covers
    one half of the tube only, so a face wrapping over the top of the tube is not a
    function there. This keeps the torus itself: points project onto the true tube.
    """

    def __init__(self, frame, center, axis, R, r):
        self.frame = frame
        self.center = np.asarray(center, float)
        a = np.asarray(axis, float)
        self.axis = a / np.linalg.norm(a)
        self.R, self.r = float(R), float(r)
        self.symbols = sp.symbols(frame.var_names)
        self.expr = sp.Symbol("torus")

    def __call__(self, X: np.ndarray) -> np.ndarray:  # outer half, for frame-based callers
        hc = float((self.center - self.frame.origin) @ self.frame.e3) / self.frame.scale
        s = self.frame.scale
        with np.errstate(all="ignore"):
            return self.R / s + np.sqrt((self.r / s) ** 2 - (X[:, 1] - hc) ** 2)

    def project_points(self, P: np.ndarray) -> np.ndarray:
        v = P - self.center
        h = v @ self.axis
        rad = v - np.outer(h, self.axis)
        rho = np.maximum(np.linalg.norm(rad, axis=1), 1e-300)
        core = self.center + rad / rho[:, None] * self.R
        w = P - core
        wn = np.maximum(np.linalg.norm(w, axis=1), 1e-300)
        return core + w / wn[:, None] * self.r

    def normals(self, P: np.ndarray) -> np.ndarray:
        q = self.project_points(P)
        v = q - self.center
        h = v @ self.axis
        rad = v - np.outer(h, self.axis)
        rho = np.maximum(np.linalg.norm(rad, axis=1), 1e-300)
        core = self.center + rad / rho[:, None] * self.R
        n = q - core
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    def latex(self) -> str:
        return rf"\left(\sqrt{{\rho^2}} - {self.R:.6g}\right)^2 + (h - h_c)^2 = {self.r:.6g}^2"

    def local_world(self) -> tuple[str, str]:
        return (f"torus R = {self.R:.6g}, r = {self.r:.6g}",
                rf"\text{{torus }} R = {self.R:.6g},\ r = {self.r:.6g}")


def project(eq: Equation, P: np.ndarray) -> np.ndarray:
    """Move points onto the fitted surface along the frame's 'height' direction."""
    if hasattr(eq, "project_points"):  # an exact surface that projects itself (a torus)
        return eq.project_points(P)
    X, _ = eq.frame.to_param(P)
    y = eq(X)
    out = eq.frame.from_param(X, y)
    bad = ~np.isfinite(out).all(axis=1)
    out[bad] = P[bad]
    return out


def param_domain(frame: Frame, X: np.ndarray, target_cells: int = 2400):
    """Grid over the region's parameter domain plus an occupancy mask."""
    a, b = X[:, 0], X[:, 1]
    a0, a1 = float(a.min()), float(a.max())
    b0, b1 = float(b.min()), float(b.max())
    wrap = False
    if frame.kind in ("cylindrical", "spherical"):
        # Does the region go all the way around? Check for an angular gap.
        srt = np.sort(a)
        gaps = np.diff(np.r_[srt, srt[0] + 2 * np.pi])
        if gaps.max() < np.radians(20):
            a0, a1, wrap = -np.pi, np.pi, True
    ea, eb = max(a1 - a0, 1e-6), max(b1 - b0, 1e-6)
    na = int(np.clip(np.sqrt(target_cells * ea / eb), 4, 160))
    nb = int(np.clip(target_cells / na, 4, 160))
    pad_a = 0 if wrap else ea / na
    ga = np.linspace(a0 - pad_a, a1 + pad_a, na + 1)
    gb = np.linspace(b0 - eb / nb, b1 + eb / nb, nb + 1)
    ia = np.clip(np.searchsorted(ga, a) - 1, 0, na - 1)
    ib = np.clip(np.searchsorted(gb, b) - 1, 0, nb - 1)
    occ = np.zeros((na, nb), bool)
    occ[ia, ib] = True
    # close small gaps between samples
    from scipy.ndimage import binary_closing, binary_dilation
    occ = binary_closing(occ, iterations=1) | occ
    occ = binary_dilation(occ, iterations=1) if occ.sum() < 0.5 * occ.size else occ
    return ga, gb, occ, wrap


def preview_mesh(eq: Equation, X: np.ndarray, target_cells: int = 2400):
    """Triangulated surface generated from the equation over the region's domain."""
    ga, gb, occ, wrap = param_domain(eq.frame, X, target_cells)
    A, B = np.meshgrid(ga, gb, indexing="ij")
    G = np.c_[A.ravel(), B.ravel()]
    y = eq(G)
    V = eq.frame.from_param(G, y)
    nb1 = len(gb)
    ii, jj = np.nonzero(occ)
    v00 = ii * nb1 + jj
    v10 = (ii + 1) * nb1 + jj
    v01 = ii * nb1 + jj + 1
    v11 = (ii + 1) * nb1 + jj + 1
    F = np.r_[np.c_[v00, v10, v11], np.c_[v00, v11, v01]]
    ok = np.isfinite(V).all(axis=1)
    F = F[ok[F].all(axis=1)]
    used = np.unique(F)
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]
