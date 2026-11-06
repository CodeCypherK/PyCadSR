"""CAD output built from the discovered surface equations.

* STEP (AP214) B-rep: one ADVANCED_FACE per region. Analytic surfaces (PLANE,
  CYLINDRICAL_SURFACE, CONICAL_SURFACE, SPHERICAL_SURFACE) when the discovered
  equation reduces to one, otherwise a B_SPLINE_SURFACE sampled from the
  equation. Faces are trimmed by edges that run along the boundary between
  neighbouring regions; boundary points are projected onto both adjacent
  surfaces so shared edges sit on their intersection.
* STL: the input mesh snapped onto the fitted surfaces, and a mesh generated
  purely from the equations.
* JSON: frames, equations, classification and error for every region.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sympy as sp
import trimesh
from scipy.interpolate import RectBivariateSpline, splev, splprep
from scipy.spatial import cKDTree

from .segmentation import _perp
from .surfaces import (DataSurface, Equation, ExactTorus, Frame, _height_frame, classify, classify_region, param_domain, polish,
                       preview_mesh, project, torus_equation)


# ===================================================================== fits

class Fit:
    def __init__(self, rid, eq: Equation, cls: dict, discovered: bool, rms: float):
        self.rid, self.eq, self.cls, self.discovered, self.rms = rid, eq, cls, discovered, rms


def fallback_equation(prim: str, frame, X: np.ndarray, y: np.ndarray) -> Equation:
    """Constant / linear / quadratic least-squares fit used for regions not searched by PySR."""
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    a, b = sp.symbols(frame.var_names)
    if prim == "cone":
        terms = [sp.Integer(1), b]
    elif prim == "freeform" and frame.kind == "height":
        terms = [sp.Integer(1), a, b, a * a, a * b, b * b]
    else:
        terms = [sp.Integer(1)]
    cols = [np.broadcast_to(np.asarray(sp.lambdify((a, b), t)(X[:, 0], X[:, 1]), float), y.shape)
            for t in terms]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    return Equation(sum(sp.Float(float(c), 8) * t for c, t in zip(coef, terms)), frame)


def _region_vertices(s):
    """Mesh vertices of each region's triangles (exact data for data-fitted surfaces)."""
    FL = getattr(s, "face_labels", None)
    if FL is None or s.mesh is None:
        return lambda rid: np.zeros((0, 3))
    order = np.argsort(FL, kind="stable")
    starts = np.searchsorted(FL[order], np.arange(len(s.regions) + 1))
    V, F = s.mesh.vertices, s.mesh.faces

    def get(rid):
        f = order[starts[rid]:starts[rid + 1]]
        return V[np.unique(F[f])] if len(f) else np.zeros((0, 3))
    return get


def _wraps_tube(r, P) -> bool:
    """Does this torus face reach both the inner and the outer half of its tube? The
    profile equation r = R +- sqrt(r0^2 - (h - hc)^2) holds one half only."""
    p = r.params
    a = np.asarray(p["axis"], float)
    a = a / np.linalg.norm(a)
    v = P - np.asarray(p["center"], float)
    h = v @ a
    side = np.linalg.norm(v - np.outer(h, a), axis=1) - p["R"]
    eps = 0.05 * p["r"]
    return bool((side > eps).mean() > 0.02 and (side < -eps).mean() > 0.02)


class TubeTorus:
    """Angles on a torus: theta round its axis (from `ref`), phi round its tube
    (0 on the outer equator)."""

    def __init__(self, center, axis, R, r, ref):
        self.c = np.asarray(center, float)
        a = np.asarray(axis, float)
        self.a = a / np.linalg.norm(a)
        u = np.asarray(ref, float) - (np.asarray(ref, float) @ self.a) * self.a
        self.u = u / np.linalg.norm(u)
        self.w = np.cross(self.a, self.u)
        self.R, self.r = float(R), float(r)

    @classmethod
    def from_cls(cls, c, P):
        """Reference direction opposite the face (P its points): theta never wraps on it."""
        a = np.asarray(c["axis"], float)
        a = a / np.linalg.norm(a)
        v = P - np.asarray(c["center"], float)
        rad = v - np.outer(v @ a, a)
        rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-300)
        m = rad.mean(axis=0)
        if np.linalg.norm(m) < 1e-9:
            m = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        return cls(c["center"], a, c["R"], c["r"], -m)

    def angles(self, P):
        v = np.atleast_2d(P) - self.c
        h = v @ self.a
        x, y = v @ self.u, v @ self.w
        return np.arctan2(y, x), np.arctan2(h, np.hypot(x, y) - self.R)

    def point(self, th, ph):
        th, ph = np.atleast_1d(np.asarray(th, float)), np.atleast_1d(np.asarray(ph, float))
        rr = (self.R + self.r * np.cos(ph))[:, None]
        return (self.c + rr * (np.cos(th)[:, None] * self.u + np.sin(th)[:, None] * self.w)
                + (self.r * np.sin(ph))[:, None] * self.a)


def _winding(ang: np.ndarray) -> float:
    """Total turn of a closed loop of angles (2 pi when it goes round once)."""
    return float(np.angle(np.exp(1j * np.diff(np.r_[ang, ang[:1]]))).sum())


def _full_tube(c, P) -> bool:
    """Does a torus face cover its tube all the way round (a pipe elbow)?"""
    if len(P) < 8:
        return False
    _, ph = TubeTorus.from_cls(c, P).angles(P)
    s = np.sort(ph)
    gap = max(float(np.diff(s).max()) if len(s) > 1 else 2 * np.pi, float(s[0] + 2 * np.pi - s[-1]))
    return gap < np.pi / 3


def collect_fits(s) -> dict[int, Fit]:
    fits = {}
    region_vertices = _region_vertices(s)
    exact_tol = 2 * getattr(s, "seg_eps", 0.0)
    # Points on the mesh triangles between the vertices (centres and edge midpoints):
    # a surface that passes through every vertex can still swing far off between them
    # where a coarse mesh has few (the rounded frame round a drawer front, meshed with
    # vertices only at the corners of 0.95 m sides, fitted a saddle that hit them all).
    mesh_, FL_ = getattr(s, "mesh", None), getattr(s, "face_labels", None)
    if mesh_ is not None and FL_ is not None and len(FL_) == len(mesh_.faces):
        _ord = np.argsort(FL_, kind="stable")
        _at = np.searchsorted(FL_[_ord], np.arange(int(FL_.max()) + 2)) if len(FL_) else np.zeros(1, int)
    else:
        _ord = None

    def facet_samples(rid: int) -> np.ndarray:
        if _ord is None or rid + 1 >= len(_at):
            return np.zeros((0, 3))
        T = mesh_.vertices[mesh_.faces[_ord[_at[rid]:_at[rid + 1]]]]
        if len(T) > 4000:
            T = T[np.random.default_rng(0).choice(len(T), 4000, replace=False)]
        return np.vstack([T.mean(axis=1), 0.5 * (T[:, 0] + T[:, 1]), 0.5 * (T[:, 1] + T[:, 2]), 0.5 * (T[:, 0] + T[:, 2])])

    def facet_dev(eq_, Pf) -> float:
        if not len(Pf):
            return 0.0
        X_, y_ = eq_.frame.to_param(Pf)
        d_ = np.abs(eq_(X_) - y_) * eq_.frame.scale
        return float(np.nanmax(d_)) if np.isfinite(d_).any() else np.inf
    for r in s.regions:
        frame = s.frames[r.id]
        if r.prim == "freeform" and "tube_r" in r.params:
            from .tube import TubeSurface
            eq = TubeSurface.from_params(frame, r.params)
            res_ = eq.residual(s.P[r.idx])
            fits[r.id] = Fit(r.id, eq, {"type": "tube", "r": eq.r}, False, float(np.sqrt(np.mean(res_ ** 2))))
            continue
        if (r.prim == "torus" and getattr(s, "strategy", None) == "cad" and r.rms <= exact_tol
                and frame.kind == "cylindrical" and _wraps_tube(r, s.P[r.idx])):
            # found exactly on the mesh: keep the torus itself (its equation covers half the tube)
            p = r.params
            eq = ExactTorus(frame, p["center"], p["axis"], p["R"], p["r"])
            fits[r.id] = Fit(r.id, eq, {"type": "torus", "center": np.asarray(p["center"]), "axis": eq.axis,
                                        "R": float(p["R"]), "r": float(p["r"]), "exact": True}, False, float(r.rms))
            continue
        X, y = frame.to_param(s.P[r.idx])
        res = s.results.get(r.id) or {}
        eq = res.get("eq")
        discovered = eq is not None
        if eq is None:  # not searched: stand-in so export still works
            eq = (torus_equation(r, frame, s.P[r.idx]) if r.prim == "torus" and frame.kind == "cylindrical"
                  else fallback_equation(r.prim, frame, X, y))
        if hasattr(eq, "project_points"):
            # an exact surface that is no function of its frame (a torus over the top of
            # its tube): its frame formula holds one half only, so measure true distance
            rms = float(np.sqrt(np.mean(np.sum((eq.project_points(s.P[r.idx]) - s.P[r.idx]) ** 2, axis=1))))
        else:
            err = (eq(X) - y) * frame.scale
            rms = float(np.sqrt(np.nanmean(err ** 2)))
        cls = classify_region(eq, rms, r, s.diag)
        if type(eq).__name__ == "HelixSurface":  # found as a thread when segmenting
            fits[r.id] = Fit(r.id, eq, {"type": "helix", "lead": eq.L, "axis": eq.a, "point": eq.c}, True, rms)
            _mark_nearest(eq, fits[r.id].cls)
            continue
        if getattr(s, "strategy", None) == "cad":  # exact vertices: hold surfaces to them
            tol = max(20 * getattr(s, "seg_eps", 0.0), 1e-6 * s.diag)
        else:
            tol = max(getattr(s, "seg_eps", 0.0), 0.002 * s.diag)
        # Any surface that misses its region (a freeform patch, or a primitive the
        # data does not really follow) gets a data-fitted spline instead when that
        # fits better. Exact planes never do. A freeform region nobody searched has
        # only a quadratic stand-in: it takes the spline whenever that follows its
        # vertices clearly closer (a quarter-round strip along a drawer front missed
        # its stand-in by 0.26 mm, and its true edges then folded on the patch).
        stand_in = not discovered and r.prim == "freeform"
        # a freeform surface that leaves the mesh triangles far behind between its
        # vertices is checked against a data-fitted spline, which follows them
        Pf, eq_dev, facet_bad = np.zeros((0, 3)), 0.0, False
        if (r.prim == "freeform" and cls.get("type") == "freeform" and res.get("status") != "exact"
                and rms <= 0.5 * tol and not hasattr(eq, "project_points")):
            Pf = facet_samples(r.id)
            eq_dev = facet_dev(eq, Pf)
            facet_bad = FACET_CHECK and eq_dev > 10 * tol
        if r.prim == "freeform" and res.get("status") != "exact" and rms > 0.02 * tol and not facet_bad:
            # a thread flank: a profile swept along a helix, exact over all its turns,
            # where no height-field patch can hold the stacked turns
            from .helix import fit_helix
            try:
                hx = fit_helix(s.P[r.idx], s.N[r.idx], frame, tol)
            except Exception:
                hx = None
            # on an exact CAD mesh a real thread fits its helix to float precision (the
            # T8 nut's flanks to 1e-5 of the tolerance); a wavy near-cylinder that a
            # flexible profile follows to a few percent of it is no thread
            hx_lim = (0.002 if getattr(s, "strategy", None) == "cad" else 0.05) * tol
            if hx is not None and hx.rms <= hx_lim and hx.rms < rms:
                fits[r.id] = Fit(r.id, hx, {"type": "helix", "lead": hx.L, "axis": hx.a, "point": hx.c},
                                 False, hx.rms)
                _mark_nearest(hx, fits[r.id].cls)
                continue
        if (rms > 0.5 * tol or stand_in or facet_bad) and res.get("status") != "exact":
            # No equation is accurate enough here: the CAD surface follows the data
            # (a smoothing B-spline) so the STEP is still right; the UI keeps showing
            # the best equation found.
            try:
                # the region's own mesh vertices are exact: fit to them as well. Try the
                # region's frame and a plain height field over its mean normal (a
                # misfit cone's axis is no good frame for its data either).
                Pv = region_vertices(r.id)
                best = None
                frames = [frame]
                if frame.kind != "height":
                    frames.append(_height_frame(s.P[r.idx], s.N[r.idx], None))
                if FIT_HELPERS and getattr(s, "strategy", None) == "cad" and not len(Pf):
                    Pf = facet_samples(r.id)
                fail_ = None
                for fr_ in frames:
                    Xf, yf = fr_.to_param(s.P[r.idx])
                    Xv, yv = fr_.to_param(Pv)
                    helpers = fr_.to_param(Pf) if FIT_HELPERS and getattr(s, "strategy", None) == "cad" and len(Pf) else None
                    try:
                        cand = DataSurface(fr_, np.r_[Xf, Xv], np.r_[yf, yv], 0.25 * tol / fr_.scale, helpers=helpers)
                        e_ = float(np.sqrt(np.nanmean(((cand(Xf) - yf) * fr_.scale) ** 2)))
                    except Exception as exc:
                        # samples that fall on one line in this frame's parameters (a flat
                        # triangle at one height round a torus's axis) give no surface there;
                        # the next frame may hold them (its plane does)
                        fail_ = exc
                        continue
                    if best is None or e_ < best[1]:
                        best = (cand, e_)
                if best is None:
                    raise fail_ or ValueError("no frame holds the samples")
                ds, drms = best
                if facet_bad:
                    ds_dev = facet_dev(ds, Pf)
                    if ds_dev < 0.5 * eq_dev and drms <= 0.5 * tol:
                        s.export_notes.append(f"region {r.id}: its surface swings {eq_dev:.3g} off the mesh between "
                                              f"vertices; CAD uses a data-fitted B-spline ({ds_dev:.3g})")
                        eq, rms, cls = ds, drms, {"type": "freeform", "data_fit": True}
                elif stand_in and rms <= 0.5 * tol:
                    if drms < 0.5 * rms:
                        eq, rms, cls = ds, drms, {"type": "freeform", "data_fit": True}
                elif drms >= 0.8 * rms:
                    s.export_notes.append(f"region {r.id}: no equation or data fit within tolerance "
                                          f"(rms {rms:.3g} / {drms:.3g}); try another frame")
                else:
                    s.export_notes.append(f"region {r.id}: best equation misses tolerance (rms {rms:.3g}); "
                                          f"CAD uses a data-fitted B-spline (rms {drms:.3g})")
                    eq, rms, cls = ds, drms, {"type": "freeform", "data_fit": True}
            except Exception as exc:
                s.export_notes.append(f"region {r.id}: data-fitted spline failed ({exc})")
        fits[r.id] = Fit(r.id, eq, cls, discovered, rms)
        _mark_nearest(eq, cls)
    return fits


def _mark_nearest(eq, cls: dict) -> None:
    """Surfaces written as exact unbounded geometry (planes, cylinders, cones, spheres,
    surfaces of revolution) take true nearest-point projection. A B-spline patch only
    covers its region's parameter box, so its points keep their parameters and move
    along the height direction only."""
    t = cls.get("type")
    ok = (t == "plane" or (t in ("cylinder", "sphere") and cls.get("radius", 0) > 0)
          or (t == "cone" and cls.get("radius0", 0) > 0 and abs(cls.get("slope", 0)) > 1e-6)
          or (t in ("revolve", "torus") and eq.frame.kind == "cylindrical"))
    try:
        eq.nearest = bool(ok)
    except AttributeError:
        pass


# ------------------------------------------------------------ shared axes

def _turned_axis(fit):
    """(point on the axis, axis direction) of a turned surface; direction None for a
    sphere (any line through its centre is an axis); None for other surfaces."""
    c, eq = fit.cls, fit.eq
    t = c.get("type")
    fr_ = eq.frame
    if t == "sphere" and c.get("radius", 0) > 0:
        return fr_.origin, None
    if t == "torus":
        if isinstance(eq, ExactTorus):
            return eq.center, eq.axis
        if fr_.kind == "cylindrical" and "axis" in c:
            a = np.asarray(c["axis"], float)
            return np.asarray(c["center"], float), a / np.linalg.norm(a)
        return None
    if fr_.kind == "cylindrical" and (
            (t == "cylinder" and c.get("radius", 0) > 0)
            or (t == "cone" and c.get("radius0", 0) > 0 and abs(c.get("slope", 0)) > 1e-6)
            or (t == "revolve" and isinstance(eq, Equation))):
        return fr_.origin, fr_.e3
    return None


def _profile_fit(kind: str, rho: np.ndarray, h: np.ndarray):
    """Best profile of a turned surface round a given axis, from its points' distance
    to the axis (rho) and height along it (h): (parameters, residuals)."""
    one = np.ones_like(h)
    if kind == "cylinder":
        R = float(rho.mean())
        return (R,), rho - R
    if kind == "cone":
        (r0, k), *_ = np.linalg.lstsq(np.c_[one, h], rho, rcond=None)
        return (float(r0), float(k)), (rho - r0 - k * h) / np.sqrt(1 + k * k)
    if kind == "sphere":  # centre on the axis at height hc
        (hc, c), *_ = np.linalg.lstsq(np.c_[2 * h, one], rho ** 2 + h ** 2, rcond=None)
        R = float(np.sqrt(max(c + hc * hc, 0.0)))
        return (float(hc), R), np.hypot(rho, h - hc) - R
    # torus: its profile is a circle in the (rho, h) half-plane
    (Rt, hc, c), *_ = np.linalg.lstsq(np.c_[2 * rho, 2 * h, one], rho ** 2 + h ** 2, rcond=None)
    r = float(np.sqrt(max(c + Rt * Rt + hc * hc, 0.0)))
    return (float(Rt), float(hc), r), np.hypot(rho - Rt, h - hc) - r


class _TorusParams:  # what torus_equation reads from a region
    def __init__(self, center, axis, R, r):
        self.params = {"center": np.asarray(center, float), "axis": np.asarray(axis, float),
                       "R": float(R), "r": float(r)}


def _refit_on_axis(fit, P: np.ndarray, p: np.ndarray, a: np.ndarray):
    """A turned surface refit round the axis through p along a: (Equation, cls, rms),
    or None when it no longer is the same kind of surface."""
    c, eq = fit.cls, fit.eq
    t = c["type"]
    f0 = eq.frame
    s_ = f0.scale
    if t == "sphere":
        v = P - p
        h = v @ a
        rho = np.linalg.norm(v - np.outer(h, a), axis=1)
        (hc, R), _ = _profile_fit("sphere", rho, h)
        if R <= 0:
            return None
        ctr = p + hc * a
        frame = Frame(f0.kind, ctr, f0.e1, f0.e2, f0.e3, s_)
        rms = float(np.sqrt(np.mean((np.linalg.norm(P - ctr, axis=1) - R) ** 2)))
        return Equation(sp.Float(R / s_, 15), frame), {**c, "radius": R}, rms
    e3 = a if a @ f0.e3 >= 0 else -a
    o = p + ((f0.origin - p) @ e3) * e3
    e1 = f0.e1 - (f0.e1 @ e3) * e3
    if np.linalg.norm(e1) < 1e-6:
        e1 = _perp(e3)
    e1 = e1 / np.linalg.norm(e1)
    frame = Frame("cylindrical", o, e1, np.cross(e3, e1), e3, s_)
    X, y = frame.to_param(P)
    rho, h = y * s_, X[:, 1] * s_
    th_, hb = sp.symbols(frame.var_names)
    if t == "cylinder":
        (R,), _ = _profile_fit("cylinder", rho, h)
        new, cls = Equation(sp.Float(R / s_, 15), frame), {**c, "radius": R}
    elif t == "cone":
        (r0, k), _ = _profile_fit("cone", rho, h)
        if r0 <= 0 or abs(k) <= 1e-6:
            return None
        new, cls = Equation(sp.Float(k, 15) * hb + sp.Float(r0 / s_, 15), frame), {**c, "slope": k, "radius0": r0}
    elif t == "torus":
        (Rt, hc, r), _ = _profile_fit("torus", rho, h)
        if r <= 0:
            return None
        ctr = o + hc * e3
        cls = {**c, "R": Rt, "r": r, "center": [float(x) for x in ctr], "axis": [float(x) for x in e3]}
        if isinstance(eq, ExactTorus):
            new = ExactTorus(frame, ctr, e3, Rt, r)
            rms = float(np.sqrt(np.mean(np.sum((new.project_points(P) - P) ** 2, axis=1))))
            return new, cls, rms
        new = torus_equation(_TorusParams(ctr, e3, Rt, r), frame, P)
    else:  # revolve: the same profile expression, its constants refit in the new frame
        new, cls = polish(Equation(eq.expr, frame), X, y), dict(c)
    err = (new(X) - y) * s_
    return new, cls, float(np.sqrt(np.nanmean(err ** 2)))


def share_axes(s, fits: dict[int, Fit], tol: float) -> int:
    """Touching turned faces (cylinders, cones, spheres, tori, surfaces of revolution)
    that are coaxial in the model get one axis, fitted to all their vertices at once.

    Each face fitted on its own gets an axis a hair off its neighbours' (0.001 to 1
    degree), so two of them meet in a slightly wavy curve instead of the circle the
    model has, and where they meet smoothly that curve can wander by a sizeable share
    of the tolerance. The axis is fitted with each face weighted by its own fit error,
    and a face stays in the group only while it still fits its own points (within 2x
    its own rms, or 5 % of the tolerance). Returns how many faces got a shared axis."""
    from scipy.optimize import least_squares

    FL = getattr(s, "face_labels", None)
    if FL is None or s.mesh is None or len(fits) < 2:
        return 0
    axes = {rid: ax for rid, f in fits.items() if (ax := _turned_axis(f)) is not None}
    if len(axes) < 2:
        return 0
    adj = s.mesh.face_adjacency
    la, lb = FL[adj[:, 0]], FL[adj[:, 1]]
    m = la != lb
    V = s.mesh.vertices
    junction: dict[tuple, list] = {}
    for a_, b_, e_ in zip(la[m], lb[m], s.mesh.face_adjacency_edges[m]):
        a_, b_ = int(a_), int(b_)
        if a_ in axes and b_ in axes:
            junction.setdefault((min(a_, b_), max(a_, b_)), []).append(e_[0])
    parent = {rid: rid for rid in axes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def coaxial(ra, rb, J):
        (pa, xa), (pb, xb) = axes[ra], axes[rb]
        if xa is None and xb is None:
            return False  # two spheres: no axis between them
        if xa is None or xb is None:
            pc, (pl, xl) = (pa, axes[rb]) if xa is None else (pb, axes[ra])
            R = float(fits[ra if xa is None else rb].cls["radius"])
            return np.linalg.norm(np.cross(pc - pl, xl)) <= max(0.1 * R, 20 * tol)
        if abs(float(xa @ xb)) < np.cos(np.radians(3)):
            return False
        fa = pa + ((J - pa) @ xa) * xa
        fb = pb + ((J - pb) @ xb) * xb
        rj = max(float(np.linalg.norm(J - fa)), tol)
        return float(np.linalg.norm(fa - fb)) <= max(0.1 * rj, 20 * tol)

    links = []
    for (ra, rb), vs in junction.items():
        J = V[np.asarray(vs)].mean(axis=0)
        if coaxial(ra, rb, J):
            links.append((ra, rb))
            x, y = find(ra), find(rb)
            if x != y:
                parent[max(x, y)] = min(x, y)
    groups: dict[int, list[int]] = {}
    for rid in axes:
        groups.setdefault(find(rid), []).append(rid)
    pts = {}

    def points(rid):
        if rid not in pts:
            pts[rid] = s.P[s.regions[rid].idx]
        return pts[rid]

    done = 0
    gid = 0
    todo = [g for g in groups.values() if len(g) > 1]
    while todo:
        grp = sorted(todo.pop())
        # the axis is fitted to the faces whose profile is known in closed form
        est = [r for r in grp if fits[r].cls["type"] in ("cylinder", "cone", "sphere", "torus")]
        if not any(axes[r][1] is not None for r in est):
            continue
        # each face weighted by its own fit error: a sphere fitted to 1e-10 must not be
        # pulled onto the axis of a cylinder fitted to 1e-7
        sub = []
        for r in est:
            P = points(r)
            if len(P) > 1500:
                # each face's sample from its own points only: one shared generator made it
                # depend on how many faces came before, so a change anywhere re-rolled it
                P = P[np.random.default_rng(len(P)).choice(len(P), 1500, replace=False)]
            sub.append((fits[r].cls["type"], P, 1.0 / max(fits[r].rms, 1e-4 * tol)))
        wts = np.array([len(points(r)) for r in grp if axes[r][1] is not None], float)
        dirs = np.array([axes[r][1] for r in grp if axes[r][1] is not None])
        big = int(np.argmax(wts))
        dirs = dirs * np.sign(dirs @ dirs[big])[:, None]
        a0 = dirs.T @ wts
        a0 /= np.linalg.norm(a0)
        p0 = [axes[r][0] for r in grp if axes[r][1] is not None][big]
        u = _perp(a0)
        w = np.cross(a0, u)
        L = max(float(np.median([np.linalg.norm(np.cross(P.mean(axis=0) - p0, a0)) for _, P, _ in sub])), tol)

        def axis_of(x):
            a = a0 + x[0] * u + x[1] * w
            return p0 + L * (x[2] * u + x[3] * w), a / np.linalg.norm(a)

        def resid(x):
            p, a = axis_of(x)
            out = []
            for kind, P, wt in sub:
                v = P - p
                h = v @ a
                out.append(wt * _profile_fit(kind, np.linalg.norm(v - np.outer(h, a), axis=1), h)[1])
            return np.concatenate(out)

        try:
            sol = least_squares(resid, np.zeros(4), method="lm", xtol=1e-14, ftol=1e-14, max_nfev=400)
        except Exception:
            continue
        p, a = axis_of(sol.x)
        refit, worst = {}, None
        for r in grp:
            try:
                got = _refit_on_axis(fits[r], points(r), p, a)
            except Exception:
                got = None
            limit = max(2 * fits[r].rms, 0.05 * tol)
            ratio = np.inf if got is None or not np.isfinite(got[2]) else got[2] / limit
            if ratio > 1 and (worst is None or ratio > worst[1]):
                worst = (r, ratio)
            refit[r] = got
        if worst is not None:
            # one face does not follow the shared axis: leave it out and try again with
            # the rest (still touching each other)
            rest = [r for r in grp if r != worst[0]]
            par = {r: r for r in rest}

            def fnd(x):
                while par[x] != x:
                    x = par[x]
                return x
            for ra, rb in links:
                if ra in par and rb in par:
                    x, y = fnd(ra), fnd(rb)
                    if x != y:
                        par[max(x, y)] = min(x, y)
            comps: dict[int, list[int]] = {}
            for r in rest:
                comps.setdefault(fnd(r), []).append(r)
            todo.extend(c for c in comps.values() if len(c) > 1)
            continue
        gid += 1
        for r in grp:
            new, cls, rms = refit[r]
            f = Fit(r, new, cls, fits[r].discovered, rms)
            f.shared_axis = (gid, p, a)
            _mark_nearest(new, cls)
            fits[r] = f
            done += 1
    return done


def surface_normals(eq, S: np.ndarray) -> np.ndarray:
    """Unit normals (sign arbitrary) of a surface at points on it; NaN where undefined."""
    if hasattr(eq, "normals"):
        return eq.normals(S)
    f = eq.frame
    X, _ = f.to_param(S)
    d = 1e-4
    G = lambda Y: f.from_param(Y, eq(Y))
    with np.errstate(all="ignore"):
        n = np.cross(G(X + [d, 0]) - G(X - [d, 0]), G(X + [0, d]) - G(X - [0, d]))
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(ln > 0, ln, np.nan)


def closest(eq, P: np.ndarray, iters: int = 4) -> np.ndarray:
    """Nearest points on a surface.

    `project` moves a point along its frame's height direction, which is the nearest
    point only for planes, cylinders and spheres: off a steep cone it moves many times
    the true distance. Foot-point steps along the surface normal fix that.
    """
    S0 = project(eq, P)
    if hasattr(eq, "project_points") or not getattr(eq, "nearest", False):
        return S0
    S = S0.copy()
    for _ in range(iters):
        n = surface_normals(eq, S)
        T = P - np.einsum("ij,ij->i", P - S, n)[:, None] * n
        with np.errstate(all="ignore"):
            S2 = project(eq, T)
        ok = np.isfinite(S2).all(axis=1)
        step = np.linalg.norm(np.where(ok[:, None], S2 - S, 0.0), axis=1)
        S[ok] = S2[ok]
        if step.max(initial=0.0) <= 1e-10 * eq.frame.scale:
            break
    d0 = np.linalg.norm(P - S0, axis=1)
    d1 = np.linalg.norm(P - S, axis=1)
    worse = ~np.isfinite(d1) | (d1 > d0 * (1 + 1e-9))
    S[worse] = S0[worse]
    return S


MU_TIE = 0.01  # moving 10x a residual costs as much as the residual itself
READER_SEAMED = ("cylinder", "cone", "sphere", "torus", "revolve", "tube")  # periodic surface types


def _corner_solve(P0, Q, surfs, wts, mu, iters):
    """Gauss-Newton steps towards the point on all surfaces: each surface linearised at
    the nearest point, the tangent planes solved together, tied to P0 with weight mu."""
    Q = Q.copy()
    for _ in range(iters):
        A = np.repeat(mu * np.eye(3)[None], len(Q), axis=0)
        b = mu * P0
        for e, w in zip(surfs, wts):
            Fi = closest(e, Q)
            ni = surface_normals(e, Fi)
            # where the surface has no usable normal (a half-tube torus at its rim) the
            # direction to the nearest point is the normal
            dv = Q - Fi
            dl = np.linalg.norm(dv, axis=1)
            fb = ~np.isfinite(ni).all(axis=1) & (dl > 0)
            ni[fb] = dv[fb] / dl[fb, None]
            ok = (np.isfinite(ni).all(axis=1) & np.isfinite(Fi).all(axis=1))[:, None]
            ni = np.where(ok, ni, 0.0)
            Fi = np.where(ok, Fi, 0.0)
            A += w * ni[:, :, None] * ni[:, None, :]
            b += w * ni * np.einsum("ij,ij->i", ni, Fi)[:, None]
        with np.errstate(all="ignore"):
            Qn = np.linalg.solve(A, b[..., None])[..., 0]
        good = np.isfinite(Qn).all(axis=1)
        step = np.linalg.norm(np.where(good[:, None], Qn - Q, 0.0), axis=1)
        Q[good] = Qn[good]
        if step.max(initial=0.0) <= 1e-10 * max(1.0, float(np.abs(P0).max(initial=1.0))):
            break
    return Q


def alt_project(P: np.ndarray, eqs: list[Equation], max_move: float, iters: int = 10,
                final: Equation | None = None) -> np.ndarray:
    """Points onto several surfaces at once -> their common curve or corner.

    Each step linearises every surface at its nearest point and solves for the point
    closest to all those tangent planes, tied to the start: on a CAD mesh the mesh
    corner is the true corner, and where surfaces meet at a small angle their exact
    common point can be far off. Alternating projections drift there (a corner of a
    nearly flat cone, a sphere and a plane ended up 39 mm from all three).

    `final`: a surface the result must lie on exactly (an exact plane next to a
    fitted surface): a plane's boundary curves have to be in the plane, or CAD
    kernels cannot build the face.
    """
    P0 = np.asarray(P, float).copy()
    surfs = list(eqs)
    if final is not None and not any(e is final for e in surfs):
        surfs.append(final)

    def resid(Q):
        return np.max([np.linalg.norm(closest(e, Q) - Q, axis=1) for e in surfs], axis=0)

    if len(surfs) == 1:
        Q = closest(surfs[0], P0)
    else:
        wts = [100.0 if (final is not None and e is final) else 1.0 for e in surfs]
        Q = _corner_solve(P0, P0, surfs, wts, MU_TIE, iters)
        # never end further from the surfaces than the mesh point was
        worse = ~(resid(Q) <= resid(P0))
        Q[worse] = P0[worse]
    if final is not None:
        Q = closest(final, Q)
    bad = ~np.isfinite(Q).all(axis=1) | (np.linalg.norm(Q - P0, axis=1) > max_move)
    Q[bad] = P0[bad]
    return Q


def vertex_labels(F: np.ndarray, FL: np.ndarray, nV: int):
    pairs = np.unique(np.c_[F.ravel(), np.repeat(FL, 3)], axis=0)
    starts = np.searchsorted(pairs[:, 0], np.arange(nV + 1))
    return pairs, starts


def snap_mesh(s, fits) -> np.ndarray:
    V, F, FL = s.mesh.vertices, s.mesh.faces, s.face_labels
    pairs, starts = vertex_labels(F, FL, len(V))
    counts = np.diff(starts)
    out = V.copy()
    max_move = 0.05 * s.diag
    single = np.where(counts == 1)[0]
    lab1 = pairs[starts[single], 1]
    for rid in np.unique(lab1):
        vs = single[lab1 == rid]
        out[vs] = alt_project(V[vs], [fits[rid].eq], max_move)
    groups: dict[tuple, list[int]] = {}
    for v in np.where(counts > 1)[0]:
        groups.setdefault(tuple(pairs[starts[v]:starts[v + 1], 1]), []).append(v)
    for labs, vs in groups.items():
        vs = np.asarray(vs)
        out[vs] = alt_project(V[vs], [fits[l].eq for l in labs], max_move)
    return out


# ===================================================================== topology

def crack_pairs(V: np.ndarray, F: np.ndarray, eps_rel: float = 1e-7) -> np.ndarray:
    """T-junction cracks: pairs of unmatched half-edges (indices into the 3-per-face
    half-edge list) that lie on one line, overlap and run in opposite directions.

    A CAD face tessellated in pieces (a drill-point cone split into two halves) has
    a crack inside it where one side's edge is cut by vertices the other side lacks.
    Without twins both sides look like boundary edges, and the face gets a slit."""
    nV = len(V)
    e = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).astype(np.int64)
    key = e[:, 0] * nV + e[:, 1]
    tl = np.flatnonzero(~np.isin(e[:, 1] * nV + e[:, 0], key))
    if len(tl) < 2:
        return np.zeros((0, 2), np.int64)
    V = np.asarray(V, float)
    eps = eps_rel * float(np.linalg.norm(np.ptp(V, axis=0)))
    a, b = V[e[tl, 0]], V[e[tl, 1]]
    L = np.linalg.norm(b - a, axis=1)
    u = (b - a) / np.maximum(L, 1e-300)[:, None]
    at: dict[int, list[int]] = {}
    for k, i in enumerate(tl):
        at.setdefault(int(e[i, 0]), []).append(k)
        at.setdefault(int(e[i, 1]), []).append(k)
    wv = np.fromiter(at.keys(), np.int64)
    tree = cKDTree(V[wv])
    pairs = set()
    for k in range(len(tl)):
        if L[k] <= eps:
            continue
        for w in tree.query_ball_point(0.5 * (a[k] + b[k]), 0.5 * L[k] + eps):
            r = V[wv[w]] - a[k]
            t = float(r @ u[k])
            if t < -eps or t > L[k] + eps or np.linalg.norm(r - t * u[k]) > eps:
                continue
            for m in at[int(wv[w])]:
                if m == k or (b[m] - a[m]) @ u[k] >= 0:
                    continue
                ra, rb = a[m] - a[k], b[m] - a[k]
                ta, tb = float(ra @ u[k]), float(rb @ u[k])
                if np.linalg.norm(ra - ta * u[k]) > eps or np.linalg.norm(rb - tb * u[k]) > eps:
                    continue
                if min(max(ta, tb), L[k]) - max(min(ta, tb), 0.0) > eps:
                    pairs.add((int(min(tl[k], tl[m])), int(max(tl[k], tl[m]))))
    return np.array(sorted(pairs), np.int64).reshape(-1, 2)


def zip_cracks(s, sag: float = 0.03, rounds: int = 12):
    """Close T-junction cracks in the mesh: a triangle edge with no twin is split at the
    vertices of the crack's other side that lie along it, so both sides share them.

    CAD tessellators sometimes mesh neighbouring faces (or the halves of one face) on
    their own, so a shared edge is one long chord on one side and several shorter
    ones on the other. Those edges have no twins and looked like open boundary: slits
    inside faces, and neighbours split into many short pieces. A vertex counts as on
    the edge within `sag` of the edge's length (a curved seam's chords) when it has an
    unmatched edge of its own running the opposite way, in the same part (touching
    parts are never joined). If any split leaves an edge with more than two triangles,
    nothing is changed. Returns (session copy, number of triangles split)."""
    import copy
    mesh = s.mesh
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64).copy()
    FL = np.asarray(s.face_labels).copy()
    nF0 = len(F)
    extra = {k: np.asarray(getattr(s, k)).copy() for k in ("face_part", "face_body")
             if getattr(s, k, None) is not None and len(getattr(s, k)) == nF0}
    part = extra.get("face_part", np.zeros(nF0, np.int64))
    nV = len(V)
    diag = float(np.linalg.norm(np.ptp(V, axis=0)))
    eps = 1e-9 * diag
    n_split = 0
    for _ in range(rounds):
        e = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        key = e[:, 0] * nV + e[:, 1]
        tl = np.flatnonzero(~np.isin(e[:, 1] * nV + e[:, 0], key))
        if len(tl) < 2:
            break
        a, b = V[e[tl, 0]], V[e[tl, 1]]
        L = np.linalg.norm(b - a, axis=1)
        u = (b - a) / np.maximum(L, 1e-300)[:, None]
        tpart = part[tl // 3]
        at: dict[int, list[int]] = {}
        for k, h in enumerate(tl):
            at.setdefault(int(e[h, 0]), []).append(k)
            at.setdefault(int(e[h, 1]), []).append(k)
        wv = np.fromiter(at.keys(), np.int64)
        tree = cKDTree(V[wv])
        splits: dict[int, tuple[int, list[tuple[float, int]]]] = {}
        for k, h in enumerate(tl):
            t = int(h // 3)
            if t in splits or L[k] <= 10 * eps:
                continue
            tol_k = max(sag * L[k], eps)
            ins = []
            for w in tree.query_ball_point(0.5 * (a[k] + b[k]), 0.5 * L[k] + tol_k):
                vw = int(wv[w])
                if vw == e[h, 0] or vw == e[h, 1]:
                    continue
                r = V[vw] - a[k]
                tt = float(r @ u[k])
                if tt <= 1e-6 * L[k] or tt >= L[k] * (1 - 1e-6) or np.linalg.norm(r - tt * u[k]) > tol_k:
                    continue
                # the other side of the crack: an unmatched edge at w running back along this one
                if not any(tpart[m] == tpart[k] and u[m] @ u[k] < -0.9 for m in at[vw]):
                    continue
                ins.append((tt, vw))
            if ins:
                splits[t] = (int(h % 3), sorted(ins))
        if not splits:
            break
        newF, newL, keep = [], [], np.ones(len(F), bool)
        newX = {k: [] for k in extra}
        for t, (j, ins) in splits.items():
            p, q, c = F[t, j], F[t, (j + 1) % 3], F[t, (j + 2) % 3]
            chain = [p] + [vw for _, vw in ins] + [q]
            for x, y in zip(chain[:-1], chain[1:]):
                newF.append((x, y, c))
                newL.append(FL[t])
                for k in extra:
                    newX[k].append(extra[k][t])
            keep[t] = False
            n_split += 1
        F = np.vstack([F[keep], np.array(newF, np.int64)])
        FL = np.concatenate([FL[keep], np.array(newL, FL.dtype)])
        for k in extra:
            extra[k] = np.concatenate([extra[k][keep], np.array(newX[k], extra[k].dtype)])
        part = extra.get("face_part", np.zeros(len(F), np.int64))
    if not n_split:
        return s, 0

    def overused(F_):
        # undirected edges with more than two triangles: a wrong split folds triangles over
        ue = np.sort(F_[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        _, cnt = np.unique(ue[:, 0] * nV + ue[:, 1], return_counts=True)
        return int((cnt > 2).sum())

    if overused(F) > overused(np.asarray(mesh.faces, np.int64)):
        return s, 0  # some split was wrong (a vertex across a gap, not a crack): leave the mesh as it is
    s2 = copy.copy(s)
    s2.mesh = trimesh.Trimesh(V, F, process=False)
    s2.face_labels = FL
    for k, v in extra.items():
        setattr(s2, k, v)
    for k in ("_area_cache_key", "_area_by_label"):
        s2.__dict__.pop(k, None)
    return s2, n_split


def absorb_crumbs(s, max_tris: int = 3, max_frac: float = 0.05):
    """Detached crumbs of a region (a piece of at most `max_tris` triangles away from
    its main piece, under `max_frac` of its area) join the neighbour they share the
    most edge with. Each piece of a region becomes its own face, and a one-triangle
    face of 0.001 mm2 is one the CAD kernel cannot even mesh. A region's only piece
    is never touched. Returns (session copy, number of crumbs)."""
    import copy
    F = np.asarray(s.mesh.faces)
    FL = np.asarray(s.face_labels).copy()
    adj = np.asarray(s.mesh.face_adjacency)
    if not len(adj):
        return s, 0
    edges = np.asarray(s.mesh.face_adjacency_edges)
    V = np.asarray(s.mesh.vertices)
    elen = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    area = np.asarray(s.mesh.area_faces, float)
    PL, region_of, _ = face_patches(F, adj, FL)
    npatch = len(region_of)
    ntri = np.bincount(PL, minlength=npatch)
    parea = np.bincount(PL, weights=area, minlength=npatch)
    main = {}
    for p in range(npatch):
        r = int(region_of[p])
        if r not in main or parea[p] > parea[main[r]]:
            main[r] = p
    rarea = np.bincount(region_of, weights=parea, minlength=int(region_of.max()) + 1)
    crumbs = [p for p in range(npatch) if main[int(region_of[p])] != p and ntri[p] <= max_tris
              and parea[p] < max_frac * rarea[int(region_of[p])]]
    if not crumbs:
        return s, 0
    a, b = adj[:, 0], adj[:, 1]
    cross = PL[a] != PL[b]
    share: dict[int, dict[int, float]] = {}
    for x, y, L in zip(a[cross], b[cross], elen[cross]):
        for p, q in ((PL[x], y), (PL[y], x)):
            d = share.setdefault(int(p), {})
            d[int(FL[q])] = d.get(int(FL[q]), 0.0) + float(L)
    n = 0
    for p in crumbs:
        own = int(region_of[p])
        cand = {r: L for r, L in share.get(p, {}).items() if r != own}
        if cand:
            FL[PL == p] = max(cand, key=lambda r: (cand[r], -r))
            n += 1
    if not n:
        return s, 0
    s2 = copy.copy(s)
    s2.face_labels = FL
    for k in ("_area_cache_key", "_area_by_label"):
        s2.__dict__.pop(k, None)
    return s2, n


def slit_halfedges(F: np.ndarray, PL: np.ndarray, cracks: np.ndarray) -> np.ndarray:
    """Half-edges of cracks lying inside one patch, to be dropped from its boundary.

    A connected group of crack half-edges goes only when all of it is in the same
    patch and dropping it keeps every vertex's boundary in/out count balanced, so
    the patch's loops stay closed."""
    if len(cracks) == 0:
        return np.zeros(0, np.int64)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    hs, inv = np.unique(cracks.ravel(), return_inverse=True)
    inv = inv.reshape(-1, 2)
    G = csr_matrix((np.ones(len(inv)), (inv[:, 0], inv[:, 1])), shape=(len(hs), len(hs)))
    _, comp = connected_components(G, directed=False)
    e = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).astype(np.int64)
    lab = PL[hs // 3]
    drop = []
    for c in np.unique(comp):
        g = hs[comp == c]
        if len(set(lab[comp == c].tolist())) != 1:
            continue
        bal: dict[int, int] = {}
        for h in g:
            bal[int(e[h, 0])] = bal.get(int(e[h, 0]), 0) + 1
            bal[int(e[h, 1])] = bal.get(int(e[h, 1]), 0) - 1
        if any(bal.values()):
            continue
        drop.extend(g.tolist())
    return np.array(drop, np.int64)


def boundary_halfedges(F: np.ndarray, FL: np.ndarray, nV: int, interior=None):
    e = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).astype(np.int64)
    ef = np.repeat(np.arange(len(F)), 3)
    key = e[:, 0] * nV + e[:, 1]
    tkey = e[:, 1] * nV + e[:, 0]
    order = np.argsort(key)
    sk = key[order]
    pos = np.clip(np.searchsorted(sk, tkey), 0, len(sk) - 1)
    found = sk[pos] == tkey
    twin_face = np.where(found, ef[order[pos]], -1)
    lab = FL[ef]
    nb = np.where(twin_face >= 0, FL[np.maximum(twin_face, 0)], -1)
    m = lab != nb
    if interior is not None and len(interior):
        m[interior] = False  # cracks inside a patch are not boundary
    return e[m, 0], e[m, 1], lab[m], nb[m]


def region_loops(A, B, LAB, NB, rid):
    sel = np.where(LAB == rid)[0]
    return patch_loops(A[sel], B[sel], NB[sel])


def patch_loops(a, b, nb):
    """Closed boundary loops of one patch from its boundary half-edges."""
    nxt: dict[int, list[int]] = {}
    for i, v in enumerate(a):
        nxt.setdefault(int(v), []).append(i)
    used = np.zeros(len(a), bool)
    loops = []
    for i0 in range(len(a)):
        if used[i0]:
            continue
        verts, nbs = [], []
        i = i0
        while True:
            used[i] = True
            verts.append(int(a[i]))
            nbs.append(int(nb[i]))
            cand = [j for j in nxt.get(int(b[i]), []) if not used[j]]
            if int(b[i]) == int(a[i0]) or not cand:
                break
            i = cand[0]
        closed = int(b[i]) == int(a[i0])
        if closed and len(verts) >= 3:
            loops.extend(_split_pinched(verts, nbs))
    return loops


def _split_pinched(verts: list[int], nbs: list[int]):
    """Split a boundary walk that passes through the same vertex twice (a pinch,
    where two boundary circles touch) into simple loops."""
    out, stack_v, stack_n, seen = [], [], [], {}
    for v, n in zip(verts, nbs):
        if v in seen:
            k = seen[v]
            loop_v, loop_n = stack_v[k:], stack_n[k:]
            if len(loop_v) >= 3:
                out.append((np.array(loop_v), np.array(loop_n)))
            for u in loop_v:
                seen.pop(u, None)
            del stack_v[k:], stack_n[k:]
        seen[v] = len(stack_v)
        stack_v.append(v)
        stack_n.append(n)
    if len(stack_v) >= 3:
        out.append((np.array(stack_v), np.array(stack_n)))
    return out


def face_patches(F: np.ndarray, adjacency: np.ndarray, FL: np.ndarray):
    """Connected pieces of each region (one CAD face each) and the mesh bodies they lie on.

    Returns (patch label per face, region per patch, body per patch). A region of
    coplanar panels spread over many separate parts becomes many faces; a CAD
    face must be one connected patch.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(F)
    a, b = adjacency[:, 0], adjacency[:, 1]
    same = FL[a] == FL[b]
    G = csr_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(n, n))
    _, patch = connected_components(G, directed=False)
    Gb = csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, body = connected_components(Gb, directed=False)
    npatch = int(patch.max()) + 1 if n else 0
    region_of = np.zeros(npatch, np.int64)
    body_of = np.zeros(npatch, np.int64)
    region_of[patch] = FL
    body_of[patch] = body
    return patch, region_of, body_of


def split_loop(verts: np.ndarray, nbs: np.ndarray):
    n = len(verts)
    change = np.where(nbs != np.roll(nbs, 1))[0]
    if len(change) == 0:
        k = int(np.argmin(verts))
        verts, nbs = np.roll(verts, -k), np.roll(nbs, -k)
        return [(np.r_[verts, verts[0]], int(nbs[0]), True)]
    segs = []
    for ci, start in enumerate(change):
        end = change[(ci + 1) % len(change)]
        stop = end if end > start else end + n
        idx = np.arange(start, stop) % n
        segs.append((np.r_[verts[idx], verts[end % n]], int(nbs[start]), False))
    return segs


# ===================================================================== STEP

def fr(x: float) -> str:
    x = float(x)
    if x == 0 or abs(x) < 1e-14:
        return "0."
    s = f"{x:.12G}"
    if "E" in s:
        m, e = s.split("E")
        if "." not in m:
            m += "."
        return f"{m}E{e}"
    return s if "." in s else s + "."


def knots_mult(t: np.ndarray):
    u, m = np.unique(np.round(np.asarray(t, float), 12), return_counts=True)
    return u, m


def douglas_peucker(pts: np.ndarray, tol: float) -> np.ndarray:
    """Polyline simplification keeping the end points; closed loops keep their start."""
    if len(pts) <= 2:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        L = np.linalg.norm(seg)
        rel = pts[a + 1:b] - pts[a]
        if L < 1e-12:
            d = np.linalg.norm(rel, axis=1)
        else:
            d = np.linalg.norm(np.cross(rel, seg / L), axis=1)
        i = int(np.argmax(d))
        if d[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return pts[keep]


# Hold every boundary point to the neck limit (round 11), not only the edges the
# crossing check redoes.
NECK_LIMIT_ALL = False
# Round-12 changes, each switchable for comparison runs
CROSSING_REPAIR = True   # redo edges whose curves cross in a face
FACET_CHECK = False       # judge freeform fits between vertices too
FIT_HELPERS = False       # data fits also take points on the triangles
PCURVES = False           # write exact (u, v) curves on freeform patches
HELIX_PCURVES = True      # ... and on helical faces, whose (s, x) of every point is exact
CRACK_TWINS = True        # T-junction cracks inside a face are not boundary
ZIP_CRACKS = True         # T-junction cracks are closed before writing (mesh edges split)
ABSORB_CRUMBS = True      # detached 1-3 triangle pieces of a face join their neighbour
FACE_SENSE_AT_FACE = True # torus-as-revolution orientation judged where the face is
MERGE_EXACT_STRIPS = True  # a small exact region lying on a bigger exact neighbour's surface joins it
SEAM_IN_GAP = True        # a surface of revolution's seam goes in the widest gap of its face
MERGE_CONTINUATIONS = False  # freeform lying on a neighbour's surface joins it
VERTEX_REFIT = False      # freeform regions retried as exact shapes on their mesh vertices
SPLIT_FREEFORM = False    # freeform regions cut into neighbour continuations and fillet cylinders
SLIVER_ASK_ENDS = True   # no sliver collapse where a tiny face has a corner at either end
WRITER_TOL_FACTOR = 8   # the writer's edge and vertex tolerance, in segmentation tolerances
REPAIR_MAX_LEVEL = 3     # crossing repair: 1 neck limit, 2 every point kept, 3 raw mesh boundary


def loop_crossings(loops, normal_at, step: float):
    """Pairs of edges of one face whose curves cross each other.

    loops: per boundary loop, a list of (edge key, points along the edge in loop
    order). Judged in the plane square to the surface normal where two pieces come
    close, which is how a reader sees the face's boundary in its parameter space.
    Pieces longer than `step` are split for the search.
    """
    P, Q, own, seq, lid, nseq = [], [], [], [], [], []
    for li, edges in enumerate(loops):
        k = 0
        start = len(P)
        for ek, pts in edges:
            pts = np.asarray(pts, float)
            for a, b in zip(pts[:-1], pts[1:]):
                L = float(np.linalg.norm(b - a))
                n = max(1, int(np.ceil(L / step))) if step > 0 else 1
                for j in range(n):
                    P.append(a + (b - a) * (j / n))
                    Q.append(a + (b - a) * ((j + 1) / n))
                    own.append(ek)
                    seq.append(k)
                    lid.append(li)
                    k += 1
        nseq += [k] * (len(P) - start)
    if len(P) < 4:
        return []
    P, Q = np.asarray(P), np.asarray(Q)
    seq, lid, nseq = np.asarray(seq), np.asarray(lid), np.asarray(nseq)
    M = 0.5 * (P + Q)
    half = 0.5 * np.linalg.norm(Q - P, axis=1)
    r = 2.0 * float(half.max()) + 1e-12
    pairs = cKDTree(M).query_pairs(r, output_type="ndarray")
    if not len(pairs):
        return []
    i, j = pairs[:, 0], pairs[:, 1]
    same = lid[i] == lid[j]
    gap = np.abs(seq[i] - seq[j])
    adj = same & ((gap <= 1) | (gap >= nseq[i] - 1))
    near = np.linalg.norm(M[i] - M[j], axis=1) <= half[i] + half[j] + 1e-12
    keep = ~adj & near
    i, j = i[keep], j[keep]
    if not len(i):
        return []
    # pieces that share an end point (a loop through the same vertex twice) touch
    share = ((np.linalg.norm(P[i] - P[j], axis=1) == 0) | (np.linalg.norm(P[i] - Q[j], axis=1) == 0)
             | (np.linalg.norm(Q[i] - P[j], axis=1) == 0) | (np.linalg.norm(Q[i] - Q[j], axis=1) == 0))
    i, j = i[~share], j[~share]
    if not len(i):
        return []
    N = normal_at(0.25 * (P[i] + Q[i] + P[j] + Q[j]))
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
    U = np.cross(N, np.where(np.abs(N[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-300)
    W = np.cross(N, U)
    o = P[i]
    def uv(X):
        return np.c_[np.einsum("ij,ij->i", X - o, U), np.einsum("ij,ij->i", X - o, W)]
    a, b, c, d = uv(P[i]), uv(Q[i]), uv(P[j]), uv(Q[j])
    def orient(p, q, r_):
        return (q[:, 0] - p[:, 0]) * (r_[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r_[:, 0] - p[:, 0])
    hit = (orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a) * orient(c, d, b) < 0)
    return sorted({(own[int(x)], own[int(y)]) for x, y in zip(i[hit], j[hit])}, key=str)


def circle_through(pts: np.ndarray, tol: float):
    """The circle the points run along, within tol, going one way round.

    Returns (centre, axis, radius) with the axis oriented so the points run
    counterclockwise (the circle's own direction) from first to last, or None. A
    closed run (first point = last) must go once all the way round.
    """
    closed = bool(np.linalg.norm(pts[0] - pts[-1]) <= tol)
    P = pts[:-1] if closed else pts
    if len(P) < 3:
        return None
    c0 = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c0, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    n = np.cross(e1, e2)  # right-handed: SVD alone may give the opposite normal
    n /= np.linalg.norm(n)
    if np.abs((pts - c0) @ n).max() > tol:
        return None  # not in one plane
    x, y = (pts - c0) @ e1, (pts - c0) @ e2
    # algebraic circle fit, then the radius from the distances
    A = np.c_[x, y, np.ones_like(x)]
    try:
        sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = 0.5 * sol[0], 0.5 * sol[1]
    # through the end vertices exactly: CAD readers end the edge where its vertex
    # projects onto the circle, and two arcs meeting nearly tangentially crossed
    # just short of their shared vertex when a fit missed it slightly
    if closed:
        r = float(np.hypot(x[0] - cx, y[0] - cy))
    else:
        mx, my = 0.5 * (x[0] + x[-1]), 0.5 * (y[0] + y[-1])
        wx, wy = -(y[-1] - y[0]), x[-1] - x[0]
        wl = float(np.hypot(wx, wy))
        if wl <= 0:
            return None
        wx, wy = wx / wl, wy / wl
        t_ = (cx - mx) * wx + (cy - my) * wy  # the fitted centre moved onto the bisector
        cx, cy = mx + t_ * wx, my + t_ * wy
        r = float(np.hypot(x[0] - cx, y[0] - cy))
    d = np.hypot(x - cx, y - cy)
    if not np.isfinite(r) or r <= 0 or np.abs(d - r).max() > tol:
        return None
    ang = np.unwrap(np.arctan2(y - cy, x - cx))
    step = np.diff(ang)
    big = np.abs(step) > 1e-9
    if not big.any() or (step[big] > 0).any() and (step[big] < 0).any():
        return None  # turns back on itself
    if np.abs(step).max() > np.radians(45):
        # a few nearly collinear points also fit a big arc that swings far away
        # between them; a meshed circle has its points at most an eighth turn apart
        return None
    sweep = float(ang[-1] - ang[0])
    if closed and abs(abs(sweep) - 2 * np.pi) > np.radians(5):
        return None
    if not closed and not np.radians(3) <= abs(sweep) <= 2 * np.pi - np.radians(3):
        # a sliver of arc between two nearly coincident vertices: a reader cannot tell
        # no turn from a full turn there
        return None
    if sweep < 0:
        n = -n
    return c0 + cx * e1 + cy * e2, n, r


class Step:
    def __init__(self):
        self.lines: list[str] = []

    def add(self, text: str) -> int:
        self.lines.append(text)
        return len(self.lines)

    def ref(self, ids) -> str:
        return "(" + ",".join(f"#{i}" for i in ids) + ")"

    def point(self, p) -> int:
        return self.add(f"CARTESIAN_POINT('',({fr(p[0])},{fr(p[1])},{fr(p[2])}))")

    def direction(self, d) -> int:
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        return self.add(f"DIRECTION('',({fr(d[0])},{fr(d[1])},{fr(d[2])}))")

    def axis2(self, loc, z, x) -> int:
        z = np.asarray(z, float) / np.linalg.norm(z)
        x = np.asarray(x, float)
        x = x - (x @ z) * z
        if np.linalg.norm(x) < 1e-9:
            t = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
            x = t - (t @ z) * z
        return self.add(f"AXIS2_PLACEMENT_3D('',#{self.point(loc)},#{self.direction(z)},#{self.direction(x)})")

    def bspline_curve(self, deg: int, C: np.ndarray, t: np.ndarray) -> int:
        u, m = knots_mult(t)
        pts = [self.point(c) for c in C]
        return self.add(
            f"B_SPLINE_CURVE_WITH_KNOTS('',{deg},{self.ref(pts)},.UNSPECIFIED.,.F.,.F.,"
            f"({','.join(map(str, m))}),({','.join(fr(x) for x in u)}),.UNSPECIFIED.)")

    def pcurve(self, surface: int, UV: np.ndarray, t: np.ndarray) -> int:
        """A degree-1 curve in a surface's (u, v) parameters, as a PCURVE on it."""
        if not getattr(self, "_ctx2d", None):
            self._ctx2d = self.add("( GEOMETRIC_REPRESENTATION_CONTEXT(2) PARAMETRIC_REPRESENTATION_CONTEXT() "
                                   "REPRESENTATION_CONTEXT('2D SPACE','') )")
        u, m = knots_mult(t)
        pts = [self.add(f"CARTESIAN_POINT('',({fr(q[0])},{fr(q[1])}))") for q in UV]
        bs = self.add(f"B_SPLINE_CURVE_WITH_KNOTS('',1,{self.ref(pts)},.UNSPECIFIED.,.F.,.F.,"
                      f"({','.join(map(str, m))}),({','.join(fr(x) for x in u)}),.UNSPECIFIED.)")
        rep_ = self.add(f"DEFINITIONAL_REPRESENTATION('',(#{bs}),#{self._ctx2d})")
        return self.add(f"PCURVE('',#{surface},#{rep_})")

    def bspline_surface(self, ku, kv, C: np.ndarray, tu, tv) -> int:
        uu, mu = knots_mult(tu)
        vv, mv = knots_mult(tv)
        rows = []
        for i in range(C.shape[0]):
            rows.append(self.ref([self.point(C[i, j]) for j in range(C.shape[1])]))
        return self.add(
            f"B_SPLINE_SURFACE_WITH_KNOTS('',{ku},{kv},({','.join(rows)}),.UNSPECIFIED.,.F.,.F.,.F.,"
            f"({','.join(map(str, mu))}),({','.join(map(str, mv))}),"
            f"({','.join(fr(x) for x in uu)}),({','.join(fr(x) for x in vv)}),.UNSPECIFIED.)")

    def write(self, path: Path, name: str):
        # Product structure + representation context appended at the end of DATA.
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        head = [
            "ISO-10303-21;", "HEADER;",
            "FILE_DESCRIPTION(('PyCadSR surface reconstruction'),'2;1');",
            f"FILE_NAME('{name}.step','{now}',('PyCadSR'),(''),'PyCadSR','PyCadSR','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;",
        ]
        body = [f"#{i + 1}={t};" for i, t in enumerate(self.lines)]
        path.write_text("\n".join(head + body + ["ENDSEC;", "END-ISO-10303-21;", ""]), encoding="ascii")


UNITS = {
    "mm": "( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )",
    "cm": "( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.CENTI.,.METRE.) )",
    "m": "( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) )",
}


class StepExporter:
    def __init__(self, s, fits: dict[int, Fit], units: str = "mm", progress=None):
        self.s, self.fits = s, fits
        self.units = units
        self.face_name = lambda g: f"R{g}"  # export_all names faces by the caller's ids
        # Smallest distance kept apart: CAD kernels merge points closer than about
        # 1e-7 mm (OpenCASCADE's confusion tolerance), whatever the model's size.
        mm_per_unit = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4}.get(units, 1.0)
        self.min_gap = max(1e-9 * s.diag, 1e-5 / mm_per_unit)
        self.progress = progress or (lambda *a: None)
        self.st = Step()
        self.diag = s.diag
        # never tighter than the data: on scans the segmentation tolerance tracks the noise
        if getattr(s, "strategy", None) == "cad":
            # a CAD mesh: regions border exactly on the true edges, so edges and
            # vertices are kept to the (micron-level) segmentation tolerance
            self.tol = max(WRITER_TOL_FACTOR * getattr(s, "seg_eps", 0.0), 1e-6 * s.diag)
            # surfaces are sampled at the looser fitting tolerance: a tighter one only
            # multiplies a freeform patch's control points (40 % bigger files)
            self.surf_tol = max(20 * getattr(s, "seg_eps", 0.0), 1e-6 * s.diag)
        else:
            self.tol = max(0.0015 * s.diag, getattr(s, "seg_eps", 0.0))
            self.surf_tol = self.tol
        self.warnings: list[str] = []

    # ---------------------------------------------------------- surfaces
    def _mean_normal_dot(self, rid, dirs_fn) -> float:
        # orientation is decided per face in build(); remember how to get the normal
        self._normal_fn = dirs_fn
        r = self.s.regions[rid]
        P, N = self.s.P[r.idx], self.s.N[r.idx]
        return float(np.mean(np.einsum("ij,ij->i", dirs_fn(P), N)))

    def _equation_normals(self, eq):
        """Surface normals (d/dx0 x d/dx1 of the equation's parametrisation) at points.

        Matches the STEP parametrisation of surfaces built from equations: a
        B-spline patch in (x0, x1) and a surface of revolution in (angle, profile).
        """
        f = eq.frame

        def fn(P):
            X, _ = f.to_param(P)
            d = 1e-3
            S = lambda G: f.from_param(G, eq(G))
            du = S(X + [d, 0]) - S(X - [d, 0])
            dv = S(X + [0, d]) - S(X - [0, d])
            n = np.cross(du, dv)
            return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

        return fn

    def surface(self, rid: int) -> tuple[int, bool]:
        fit = self.fits[rid]
        fr_ = fit.eq.frame
        c = fit.cls
        st = self.st
        o, e1, e2, e3, s = fr_.origin, fr_.e1, fr_.e2, fr_.e3, fr_.scale
        t = c.get("type")
        # Periodic surfaces have their seam (u = 0) along the reference direction.
        # The frame's e1 points at the middle of the region, so put the seam on the
        # opposite side: a face straddling its own seam makes CAD kernels heal it badly.
        seam = getattr(self, "_seam_dir", {}).get(rid, -e1)

        def radial(P):
            v = P - o
            rad = v - np.outer(v @ e3, e3)
            return rad / np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-300)

        if t == "plane":
            a, b = c.get("slope_u", 0.0), c.get("slope_v", 0.0)
            n = -a * e1 - b * e2 + e3
            n /= np.linalg.norm(n)
            loc = o + c.get("offset", 0.0) * e3
            same = self._mean_normal_dot(rid, lambda P: np.broadcast_to(n, P.shape)) >= 0
            return st.add(f"PLANE('',#{st.axis2(loc, n, e1)})"), same
        if t == "cylinder" and c["radius"] > 0:
            same = self._mean_normal_dot(rid, radial) >= 0
            return st.add(f"CYLINDRICAL_SURFACE('',#{st.axis2(o, e3, seam)},{fr(c['radius'])})"), same
        if t == "sphere" and c["radius"] > 0:
            same = self._mean_normal_dot(rid, lambda P: (P - o) / np.linalg.norm(P - o, axis=1, keepdims=True)) >= 0
            ax, ref = getattr(self, "_sphere_axes", {}).get(rid) or self._sphere_axis(rid, o, e3, seam)
            if rid in getattr(self, "_seam_dir", {}):
                ref = seam - (seam @ ax) * ax
                ref /= np.linalg.norm(ref)
            return st.add(f"SPHERICAL_SURFACE('',#{st.axis2(o, ax, ref)},{fr(c['radius'])})"), same
        if t == "cone" and c["radius0"] > 0 and abs(c["slope"]) > 1e-6:
            axis = e3 if c["slope"] > 0 else -e3
            ang = float(np.arctan(abs(c["slope"])))

            def cone_n(P):
                return radial(P) * np.cos(ang) - np.outer(np.ones(len(P)), axis) * np.sin(ang)

            same = self._mean_normal_dot(rid, cone_n) >= 0
            return st.add(f"CONICAL_SURFACE('',#{st.axis2(o, axis, seam)},{fr(c['radius0'])},{fr(ang)})"), same
        if t == "tube":
            return self.tube_surface(rid, fit.eq)
        if t == "helix":
            return self.helix_surface(rid, fit.eq)
        full_tube = t == "torus" and _full_tube(c, self.s.P[self.s.regions[rid].idx])
        if full_tube:
            # a pipe elbow: the face is closed round the tube, which no single profile
            # covers; a TOROIDAL_SURFACE with its axis seam away from the face is what
            # CAD systems write, and the seam edge runs along the tube (phi = 0)
            tor = TubeTorus.from_cls(c, self.s.P[self.s.regions[rid].idx])

            def tube_n(P):
                v = P - tor.c
                h = v @ tor.a
                rad = v - np.outer(h, tor.a)
                rho = np.maximum(np.linalg.norm(rad, axis=1), 1e-300)
                n = rad / rho[:, None] * (rho - tor.R)[:, None] + np.outer(h, tor.a)
                return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

            same = self._mean_normal_dot(rid, tube_n) >= 0
            return st.add(f"TOROIDAL_SURFACE('',#{st.axis2(tor.c, tor.a, tor.u)},{fr(tor.R)},{fr(tor.r)})"), same
        if t == "torus" and isinstance(fit.eq, ExactTorus):
            try:
                return self.torus_revolution(rid, fit.eq)
            except Exception as exc:
                self.warnings.append(f"region {rid}: exact torus as revolution failed ({exc}); using TOROIDAL_SURFACE")
        if t == "torus" and fr_.kind == "cylindrical" and not isinstance(fit.eq, ExactTorus):
            # Written as the exact circular-arc profile swept round the axis: the same
            # torus, but OpenCascade trims a doubly periodic TOROIDAL_SURFACE band
            # unreliably, while a surface of revolution closes cleanly.
            try:
                tp = _TorusParams(c["center"], c["axis"], c["R"], c["r"])  # the fit's torus (its axis may be shared)
                return self.revolution_from_equation(rid, torus_equation(tp, fr_, self.s.P[self.s.regions[rid].idx]))
            except Exception as exc:
                self.warnings.append(f"region {rid}: torus as revolution failed ({exc}); using TOROIDAL_SURFACE")
        if t == "torus":
            ca, ax_, R, r0 = np.asarray(c["center"]), np.asarray(c["axis"]), c["R"], c["r"]

            def torus_n(P):
                v = P - ca
                h = v @ ax_
                rad = v - np.outer(h, ax_)
                rho = np.maximum(np.linalg.norm(rad, axis=1), 1e-300)
                n = rad / rho[:, None] * (rho - R)[:, None] + np.outer(h, ax_)
                return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

            same = self._mean_normal_dot(rid, torus_n) >= 0
            return st.add(f"TOROIDAL_SURFACE('',#{st.axis2(ca, ax_, seam)},{fr(R)},{fr(r0)})"), same
        if t == "revolve" and fr_.kind == "cylindrical":
            try:
                return self.revolution_from_equation(rid)
            except Exception as exc:  # fall back to a sampled B-spline patch
                self.warnings.append(f"region {rid}: surface of revolution failed ({exc}); using B-spline")
        return self.bspline_from_equation(rid)

    def _sphere_axis(self, rid: int, o: np.ndarray, e3: np.ndarray, seam: np.ndarray):
        """Axis and seam direction of a sphere face.

        A face that goes all the way round an axis through the centre (a dome, a ring)
        gets that axis as the sphere's own, the way CAD systems write it: its boundary
        circles are then latitude lines. Lying across the sphere's equator instead, such
        a face is meshed by OpenCASCADE with triangles far coarser than asked for.
        Partial patches (a corner ball) keep the frame's axis, away from the poles."""
        P = self.s.P[self.s.regions[rid].idx]
        R = float(np.median(np.linalg.norm(P - o, axis=1)))
        d = P.mean(axis=0) - o
        if np.linalg.norm(d) < 0.05 * R:
            return e3, seam
        a = d / np.linalg.norm(d)
        u = seam - (seam @ a) * a
        if np.linalg.norm(u) < 1e-6:
            u = e3 - (e3 @ a) * a
        u /= np.linalg.norm(u)
        w = np.cross(a, u)
        v = P - o
        ang = np.sort(np.arctan2(v @ w, v @ u))
        if np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max() > np.radians(30):
            return e3, seam  # not all the way round
        return a, u

    def tube_surface(self, rid: int, tube) -> tuple[int, bool]:
        """A closed B-spline surface: the tube's circle swept along its spine, closing
        on its seam line (so the seam edge is an exact iso-line)."""
        r = self.s.regions[rid]
        t, _ = tube.angles(self.s.P[r.idx])
        lo, hi = float(t.min()), float(t.max())
        pad = 0.01 * (hi - lo) + 1e-4
        a0, a1 = max(lo - pad, 0.0), min(hi + pad, 1.0)
        L = tube._length() * (a1 - a0)
        nu = int(np.clip(L / tube.r, 24, 600))
        nv = 17
        A = np.linspace(a0, a1, nu)
        Bv = np.linspace(0.0, 2 * np.pi, nv)
        TT, VV = np.meshgrid(A, Bv, indexing="ij")
        W = tube.point(TT.ravel(), tube.seam_phi(TT.ravel()) + VV.ravel()).reshape(nu, nv, 3)
        C = None
        for k in range(3):
            spl = RectBivariateSpline(A, Bv, W[:, :, k], kx=3, ky=3, s=0)
            tu, tv, coef = spl.tck
            if C is None:
                C = np.empty((len(tu) - 4, len(tv) - 4, 3))
            C[:, :, k] = np.asarray(coef).reshape(len(tu) - 4, len(tv) - 4)
        C[:, -1] = C[:, 0]  # closed exactly round the tube
        sid = self.st.bspline_surface(3, 3, C, tu, tv)
        i, j = nu // 2, nv // 4
        n_nat = np.cross(W[i + 1, j] - W[i - 1, j], W[i, j + 1] - W[i, j - 1])
        sign = 1.0 if float(n_nat @ tube.normals(W[i, j][None])[0]) >= 0 else -1.0
        self._normal_fn = lambda X: sign * tube.normals(X)
        same = self._mean_normal_dot(rid, self._normal_fn) >= 0
        return sid, same

    def helix_surface(self, rid: int, hx) -> tuple[int, bool]:
        """A B-spline surface sampled from a helical sweep in its (s, x) coordinates,
        over the region's turns plus a margin (16 samples a turn hold the helix to well
        under a micron on a thread)."""
        r = self.s.regions[rid]
        s_, x_ = hx.coords(self.s.P[r.idx])
        lo, hi = float(s_.min()), float(s_.max())
        xl, xh = float(x_.min()), float(x_.max())
        ps, px = 0.02 * (hi - lo) + 0.05, 0.05 * (xh - xl) + 1e-9 * hx.xs
        A = np.linspace(lo - ps, hi + ps, int(np.clip((hi - lo + 2 * ps) / (2 * np.pi) * 16, 8, 1200)))
        # a polynomial profile needs 9 samples across; a spline profile (a round groove)
        # one or two per knot span
        Bv = np.linspace(xl - px, xh + px, int(getattr(hx, "n_profile", 9)))
        SS, XX = np.meshgrid(A, Bv, indexing="ij")
        W = hx.point(SS.ravel(), XX.ravel()).reshape(len(A), len(Bv), 3)
        # u runs along the turns as arc length at the face's mean radius, not as an angle:
        # with u in radians (dozens over the turns) and v in metres (a millimetre across),
        # a CAD mesher spanned a narrow curved crest with triangles that cut across it
        _, rho_, _ = hx._raw(self.s.P[r.idx])
        su = max(float(np.median(rho_)), 1e-9 * max(self.s.diag, 1e-300))
        C = None
        for k in range(3):
            spl = RectBivariateSpline(A * su, Bv, W[:, :, k], kx=3, ky=3, s=0)
            tu, tv, coef = spl.tck
            if C is None:
                C = np.empty((len(tu) - 4, len(tv) - 4, 3))
            C[:, :, k] = np.asarray(coef).reshape(len(tu) - 4, len(tv) - 4)
        sid = self.st.bspline_surface(3, 3, C, tu, tv)
        if not hasattr(self, "_pc_helix"):
            self._pc_helix = {}
        self._pc_helix[rid] = (hx, float(A[0]), float(A[-1]), float(Bv[0]), float(Bv[-1]), su)
        i, j = len(A) // 2, len(Bv) // 2
        n_nat = np.cross(W[i + 1, j] - W[i - 1, j], W[i, j + 1] - W[i, j - 1])
        sign = 1.0 if float(n_nat @ hx.normals(W[i, j][None])[0]) >= 0 else -1.0
        self._normal_fn = lambda X: sign * hx.normals(X)
        same = self._mean_normal_dot(rid, self._normal_fn) >= 0
        return sid, same

    def torus_seam(self, rid: int, tor, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
        """Points along the tube's circular cross-section from pa to pb, going the way
        that passes through the face (the face may cover more than half the tube)."""
        a, cen, R, r0 = tor.axis, tor.center, tor.R, tor.r

        def phi(P):
            v = P - cen
            h = v @ a
            rho = np.linalg.norm(v - np.outer(h, a), axis=1)
            return np.arctan2(h, rho - R)

        P = self.s.P[self.s.regions[rid].idx]
        ref = float(np.angle(np.mean(np.exp(1j * phi(P)))))
        fa, fb = phi(np.vstack([pa, pb]))
        d1 = float(np.angle(np.exp(1j * (fb - fa))))  # short way
        d2 = d1 - np.sign(d1) * 2 * np.pi if d1 != 0 else 2 * np.pi
        mid1 = np.angle(np.exp(1j * (fa + d1 / 2 - ref)))
        mid2 = np.angle(np.exp(1j * (fa + d2 / 2 - ref)))
        d = d1 if abs(mid1) <= abs(mid2) else d2
        ph = fa + np.linspace(0.0, d, 32)
        v = pa - cen
        rad = v - (v @ a) * a
        u = rad / max(np.linalg.norm(rad), 1e-300)
        return cen[None, :] + np.outer(R + r0 * np.cos(ph), u) + np.outer(r0 * np.sin(ph), a)

    def torus_revolution(self, rid: int, tor) -> tuple[int, bool]:
        """The torus as its exact circular profile, over the arc of the tube the face
        actually covers, swept round the axis (SURFACE_OF_REVOLUTION closes full
        turns cleanly in CAD kernels)."""
        r = self.s.regions[rid]
        P = self.s.P[r.idx]
        a, cen, R, r0 = tor.axis, tor.center, tor.R, tor.r
        u = getattr(self, "_seam_dir", {}).get(rid)  # a band in a chain: the chain's seam
        if u is None:
            u = -tor.frame.e1  # profile opposite the face: seam away from it
        u = u - (u @ a) * a
        u /= np.linalg.norm(u)
        w = np.cross(a, u)
        v = P - cen
        h = v @ a
        rho = np.linalg.norm(v - np.outer(h, a), axis=1)
        phi = np.arctan2(h, rho - R)
        ref = np.angle(np.mean(np.exp(1j * phi)))
        d = np.angle(np.exp(1j * (phi - ref)))
        lo, hi = ref + d.min(), ref + d.max()
        # profile a little past the face both ways (OpenCASCADE's mesher fills the hole of
        # a ring face whose profile ends just past its boundary; three times this margin
        # meshed right on every model tried)
        pad = min(3 * (0.05 * (hi - lo) + 0.02), 0.5 * max(0.0, 2 * np.pi - (hi - lo)))
        p0, p1 = lo - pad, hi + pad
        if R < r0:
            # the tube crosses the axis (a dome on a spindle torus, R < r): a face that
            # reaches it ends there; past the axis the profile would sweep back over it
            for ph_ax in (np.arccos(-R / r0), -np.arccos(-R / r0)):
                ph_ax = ref + float(np.angle(np.exp(1j * (ph_ax - ref))))
                if hi <= ph_ax <= p1 or (hi > ph_ax - 1e-9 and ph_ax >= hi - 0.02):
                    p1 = ph_ax
                if p0 <= ph_ax <= lo or (lo < ph_ax + 1e-9 and ph_ax <= lo + 0.02):
                    p0 = ph_ax
        ph = np.linspace(p0, p1, 72)
        prof = cen[None, :] + np.outer(R + r0 * np.cos(ph), u) + np.outer(r0 * np.sin(ph), a)
        tck, _ = splprep(prof.T, s=0, k=3)
        t_, c_, k_ = tck
        cid = self.st.bspline_curve(k_, np.asarray(c_).T[: len(t_) - k_ - 1], t_)
        axis1 = self.st.add(f"AXIS1_PLACEMENT('',#{self.st.point(cen)},#{self.st.direction(a)})")
        sid = self.st.add(f"SURFACE_OF_REVOLUTION('',#{cid},#{axis1})")

        def S_(th, ph_):
            return cen + (R + r0 * np.cos(ph_)) * (np.cos(th) * u + np.sin(th) * w) + r0 * np.sin(ph_) * a

        # natural normal (d/d angle) x (d/d profile) against the outward tube normal, where
        # the face is: on a spindle torus the part inside the axis (R + r cos < 0) sweeps
        # the other way round, so the two disagree there
        pf = float(np.clip(ref, p0, p1)) if FACE_SENSE_AT_FACE else 0.3
        n_nat = np.cross(S_(1e-3, pf) - S_(-1e-3, pf), S_(0, pf + 1e-3) - S_(0, pf - 1e-3))
        q = S_(0.0, pf)
        sign = 1.0 if float(n_nat @ tor.normals(q[None])[0]) >= 0 else -1.0
        self._normal_fn = lambda X: sign * tor.normals(X)
        same = self._mean_normal_dot(rid, self._normal_fn) >= 0
        return sid, same

    def revolution_from_equation(self, rid: int, eq=None) -> tuple[int, bool]:
        """r = f(h) discovered in a cylindrical frame -> SURFACE_OF_REVOLUTION of the profile.

        Unlike a sampled B-spline patch this is periodic around the axis, so CAD
        kernels close full turns cleanly (a closed non-periodic patch gets its
        seam healed and can come out inside-out).
        """
        eq = eq or self.fits[rid].eq
        f = eq.frame
        r = self.s.regions[rid]
        P = self.s.P[r.idx]
        X, _ = f.to_param(P)
        h0, h1 = float(X[:, 1].min()), float(X[:, 1].max())
        pad = 0.05 * (h1 - h0)
        # padding beyond the region, plus the region's exact end heights (a torus arc
        # can end right at its tube, where the next sample out is undefined)
        hs = np.r_[np.linspace(h0 - pad, h0, 4, endpoint=False), np.linspace(h0, h1, 72),
                   np.linspace(h1, h1 + pad, 5)[1:]]
        sd = getattr(self, "_seam_dir", {}).get(rid)  # a band in a chain: the chain's seam
        if sd is not None:
            th_seam = float(np.arctan2(sd @ f.e2, sd @ f.e1))
        elif SEAM_IN_GAP and len(X) > 1:
            # in the middle of the widest angle the face leaves free: the frame's e1 need
            # not point at the face (a torus refit on a shared axis), and a seam through
            # the face makes the reader wrap its edges the long way round
            srt = np.sort(np.angle(np.exp(1j * X[:, 0])))
            gaps = np.diff(np.r_[srt, srt[0] + 2 * np.pi])
            k_ = int(np.argmax(gaps))
            th_seam = float(np.angle(np.exp(1j * (srt[k_] + 0.5 * gaps[k_])))) if gaps[k_] > np.radians(30) else np.pi
        else:
            th_seam = np.pi
        G = np.c_[np.full_like(hs, th_seam), hs]  # else opposite the region: seam away from the face
        y = eq(G)
        good = np.isfinite(y) & (y > 0)
        if not good.all():
            # trim padding the profile is not defined on (a torus arc ends at its tube)
            lo, hi = int(np.argmax(good)), len(good) - int(np.argmax(good[::-1]))
            if not good[lo:hi].all() or hs[lo] > h0 + 1e-9 or hs[hi - 1] < h1 - 1e-9:
                raise ValueError("profile undefined on the region's height range")
            hs, G, y = hs[lo:hi], G[lo:hi], y[lo:hi]
        C = f.from_param(G, y)
        tck, _ = splprep(C.T, s=len(C) * (0.05 * self.surf_tol) ** 2, k=3)
        t, c, k = tck
        cid = self.st.bspline_curve(k, np.asarray(c).T[: len(t) - k - 1], t)
        axis1 = self.st.add(f"AXIS1_PLACEMENT('',#{self.st.point(f.origin)},#{self.st.direction(f.e3)})")
        sid = self.st.add(f"SURFACE_OF_REVOLUTION('',#{cid},#{axis1})")
        self._normal_fn = self._equation_normals(eq)
        # orientation: (d/d angle) x (d/d profile) against the mesh normal near the centroid
        i0 = int(np.argmin(np.linalg.norm(P - P.mean(axis=0), axis=1)))
        th, hh = X[i0]
        d = 1e-3
        S = lambda a, b: f.from_param(np.array([[a, b]]), eq(np.array([[a, b]])))[0]
        n = np.cross(S(th + d, hh) - S(th - d, hh), S(th, hh + d) - S(th, hh - d))
        same = float(n @ self.s.N[r.idx][i0]) >= 0
        return sid, same

    def bspline_from_equation(self, rid: int) -> tuple[int, bool]:
        fit = self.fits[rid]
        eq = fit.eq
        r = self.s.regions[rid]
        X, yd = eq.frame.to_param(self.s.P[r.idx])
        ga, gb, _, wrap = param_domain(eq.frame, X, target_cells=400)
        # The patch spans the region's whole parameter box, and away from the data an
        # equation can run off to absurd values (a steep parabola fitted to a thin
        # strip reached metres on a 75 mm part). Only the patch near the data is used.
        yd = yd[np.isfinite(yd)]
        if len(yd):
            pad = max(float(np.ptp(yd)), 0.1 * max(ga[-1] - ga[0], gb[-1] - gb[0]))
            y_lo, y_hi = float(yd.min()) - pad, float(yd.max()) + pad
        else:
            y_lo, y_hi = -np.inf, np.inf
        na = int(np.clip(len(ga), 6, 40))
        nb = int(np.clip(len(gb), 6, 40))
        # check points: the region's samples that lie inside the parameter box
        inside = np.isfinite(X).all(axis=1) & (X[:, 0] >= ga[0]) & (X[:, 0] <= ga[-1]) & \
            (X[:, 1] >= gb[0]) & (X[:, 1] <= gb[-1])
        Xc = X[inside][:: max(1, int(inside.sum()) // 4000)]
        target = eq.frame.from_param(Xc, eq(Xc)) if len(Xc) else None
        for attempt in range(4):
            a = np.linspace(ga[0], ga[-1], na)
            b = np.linspace(gb[0], gb[-1], nb)
            A, B = np.meshgrid(a, b, indexing="ij")
            G = np.c_[A.ravel(), B.ravel()]
            y = eq(G)
            if not np.isfinite(y).all():
                good = np.isfinite(y)
                if good.sum() < 4:
                    raise ValueError(f"region {rid}: equation not finite on its domain")
                from scipy.spatial import cKDTree
                _, j = cKDTree(G[good]).query(G[~good])
                y[~good] = y[good][j]
                if attempt == 0:
                    self.warnings.append(f"region {rid}: equation undefined on part of its domain, patched")
            y = np.clip(y, y_lo, y_hi)
            W = eq.frame.from_param(G, y).reshape(na, nb, 3)
            C = np.empty((na, nb, 3))
            tu = tv = None
            spls = []
            for k in range(3):
                spl = RectBivariateSpline(a, b, W[:, :, k], kx=3, ky=3, s=0)
                spls.append(spl)
                tu, tv, coef = spl.tck
                C[:, :, k] = np.asarray(coef).reshape(len(tu) - 4, len(tv) - 4)
            # Detailed surfaces (a sculpted relief) need more control points than a
            # smooth one: refine until the spline follows the surface within tolerance.
            if target is None or not np.isfinite(target).all() or (na >= 160 and nb >= 160):
                break
            got = np.stack([sp_.ev(Xc[:, 0], Xc[:, 1]) for sp_ in spls], axis=1)
            if np.percentile(np.linalg.norm(got - target, axis=1), 95) <= 0.5 * self.surf_tol:
                break
            na, nb = min(160, int(na * 1.8)), min(160, int(nb * 1.8))
        sid = self.st.bspline_surface(3, 3, C, tu, tv)
        if eq.frame.kind == "height" or (eq.frame.kind == "cylindrical" and not wrap):
            # its (u, v) are the frame's own parameters: curves on it are known exactly
            self._pc_frame[rid] = (eq.frame, float(a[0]), float(a[-1]), float(b[0]), float(b[-1]))
        self._normal_fn = self._equation_normals(eq)
        # orientation: dS/da x dS/db vs mesh normals
        i, j = na // 2, nb // 2
        du = W[min(i + 1, na - 1), j] - W[max(i - 1, 0), j]
        dv = W[i, min(j + 1, nb - 1)] - W[i, max(j - 1, 0)]
        n = np.cross(du, dv)
        from scipy.spatial import cKDTree
        _, k = cKDTree(self.s.P[r.idx]).query(W[i, j], k=min(20, len(r.idx)))
        same = float(n @ self.s.N[r.idx][np.atleast_1d(k)].mean(axis=0)) >= 0
        return sid, same

    # ---------------------------------------------------------- curves
    def curve(self, pts: np.ndarray, circles: bool = True, max_tol: float = np.inf) -> int:
        return self.curve_write(self.curve_plan(pts, circles, max_tol))

    def curve_write(self, plan) -> int:
        kind, data, _ = plan
        if kind == "circle":
            c, n, r, p0 = data
            return self.st.add(f"CIRCLE('',#{self.st.axis2(c, n, p0 - c)},{fr(r)})")
        k, C, t = data
        return self.st.bspline_curve(k, C, t)

    def pcurve_uv(self, plan, rid: int):
        """The written curve in the (u, v) of region rid's freeform patch, at the curve's
        own parameters: (UV, knots) of a degree-1 curve, or None if it leaves the patch."""
        helix = getattr(self, "_pc_helix", {}).get(rid)
        if helix is None:
            frame, a0, a1, b0, b1 = self._pc_frame[rid]
        kind, data, _ = plan
        if kind == "poly":
            _, P, t = data
            ps = np.asarray(t[1:-1], float)
        elif kind == "spline":
            k, C, t = data
            ps = np.linspace(t[k], t[len(t) - k - 1], max(16, 4 * len(C)))
            P = np.asarray(splev(ps, (t, np.asarray(C).T, k))).T
        else:
            c, n, r, p0 = data
            u_ = (p0 - c) / max(np.linalg.norm(p0 - c), 1e-300)
            w_ = np.cross(n / np.linalg.norm(n), u_)
            q_ = plan[2][-1] - c
            sweep = float(np.arctan2(q_ @ w_, q_ @ u_)) % (2 * np.pi)
            if sweep < 1e-9 or np.linalg.norm(plan[2][-1] - plan[2][0]) <= self.min_gap:
                sweep = 2 * np.pi
            ps = np.linspace(0.0, sweep, max(9, int(np.ceil(48 * sweep / (2 * np.pi))) + 1))
            P = c + r * (np.outer(np.cos(ps), u_) + np.outer(np.sin(ps), w_))
        if len(ps) != len(P) or len(P) < 2 or np.any(np.diff(ps) <= 0):
            return None
        if helix is not None:
            # a helical face's B-spline is sampled in (s, x): every point's own coordinates
            hx, a0, a1, b0, b1, scale_u = helix
            sv, xu = hx.coords(np.asarray(P, float))
            X = np.c_[sv, xu]  # s is continuous over the turns already (no unwrap: thinned edges skip more than half a turn)
            if not np.isfinite(X).all():
                return None
            X[:, 0] += 2 * np.pi * np.round((0.5 * (a0 + a1) - X[:, 0].mean()) / (2 * np.pi)) if (
                X[:, 0].min() < a0 or X[:, 0].max() > a1) else 0.0
            if X[:, 0].min() < a0 or X[:, 0].max() > a1 or X[:, 1].min() < b0 or X[:, 1].max() > b1:
                return None
            X[:, 0] *= scale_u
            return X, np.r_[ps[0], ps, ps[-1]]
        X, _ = frame.to_param(np.asarray(P, float))
        if not np.isfinite(X).all():
            return None
        if frame.kind == "cylindrical":
            X[:, 0] = np.unwrap(X[:, 0])
            mid = 0.5 * (a0 + a1)
            X[:, 0] += 2 * np.pi * np.round((mid - X[:, 0].mean()) / (2 * np.pi))
        ma, mb = 0.02 * (a1 - a0), 0.02 * (b1 - b0)
        if X[:, 0].min() < a0 - ma or X[:, 0].max() > a1 + ma or X[:, 1].min() < b0 - mb or X[:, 1].max() > b1 + mb:
            return None
        return X, np.r_[ps[0], ps, ps[-1]]

    def curve_plan(self, pts: np.ndarray, circles: bool = True, max_tol: float = np.inf):
        """What curve to write through these edge points: (kind, data, points it runs through)."""
        # points a CAD kernel cannot tell apart would make zero-length spans
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > self.min_gap]
        pts = pts[keep]
        if len(pts) < 2:
            pts = np.vstack([pts, pts])
        # The tolerance never exceeds a small share of the curve itself: a hole 2 mm
        # across must not be smoothed into a line because the model is 400 mm big.
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        tol = min(self.tol, 0.02 * length) if length > 0 else self.tol
        # A circle or arc (a hole's rim, a fillet's end) is written as an exact CIRCLE,
        # as CAD systems write it, so CAD tools can measure it and build on it. Judged
        # at the curve's own tolerance: the exact circle is what the model has.
        if circles and len(pts) >= 6 and len(douglas_peucker(pts, 0.5 * tol)) >= 3:  # fewer fit any circle
            circ = circle_through(pts, 0.25 * tol)
            if circ is not None:
                c, n, r = circ  # parameter 0 at the start vertex, as CAD systems write it
                return "circle", (c, n, r, pts[0]), pts
        # Otherwise the curve is thinned, never more than a share of the faces it
        # bounds: the two long edges of a strip narrower than the tolerance, each
        # thinned on its own, crossed each other. Dense, noisy boundaries (scans) are
        # thinned first: hundreds of poles add nothing below the tolerance.
        tol = min(tol, max_tol)
        simple = douglas_peucker(pts, 0.5 * tol)
        if len(pts) >= 6:
            try:
                tck, _ = splprep(pts.T, s=len(pts) * (0.4 * tol) ** 2, k=3)
                t, c, k = tck
                C = np.asarray(c).T[: len(t) - k - 1].copy()
                C[0], C[-1] = pts[0], pts[-1]
                # Keep the spline only if it follows the boundary: every sample within
                # tolerance, and no overshoot away from the (thinned) polyline. A short
                # sparse arc can overshoot badly, especially with pinned end poles.
                dense = np.asarray(splev(np.linspace(t[k], t[-k - 1], max(200, 10 * len(simple))), (t, C.T, k))).T
                fine = np.vstack([np.linspace(a, b, 8, endpoint=False) for a, b in zip(simple[:-1], simple[1:])]
                                 + [simple[-1:]])
                to_pts = np.percentile(cKDTree(dense).query(pts)[0], 98)
                overshoot = cKDTree(fine).query(dense)[0].max()
                if to_pts <= tol and overshoot <= tol and len(C) <= max(8, len(simple) + 4):
                    smp = dense[:: max(1, len(dense) // max(8, 4 * len(simple)))]
                    if np.linalg.norm(smp[-1] - dense[-1]) > 0:
                        smp = np.vstack([smp, dense[-1:]])
                    return "spline", (k, C, t), smp
            except Exception:
                pass
        pts = simple
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        d = d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(pts))
        t = np.r_[d[0], d, d[-1]]
        return "poly", (1, pts, t), pts

    # ---------------------------------------------------------- build
    def build(self):
        s = self.s
        st = self.st
        V, F, FL = s.mesh.vertices, s.mesh.faces, s.face_labels
        self._seam_dir = {}  # region -> seam direction shared along a chain of bands
        self._pc_frame = {}  # freeform region -> its patch's parameter frame and box
        self._pc_helix = {}  # helical region -> (surface, s range, x range) of its B-spline
        self.pcurves_written = 0
        self._sphere_axes = {}  # sphere region -> (axis, seam direction) from its boundary circle
        pairs, starts = vertex_labels(F, FL, len(V))
        adjacency = s.mesh.face_adjacency
        slits = None
        if CRACK_TWINS:
            # T-junction cracks join the triangles on their two sides (one patch, one body)
            # and, inside a patch, are no boundary
            cracks = crack_pairs(np.asarray(V), np.asarray(F))
            if len(cracks):
                ca, cb = cracks[:, 0] // 3, cracks[:, 1] // 3
                adjacency = np.vstack([adjacency, np.c_[ca, cb]])
        PL, region_of, body_of = face_patches(F, adjacency, FL)
        if CRACK_TWINS and len(cracks):
            slits = slit_halfedges(np.asarray(F), PL, cracks)
            self.warnings.append(f"{len(cracks)} crack edge pairs; {len(slits)} half-edges inside faces dropped")
        A, B, LAB, NB = boundary_halfedges(F, PL, len(V), slits)
        max_move = 0.05 * self.diag
        R = len(region_of)  # number of patches (one CAD face each)
        order = np.argsort(LAB, kind="stable")
        bounds_at = np.searchsorted(LAB[order], np.arange(R + 1))
        region_segs = {}
        for pid in range(R):
            sel = order[bounds_at[pid]:bounds_at[pid + 1]]
            region_segs[pid] = [split_loop(v, n) for v, n in patch_loops(A[sel], B[sel], NB[sel])]
        reg = lambda pid: int(region_of[pid])  # region (surface) of a patch
        tri_area, tri_n, tri_c = s.mesh.area_faces, s.mesh.face_normals, s.mesh.triangles_center
        patch_faces = np.argsort(PL, kind="stable")
        patch_at = np.searchsorted(PL[patch_faces], np.arange(R + 1))

        # Slivers: where coarse triangles meet at a corner, two regions can touch along
        # a single tiny mesh edge (two fillet cylinders that really meet only at the
        # corner sphere). Such a segment would become a degenerate B-rep edge, so it is
        # collapsed into one vertex - identically for every face that uses it.
        # A sliver is short for the model AND for the faces it borders: on a detailed
        # model (a building) whole faces are smaller than 1 % of the model, and
        # collapsing their edges would leave them without any boundary.
        short_len = 0.01 * self.diag
        seg_len = lambda seg: float(np.linalg.norm(np.diff(V[seg], axis=0), axis=1).sum())
        perim = np.zeros(R)
        for pid, loops in region_segs.items():
            perim[pid] = sum(seg_len(seg) for segs in loops for seg, _, _ in segs)
        # a face's typical width (2 x area / perimeter): the end edge of a thin strip
        # is as long as the strip is wide, and must not be collapsed
        patch_area = np.bincount(PL, weights=tri_area, minlength=R)
        width = 2 * patch_area / np.maximum(perim, 1e-300)
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # the smallest face at each boundary vertex: collapsing an edge moves its ends,
        # and a tiny face with a corner there (a pin's 1 mm cross-section next to two
        # long side faces) lost that corner when only the two faces along the edge
        # were asked
        v_perim: dict[int, float] = {}
        v_width: dict[int, float] = {}
        for pid_, loops_ in region_segs.items():
            for segs_ in loops_:
                for seg_, _, _ in segs_:
                    for v_ in (int(seg_[0]), int(seg_[-1])):
                        v_perim[v_] = min(v_perim.get(v_, np.inf), perim[pid_])
                        v_width[v_] = min(v_width.get(v_, np.inf), width[pid_])

        def small_at(v_, L) -> bool:
            # small all round (a long thin strip is narrow but not small: collapsing at
            # its end only shortens it)
            return v_perim.get(v_, np.inf) <= 10 * L

        def is_sliver(seg, closed, pid, nb) -> bool:
            if closed or len(seg) > 3:
                return False
            L = seg_len(seg)
            local = perim[pid] if nb < 0 else min(perim[pid], perim[nb])
            w = width[pid] if nb < 0 else min(width[pid], width[nb])
            if not (L < short_len and L < 0.1 * local and L < 0.25 * w):
                return False
            # A tiny face with a corner at either end (a pin's 1 mm cross-section next
            # to two long side faces) would lose that corner: the edge stays.
            return not (SLIVER_ASK_ENDS and (small_at(int(seg[0]), L) or small_at(int(seg[-1]), L)))

        for pid, loops in region_segs.items():
            for segs in loops:
                for seg, nb, closed in segs:
                    if is_sliver(seg, closed, pid, nb):
                        a, b = find(int(seg[0])), find(int(seg[-1]))
                        if a != b:
                            parent[max(a, b)] = min(a, b)
        members: dict[int, list[int]] = {}
        for v in list(parent):
            members.setdefault(find(v), []).append(v)

        vert_ids: dict[int, tuple[int, np.ndarray]] = {}
        seam_vertex: dict[int, object] = {}  # vertex -> frame whose seam it anchors

        reg_area = np.bincount(FL, weights=tri_area, minlength=max(self.fits) + 1 if self.fits else 0)

        def plane_of(labs):
            """The exact plane among these regions that holds their boundary exactly.

            Several planes can meet at a corner, and where two of them are parallel but
            offset (a 0.01 mm step: a riser between two big faces) no point is on both.
            The smallest plane gets the corner exactly: a big face hardly notices a
            corner a step height off, but a riser put on a big face collapses into a
            zero-area face. Chosen by area, not numbering, so the face order cannot
            change it."""
            pl = [int(l) for l in labs if self.fits[int(l)].cls.get("type") == "plane"]
            if not pl:
                return None
            return self.fits[min(pl, key=lambda l: (float(reg_area[l]), l))].eq

        pos_cache: dict[int, np.ndarray] = {}
        clamp_verts: set[int] = set()  # vertices kept near the mesh (crossing repair)
        def vertex_pos(v: int) -> np.ndarray:
            if v not in pos_cache:
                group = members.get(v, [v])
                labs = np.unique(np.concatenate([pairs[starts[m]:starts[m + 1], 1] for m in group]))
                p0 = V[group].mean(axis=0)[None]
                p = alt_project(p0, [self.fits[l].eq for l in labs], max_move, final=plane_of(labs))[0]
                if NECK_LIMIT_ALL or v in clamp_verts:
                    p = clamp_move(p0, p[None], [min(group, key=lambda m: move_lim[m])])[0]
                if v in ring_of:
                    p = on_ring(p, ring_of[v])
                fr_ = seam_vertex.get(v)
                if isinstance(fr_, TubeTorus):
                    # along the end circle onto the outer equator: still on the circle
                    # (and the end plane or straight pipe it joins)
                    th, _ = fr_.angles(p[None])
                    p = fr_.point(th, np.full(1, getattr(fr_, "pipe_phi", 0.0)))[0]
                elif fr_ is not None:
                    # rotate onto the seam half-plane: still on every coaxial surface it
                    # touches, and the seam becomes an exact iso-parameter line
                    X, yv = fr_.to_param(p[None])
                    X[0, 0] = vertex_seam_angle.get(v, np.pi)
                    p = fr_.from_param(X, yv)[0]
                if v in cone_snap:
                    # exactly onto the written cone's generator in the seam half-plane, so
                    # the seam is that straight line: projected onto the cone, any other
                    # seam wobbles across u = 0 and OpenCASCADE's mesher fails the face
                    cones_ = cone_snap[v]
                    o_, a_, r0_, k_, d_ = cones_[0]
                    w_ = p - o_
                    h_ = float(w_ @ a_)
                    rho_ = float(np.linalg.norm(w_ - h_ * a_))
                    c_ = 1.0 / np.sqrt(1.0 + k_ * k_)
                    t_ = ((rho_ - r0_) * k_ + h_) * c_
                    hf_, rf_ = t_ * c_, r0_ + t_ * k_ * c_
                    if len(cones_) > 1:
                        # two coaxial cones (a shared axis): where their seams cross
                        o2, a2, r2, k2, _ = cones_[1]
                        sg_ = 1.0 if float(a2 @ a_) > 0 else -1.0
                        cc_ = float((o2 - o_) @ a_)
                        den_ = k_ - k2 * sg_
                        if (abs(float(a2 @ a_)) > 1 - 1e-12 and np.linalg.norm(np.cross(o2 - o_, a_)) <= 1e-9 * self.diag
                                and abs(den_) > 1e-6):
                            hx_ = (r2 - k2 * sg_ * cc_ - r0_) / den_
                            rx_ = r0_ + k_ * hx_
                            if rx_ > 0 and np.hypot(hx_ - hf_, rx_ - rf_) <= self.tol:
                                hf_, rf_ = hx_, rx_
                    p = o_ + hf_ * a_ + rf_ * d_
                pos_cache[v] = p
            return pos_cache[v]

        def vertex(v: int) -> tuple[int, np.ndarray]:
            if v not in vert_ids:
                p = vertex_pos(v)
                vert_ids[v] = (st.add(f"VERTEX_POINT('',#{st.point(p)})"), p)
            return vert_ids[v]

        # How far a boundary point may move when it is put onto its surfaces: under half
        # the gap to the face's other boundary points (another loop, or the far side of
        # a thin neck of the same loop). Where a plane meets a slightly misfitting
        # surface at a shallow angle their intersection runs sideways of the true edge
        # by several times the misfit, and across a thin ring that crossed the loop
        # over itself; the mesh boundary of a CAD export is on the true edge already.
        move_lim = np.full(len(V), np.inf)
        for pid_, loops_ in region_segs.items():
            seq = []
            for li_, segs_ in enumerate(loops_):
                core_ = np.concatenate([np.asarray(sg_[0][:-1]) for sg_ in segs_]) if segs_ else np.zeros(0, int)
                if not len(core_):
                    continue
                arc_ = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(V[np.r_[core_, core_[:1]]], axis=0), axis=1))]
                for k_, v_ in enumerate(core_):
                    seq.append((int(v_), li_, float(arc_[k_]), float(arc_[-1])))
            if len(seq) < 4:
                continue
            ids_ = np.array([q[0] for q in seq])
            lid_ = np.array([q[1] for q in seq])
            pos_ = np.array([q[2] for q in seq])
            n_ = np.array([q[3] for q in seq])
            tree_ = cKDTree(V[ids_])
            dd_, jj_ = tree_.query(V[ids_], k=min(12, len(seq)))
            for i_ in range(len(seq)):
                for d__, j__ in zip(dd_[i_], jj_[i_]):
                    if j__ == i_ or d__ <= 0:
                        continue
                    if lid_[j__] == lid_[i_]:
                        sep = abs(pos_[j__] - pos_[i_])
                        if min(sep, n_[i_] - sep) <= 3 * d__:
                            continue  # near along the loop too: a curve, not a neck
                    move_lim[ids_[i_]] = min(move_lim[ids_[i_]], 0.45 * float(d__))
                    break

        def clamp_move(P0, Q, vs):
            # moves within a tenth of the tolerance (an exact surface's own projection)
            # always go through
            lim = np.maximum(move_lim[np.asarray(vs, int)], 0.1 * self.tol)
            d = Q - P0
            n = np.linalg.norm(d, axis=1)
            k = np.where((n > lim) & np.isfinite(lim), lim / np.maximum(n, 1e-300), 1.0)
            return P0 + d * k[:, None]

        # Full-turn bands (hole walls, hubs, fillet rings): two closed boundary circles
        # on a periodic surface. Give each a seam edge like a CAD system would, and
        # start both circles at the seam so the shared circle edges line up.
        seam_regions = {}
        sphere_R: dict[int, float] = {}  # sphere rings: radius (their seam is a meridian)

        def cone_fits(cls, fr_, vs) -> bool:
            """Does the cone written to STEP (from the classification) pass through these
            boundary vertices? Only then can its seam be its exact straight generator."""
            r0, k = float(cls.get("radius0", 0)), float(cls.get("slope", 0))
            if r0 <= 0 or abs(k) <= 1e-6:
                return False  # written as a B-spline patch
            v_ = V[vs] - fr_.origin
            h_ = v_ @ fr_.e3
            rho_ = np.linalg.norm(v_ - np.outer(h_, fr_.e3), axis=1)
            return float(np.abs(rho_ - r0 - k * h_).max()) / np.sqrt(1 + k * k) <= 0.25 * self.tol

        cone_snap: dict[int, list] = {}  # seam vertex -> cones [(origin, axis, r0, slope, seam direction)]
        ring_of: dict[int, tuple] = {}  # vertex -> circle it sits on (axis point, axis, radius, height)

        def on_ring(p, ring):
            p_, a_, rq, hq = ring
            w_ = p - p_
            d_ = w_ - (w_ @ a_) * a_
            n_ = float(np.linalg.norm(d_))
            return p if n_ <= 0 else p_ + hq * a_ + (rq / n_) * d_
        tube_regions: dict[int, TubeTorus] = {}  # pipe elbows: loops round the tube
        for rid in range(R):
            fit_ = self.fits[reg(rid)]
            fr_ = fit_.eq.frame
            loops = region_segs[rid]
            typ = fit_.cls.get("type")
            if (typ == "sphere" and fit_.cls.get("radius", 0) > 0 and len(loops) in (1, 2)
                    and all(len(sg) == 1 and sg[0][2] for sg in loops) and reg(rid) not in self._sphere_axes):
                # a dome or ring bounded by whole circles: the sphere's axis is the
                # circles' normal (exact on a CAD mesh), so they become latitude lines
                big = max((np.asarray(sg[0][0][:-1]) for sg in loops), key=len)
                Pc = V[big]
                sa_ = getattr(fit_, "shared_axis", None)
                if len(Pc) >= 6 or sa_ is not None:
                    ax = (np.array(sa_[2], float) if sa_ is not None
                          else np.linalg.svd(Pc - Pc.mean(axis=0), full_matrices=False)[2][-1])
                    Pr = s.P[s.regions[reg(rid)].idx]
                    if ax @ (Pr.mean(axis=0) - fr_.origin) < 0:
                        ax = -ax
                    ref = -fr_.e1 - (-fr_.e1 @ ax) * ax
                    if np.linalg.norm(ref) < 1e-6:
                        ref = fr_.e3 - (fr_.e3 @ ax) * ax
                    ref /= np.linalg.norm(ref)
                    w_ = np.cross(ax, ref)
                    ang = np.sort(np.arctan2((Pr - fr_.origin) @ w_, (Pr - fr_.origin) @ ref))
                    if np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max() <= np.radians(30):
                        self._sphere_axes[reg(rid)] = (ax, ref)
            if len(loops) != 2:
                continue
            single = all(len(sg) == 1 and sg[0][2] for sg in loops)
            cores = [np.concatenate([np.asarray(seg[:-1]) for seg, _, _ in sg]) for sg in loops]
            if typ == "tube":
                # no seam of ours: OpenCASCADE heals a closed tube face's missing seam and
                # meshes it right, while one along our seam line meshed only a sliver
                continue
            if typ == "torus" and _full_tube(fit_.cls, s.P[s.regions[reg(rid)].idx]):
                tor = TubeTorus.from_cls(fit_.cls, s.P[s.regions[reg(rid)].idx])
                if all(abs(_winding(tor.angles(V[cr])[1])) > np.pi for cr in cores):
                    tube_regions[rid] = tor
                continue
            if single and typ == "sphere" and fit_.cls.get("radius", 0) > 0:
                # a ring round an axis through the centre: written with that axis, and
                # seamed like a cylinder band rather than left to the reader to heal
                if reg(rid) in self._sphere_axes:
                    ax, ref = self._sphere_axes[reg(rid)]
                    sf = Frame("cylindrical", fr_.origin, -ref, np.cross(ax, -ref), ax, fr_.scale)
                    if all(abs(_winding(sf.to_param(V[cr])[0][:, 0])) > np.pi for cr in cores):
                        seam_regions[rid] = sf
                        sphere_R[rid] = float(fit_.cls["radius"])
                continue
            if single and typ == "cone" and not cone_fits(fit_.cls, fr_, np.concatenate(cores)):
                continue  # the written cone misses its own circles: the reader seams it
            if single and (fr_.kind == "cylindrical" and typ in ("cylinder", "cone", "revolve", "torus", "freeform")
                    and all(abs(_winding(fr_.to_param(V[cr])[0][:, 0])) > np.pi for cr in cores)):
                seam_regions[rid] = fr_
        # Two cone bands meeting at a circle, each fitted round its own axis, are never
        # exactly coaxial, so one seam vertex cannot sit exactly on both their straight
        # seams, and a seam that misses its cone by even 1e-6 rad fails OpenCASCADE's
        # mesher: the second keeps the reader's seam. On one shared axis (share_axes)
        # the vertex goes where both seams cross.
        cone_circles: dict[frozenset, object] = {}
        for rid in sorted(seam_regions):
            if self.fits[reg(rid)].cls.get("type") != "cone":
                continue
            g_ = getattr(self.fits[reg(rid)], "shared_axis", (None,))[0]
            keys_ = [frozenset(np.asarray(sg[0][0][:-1]).tolist()) for sg in region_segs[rid]]
            if any(k_ in cone_circles and (g_ is None or cone_circles[k_] != g_) for k_ in keys_):
                del seam_regions[rid]
            else:
                for k_ in keys_:
                    cone_circles.setdefault(k_, g_)
        # Bands stacked along one axis share their circles, and a circle has one start
        # vertex. Each band's seam runs between the start vertices of its two circles,
        # so every band in such a chain takes the same seam half-plane (the first
        # band's): with each band's own, a seam twists round the band from one circle's
        # start to the other's, and CAD kernels build a broken face.
        circle_bands: dict[frozenset, list[int]] = {}
        for rid in seam_regions:
            for sg in region_segs[rid]:
                if len(sg) == 1 and sg[0][2]:
                    circle_bands.setdefault(frozenset(np.asarray(sg[0][0][:-1]).tolist()), []).append(rid)
        chain: dict[int, int] = {rid: rid for rid in seam_regions}

        def chain_root(x: int) -> int:
            while chain[x] != x:
                chain[x] = chain[chain[x]]
                x = chain[x]
            return x

        for bands in circle_bands.values():
            for b_ in bands[1:]:
                ra_, rb_ = chain_root(bands[0]), chain_root(b_)
                if ra_ != rb_:
                    chain[max(ra_, rb_)] = min(ra_, rb_)
        # A pipe: straight bands joined by elbows. A circle between two faces has one
        # start vertex; an elbow's seam runs along the tube between the start vertices of
        # its two circles and a band's along its straight side. With each band seamed on
        # its own side the elbow's seam spirals round the tube from one start to the other
        # (a quarter turn is typical), and readers mesh that face with holes. So seams are
        # chosen along the pipe: from a band's seam the elbow's seam keeps that tube angle
        # round the bend, which gives the seam of the band after it, and so on.
        pipe_ref: dict[int, np.ndarray] = {}  # chain root -> seam direction
        circle_regs: dict[frozenset, list[int]] = {}
        for rid in sorted(set(seam_regions) | set(tube_regions)):
            for sg in region_segs[rid]:
                if len(sg) == 1 and sg[0][2]:
                    circle_regs.setdefault(frozenset(np.asarray(sg[0][0][:-1]).tolist()), []).append(rid)

        def band_dir(rid):
            f0 = seam_regions[chain_root(rid)]
            d_ = pipe_ref.get(chain_root(rid), -f0.e1)
            fr_ = seam_regions[rid]
            d_ = d_ - (d_ @ fr_.e3) * fr_.e3
            return d_ / np.linalg.norm(d_) if np.linalg.norm(d_) > 1e-6 else -fr_.e1

        def circle_point(rid, key, d_):
            """The point of circle `key` (on band rid) in the direction d_ from its axis."""
            fr_ = seam_regions[rid]
            Q = V[np.fromiter(key, np.int64)]
            w_ = Q - fr_.origin
            h_ = w_ @ fr_.e3
            rad_ = w_ - np.outer(h_, fr_.e3)
            return fr_.origin + float(h_.mean()) * fr_.e3 + float(np.linalg.norm(rad_, axis=1).mean()) * d_

        if tube_regions and seam_regions:
            seen_ = set()
            for start in sorted(seam_regions):
                if chain_root(start) in seen_:
                    continue
                queue = [r_ for r_ in sorted(seam_regions) if chain_root(r_) == chain_root(start)]
                seen_.add(chain_root(start))
                while queue:
                    rid = queue.pop(0)
                    d_ = band_dir(rid)
                    for sg in region_segs[rid]:
                        if not (len(sg) == 1 and sg[0][2]):
                            continue
                        key = frozenset(np.asarray(sg[0][0][:-1]).tolist())
                        for t_ in circle_regs.get(key, []):
                            tor = tube_regions.get(t_)
                            if tor is None or hasattr(tor, "pipe_phi") or hasattr(tor, "seam_points"):
                                continue
                            tor.pipe_phi = float(tor.angles(circle_point(rid, key, d_)[None])[1][0])
                            for sg2 in region_segs[t_]:
                                if not (len(sg2) == 1 and sg2[0][2]):
                                    continue
                                key2 = frozenset(np.asarray(sg2[0][0][:-1]).tolist())
                                if key2 == key:
                                    continue
                                th2 = tor.angles(V[np.fromiter(key2, np.int64)])[0]
                                q_ = tor.point(float(np.angle(np.mean(np.exp(1j * th2)))), tor.pipe_phi)[0]
                                for r2 in circle_regs.get(key2, []):
                                    if r2 not in seam_regions or chain_root(r2) in seen_:
                                        continue
                                    fr2 = seam_regions[r2]
                                    w2 = q_ - fr2.origin
                                    d2 = w2 - (w2 @ fr2.e3) * fr2.e3
                                    if np.linalg.norm(d2) < 1e-9:
                                        continue
                                    pipe_ref[chain_root(r2)] = d2 / np.linalg.norm(d2)
                                    seen_.add(chain_root(r2))
                                    queue += [r_ for r_ in sorted(seam_regions) if chain_root(r_) == chain_root(r2)]
        seam_angle: dict[int, float] = {}  # band -> seam angle in its own frame
        for rid in sorted(seam_regions):
            fr_, f0 = seam_regions[rid], seam_regions[chain_root(rid)]
            ref = pipe_ref.get(chain_root(rid), -f0.e1)
            d_ = ref - (ref @ fr_.e3) * fr_.e3
            if np.linalg.norm(d_) < 1e-6:
                d_ = -fr_.e1
            d_ = d_ / np.linalg.norm(d_)
            seam_angle[rid] = float(np.arctan2(d_ @ fr_.e2, d_ @ fr_.e1))
            self._seam_dir.setdefault(reg(rid), d_)
        vertex_seam_angle: dict[int, float] = {}
        pref: dict[frozenset, int] = {}
        for rid, fr_ in list(seam_regions.items()) + list(tube_regions.items()):
            for sg in region_segs[rid]:
                if not (len(sg) == 1 and sg[0][2]):
                    continue  # several faces along this circle: the seam starts at a junction
                core = np.asarray(sg[0][0][:-1])
                if isinstance(fr_, TubeTorus) or hasattr(fr_, "seam_points"):
                    ang = fr_.angles(V[core])[1] - getattr(fr_, "pipe_phi", 0.0)  # at the seam's tube angle
                    k = int(np.argmin(np.abs(np.angle(np.exp(1j * ang)))))
                else:
                    th = fr_.to_param(V[core])[0][:, 0]
                    k = int(np.argmin(np.abs(np.angle(np.exp(1j * (th - seam_angle[rid]))))))
                key = frozenset(core.tolist())
                if key not in pref:
                    pref[key] = int(core[k])
                    if not hasattr(fr_, "seam_points"):
                        seam_vertex[int(core[k])] = fr_
                        if rid in seam_angle:
                            vertex_seam_angle[int(core[k])] = seam_angle[rid]
                c_ = self.fits[reg(rid)].cls
                if rid in seam_angle and c_.get("type") == "cone":
                    cone_snap.setdefault(pref[key], []).append(
                        (fr_.origin, fr_.e3, float(c_["radius0"]), float(c_["slope"]), self._seam_dir[reg(rid)]))

        # Caps with holes: a cone or dome face that covers its apex or pole has one loop
        # round the axis. Without holes the reader adds the seam and the point at the
        # apex; with holes (the slots in a tank roof) that repair fails and the face is
        # lost. Such a face gets our own seam, from its rim's start vertex along the
        # cone's straight generator (a sphere's meridian) to the apex, at an angle clear
        # of the holes; the reader then only adds the point.
        cap_seams: dict[int, tuple] = {}  # patch -> (rim loop index, apex, kind, frame)
        for rid in range(R):
            if rid in seam_regions or rid in tube_regions:
                continue
            fit_ = self.fits[reg(rid)]
            c_ = fit_.cls
            typ = c_.get("type")
            if (typ == "cone" and c_.get("radius0", 0) > 0 and abs(c_.get("slope", 0)) > 1e-6
                    and fit_.eq.frame.kind == "cylindrical"):
                cf = fit_.eq.frame
            elif typ == "sphere" and c_.get("radius", 0) > 0 and reg(rid) in self._sphere_axes:
                ax_, ref_ = self._sphere_axes[reg(rid)]
                cf = Frame("cylindrical", fit_.eq.frame.origin, ref_, np.cross(ax_, ref_), ax_, fit_.eq.frame.scale)
            elif (typ == "torus" and isinstance(fit_.eq, ExactTorus) and fit_.eq.R < fit_.eq.r
                  and not _full_tube(c_, s.P[s.regions[reg(rid)].idx])):
                # a dome on a spindle torus: written as a surface of revolution, which the
                # reader cannot close round its pole even without holes
                tq = fit_.eq
                e1_ = _perp(tq.axis)
                cf = Frame("cylindrical", tq.center, e1_, np.cross(tq.axis, e1_), tq.axis, fit_.eq.frame.scale)
            else:
                continue
            loops = region_segs[rid]
            if len(loops) < (1 if typ == "torus" else 2):
                continue
            ths = []
            for sg in loops:
                core = np.concatenate([np.asarray(seg[:-1]) for seg, _, _ in sg])
                ths.append(cf.to_param(V[core])[0][:, 0] if len(core) else np.zeros(0))
            wind = [len(t) >= 3 and abs(_winding(t)) > np.pi for t in ths]
            if sum(wind) != 1:
                continue
            wi = wind.index(True)
            sg = loops[wi]
            if not (len(sg) == 1 and sg[0][2]):
                continue  # the rim must be one closed edge: the seam starts at its vertex
            hole_th = np.concatenate([t for i, t in enumerate(ths) if i != wi] + [np.zeros(0)])
            core = np.asarray(sg[0][0][:-1])
            key = frozenset(core.tolist())
            if key in pref:
                v0 = pref[key]
                fv = seam_vertex.get(v0)
                if fv is None or isinstance(fv, TubeTorus) or v0 not in vertex_seam_angle:
                    continue
                a0 = vertex_seam_angle[v0]
                dv = np.cos(a0) * fv.e1 + np.sin(a0) * fv.e2
                th0 = float(np.arctan2(dv @ cf.e2, dv @ cf.e1))
            else:
                if len(hole_th):
                    srt = np.sort(np.angle(np.exp(1j * hole_th)))
                    gaps = np.diff(np.r_[srt, srt[0] + 2 * np.pi])
                    k_ = int(np.argmax(gaps))
                    th0 = float(np.angle(np.exp(1j * (srt[k_] + 0.5 * gaps[k_]))))
                else:
                    th0 = float(ths[wi][0])
                k0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (ths[wi] - th0))))))
                v0 = int(core[k0])
                pref[key] = v0
                seam_vertex[v0] = cf
                vertex_seam_angle[v0] = th0
            if len(hole_th) and np.abs(np.angle(np.exp(1j * (hole_th - th0)))).min() < np.radians(2):
                continue  # the seam would run through a hole
            d0 = np.cos(th0) * cf.e1 + np.sin(th0) * cf.e2
            if typ == "torus":
                tq = fit_.eq
                Pr_ = s.P[s.regions[reg(rid)].idx]
                sgn = 1.0 if float((Pr_.mean(axis=0) - tq.center) @ tq.axis) >= 0 else -1.0
                apex = tq.center + sgn * float(np.sqrt(max(tq.r ** 2 - tq.R ** 2, 0.0))) * tq.axis
                self._seam_dir[reg(rid)] = d0  # the surface's profile is the seam
                cap_seams[rid] = (wi, apex, typ, cf)
                continue
            if typ == "cone":
                r0_, k_s = float(c_["radius0"]), float(c_["slope"])
                apex = cf.origin - (r0_ / k_s) * cf.e3
                cone_snap.setdefault(v0, []).append((cf.origin, cf.e3, r0_, k_s, d0))
            else:
                Pr_ = s.P[s.regions[reg(rid)].idx]
                sgn = 1.0 if float((Pr_.mean(axis=0) - cf.origin) @ cf.e3) >= 0 else -1.0
                apex = cf.origin + sgn * float(c_["radius"]) * cf.e3
            cap_seams[rid] = (wi, apex, typ, cf)

        # turned faces that go all the way round without a seam of ours
        reader_seamed: set[int] = set()
        for pid in range(R):
            fit_ = self.fits[reg(pid)]
            typ_ = fit_.cls.get("type")
            if pid in seam_regions or pid in tube_regions or pid in cap_seams or typ_ not in READER_SEAMED:
                continue
            if typ_ == "tube" or (typ_ == "sphere" and reg(pid) in self._sphere_axes):
                reader_seamed.add(pid)
                continue
            fr_ = fit_.eq.frame
            if fr_.kind != "cylindrical":
                continue
            for sg in region_segs[pid]:
                core = np.concatenate([np.asarray(seg[:-1]) for seg, _, _ in sg])
                if len(core) >= 3 and abs(_winding(fr_.to_param(V[core])[0][:, 0])) > np.pi:
                    reader_seamed.add(pid)
                    break

        # exact circles only next to faces whose curves OpenCASCADE places analytically
        # or that do not go all the way round: on a surface of revolution with our seam
        # (a rim torus) a full circle starting at the seam came out a turn off
        no_circle = reader_seamed | set(tube_regions) | {
            p_ for p_ in seam_regions if self.fits[reg(p_)].cls.get("type") not in ("cylinder", "sphere", "cone")}

        # Loops of one face that touch or all but touch (a round hole tangent to a
        # cutout, two cutouts meeting at a point): the exact circle can cross the other
        # loop where the polyline, running just inside it, did not. Their edges stay
        # polylines (pinch holds every vertex of such loops).
        pinch: set[int] = set()
        near_ = max(10 * self.tol, 1e-3 * self.diag)
        for loops_ in region_segs.values():
            lv_ = []
            for segs_ in loops_:
                vv_ = sorted({find(int(v_)) for seg_, _, _ in segs_ for v_ in seg_})
                lv_.append(vv_)
                ends_: dict[int, int] = {}
                for seg_, _, _ in segs_:
                    for v_ in (find(int(seg_[0])), find(int(seg_[-1]))):
                        ends_[v_] = ends_.get(v_, 0) + 1
                if any(k_ > 2 for k_ in ends_.values()):  # visits a vertex twice
                    pinch.update(vv_)
            if len(lv_) > 1:
                trees_ = [cKDTree(V[vv_]) for vv_ in lv_]
                for a_ in range(len(lv_)):
                    for b_ in range(a_ + 1, len(lv_)):
                        if trees_[a_].query(V[lv_[b_]], distance_upper_bound=near_)[0].min() < near_:
                            pinch.update(lv_[a_])
                            pinch.update(lv_[b_])

        # Coaxial turned faces (one shared axis, see share_axes) meet in a circle round
        # that axis. Where they meet smoothly (a fillet running into a wall) the mesh
        # boundary wanders across the join, and projecting each point onto both
        # surfaces keeps that waviness: every point of such a loop goes onto one circle,
        # at the loop's median height and radius solved onto both surfaces.
        self._ring_debug = []
        bverts = {pid: {int(v_) for sg in loops for seg, _, _ in sg for v_ in seg} for pid, loops in region_segs.items()}
        for pid in range(R):
            sa_ = getattr(self.fits[reg(pid)], "shared_axis", None)
            if sa_ is None:
                continue
            _, p_, a_ = sa_
            for sg in region_segs[pid]:
                nbs = [nb for _, nb, _ in sg]
                if any(nb < 0 or reg(nb) == reg(pid) for nb in nbs):
                    continue
                if any(getattr(self.fits[reg(nb)], "shared_axis", (None,))[0] != sa_[0] for nb in nbs):
                    self._ring_debug.append((reg(pid), [reg(x) for x in nbs], "other axis"))
                    continue
                core = np.concatenate([np.asarray(seg[:-1]) for seg, _, _ in sg])
                if len(core) < 6 or any(int(v_) in ring_of or find(int(v_)) != int(v_) for v_ in core):
                    continue
                u_ = _perp(a_)
                w_ = V[core] - p_
                if abs(_winding(np.arctan2(w_ @ np.cross(a_, u_), w_ @ u_))) < np.pi:
                    continue  # not round the axis
                eq0 = self.fits[reg(pid)].eq
                Q_ = np.vstack([alt_project(V[np.asarray(seg[:-1])], [eq0, self.fits[reg(nb)].eq], max_move)
                                for seg, nb, _ in sg if len(seg) > 1])
                eqs_ = [eq0] + [self.fits[g_].eq for g_ in sorted({reg(nb) for nb in nbs})]
                w_ = Q_ - p_
                h_ = w_ @ a_
                rho_ = np.linalg.norm(w_ - np.outer(h_, a_), axis=1)
                d0_ = (w_[0] - h_[0] * a_) / max(float(rho_[0]), 1e-300)
                q0_ = p_ + float(np.median(h_)) * a_ + float(np.median(rho_)) * d0_
                q_ = alt_project(q0_[None], eqs_, max_move)[0]
                hq = float((q_ - p_) @ a_)
                rq = float(np.linalg.norm((q_ - p_) - hq * a_))
                spread_ = float(np.hypot(rho_ - rq, h_ - hq).max())
                # the circle must lie on the two faces of every edge along the loop (a
                # loop stepping between two heights is no circle)
                off_ = 0.0
                for seg, nb, _ in sg:
                    P_ = V[np.asarray(seg)]
                    d_ = P_ - p_
                    d_ = d_ - np.outer(d_ @ a_, a_)
                    d_ /= np.maximum(np.linalg.norm(d_, axis=1, keepdims=True), 1e-300)
                    Rg_ = p_ + hq * a_ + rq * d_
                    for e_ in (eq0, self.fits[reg(nb)].eq):
                        dd_ = np.linalg.norm(closest(e_, Rg_) - Rg_, axis=1)
                        off_ = max(off_, float(np.nanmax(dd_)) if np.isfinite(dd_).any() else np.inf)
                if rq <= 0 or not off_ <= 0.25 * self.tol:
                    self._ring_debug.append((reg(pid), [reg(x) for x in nbs], f"off {off_:.3g} spread {spread_:.3g}"))
                    continue
                # points move along the faces by up to the spread: never as far as half
                # the way to another boundary of these faces (a thin band would fold,
                # an edge meeting the loop would kink back)
                if spread_ > 0.25 * self.tol:
                    others_ = set().union(bverts[pid], *(bverts[nb] for nb in nbs)) - {int(v_) for v_ in core}
                    gap_ = (float(cKDTree(V[sorted(others_)]).query(V[core])[0].min()) if others_ else np.inf)
                    if spread_ > 0.5 * gap_:
                        self._ring_debug.append((reg(pid), [reg(x) for x in nbs], f"spread {spread_:.3g} gap {gap_:.3g}"))
                        continue
                self._ring_debug.append((reg(pid), [reg(x) for x in nbs], f"ring r={rq:.5g} h={hq:.5g} spread {spread_:.3g} off {off_:.3g}"))
                for v_ in core:
                    ring_of[int(v_)] = (p_, a_, rq, hq)
        if ring_of:
            self.warnings.append(f"{sum(1 for d in self._ring_debug if d[2].startswith('ring'))} junction loops "
                                 "between coaxial faces put on exact circles")

        tube_start: dict[tuple, int] = {}
        for rid, tor in tube_regions.items():
            for li, sg in enumerate(region_segs[rid]):
                if len(sg) == 1 and sg[0][2]:
                    continue
                best = None
                for si, (seg, nb, closed) in enumerate(sg):
                    ph = np.abs(np.angle(np.exp(1j * (tor.angles(V[np.asarray(seg)])[1] - getattr(tor, "pipe_phi", 0.0)))))
                    k = int(np.argmin(ph))
                    if best is None or ph[k] < best[0]:
                        best = (float(ph[k]), si, k)
                _, si, k = best
                seg, nb, closed = sg[si]
                seg = np.asarray(seg)
                v = int(seg[k])
                if 0 < k < len(seg) - 1:
                    # a vertex inside one edge: split that edge here (on both faces) and
                    # move the vertex along the end circle onto the outer equator
                    sg[si:si + 1] = [(seg[:k + 1], nb, False), (seg[k:], nb, False)]
                    if nb >= 0:
                        done = False
                        for sg2 in region_segs[nb]:
                            for sj, (s2, nb2, c2) in enumerate(sg2):
                                s2 = np.asarray(s2)
                                if nb2 == rid and not c2 and v in s2[1:-1].tolist():
                                    j = int(np.where(s2 == v)[0][0])
                                    sg2[sj:sj + 1] = [(s2[:j + 1], nb2, False), (s2[j:], nb2, False)]
                                    done = True
                                    break
                            if done:
                                break
                    if not hasattr(tor, "seam_points"):
                        seam_vertex[v] = tor
                tube_start[(rid, li)] = v

        def seam_edge(rid: int, a: int, b: int) -> int:
            fr_ = seam_regions[rid]
            eq = self.fits[reg(rid)].eq
            va, pa = vertex(a)
            vb, pb = vertex(b)
            if self.fits[reg(rid)].cls.get("type") == "cone" and np.linalg.norm(pb - pa) > 0:
                # the cone's straight generator (both vertices sit on it, see cone_snap)
                d_ = pb - pa
                vec_ = st.add(f"VECTOR('',#{st.direction(d_)},{fr(float(np.linalg.norm(d_)))})")
                line_ = st.add(f"LINE('',#{st.point(pa)},#{vec_})")
                return st.add(f"EDGE_CURVE('',#{va},#{vb},#{line_},.T.)")
            Xa, Xb = fr_.to_param(pa[None])[0][0], fr_.to_param(pb[None])[0][0]
            dth = float(np.angle(np.exp(1j * (Xb[0] - Xa[0]))))
            G = np.c_[np.linspace(Xa[0], Xa[0] + dth, 24), np.linspace(Xa[1], Xb[1], 24)]
            if isinstance(eq, ExactTorus):
                pts = self.torus_seam(reg(rid), eq, pa, pb)  # by region: rid is a patch
            elif rid in sphere_R:  # a meridian
                hh = G[:, 1] * fr_.scale
                pts = fr_.from_param(G, np.sqrt(np.maximum(sphere_R[rid] ** 2 - hh ** 2, 0.0)) / fr_.scale)
            else:
                pts = fr_.from_param(G, eq(G))
            if not np.isfinite(pts).all():
                pts = np.linspace(pa, pb, 24)
            pts[0], pts[-1] = pa, pb
            return st.add(f"EDGE_CURVE('',#{va},#{vb},#{self.curve(pts)},.T.)")

        def cap_seam_edge(rid: int, a: int) -> int:
            """A cap's seam: from its rim vertex to the apex (the cone's straight
            generator, a sphere's meridian)."""
            _, apex, typ, cf = cap_seams[rid]
            va, pa = vertex(a)
            vb = st.add(f"VERTEX_POINT('',#{st.point(apex)})")
            if typ == "cone":
                d_ = apex - pa
                vec_ = st.add(f"VECTOR('',#{st.direction(d_)},{fr(float(np.linalg.norm(d_)))})")
                line_ = st.add(f"LINE('',#{st.point(pa)},#{vec_})")
                return st.add(f"EDGE_CURVE('',#{va},#{vb},#{line_},.T.)")
            if typ == "torus":
                tq = self.fits[reg(rid)].eq
                w0 = pa - tq.center
                h0 = float(w0 @ tq.axis)
                dd = w0 - h0 * tq.axis
                dd /= max(np.linalg.norm(dd), 1e-300)
                rho0 = float(np.linalg.norm(w0 - h0 * tq.axis))
                hp = float((apex - tq.center) @ tq.axis)
                f0 = np.arctan2(h0, rho0 - tq.R)
                f1 = np.arctan2(hp, -tq.R)
                df = float(np.angle(np.exp(1j * (f1 - f0))))
                ph_ = f0 + df * np.linspace(0.0, 1.0, 32)
                pts = (tq.center + np.outer(tq.R + tq.r * np.cos(ph_), dd) + np.outer(tq.r * np.sin(ph_), tq.axis))
                pts[0], pts[-1] = pa, apex
                return st.add(f"EDGE_CURVE('',#{va},#{vb},#{self.curve(pts)},.T.)")
            c0 = cf.origin
            u0 = (pa - c0) / np.linalg.norm(pa - c0)
            u1 = (apex - c0) / np.linalg.norm(apex - c0)
            om = float(np.arccos(np.clip(u0 @ u1, -1.0, 1.0)))
            t_ = np.linspace(0.0, 1.0, 24)
            dirs = (np.sin((1 - t_) * om)[:, None] * u0 + np.sin(t_ * om)[:, None] * u1) / max(np.sin(om), 1e-300)
            pts = c0 + float(np.linalg.norm(pa - c0)) * dirs
            pts[0], pts[-1] = pa, apex
            return st.add(f"EDGE_CURVE('',#{va},#{vb},#{self.curve(pts)},.T.)")

        def tube_seam_edge(rid: int, a: int, b: int) -> int:
            """The seam of a pipe elbow: along the tube at the seam vertices' tube angle,
            the way round the axis that crosses the face."""
            tor = tube_regions[rid]
            va, pa = vertex(a)
            vb, pb = vertex(b)
            if hasattr(tor, "seam_points"):
                pts = tor.seam_points(pa, pb)
                return st.add(f"EDGE_CURVE('',#{va},#{vb},#{self.curve(pts)},.T.)")
            (ta, tb), (fa, fb) = tor.angles(np.vstack([pa, pb]))
            ref = float(np.angle(np.mean(np.exp(1j * tor.angles(s.P[s.regions[reg(rid)].idx])[0]))))
            d1 = float(np.angle(np.exp(1j * (tb - ta))))
            d2 = d1 - np.sign(d1) * 2 * np.pi if d1 != 0 else 2 * np.pi
            m1 = abs(np.angle(np.exp(1j * (ta + d1 / 2 - ref))))
            m2 = abs(np.angle(np.exp(1j * (ta + d2 / 2 - ref))))
            d = d1 if m1 <= m2 else d2
            t_ = np.linspace(0.0, 1.0, 32)
            pts = tor.point(ta + d * t_, fa + float(np.angle(np.exp(1j * (fb - fa)))) * t_)
            pts[0], pts[-1] = pa, pb
            return st.add(f"EDGE_CURVE('',#{va},#{vb},#{self.curve(pts)},.T.)")

        # Segments whose two ends land on the same point once projected onto their
        # surfaces (a short side edge of a thin step, seen from the plane it bounds)
        # would become zero-length edges, which CAD kernels cannot build. Merge their
        # ends like slivers before anything is written.
        eps_pos = max(1e-6 * self.diag, 1e-3 * self.tol)
        extent = lambda seg: float(np.ptp(V[seg], axis=0).max())
        merged = 0
        for pid, loops in region_segs.items():
            for segs in loops:
                for seg, nb, closed in segs:
                    if closed or is_sliver(seg, closed, pid, nb):
                        continue
                    a, b = find(int(seg[0])), find(int(seg[-1]))
                    if a == b or extent(seg) >= short_len:
                        continue
                    if np.linalg.norm(vertex_pos(a) - vertex_pos(b)) < eps_pos:
                        parent[max(a, b)] = min(a, b)
                        merged += 1
        if merged:
            members = {}
            for v in list(parent):
                members.setdefault(find(v), []).append(v)
            pos_cache.clear()
            self.warnings.append(f"{merged} edges that collapse to a point were merged into their vertex")
        edges: dict[tuple, int] = {}
        edge_uses: dict[tuple, int] = {}
        faces = []
        face_body = []
        face_part = getattr(s, "face_part", None)
        body_part: dict[int, int] = {}
        edge_dev: list[float] = []
        dropped = 0
        edge_pts: dict = {}
        face_loops: dict[int, list] = {}
        surfaces: dict[int, tuple[int, object]] = {}
        # Pass 1: every face's loops and every edge's points; nothing is written yet, so
        # edges whose curves cross in a face can be redone before the file is built.
        face_plan: dict[int, tuple] = {}
        edge_geo: dict[tuple, dict] = {}
        vert_edges: dict[int, list] = {}
        fixed_verts: set[int] = set()  # positions other entities were already built on
        for rid in range(R):
            if rid % 50 == 0:
                self.progress(rid / max(R, 1), f"STEP face {rid + 1}/{R}")
            g = reg(rid)
            if g not in surfaces and rid in tube_regions and hasattr(tube_regions[rid], "seam_points"):
                ends = []
                for li_, sg_ in enumerate(region_segs[rid]):
                    if len(sg_) == 1 and sg_[0][2]:
                        w_ = pref.get(frozenset(np.asarray(sg_[0][0][:-1]).tolist()))
                    else:
                        w_ = tube_start.get((rid, li_))
                    if w_ is not None:
                        ends.append(vertex_pos(find(int(w_))))
                        fixed_verts.add(find(int(w_)))
                if len(ends) == 2:
                    tube_regions[rid].set_seam(ends[0], ends[1])
            if g not in surfaces:
                self._normal_fn = None
                sid, same_region = self.surface(g)
                surfaces[g] = (sid, self._normal_fn, same_region)
            surf, nfn, same = surfaces[g]
            fids = patch_faces[patch_at[rid]:patch_at[rid + 1]]
            if nfn is not None and len(fids):
                try:
                    dots = np.einsum("ij,ij->i", nfn(tri_c[fids]), tri_n[fids])
                    same = float(np.nansum(dots * tri_area[fids])) >= 0
                except Exception:
                    pass
            loops = []
            for li_, segs in enumerate(region_segs[rid]):
                cleaned = []
                for seg, nb, closed in segs:
                    if is_sliver(seg, closed, rid, nb):
                        continue
                    seg = np.array(seg)
                    if closed and seg[0] == seg[-1]:
                        core = seg[:-1]
                        want = pref.get(frozenset(core.tolist()))
                        if want is not None:
                            k = int(np.where(core == want)[0][0])
                            core = np.roll(core, -k)
                            seg = np.r_[core, core[0]]
                    seg[0], seg[-1] = find(int(seg[0])), find(int(seg[-1]))
                    if len(seg) == 2 and seg[0] == seg[-1]:
                        continue
                    if not closed and seg[0] == seg[-1] and extent(seg) < short_len:
                        continue  # collapsed to a point (merged above)
                    cleaned.append((seg, nb, closed or seg[0] == seg[-1]))
                if not cleaned:
                    continue
                if rid in tube_regions and len(cleaned) > 1:
                    # start the loop where the seam leaves: the vertex split onto the
                    # outer equator (else the junction nearest it)
                    tor_ = tube_regions[rid]
                    want_ = tube_start.get((rid, li_))
                    starts_ = [int(sg_[0][0]) for sg_ in cleaned]
                    if want_ is not None and find(want_) in starts_:
                        k_ = starts_.index(find(want_))
                    else:
                        k_ = int(np.argmin([abs(float(np.angle(np.exp(1j * (tor_.angles(vertex_pos(v_)[None])[1][0]
                                                                            - getattr(tor_, "pipe_phi", 0.0))))))
                                            for v_ in starts_]))
                    fixed_verts.update(starts_)
                    cleaned = cleaned[k_:] + cleaned[:k_]
                keys = []
                for seg, nb, closed in cleaned:
                    if nb < 0:
                        key, fwd, canon = (rid, -1, int(seg[0]), int(seg[1]), int(seg[-1])), True, seg
                    elif rid < nb:
                        key, fwd, canon = (rid, nb, int(seg[0]), int(seg[1]), int(seg[-1])), True, seg
                    else:
                        key, fwd, canon = (nb, rid, int(seg[-1]), int(seg[-2]), int(seg[0])), False, seg[::-1]
                    if key not in edge_geo:
                        # Edges of a turned face that goes all the way round stay
                        # polylines unless it is a cylinder or sphere with our seam
                        # (see no_circle): OpenCASCADE's seam repair, and its curves on
                        # surfaces of revolution, went wrong with exact circles. A loop
                        # put on its circle round a shared axis (ring_of) is the
                        # exception: it is coaxial with both faces, so it projects onto
                        # them as a straight line, and it cannot cross another loop.
                        circ_ok = rid not in no_circle and nb not in no_circle and (
                            int(canon[0]) not in pinch and int(canon[-1]) not in pinch)
                        edge_geo[key] = {
                            "canon": np.asarray(canon), "rids": [g] if nb < 0 else [reg(key[0]), reg(key[1])],
                            "cap": 0.25 * float(min(width[rid], width[nb] if nb >= 0 else np.inf)),
                            "circ": circ_ok, "level": 0}
                        for v_ in {int(canon[0]), int(canon[-1])}:
                            vert_edges.setdefault(v_, []).append(key)
                    keys.append((key, fwd))
                loops.append((keys, int(cleaned[0][0][0]), li_))
            face_plan[rid] = (g, surf, same, fids, loops)

        def edge_points(key):
            """The edge's points on its surfaces, at its repair level: 0 projected onto
            both surfaces, 1 moves held to the neck limit, 3 the mesh boundary as it is."""
            e = edge_geo[key]
            canon, lev = e["canon"], e["level"]
            p0, p1 = vertex_pos(int(canon[0])), vertex_pos(int(canon[-1]))
            if len(canon) > 2:
                raw = V[canon[1:-1]]
                if lev >= 3:
                    inner = raw.copy()
                else:
                    inner = alt_project(raw, [self.fits[x].eq for x in e["rids"]], max_move, final=plane_of(e["rids"]))
                    if lev >= 1 or NECK_LIMIT_ALL:
                        inner = clamp_move(raw, inner, canon[1:-1])
            else:
                inner = np.zeros((0, 3))
            for i_, v_ in enumerate(canon[1:-1]):
                if int(v_) in ring_of:
                    inner[i_] = on_ring(inner[i_], ring_of[int(v_)])
            ring_e = len(canon) > 2 and all(int(v_) in ring_of for v_ in canon)
            if ring_e:
                # Along a smooth join the mesh boundary zigzags, so its points
                # put on the circle can step back and forth round it: spaced
                # evenly instead, the same way round from end to end.
                p_, a_, rq, hq = ring_of[int(canon[1])]
                u_ = _perp(a_)
                w_ = np.cross(a_, u_)
                Q_ = np.vstack([p0, inner, p1]) - p_
                th_ = np.arctan2(Q_ @ w_, Q_ @ u_)
                tot_ = float(np.angle(np.exp(1j * np.diff(th_))).sum())
                if int(canon[0]) == int(canon[-1]):
                    tot_ = 2 * np.pi * (1.0 if tot_ >= 0 else -1.0)
                t_ = th_[0] + tot_ * np.arange(1, len(canon) - 1) / (len(canon) - 1)
                inner = p_ + hq * a_ + rq * (np.outer(np.cos(t_), u_) + np.outer(np.sin(t_), w_))
            return np.vstack([p0, inner, p1]), ring_e

        def plan_edge(key):
            e = edge_geo[key]
            pts, ring_e = edge_points(key)
            if e["level"] >= 2:
                # every boundary point kept: no circle, no smoothing, no thinning
                plan = self.curve_plan(pts, False, max_tol=1e-300)
            else:
                plan = self.curve_plan(pts, ring_e or e["circ"], max_tol=e["cap"])
            return plan, pts

        plan_of = {k_: plan_edge(k_) for k_ in edge_geo}
        edge_faces: dict[tuple, set] = {}
        for rid, fp_ in face_plan.items():
            for keys, _, _ in fp_[4]:
                for k_, _ in keys:
                    edge_faces.setdefault(k_, set()).add(rid)

        def face_crossings(rid):
            fids_, loops_ = face_plan[rid][3], face_plan[rid][4]
            if not len(fids_) or not loops_:
                return []
            lps = [[(k_, plan_of[k_][0][2] if f_ else plan_of[k_][0][2][::-1]) for k_, f_ in keys]
                   for keys, _, _ in loops_]
            seglen = [float(np.linalg.norm(np.diff(pp, axis=0), axis=1).max()) for lp in lps for _, pp in lp if len(pp) > 1]
            if not seglen:
                return []
            tree_, nrm_ = cKDTree(tri_c[fids_]), tri_n[fids_]
            return loop_crossings(lps, lambda X: nrm_[tree_.query(X)[1]], float(np.median(seglen)))

        # Boundary curves that cross inside a face (a reader rejects the face): redo just
        # those edges, step by step - their moves held to the neck limit, then every
        # boundary point kept, then the mesh boundary itself. Elsewhere the edges stay
        # on their surfaces and simplified.
        todo = set(face_plan)
        redone: set = set()
        for _round in range(5 if CROSSING_REPAIR else 0):
            bad = set()
            for rid in todo:
                for a_, b_ in face_crossings(rid):
                    bad.update((a_, b_))
            bad = {k_ for k_ in bad if edge_geo[k_]["level"] < REPAIR_MAX_LEVEL}
            if not bad:
                break
            todo = set()
            for k_ in bad:
                e = edge_geo[k_]
                e["level"] += 1
                redone.add(k_)
                changed = {k_}
                if e["level"] == 1:
                    for v_ in {int(e["canon"][0]), int(e["canon"][-1])}:
                        if (v_ in clamp_verts or v_ in fixed_verts or v_ in ring_of or v_ in seam_vertex
                                or v_ in cone_snap):
                            continue
                        clamp_verts.add(v_)
                        pos_cache.pop(v_, None)
                        changed.update(vert_edges.get(v_, []))
                for c_ in changed:
                    plan_of[c_] = plan_edge(c_)
                    todo |= edge_faces.get(c_, set())
        self.crossings = {}
        for rid in todo:
            hits = face_crossings(rid)
            if hits:
                self.crossings[rid] = (reg(rid), hits)
        if redone:
            self.warnings.append(f"{len(redone)} edges redone where boundary curves crossed in a face; "
                                 f"{len(self.crossings)} faces still have crossings")

        self._edge_info = []
        # Pass 2: write the edges, loops and faces
        for rid in range(R):
            g, surf, same, fids, loops = face_plan[rid]
            bounds = []
            loop_list = []
            for keys, first_v, li_ in loops:
                oes = []
                for key, fwd in keys:
                    if key not in edges:
                        e = edge_geo[key]
                        v0, _ = vertex(int(e["canon"][0]))
                        v1, _ = vertex(int(e["canon"][-1]))
                        plan, pts = plan_of[key]
                        # how far this edge really is from the surfaces it joins (for the
                        # file's declared uncertainty, which CAD importers heal by)
                        devs_ = []
                        for x_ in e["rids"]:
                            dev = np.linalg.norm(closest(self.fits[x_].eq, pts) - pts, axis=1)
                            dev = dev[np.isfinite(dev)]
                            if len(dev):
                                edge_dev.append(float(dev.max()))
                                devs_.append(float(dev.max()))
                        self._edge_info.append((key, list(e["rids"]), devs_, e["level"]))
                        cid = self.curve_write(plan)
                        # On a freeform patch the reader would project the edge onto the
                        # patch to find its curve in (u, v), and on a thin or steep patch
                        # that projection went astray (a 187 mm tolerance on an edge 1.6 mm
                        # off). Its curve in (u, v) is known exactly, so it is written.
                        pcs = []
                        for x_ in dict.fromkeys(e["rids"]):
                            if (PCURVES and x_ in self._pc_frame) or (
                                    HELIX_PCURVES and x_ in getattr(self, "_pc_helix", {})):
                                got_ = self.pcurve_uv(plan, x_)
                                if got_ is not None:
                                    pcs.append(st.pcurve(surfaces[x_][0], got_[0], got_[1]))
                        if pcs:
                            cid = st.add(f"SURFACE_CURVE('',#{cid},{st.ref(pcs)},.CURVE_3D.)")
                            self.pcurves_written += len(pcs)
                        edges[key] = st.add(f"EDGE_CURVE('',#{v0},#{v1},#{cid},.T.)")
                        edge_pts[key] = plan[2]
                    edge_uses[key] = edge_uses.get(key, 0) + 1
                    oes.append(st.add(f"ORIENTED_EDGE('',*,*,#{edges[key]},{'.T.' if fwd else '.F.'})"))
                loop_list.append((oes, first_v, li_))
            if (rid in seam_regions or rid in tube_regions) and len(loop_list) == 2:
                (oes1, a, _), (oes2, b, _) = loop_list
                e = seam_edge(rid, a, b) if rid in seam_regions else tube_seam_edge(rid, a, b)
                edge_uses[("seam", rid)] = 2
                oes = (oes1 + [st.add(f"ORIENTED_EDGE('',*,*,#{e},.T.)")]
                       + oes2 + [st.add(f"ORIENTED_EDGE('',*,*,#{e},.F.)")])
                loop_id = st.add(f"EDGE_LOOP('',{st.ref(oes)})")
                bounds.append(st.add(f"FACE_BOUND('',#{loop_id},.T.)"))
            else:
                for oes, a, li_ in loop_list:
                    if rid in cap_seams and li_ == cap_seams[rid][0]:
                        e = cap_seam_edge(rid, a)
                        edge_uses[("seam", rid)] = 2
                        oes = oes + [st.add(f"ORIENTED_EDGE('',*,*,#{e},.T.)"), st.add(f"ORIENTED_EDGE('',*,*,#{e},.F.)")]
                    loop_id = st.add(f"EDGE_LOOP('',{st.ref(oes)})")
                    bounds.append(st.add(f"FACE_BOUND('',#{loop_id},.T.)"))
            if not bounds and self.fits[g].cls.get("type") not in ("sphere", "torus", "freeform"):
                # every edge collapsed: an unbounded plane or cylinder would be wrong
                dropped += 1
                continue
            if not bounds:  # closed surface without boundary (e.g. a whole sphere)
                rv = s.regions[g].idx[0]
                vp = st.add(f"VERTEX_POINT('',#{st.point(s.P[rv])})")
                vloop = st.add(f"VERTEX_LOOP('',#{vp})")
                bounds.append(st.add(f"FACE_BOUND('',#{vloop},.T.)"))
            faces.append(st.add(f"ADVANCED_FACE('{self.face_name(g)}',{st.ref(bounds)},#{surf},{'.T.' if same else '.F.'})"))
            face_body.append(int(body_of[rid]))
            if face_part is not None and len(fids):
                body_part[int(body_of[rid])] = int(face_part[fids[0]])
        if self.pcurves_written:
            self.warnings.append(f"{self.pcurves_written} edge curves on freeform patches written with their (u, v) curves")
        if dropped:
            self.warnings.append(f"{dropped} faces too small to keep (every edge shorter than the sliver limit) were left out")
        # closed bodies: every edge of theirs used exactly twice, none on a mesh border
        open_body = set()
        for k, n in edge_uses.items():
            if k[0] == "seam":
                continue
            if k[1] < 0 or n != 2:
                open_body.add(int(body_of[k[0]]))
        shells: dict[int, list[int]] = {}
        for fid, b in zip(faces, face_body):
            shells.setdefault(b, []).append(fid)
        self._edge_uses = edge_uses  # kept for diagnostics
        # declared uncertainty: what the edges actually achieve (99th percentile), so a
        # reader heals the gaps that are there and no more
        if edge_dev:
            self.uncertainty = float(np.clip(1.5 * np.percentile(edge_dev, 99), self.tol, 0.002 * self.diag))
        return shells, open_body, body_part

    def _part_rep(self, pname: str, shells: dict, bodies: list[int], open_body: set, ctx: int):
        """Shape representation of one part: solids if all its bodies close, else a surface model."""
        st = self.st
        closed = all(b not in open_body for b in bodies)
        if closed:
            items = [st.add(f"MANIFOLD_SOLID_BREP('{pname}',#{st.add(f'CLOSED_SHELL({chr(39)}{chr(39)},{st.ref(shells[b])})')})")
                     for b in bodies]
        else:
            shell_ids = [st.add(f"{'OPEN_SHELL' if b in open_body else 'CLOSED_SHELL'}('',{st.ref(shells[b])})")
                         for b in bodies]
            items = [st.add(f"SHELL_BASED_SURFACE_MODEL('{pname}',{st.ref(shell_ids)})")]
        origin = st.axis2([0, 0, 0], [0, 0, 1], [1, 0, 0])
        rep_type = "ADVANCED_BREP_SHAPE_REPRESENTATION" if closed else "MANIFOLD_SURFACE_SHAPE_REPRESENTATION"
        return st.add(f"{rep_type}('{pname}',{st.ref([origin] + items)},#{ctx})"), origin, closed

    def export(self, path: Path, name: str) -> dict:
        st = self.st
        shells, open_body, body_part = self.build()
        part_names = list(getattr(self.s, "part_names", None) or [name])
        by_part: dict[int, list[int]] = {}
        for b in shells:
            by_part.setdefault(body_part.get(b, 0), []).append(b)
        closed = not open_body
        faces = [f for fs in shells.values() for f in fs]
        if self.units == "inch":
            mm = st.add(UNITS["mm"])
            lmw = st.add(f"LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#{mm})")
            dim = st.add("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)")
            length = st.add(f"( CONVERSION_BASED_UNIT('INCH',#{lmw}) LENGTH_UNIT() NAMED_UNIT(#{dim}) )")
        else:
            length = st.add(UNITS.get(self.units, UNITS["mm"]))
        angle = st.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
        solid = st.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
        unc = st.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({fr(getattr(self, 'uncertainty', self.tol))}),#{length},"
                     f"'distance_accuracy_value','confusion accuracy')")
        ctx = st.add(f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) "
                     f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid})) "
                     f"REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )")
        app = st.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
        st.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
        pctx = st.add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
        pdc = st.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")

        def product(pname: str, pid: str, category: str) -> tuple[int, int]:
            prod = st.add(f"PRODUCT('{pid}','{pname}','',(#{pctx}))")
            st.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('{category}',$,(#{prod}))")
            pdf = st.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
            pd = st.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdc})")
            return pd, st.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")

        closed_parts = 0
        if len(by_part) <= 1:
            pname = part_names[next(iter(by_part))] if by_part and len(part_names) > 1 else name
            rep, _, c = self._part_rep(pname, shells, list(shells), open_body, ctx)
            closed_parts += c
            _, pds = product(pname, pname, "part")
            st.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
        else:
            # An assembly: a top product holding one product per part, each placed
            # (at the identity: the mesh is already in assembly coordinates).
            top_origin = st.axis2([0, 0, 0], [0, 0, 1], [1, 0, 0])
            placements = {p: st.axis2([0, 0, 0], [0, 0, 1], [1, 0, 0]) for p in by_part}
            top_rep = st.add(f"SHAPE_REPRESENTATION('{name}',{st.ref([top_origin] + list(placements.values()))},#{ctx})")
            top_pd, top_pds = product(name, name, "assembly")
            st.add(f"SHAPE_DEFINITION_REPRESENTATION(#{top_pds},#{top_rep})")
            used: dict[str, int] = {}
            for k, (p, bodies) in enumerate(sorted(by_part.items())):
                pname = part_names[p] if p < len(part_names) else f"part_{p + 1}"
                used[pname] = used.get(pname, 0) + 1
                pid = pname if used[pname] == 1 else f"{pname}_{used[pname]}"
                rep, origin, c = self._part_rep(pname, shells, bodies, open_body, ctx)
                closed_parts += c
                pd, pds = product(pname, pid, "part")
                st.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
                nauo = st.add(f"NEXT_ASSEMBLY_USAGE_OCCURRENCE('NAUO{k + 1}','{pname}','',#{top_pd},#{pd},$)")
                pds_n = st.add(f"PRODUCT_DEFINITION_SHAPE('Placement','Placement of an item',#{nauo})")
                idt = st.add(f"ITEM_DEFINED_TRANSFORMATION('','',#{origin},#{placements[p]})")
                rr = st.add(f"( REPRESENTATION_RELATIONSHIP('','',#{rep},#{top_rep}) "
                            f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{idt}) SHAPE_REPRESENTATION_RELATIONSHIP() )")
                st.add(f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{rr},#{pds_n})")
        st.write(path, name)
        return {"faces": len(faces), "closed": closed, "bodies": len(shells), "parts": len(by_part),
                "uncertainty": float(getattr(self, "uncertainty", self.tol)),
                "closed_parts": int(closed_parts), "open_bodies": len(open_body), "warnings": self.warnings}


# ===================================================================== export

_EXACT_TYPES = ("plane", "cylinder", "cone", "sphere", "torus")


def exact_distance(fit: Fit, P: np.ndarray):
    """True distance from points to an exact primitive, from its parameters - not from
    its frame formula, which runs on past the surface (a 1 mm round's torus formula
    put a whole 6 mm rim "on" the torus). None for other surfaces."""
    c, fr_ = fit.cls, fit.eq.frame
    t = c.get("type")
    o, a = np.asarray(fr_.origin, float), np.asarray(fr_.e3, float)
    W = P - o
    z = W @ a
    if t == "plane":
        ca, cb = float(c.get("slope_u", 0.0)), float(c.get("slope_v", 0.0))
        h = float(c.get("offset", 0.0)) + ca * (W @ np.asarray(fr_.e1, float)) + cb * (W @ np.asarray(fr_.e2, float))
        return np.abs(z - h) / np.sqrt(1 + ca * ca + cb * cb)
    if t == "sphere":
        return np.abs(np.linalg.norm(W, axis=1) - float(c["radius"]))
    rho = np.linalg.norm(W - np.outer(z, a), axis=1)
    if t == "cylinder":
        return np.abs(rho - float(c["radius"]))
    if t == "cone":
        k = float(c["slope"])
        return np.abs(rho - (float(c["radius0"]) + k * z)) / np.sqrt(1 + k * k)
    if t == "torus":
        ct, ax = np.asarray(c["center"], float), np.asarray(c["axis"], float)
        ax = ax / np.linalg.norm(ax)
        W = P - ct
        z = W @ ax
        rho = np.linalg.norm(W - np.outer(z, ax), axis=1)
        return np.abs(np.hypot(rho - float(c["R"]), z) - float(c["r"]))
    return None


def refit_on_vertices(s, fits: dict[int, Fit], tol: float) -> int:
    """Freeform regions that are a plane, cylinder, cone or sphere after all.

    The search fits sample points inside the triangles, and on a coarse CAD
    tessellation those sit inside the true surface by the chord height: a flange's
    plain cylindrical rim, meshed as long strips, missed every exact fit and became
    four freeform patches. Its vertices are exactly on the cylinder. So each freeform
    region is tried on its vertices: as a plane, and as a turned shape about the axis
    of any turned surface near it. Returns how many regions were refitted.
    """
    if getattr(s, "strategy", None) != "cad":
        return 0
    mesh, FL = getattr(s, "mesh", None), getattr(s, "face_labels", None)
    if mesh is None or FL is None or len(FL) != len(mesh.faces) or not len(FL):
        return 0
    F, V = mesh.faces, mesh.vertices
    adj = mesh.face_adjacency
    la, lb = FL[adj[:, 0]], FL[adj[:, 1]]
    m = la != lb
    nbrs: dict[int, set] = {}
    for a_, b_ in np.unique(np.sort(np.c_[la[m], lb[m]], axis=1), axis=0):
        nbrs.setdefault(int(a_), set()).add(int(b_))
        nbrs.setdefault(int(b_), set()).add(int(a_))
    order = np.argsort(FL, kind="stable")
    at = np.searchsorted(FL[order], np.arange(int(FL.max()) + 2))

    def axis_of(f):
        t = f.cls.get("type")
        if t in ("cylinder", "cone") and f.eq.frame.kind == "cylindrical":
            return np.asarray(f.eq.frame.origin, float), np.asarray(f.eq.frame.e3, float)
        if t == "torus" and "center" in f.cls:
            a_ = np.asarray(f.cls["axis"], float)
            return np.asarray(f.cls["center"], float), a_ / np.linalg.norm(a_)
        return None

    n_done = 0
    for rid, f in list(fits.items()):
        if f.cls.get("type") != "freeform" or rid + 1 >= len(at) or at[rid + 1] - at[rid] < 3:
            continue
        P = V[np.unique(F[order[at[rid]:at[rid + 1]]])]
        if len(P) < 6:
            continue
        scale = float(f.eq.frame.scale)
        found = None
        c0 = P.mean(axis=0)
        n_ = np.linalg.svd(P - c0, full_matrices=False)[2][-1]
        dev = np.abs((P - c0) @ n_)
        if dev.max() <= tol:
            e1 = _perp(n_)
            found = ("plane", Frame("height", c0, e1, np.cross(n_, e1), n_, scale), dev)
        if found is None:
            # axes of turned surfaces among its neighbours and theirs
            near = set(nbrs.get(rid, ()))
            for nb in list(near):
                near |= nbrs.get(nb, set())
            axes = []
            for nb in near:
                g = fits.get(nb)
                ax_ = axis_of(g) if g is not None else None
                if ax_ is None:
                    continue
                o_, a_ = ax_
                if not any(abs(float(a_ @ b_)) > 1 - 1e-9 and np.linalg.norm(np.cross(o_ - q_, b_)) <= 1e-6 * s.diag
                           for q_, b_ in axes):
                    axes.append((o_, a_))
            for o_, a_ in axes:
                W = P - o_
                z = W @ a_
                rho = np.linalg.norm(W - np.outer(z, a_), axis=1)
                zc = float(z.mean())
                oc = o_ + zc * a_
                e1 = _toward(P, oc, a_)
                fr_ = Frame("cylindrical", oc, e1, np.cross(a_, e1), a_, scale)
                d_cyl = np.abs(rho - rho.mean())
                if d_cyl.max() <= tol and rho.mean() > tol:
                    found = ("cylinder", fr_, d_cyl)
                    break
                k_, r0_ = np.linalg.lstsq(np.c_[z - zc, np.ones_like(z)], rho, rcond=None)[0]
                d_cone = np.abs(rho - (k_ * (z - zc) + r0_)) / np.sqrt(1 + k_ * k_)
                if d_cone.max() <= tol and abs(k_) > 1e-3 and r0_ > tol:
                    found = ("cone", fr_, d_cone)
                    break
                # sphere centred on the axis: rho^2 + z^2 = 2 z0 z + (R^2 - z0^2)
                sol = np.linalg.lstsq(np.c_[2 * z, np.ones_like(z)], rho ** 2 + z ** 2, rcond=None)[0]
                z0 = float(sol[0])
                R2 = float(sol[1]) + z0 * z0
                if R2 > 0:
                    d_sph = np.abs(np.hypot(rho, z - z0) - np.sqrt(R2))
                    if d_sph.max() <= tol:
                        ctr = o_ + z0 * a_
                        found = ("sphere", Frame("spherical", ctr, e1, np.cross(a_, e1), a_, scale), d_sph)
                        break
        if found is None:
            continue
        prim, fr_, dev = found
        X, y = fr_.to_param(P)
        eq = fallback_equation(prim, fr_, X, y)
        rms = float(np.sqrt(np.mean(dev ** 2)))
        cls = classify(eq, rms / s.diag)
        if cls.get("type") != prim:
            continue
        cls["vertex_refit"] = True
        fits[rid] = Fit(rid, eq, cls, False, rms)
        _mark_nearest(eq, cls)
        n_done += 1
    return n_done


def _toward(P: np.ndarray, o: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Unit direction square to axis a from o towards the middle of points P: a turned
    surface's frame points there, so its seam goes on the far side of the face."""
    w = P.mean(axis=0) - o
    w = w - (w @ a) * a
    n = np.linalg.norm(w)
    if n < 1e-12 * max(1.0, float(np.linalg.norm(np.ptp(P, axis=0)))):
        # a full turn: the mean is on the axis; any direction
        return _perp(a)
    return w / n


def _circle_ransac(Q: np.ndarray, tol: float, rmax: float, iters: int = 300, seed: int = 0):
    """Circle in 2D through most of the points (within tol): (centre, radius, inlier mask) or None."""
    n = len(Q)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = Q[rng.choice(n, 3, replace=False)]
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-300:
            continue
        ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
        ctr = np.array([ux, uy])
        r = float(np.linalg.norm(a - ctr))
        if not 0 < r < rmax:
            continue
        inl = np.abs(np.linalg.norm(Q - ctr, axis=1) - r) <= tol
        k = int(inl.sum())
        if best is None or k > best[0]:
            best = (k, ctr, r, inl)
    if best is None or best[0] < 4:
        return None
    _, ctr, r, inl = best
    # refine on the inliers (algebraic fit), then recount
    X = Q[inl]
    A = np.c_[2 * X, np.ones(len(X))]
    sol = np.linalg.lstsq(A, (X * X).sum(axis=1), rcond=None)[0]
    ctr = sol[:2]
    r = float(np.sqrt(max(sol[2] + ctr @ ctr, 0.0)))
    if not 0 < r < rmax:
        return None
    inl = np.abs(np.linalg.norm(Q - ctr, axis=1) - r) <= tol
    return ctr, r, inl


def split_freeform(s, fits: dict[int, Fit], tol: float):
    """Cut what is left freeform into the surfaces it really is.

    A freeform region on a CAD mesh is mostly a round the search did not take, or
    pieces of its neighbours run on: the rounded bend between a housing's floor and
    wall, one long cylinder, came out freeform because nothing gave the search its
    axis. Its axis is where the two planes it joins meet. Triangles whose vertices lie
    on a neighbour's exact surface join that neighbour; triangles on a cylinder round
    the meeting line of two neighbouring planes (or round a neighbour's axis) become
    that cylinder; only the rest stays freeform. Returns (session copy, pieces made).
    """
    import copy
    import dataclasses
    if getattr(s, "strategy", None) != "cad":
        return s, 0
    mesh, FL = getattr(s, "mesh", None), getattr(s, "face_labels", None)
    if mesh is None or FL is None or len(FL) != len(mesh.faces) or not len(FL):
        return s, 0
    F, V = mesh.faces, mesh.vertices
    adj = mesh.face_adjacency
    la, lb = FL[adj[:, 0]], FL[adj[:, 1]]
    m = la != lb
    nbrs: dict[int, set] = {}
    for a_, b_ in np.unique(np.sort(np.c_[la[m], lb[m]], axis=1), axis=0):
        nbrs.setdefault(int(a_), set()).add(int(b_))
        nbrs.setdefault(int(b_), set()).add(int(a_))
    order = np.argsort(FL, kind="stable")
    at = np.searchsorted(FL[order], np.arange(int(FL.max()) + 2))
    trusted: dict[int, bool] = {}

    def trust(nb: int) -> bool:
        if nb not in trusted:
            ok_ = False
            g = fits.get(nb)
            if g is not None and g.cls.get("type") in _EXACT_TYPES and nb + 1 < len(at) and at[nb + 1] > at[nb]:
                Pn = V[np.unique(F[order[at[nb]:at[nb + 1]]])]
                dn = exact_distance(g, Pn)
                ok_ = dn is not None and bool(np.all(np.isfinite(dn))) and float(np.percentile(dn, 95)) <= 0.5 * tol
            trusted[nb] = ok_
        return trusted[nb]

    FL2 = FL.copy()
    regions = list(s.regions)
    frames = dict(getattr(s, "frames", {}) or {})
    fidx = getattr(s, "fidx", None)
    new_fits: dict[int, Fit] = {}
    made = 0
    for rid, f in list(fits.items()):
        if f.cls.get("type") != "freeform" or rid + 1 >= len(at) or at[rid + 1] - at[rid] < 4:
            continue
        tris = order[at[rid]:at[rid + 1]]
        TV = F[tris]
        free = np.ones(len(tris), bool)
        assign = np.full(len(tris), -1, np.int64)
        nb_ok = [nb for nb in sorted(nbrs.get(rid, ())) if trust(nb)]
        # 1. neighbours run on
        vids = np.unique(TV)
        vpos = {int(v): i for i, v in enumerate(vids)}
        TVi = np.vectorize(vpos.get)(TV)
        PV = V[vids]
        # (single triangles are not handed to a neighbour: near a round's tangent line
        # some fall within the tolerance of the plane and some do not, and taking them
        # one by one left a sawtooth boundary that pinched the plane's loop; a whole
        # region on its neighbour joins it in merge_continuations)
        # 2. cylinders round the meeting line of two neighbouring planes, or a neighbour's axis
        axes = []
        planes = [nb for nb in nb_ok if fits[nb].cls.get("type") == "plane"]
        for i_ in range(len(planes)):
            for j_ in range(i_ + 1, len(planes)):
                na = np.asarray(fits[planes[i_]].eq.frame.e3, float)
                nb_ = np.asarray(fits[planes[j_]].eq.frame.e3, float)
                ca_, cb_ = fits[planes[i_]].cls, fits[planes[j_]].cls
                if ca_.get("slope_u") or ca_.get("slope_v") or cb_.get("slope_u") or cb_.get("slope_v"):
                    continue
                dd = np.cross(na, nb_)
                if np.linalg.norm(dd) < 0.1:
                    continue
                axes.append((dd / np.linalg.norm(dd), (planes[i_], planes[j_])))
        for nb in nb_ok:
            g = fits[nb]
            if g.cls.get("type") in ("cylinder", "cone") and g.eq.frame.kind == "cylindrical":
                axes.append((np.asarray(g.eq.frame.e3, float), None))
        uniq = []
        for d_, pp_ in axes:
            if not any(abs(float(d_ @ u_)) > 1 - 1e-6 for u_, _ in uniq):
                uniq.append((d_, pp_))
        ext = float(np.linalg.norm(np.ptp(PV, axis=0)))
        pieces = []
        for k_, (d_, pp_) in enumerate(uniq):
            if free.sum() < 4:
                break
            e1 = _perp(d_)
            e2 = np.cross(d_, e1)
            fv = np.unique(TVi[free])
            Q = np.c_[PV[fv] @ e1, PV[fv] @ e2]
            got = _circle_ransac(Q, tol, rmax=ext, seed=k_)
            if got is None:
                continue
            ctr, r, inl = got
            if pp_ is not None:
                # a round between two planes touches both: its axis is r from each
                c3 = ctr[0] * e1 + ctr[1] * e2
                touch = True
                for pl_ in pp_:
                    dpl = exact_distance(fits[pl_], (c3 + d_ * float(PV[0] @ d_))[None])
                    if dpl is None or abs(float(dpl[0]) - r) > 2 * tol:
                        touch = False
                if not touch:
                    continue
            okv = np.zeros(len(PV), bool)
            okv[fv[inl]] = True
            on = okv[TVi].all(axis=1) & free
            if on.sum() < 4:
                continue
            # the cylinder must hold its own triangles' vertices (all of them, any height)
            Qa = np.c_[PV @ e1, PV @ e2]
            dv = np.abs(np.linalg.norm(Qa - ctr, axis=1) - r)
            on &= (dv <= tol)[TVi].all(axis=1)
            if on.sum() < 4:
                continue
            pieces.append((d_, e1, e2, ctr, r, on))
            free &= ~on
        if not (assign >= 0).any() and not pieces:
            continue
        # only a region the exact pieces explain (nearly) whole is cut up: a few exact
        # strips carved out of a real blend leave a ragged freeform remnant whose
        # loops then fail, and the blend was no worse as one patch
        if free.mean() > 0.1:
            continue
        # neighbour continuations
        for nb in np.unique(assign[assign >= 0]):
            FL2[tris[assign == nb]] = nb
            if fidx is not None:
                regions[nb] = dataclasses.replace(regions[nb], idx=np.unique(np.r_[regions[nb].idx,
                                                                              np.where(np.isin(fidx, tris[assign == nb]))[0]]))
            made += 1
        # new cylinders
        for d_, e1, e2, ctr, r, on in pieces:
            new = len(regions)
            Pc = PV[np.unique(TVi[on])]
            h = Pc @ d_
            origin = ctr[0] * e1 + ctr[1] * e2 + float(h.mean()) * d_
            f1 = _toward(Pc, origin, d_)
            fr_ = Frame("cylindrical", origin, f1, np.cross(d_, f1), d_, float(f.eq.frame.scale))
            X, y = fr_.to_param(Pc)
            eq = fallback_equation("cylinder", fr_, X, y)
            dev = np.abs(np.linalg.norm(np.c_[Pc @ e1, Pc @ e2] - ctr, axis=1) - r)
            rms = float(np.sqrt(np.mean(dev ** 2)))
            cls = classify(eq, rms / s.diag)
            if cls.get("type") != "cylinder":
                continue
            cls["vertex_refit"] = True
            _mark_nearest(eq, cls)
            FL2[tris[on]] = new
            idx_ = np.where(np.isin(fidx, tris[on]))[0] if fidx is not None else np.zeros(0, int)
            regions.append(dataclasses.replace(regions[rid], id=new, idx=idx_, prim="cylinder", params={}, rms=rms))
            frames[new] = fr_
            new_fits[new] = Fit(new, eq, cls, False, rms)
            made += 1
    if not made:
        return s, 0
    fits.update(new_fits)
    s2 = copy.copy(s)
    s2.face_labels = FL2
    s2.regions = regions
    s2.frames = frames
    return s2, made


def _arc_out_and_back(fit: Fit, P: np.ndarray, tol: float) -> bool:
    """Do these points lie on one circle round a turned surface's axis (one height, one
    radius) and cover less than half a turn of it? A face that borders only that surface
    and has every vertex on such an arc is bounded by the arc, out and back: it has no
    area of its own. (A disk's rim triangle that a sliver given to the cylinder cut off
    from the disk: left a face, its out-and-back loop was a slit in the cylinder band,
    which then had three loops and no seam of ours, and the band was lost.)"""
    c, fr_ = fit.cls, fit.eq.frame
    t = c.get("type")
    if t in ("cylinder", "cone"):
        o, a = np.asarray(fr_.origin, float), np.asarray(fr_.e3, float)
    elif t == "torus":
        o, a = np.asarray(c["center"], float), np.asarray(c["axis"], float)
    else:
        return False
    a = a / np.linalg.norm(a)
    W = P - o
    h = W @ a
    D = W - np.outer(h, a)
    rho = np.linalg.norm(D, axis=1)
    if len(P) < 2 or np.ptp(h) > 2 * tol or np.ptp(rho) > 2 * tol or rho.min() <= 2 * tol:
        return False
    u = _perp(a)
    th = np.sort(np.arctan2(D @ np.cross(a, u), D @ u))
    gap = max(float(np.diff(th).max()), float(th[0] + 2 * np.pi - th[-1]))
    return gap > np.pi


def merge_continuations(s, fits: dict[int, Fit], tol: float, exact_only: bool = False, max_exact: int = 30):
    """Freeform regions that are really a neighbour's surface carried on.

    Segmentation can cut one surface into pieces (a stamped flange's round outer rim,
    one torus, came out as one torus piece and four freeform ones). A freeform region
    whose vertices lie on an adjacent exact surface - nearly all within the tolerance -
    joins that region, so the surface runs on through it and no edge is left between
    them. Returns a shallow copy of the session with merged labels and samples (the
    session itself is left alone), and how many regions were merged.

    With exact_only, only small exact regions (at most `max_exact` triangles) go, into
    a bigger exact neighbour that borders at least a third of them: a lone flat
    triangle on a sphere, a strip of a rod that came out as its own cylinder (two
    faces on one surface meet along a tangent line no kernel can intersect). Never a
    plane into a plane (a scale's tick marks are real steps), and into another kind of
    surface only when the triangles' centres lie on it too.
    """
    import copy
    import dataclasses
    mesh, FL = getattr(s, "mesh", None), getattr(s, "face_labels", None)
    if mesh is None or FL is None or len(FL) != len(mesh.faces) or not len(FL):
        return s, 0
    F, V = mesh.faces, mesh.vertices
    adj = mesh.face_adjacency
    la, lb = FL[adj[:, 0]], FL[adj[:, 1]]
    m = la != lb
    nbrs: dict[int, set] = {}
    for a, b in np.unique(np.sort(np.c_[la[m], lb[m]], axis=1), axis=0):
        nbrs.setdefault(int(a), set()).add(int(b))
        nbrs.setdefault(int(b), set()).add(int(a))
    shared: dict[tuple, float] = {}  # boundary length between two regions
    perim: dict[int, float] = {}
    if exact_only:
        ae = mesh.face_adjacency_edges[m]
        el = np.linalg.norm(V[ae[:, 0]] - V[ae[:, 1]], axis=1)
        for a, b, L in zip(la[m], lb[m], el):
            a, b, L = int(a), int(b), float(L)
            shared[(a, b)] = shared.get((a, b), 0.0) + L
            shared[(b, a)] = shared.get((b, a), 0.0) + L
            perim[a] = perim.get(a, 0.0) + L
            perim[b] = perim.get(b, 0.0) + L
    order = np.argsort(FL, kind="stable")
    at = np.searchsorted(FL[order], np.arange(int(FL.max()) + 2))
    nfaces = np.diff(at)
    # a region of a few triangles is a plane whatever it is part of (a lone triangle
    # left inside a torus): it joins a neighbour whose surface it lies on as well
    tiny = lambda r_: r_ + 1 < len(at) and nfaces[r_] <= 4
    trusted: dict[int, bool] = {}

    def trust(nb: int) -> bool:
        # the parameters must reproduce the region's own vertices
        if nb not in trusted:
            fn_ = order[at[nb]:at[nb + 1]] if nb + 1 < len(at) else []
            ok_ = False
            if len(fn_):
                Pn = V[np.unique(F[fn_])]
                dn = exact_distance(fits[nb], Pn)
                ok_ = dn is not None and bool(np.all(np.isfinite(dn))) and float(np.percentile(dn, 95)) <= 0.5 * tol
            trusted[nb] = ok_
        return trusted[nb]

    target: dict[int, int] = {}
    for rid, f in fits.items():
        if exact_only:
            if rid + 1 >= len(at) or rid not in nbrs or f.cls.get("type") not in _EXACT_TYPES or nfaces[rid] > max_exact:
                continue
        elif rid + 1 >= len(at) or rid not in nbrs or not (f.cls.get("type") == "freeform" or tiny(rid)
                                                           or f.cls.get("vertex_refit")):
            continue
        fr_ = order[at[rid]:at[rid + 1]]
        if not len(fr_):
            continue
        P = V[np.unique(F[fr_])]
        best = None
        for nb in nbrs[rid]:
            g = fits.get(nb)
            if g is None or g.cls.get("type") not in _EXACT_TYPES or tiny(nb) or not trust(nb):
                continue
            if exact_only and nfaces[nb] <= nfaces[rid]:
                continue  # the smaller piece joins the bigger
            if exact_only and f.cls.get("type") == "plane" and g.cls.get("type") == "plane":
                continue  # a plane next to a plane within tolerance is a real step (a scale's tick marks)
            if exact_only and shared.get((rid, nb), 0.0) < perim.get(rid, 0.0) / 3:
                continue  # a sliver along the neighbour's edge, mostly against other faces
            if f.cls.get("vertex_refit") and g.cls.get("vertex_refit") and nb > rid:
                continue  # two refitted pieces of one surface: the higher id joins the lower
            try:
                d = exact_distance(g, P)
                if exact_only and f.cls.get("type") != g.cls.get("type"):
                    # a fan-triangulated flat face has every vertex on its rim, and the rim
                    # on the cylinder it cuts: its triangles' centres must be on it too,
                    # unless the face has no area of its own (see _arc_out_and_back)
                    Cc = V[F[fr_]].mean(axis=1)
                    dc = exact_distance(g, Cc)
                    if (dc is None or not np.all(np.isfinite(dc)) or float(dc.max()) > 2 * tol) and not (
                            nbrs[rid] == {nb} and _arc_out_and_back(g, P, tol)):
                        continue
            except Exception:
                continue
            d = np.where(np.isfinite(d), d, np.inf)
            frac = float(np.mean(d <= tol))
            if frac >= 0.97 and float(np.percentile(d, 99)) <= 2 * tol and (best is None or frac > best[1]):
                best = (nb, frac)
        if best is not None:
            target[rid] = best[0]
    if not target:
        return s, 0
    def root(r_):
        seen_ = set()
        while r_ in target and r_ not in seen_:
            seen_.add(r_)
            r_ = target[r_]
        return r_
    target = {r_: root(r_) for r_ in target}
    target = {r_: t_ for r_, t_ in target.items() if t_ != r_}
    FL2 = FL.copy()
    groups: dict[int, list[int]] = {}
    for rid, t in target.items():
        FL2[FL == rid] = t
        groups.setdefault(t, []).append(rid)
    regions = list(s.regions)
    for t, rs in groups.items():
        regions[t] = dataclasses.replace(regions[t], idx=np.unique(np.concatenate([regions[t].idx] + [regions[r].idx for r in rs])))
    s2 = copy.copy(s)
    s2.face_labels = FL2
    s2.regions = regions
    return s2, len(target)


def canonical_order(s):
    """A copy of the session with its regions renumbered in an order fixed by the mesh:
    smallest area first (ties by first triangle).

    The writer makes many first-come decisions (which face defines a shared edge and
    its direction, a seam's direction along a chain of bands, which faces share an
    axis), so the same faces numbered differently gave different files: a renumbering
    alone turned 0 failing faces into 5. Renumbered here, every input order gives the
    same file. Small faces come first, so they win those decisions: a small face is
    the fragile one, a big face hardly notices. Returns (session copy, original id of
    each new id)."""
    import copy
    import dataclasses
    R = len(s.regions)
    FL = getattr(s, "face_labels", None)
    mesh = getattr(s, "mesh", None)
    if not R or FL is None or mesh is None or len(FL) != len(mesh.faces):
        return s, np.arange(R)
    first = np.full(R, len(FL), np.int64)
    ok = (FL >= 0) & (FL < R)
    np.minimum.at(first, FL[ok], np.nonzero(ok)[0])
    area = np.bincount(FL[ok], weights=np.asarray(mesh.area_faces, float)[ok], minlength=R)
    key = lambda g: (float(area[g]), int(first[g]), g)
    orig = np.array(sorted(range(R), key=key), np.int64)
    new_of = np.empty(R, np.int64)
    new_of[orig] = np.arange(R)

    def remap(a):
        a = np.asarray(a)
        return np.where((a >= 0) & (a < R), new_of[np.clip(a, 0, R - 1)], a)

    s2 = copy.copy(s)
    s2.regions = [dataclasses.replace(s.regions[int(o)], id=k) for k, o in enumerate(orig)]
    s2.face_labels = remap(FL)
    if getattr(s, "labels", None) is not None:
        s2.labels = remap(s.labels)
    s2.results = {int(new_of[k]): v for k, v in s.results.items() if 0 <= k < R}
    s2.frames = {int(new_of[k]): v for k, v in s.frames.items() if 0 <= k < R}
    if getattr(s, "features", None):
        s2.features = {int(new_of[k]): v for k, v in s.features.items() if 0 <= k < R}
    s2.adjacency = [tuple(sorted((int(new_of[a]), int(new_of[b])))) for a, b in getattr(s, "adjacency", []) if a < R and b < R]
    s2.latex_cache = {}
    for k in ("_area_cache_key", "_area_by_label"):
        s2.__dict__.pop(k, None)
    return s2, orig


def export_all(s, out_dir: Path, name: str, formats: list[str], units: str = "mm", progress=None) -> dict:
    progress = progress or (lambda *a: None)
    t0 = time.perf_counter()
    cn = getattr(s, "curvenet", None)
    if cn is not None and "step" in formats:  # faces built from curves: their own B-rep
        from .curvenet import write_step as curvenet_step

        progress(0.1, "Writing STEP: faces built from curves")
        p = out_dir / f"{name}.step"
        res = {"files": {"step": str(p)}, "info": {"step": curvenet_step(cn, p, name, units)}}
        rest = [f for f in formats if f != "step"]
        if rest:
            s.curvenet = None
            try:
                other = export_all(s, out_dir, name, rest, units, progress)
            finally:
                s.curvenet = cn
            res["files"].update(other["files"])
            res["info"].update(other["info"])
        progress(1.0, f"Export finished in {time.perf_counter() - t0:.1f}s")
        return res
    # every face order gives the same file; names and notes keep the caller's ids
    s, orig = canonical_order(s)
    if ZIP_CRACKS:
        try:
            s, n_zip = zip_cracks(s)
        except Exception as exc:  # the unzipped mesh still exports
            n_zip = f"failed ({exc})"
    n_crumbs = 0
    if ABSORB_CRUMBS:
        try:
            s, n_crumbs = absorb_crumbs(s)
        except Exception as exc:
            n_crumbs = f"failed ({exc})"
    R0 = len(orig)
    orig_id = lambda g: int(orig[g]) if 0 <= g < R0 else int(g)
    import re as _re
    name_back = lambda w: _re.sub(r"\bregion (\d+)", lambda m: f"region {orig_id(int(m.group(1)))}", w)
    s.export_notes = []
    if ZIP_CRACKS and n_zip:
        s.export_notes.append(f"mesh cracks closed: {n_zip} triangles split")
    if n_crumbs:
        s.export_notes.append(f"{n_crumbs} detached crumbs of faces joined their neighbours")
    fits = collect_fits(s)
    if getattr(s, "strategy", None) == "cad":
        tol = max(20 * getattr(s, "seg_eps", 0.0), 1e-6 * s.diag)
    else:
        tol = max(getattr(s, "seg_eps", 0.0), 0.002 * s.diag)
    # On a CAD mesh the vertices are on their surfaces to float precision (nine in ten
    # exact regions within 2e-5 of the model size), while the export tolerance can be
    # 40 times looser: a 0.2 mm tolerance would put a pin's 0.15 mm chamfer "on" the
    # face next to it. Whether vertices lie on a surface is judged at this precision.
    vtol = min(tol, 5e-5 * s.diag) if getattr(s, "strategy", None) == "cad" else tol
    try:
        n_refit = refit_on_vertices(s, fits, vtol) if VERTEX_REFIT else 0
        if n_refit:
            s.export_notes.append(f"{n_refit} freeform regions are exact shapes on their mesh vertices")
    except Exception as exc:
        s.export_notes.append(f"refitting freeform regions on their vertices failed ({exc})")
    try:
        s, n_split = split_freeform(s, fits, vtol) if SPLIT_FREEFORM else (s, 0)
        if n_split:
            s.export_notes.append(f"{n_split} pieces of freeform regions taken as neighbours run on or fillet cylinders")
    except Exception as exc:
        import traceback
        s.export_notes.append(f"splitting freeform regions failed ({exc}) {traceback.format_exc()[-400:]}")
    try:
        if MERGE_CONTINUATIONS:
            s, n_merged = merge_continuations(s, fits, vtol)
        elif MERGE_EXACT_STRIPS:
            s, n_merged = merge_continuations(s, fits, vtol, exact_only=True)
        else:
            n_merged = 0
        if n_merged:
            s.export_notes.append(f"{n_merged} regions joined the neighbouring surface they lie on")
    except Exception as exc:
        s.export_notes.append(f"joining freeform regions to their neighbours failed ({exc})")
    try:
        n_shared = share_axes(s, fits, tol)
        if n_shared:
            s.export_notes.append(f"{n_shared} turned faces refit on shared axes")
    except Exception as exc:
        s.export_notes.append(f"shared axes failed ({exc})")
    files: dict[str, str] = {}
    info: dict = {}

    if "stl" in formats:
        progress(0.05, "Snapping mesh onto fitted surfaces")
        V = snap_mesh(s, fits)
        p = out_dir / f"{name}_snapped.stl"
        trimesh.Trimesh(V, s.mesh.faces, process=False).export(p)
        files["snapped_stl"] = str(p)
        progress(0.2, "Generating mesh from equations")
        Vs, Fs, off = [], [], 0
        for r in (s.regions if len(s.regions) <= 1500 else []):
            if hasattr(fits[r.id].eq, "preview"):  # a tube: meshed along its spine
                Vr, Fr = fits[r.id].eq.preview(s.P[r.idx])
            else:
                X, _ = fits[r.id].eq.frame.to_param(s.P[r.idx])
                Vr, Fr = preview_mesh(fits[r.id].eq, X, target_cells=12000)
            Vs.append(Vr)
            Fs.append(Fr + off)
            off += len(Vr)
        p = out_dir / f"{name}_equations.stl"
        if Vs:
            trimesh.Trimesh(np.vstack(Vs), np.vstack(Fs), process=False).export(p)
            files["equation_stl"] = str(p)

    if "glb" in formats:
        from .mesher import build_mesh, export_glb

        progress(0.05, "Building the mesh from the surfaces")
        tol = max(getattr(s, "seg_eps", 0.0), 0.0015 * s.diag)
        Vs, Fs, Ns, _, mstats = build_mesh(s, fits, tol, progress=lambda f, m: progress(0.05 + 0.3 * f, m))
        p = out_dir / f"{name}.glb"
        mstats["bytes"] = export_glb(p, Vs, Fs, Ns, units)
        info["glb"] = mstats
        files["glb"] = str(p)

    if "step" in formats:
        progress(0.4, "Writing STEP")
        ex = StepExporter(s, fits, units, progress=lambda f, m: progress(0.4 + 0.5 * f, m))
        ex.face_name = lambda g: f"R{orig_id(g)}"
        p = out_dir / f"{name}.step"
        info["step"] = ex.export(p, name)
        info["step"]["warnings"] = [name_back(w) for w in s.export_notes + info["step"]["warnings"]]
        files["step"] = str(p)

    if "json" in formats:
        p = out_dir / f"{name}_equations.json"
        regions = []
        for r in s.regions:
            f = fits[r.id]
            res = s.results.get(r.id) or {}
            world, world_tex = f.eq.local_world()
            regions.append({
                "id": orig_id(r.id), "segmentation_primitive": r.prim, "n_points": int(len(r.idx)),
                "frame": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in f.eq.frame.to_json().items()},
                "equation_normalized": str(f.eq.expr), "equation_latex": f.eq.latex(),
                "equation_local": world, "equation_local_latex": world_tex,
                "classification": f.cls, "rms_error": f.rms, "discovered": f.discovered,
                "status": res.get("status", "not searched"),
                "hall_of_fame": [{k: h[k] for k in ("complexity", "loss", "equation")}
                                 for h in res.get("hof", [])],
            })
        regions.sort(key=lambda d: d["id"])
        doc = {"source": s.mesh_path, "units": units, "bbox_diagonal": s.diag, "regions": regions,
               "notes": "Each equation is y=f(x0,x1) in its region frame, normalised by frame.scale. "
                        "height: w=f(u,v); cylindrical: r=f(theta,h); spherical: r=f(theta,lat)."}
        # numpy values (a torus centre and axis) as plain numbers and lists
        p.write_text(json.dumps(doc, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
        files["json"] = str(p)

    progress(1.0, f"Export finished in {time.perf_counter() - t0:.1f}s")
    return {"files": files, "info": info}
