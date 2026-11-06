"""Curve-built faces for mostly freeform parts (bottles, jugs, housings, handles).

A designer builds such a part from a few big smooth surfaces whose borders are curves on
the shape's crests; the mesh segmentation instead tiles it with hundreds of patches. Here:
  shape map (principal curvatures) -> necks (openings that are circles) -> calm islands
  and crest flooding -> clean border curves (splines; neck rings are exact circles) -> the
  mesh cut exactly along them -> one trimmed B-spline surface per face, B-spline edges with
  pcurves -> STEP. A mirror-symmetric part is built as one half and its exact mirror image.
Used for inputs whose segmentation is mostly freeform (FREEFORM_SHARE of the area).
Internally everything is in mm in a frame where the mirror plane is y = 0.
"""
import time

import numpy as np
from scipy.spatial import cKDTree

FREEFORM_SHARE = 0.6
MAX_EDGE_GAP = 0.5  # mm: an edge further than this from one of its faces fails the build
MAX_FIT = 1.0  # mm: a surface further than this from its mesh vertices fails the build
# the rules (curvature radii, corner and border lengths, knot spacing, tolerances) are in mm for parts
# about SIZE_MIN..SIZE_MAX across (bounding-box diagonal); a smaller or larger part is worked on at the
# nearest size in that range and scaled back, so its layout and accuracy are the same relative to it
SIZE_MIN, SIZE_MAX = 250.0, 500.0
EDGE_MAX = 8.0  # mm (at the working size): longer mesh edges are split, so every face has data all over


def split_long_edges(V, F, emax, passes=10):
    """Edges longer than emax split, both triangles at once (no cracks); a CAD export's fan triangles
    across a flat or ruled face leave a surface fit nothing to hold on to. The new point is the middle
    of the curve the end normals describe (cubic Hermite): on a curved face a chord's midpoint lies
    inside the surface by its sag, which would read as a crest; across a crease, the chord's middle."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    for _ in range(passes):
        E = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
        Eu, inv = np.unique(E, axis=0, return_inverse=True)
        lu = np.linalg.norm(V[Eu[:, 0]] - V[Eu[:, 1]], axis=1)
        sp = lu > emax
        if not sp.any():
            break
        mid = np.full(len(Eu), -1, np.int64)
        mid[sp] = len(V) + np.arange(int(sp.sum()))
        cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        N = np.zeros_like(V)
        for q in range(3):
            np.add.at(N, F[:, q], cr)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
        p0, p1 = V[Eu[sp, 0]], V[Eu[sp, 1]]
        n0, n1 = N[Eu[sp, 0]], N[Eu[sp, 1]]
        d = p1 - p0
        t0 = d - (d * n0).sum(1, keepdims=True) * n0
        t1 = d - (d * n1).sum(1, keepdims=True) * n1
        bend = 0.125 * (t0 - t1)
        bend[(n0 * n1).sum(1) < np.cos(np.radians(30.0))] = 0.0  # a crease: no smooth curve across it
        V = np.vstack([V, 0.5 * (p0 + p1) + bend])
        m = mid[np.asarray(inv).ravel()].reshape(-1, 3)  # midpoints of (ab, bc, ca) or -1
        a, b, c = F[:, 0], F[:, 1], F[:, 2]
        m0, m1, m2 = m[:, 0], m[:, 1], m[:, 2]
        h0, h1, h2 = m0 >= 0, m1 >= 0, m2 >= 0
        out = [F[~(h0 | h1 | h2)]]
        k = h0 & h1 & h2
        out += [np.c_[a, m0, m2][k], np.c_[m0, b, m1][k], np.c_[m2, m1, c][k], np.c_[m0, m1, m2][k]]
        k = h0 & ~h1 & ~h2
        out += [np.c_[a, m0, c][k], np.c_[m0, b, c][k]]
        k = ~h0 & h1 & ~h2
        out += [np.c_[a, b, m1][k], np.c_[a, m1, c][k]]
        k = ~h0 & ~h1 & h2
        out += [np.c_[a, b, m2][k], np.c_[m2, b, c][k]]
        k = h0 & h1 & ~h2
        out += [np.c_[m0, b, m1][k], np.c_[a, m0, m1][k], np.c_[a, m1, c][k]]
        k = ~h0 & h1 & h2
        out += [np.c_[m2, m1, c][k], np.c_[a, b, m1][k], np.c_[a, m1, m2][k]]
        k = h0 & ~h1 & h2
        out += [np.c_[a, m0, m2][k], np.c_[m0, b, c][k], np.c_[m0, c, m2][k]]
        F = np.vstack(out)
    return V, F


def freeform_share(regs, mesh) -> float:
    A = np.asarray(mesh.area_faces, float)
    tot = A.sum()
    ff = sum(A[np.asarray(f, np.int64)].sum() for kind, _, f in regs if kind == "freeform")
    return float(ff / tot) if tot > 0 else 0.0


def mm_scale(mesh_path, diag: float) -> float:
    """mm per mesh unit: glTF is metres; a part smaller than 2 units is taken as metres too."""
    p = str(mesh_path or "").lower()
    if p.endswith((".glb", ".gltf")) or diag < 2.0:
        return 1000.0
    return 1.0


def _frame(mirror):
    """Rotation R and offset o with x_mod = R (x_mm - o): the mirror plane becomes y = 0."""
    if mirror is None:
        return np.eye(3), np.zeros(3)
    n = np.asarray(mirror["normal"], float)
    n /= np.linalg.norm(n)
    ey = np.array([0.0, 1.0, 0.0])
    c = float(n @ ey)
    if c > 1 - 1e-12:
        R = np.eye(3)
    elif c < -1 + 1e-12:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        v = np.cross(n, ey)
        s = np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
    return R, n * float(mirror["d"])


def prepare(V, F, mirror=None, scale=1.0, log=print, progress=None):
    """The mesh as the curve network works on it: long edges split, mirrored (or the split plane of a part
    that is not symmetric), at the working size, in the frame where the plane is y = 0; with its shape
    map and necks. Shared by build() and the training miner (scripts/curvenet_train)."""
    from .necks import find_necks
    from .shape import principal_features

    prog = progress or (lambda f, m: None)
    t0 = time.perf_counter()
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    V_in, F_in = V, F
    # the working size (see SIZE_MIN) and an even sampling of the shape at it
    size0 = float(np.linalg.norm(np.ptp(V, axis=0))) * scale
    k0 = min(max(size0, SIZE_MIN), SIZE_MAX) / max(size0, 1e-12)
    n0 = len(F)
    V, F = split_long_edges(V, F, EDGE_MAX / (scale * k0))
    if len(F) > n0:
        log(f"curve network: long mesh edges split ({n0:,} -> {len(F):,} triangles)")
    if mirror is None:  # built as one half and its exact mirror image
        import trimesh

        from ..symmetry import find_mirror, symmetrize
        prog(0.02, "Curve network: mirror symmetry")
        m = find_mirror(V, F, log=log)
        if m is None:
            log("curve network: the part is not mirror-symmetric: built whole")
        else:
            # one half kept exactly and mirrored (no averaging of the halves: measuring the gap between
            # them puts facet-sized noise into the shape; a designer models half and mirrors it too)
            msym, mirror = symmetrize(trimesh.Trimesh(V, F, process=False), m, tol=0.0, log=log)
            V, F = np.asarray(msym.vertices, float), np.asarray(msym.faces, np.int64)
    size = float(np.linalg.norm(np.ptp(V, axis=0))) * scale
    k = min(max(size, SIZE_MIN), SIZE_MAX) / max(size, 1e-12)
    if abs(k - 1.0) > 1e-9:
        log(f"curve network: the part is {size:.0f} mm across: worked on at {size * k:.0f} mm and scaled back")
    scale = scale * k
    R, o = _frame(mirror)
    split = False
    if mirror is None:
        # not symmetric: built whole, in two halves split by the plane through the area centroid across
        # the part's thinnest direction (bands and tubes that wrap round then come out as halves)
        Cm = V[F].mean(1)
        Am = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        c0 = (Cm * Am[:, None]).sum(0) / Am.sum()
        _, U = np.linalg.eigh(((Cm - c0) * Am[:, None]).T @ (Cm - c0))
        ext = [float(np.ptp((V - c0) @ U[:, k])) for k in range(3)]
        kt = int(np.argmin(ext))
        ka, kb = [k for k in range(3) if k != kt]
        ey, ex = U[:, kt], U[:, ka]
        R = np.vstack([ex, ey, np.cross(ex, ey)])
        o = c0
        split = True
    o_mm = o * scale
    X = (V * scale - o_mm) @ R.T
    if mirror is not None:
        X[np.abs(X[:, 1]) < 1e-7, 1] = 0.0
    prog(0.05, "Curve network: shape map")
    feat = principal_features(X, F)
    prog(0.2, "Curve network: necks")
    necks = find_necks(X, F, log=log)
    prog(0.3, "Curve network: faces and their border curves")
    tm = mirror.get("tri_mirror") if mirror is not None else None
    return dict(V_in=V_in, F_in=F_in, V=V, F=F, X=X, R=R, o_mm=o_mm, scale=scale, mirror=mirror, split=split,
                feat=feat, necks=necks, tm=tm, half=mirror.get("half") if mirror is not None else None, t0=t0)


def build(V, F, mirror=None, scale=1.0, log=print, progress=None):
    """The B-rep (JSON-able dict, module coordinates), a per-input-triangle face label, and
    to_part (module mm -> the mesh's own coordinates)."""
    from .brepfit import fit_brep
    from .cutter import cut_along
    from .islands import BORDER_FEATURES, island_curves
    from .learned import probability

    prog = progress or (lambda f, m: None)
    w = prepare(V, F, mirror, scale, log, progress)
    V_in, F_in, X, F, R, o_mm, scale = w["V_in"], w["F_in"], w["X"], w["F"], w["R"], w["o_mm"], w["scale"]
    mirror, split, feat, necks, tm, t0 = w["mirror"], w["split"], w["feat"], w["necks"], w["tm"], w["t0"]
    # the learned border decision (curvenet/border_model.json) lays the faces out; if its B-rep fails the
    # checks, the part is built again with the hand rules
    use_model = probability("border_model", [0.0] * len(BORDER_FEATURES)) is not None
    while True:
        curves, flab = island_curves(X, F, feat, necks, tm, mirror.get("half") if mirror is not None else None,
                                     log=log, USE_MODEL=use_model)
        prog(0.45, "Curve network: cutting the mesh along the curves")
        Vc, Fc, Lc = cut_along(X, F, curves, log=log, lab=flab)
        prog(0.55, "Curve network: surfaces")
        J = fit_brep(Vc, Fc, Lc, mirror=mirror is not None, split=split,
                     log=lambda m: None if m.startswith("surface {") else log(m), curves=curves)
        bad = []
        if J["loop_problems"]:
            bad.append(f"{J['loop_problems']} face borders do not close")
        if J["max_edge_tol"] > MAX_EDGE_GAP:
            bad.append(f"an edge is {J['max_edge_tol']:.2f} mm off its faces")
        if J["max_fit"] > MAX_FIT:
            bad.append(f"a surface is {J['max_fit']:.2f} mm off the mesh")
        if bad and use_model:
            log("curve network: the learned layout fails the checks (" + "; ".join(bad) + "): built again with the hand rules")
            use_model = False
            continue
        if bad:
            raise RuntimeError("; ".join(bad))
        break
    # face label of every input triangle (the cut only splits triangles along the curves)
    X_in = (V_in * scale - o_mm) @ R.T
    _, j = cKDTree(Vc[Fc].mean(1)).query(X_in[F_in].mean(1))
    lab = Lc[j]
    _, lab = np.unique(lab, return_inverse=True)

    def to_part(P):
        return (np.asarray(P, float) @ R + o_mm) / scale
    log(f"curve network: {2 * len(J['faces']) if mirror is not None else len(J['faces'])} faces, "
        f"{time.perf_counter() - t0:.0f} s")
    return {"brep": J, "labels": lab, "to_part": to_part, "curves": curves}


def write_step(cn, path, name, units="mm"):
    from .stepout import write_step as _w
    return _w(cn["brep"], path, name, units, to_part=cn["to_part"])
