"""Mesh output built from the discovered surfaces (written as GLB).

Every triangle of the result lies on a region's surface (its equation, or the
data-fitted spline where no equation reached the tolerance):

* Plane patches are re-triangulated from their boundary polygons: exact, and a
  handful of triangles however finely the input was tessellated.
* Curved patches keep the input's triangle layout; each vertex is moved onto the
  region's surface along the region frame's height direction (the direction the
  equation was fitted in), which is stable where closest-point searches are not.
* Vertices on a border between regions are placed once, on the common curve of
  the surfaces that meet there, and shared by both sides, so the mesh has no
  cracks. A vertex whose surface would move it further than the tolerance allows
  keeps its measured position (the surface does not describe it), so a poor fit
  can never throw out a spike.

Vertices are duplicated per patch in the output, so shading is smooth across a
surface and sharp where surfaces meet.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .cad import alt_project, boundary_halfedges, face_patches, patch_loops, vertex_labels


def lift(eq, P: np.ndarray, max_move: float):
    """Points moved onto the surface along the frame's height direction."""
    if hasattr(eq, "project_points"):  # an exact torus projects onto its own tube
        Q = eq.project_points(P)
    else:
        X, _ = eq.frame.to_param(P)
        Q = eq.frame.from_param(X, eq(X))
    bad = ~np.isfinite(Q).all(axis=1) | (np.linalg.norm(Q - P, axis=1) > max_move)
    Q[bad] = P[bad]
    return Q, bad


def _plane_of(fits, labs):
    for l in labs:
        if fits[int(l)].cls.get("type") == "plane":
            return fits[int(l)].eq
    return None


def move_limits(s, fits, tol: float) -> np.ndarray:
    """How far a vertex may move onto each region's surface.

    A surface that fits its region with rms e describes the region's vertices to
    within a few e; a vertex that would move further belongs to detail the surface
    does not capture. A surface off by more than half the tolerance does not
    describe its region well enough to reshape it at all (limit 0: the measured
    shape is kept until a better surface is found).
    """
    lim = np.zeros(len(s.regions))
    for r in s.regions:
        f = fits[r.id]
        rms = float(f.rms) if np.isfinite(f.rms) else np.inf
        lim[r.id] = 0.0 if rms > 0.5 * tol else float(np.clip(3.0 * rms, 0.05 * tol, 2.0 * tol))
    return lim


def place_vertices(s, fits, lim: np.ndarray, progress=None):
    """New positions for every mesh vertex (interior: lifted; border: shared common point)."""
    V, F, FL = s.mesh.vertices, s.mesh.faces, s.face_labels
    pairs, starts = vertex_labels(F, FL, len(V))
    counts = np.diff(starts)
    out = V.copy()
    kept = np.zeros(len(V), bool)
    single = np.where(counts == 1)[0]
    lab1 = pairs[starts[single], 1]
    order = np.argsort(lab1, kind="stable")
    bounds = np.searchsorted(lab1[order], np.unique(lab1))
    regions = np.unique(lab1)
    for k, rid in enumerate(regions):
        vs = single[order[bounds[k]:(bounds[k + 1] if k + 1 < len(bounds) else len(order))]]
        if lim[int(rid)] <= 0:
            kept[vs] = True
            continue
        out[vs], bad = lift(fits[int(rid)].eq, V[vs], lim[int(rid)])
        kept[vs] = bad
        if progress and k % 200 == 0:
            progress(0.5 * k / max(len(regions), 1), "Placing vertices on their surfaces")
    multi = np.where(counts > 1)[0]
    if len(multi):
        # group border vertices by the set of regions they touch
        width = int(counts.max())
        lab_rows = np.full((len(multi), width), -1, np.int64)
        for j in range(width):
            has = counts[multi] > j
            lab_rows[has, j] = pairs[starts[multi[has]] + j, 1]
        uniq, inv = np.unique(lab_rows, axis=0, return_inverse=True)
        inv = inv.ravel()
        g_order = np.argsort(inv, kind="stable")
        g_bounds = np.searchsorted(inv[g_order], np.arange(len(uniq) + 1))
        for g, labs in enumerate(uniq):
            vs = multi[g_order[g_bounds[g]:g_bounds[g + 1]]]
            labs = labs[labs >= 0]
            # place on the surfaces that describe their regions; a poorly fitted
            # neighbour keeps its measured shape and simply follows this vertex
            labs = labs[lim[labs] > 0]
            if len(labs) == 0:
                kept[vs] = True
                continue
            m = float(lim[labs].min())
            Q = alt_project(V[vs], [fits[int(l)].eq for l in labs], m, final=_plane_of(fits, labs))
            kept[vs] = np.all(Q == V[vs], axis=1)
            out[vs] = Q
            if progress and g % 500 == 0:
                progress(0.5 + 0.3 * g / max(len(uniq), 1), "Placing border vertices")
    return out, kept


def _triangulate_plane_patch(eq, Vn, loops, faces_orig, Vorig):
    """Earcut triangulation of a plane patch from its boundary loops, or None if unsafe."""
    import mapbox_earcut

    rings, ids = [], []
    for verts, _ in loops:
        X, _ = eq.frame.to_param(Vn[verts])
        if not np.isfinite(X).all():
            return None
        rings.append(X[:, :2])
        ids.append(verts)
    areas = [0.5 * float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])) for r in rings]
    outer = int(np.argmax(np.abs(areas)))
    order = [outer] + [i for i in range(len(rings)) if i != outer]
    pts = np.vstack([rings[i] for i in order]).astype(np.float64)
    vid = np.concatenate([ids[i] for i in order])
    ends = np.cumsum([len(rings[i]) for i in order]).astype(np.uint32)
    tri = np.asarray(mapbox_earcut.triangulate_float64(pts, ends), np.int64).reshape(-1, 3)
    if len(tri) == 0:
        return None
    # the polygon must cover the same area as the patch did (else loops were odd)
    a2 = pts[tri]
    got = 0.5 * np.abs((a2[:, 1, 0] - a2[:, 0, 0]) * (a2[:, 2, 1] - a2[:, 0, 1])
                       - (a2[:, 2, 0] - a2[:, 0, 0]) * (a2[:, 1, 1] - a2[:, 0, 1])).sum()
    Xo, _ = eq.frame.to_param(Vorig[faces_orig].reshape(-1, 3))
    o2 = Xo[:, :2].reshape(-1, 3, 2)
    want = 0.5 * np.abs((o2[:, 1, 0] - o2[:, 0, 0]) * (o2[:, 2, 1] - o2[:, 0, 1])
                        - (o2[:, 2, 0] - o2[:, 0, 0]) * (o2[:, 1, 1] - o2[:, 0, 1])).sum()
    if want <= 0 or abs(got - want) > 0.02 * want:
        return None
    T = vid[tri]
    # keep the input's winding (outward side)
    n_new = np.cross(Vn[T[:, 1]] - Vn[T[:, 0]], Vn[T[:, 2]] - Vn[T[:, 0]]).sum(axis=0)
    n_old = np.cross(Vorig[faces_orig[:, 1]] - Vorig[faces_orig[:, 0]],
                     Vorig[faces_orig[:, 2]] - Vorig[faces_orig[:, 0]]).sum(axis=0)
    if float(n_new @ n_old) < 0:
        T = T[:, ::-1]
    return T


def build_mesh(s, fits, tol: float, progress=None):
    """Returns (vertices, faces, vertex normals, face region ids, stats)."""
    progress = progress or (lambda *a: None)
    V, F, FL = s.mesh.vertices, s.mesh.faces, s.face_labels
    lim = move_limits(s, fits, tol)
    Vn, kept = place_vertices(s, fits, lim, progress)
    progress(0.8, "Re-triangulating planes")
    PL, region_of, _ = face_patches(F, s.mesh.face_adjacency, FL)
    A, B, LAB, NB = boundary_halfedges(F, PL, len(V))
    p_order = np.argsort(PL, kind="stable")
    p_at = np.searchsorted(PL[p_order], np.arange(len(region_of) + 1))
    h_order = np.argsort(LAB, kind="stable")
    h_at = np.searchsorted(LAB[h_order], np.arange(len(region_of) + 1))
    out_faces, out_patch = [], []
    n_plane, n_plane_fallback = 0, 0
    for pid in range(len(region_of)):
        fids = p_order[p_at[pid]:p_at[pid + 1]]
        if len(fids) == 0:
            continue
        rid = int(region_of[pid])
        T = None
        if fits[rid].cls.get("type") == "plane" and len(fids) > 2 and lim[rid] > 0:
            sel = h_order[h_at[pid]:h_at[pid + 1]]
            loops = patch_loops(A[sel], B[sel], NB[sel])
            # every interior vertex must really lie on the plane, or re-triangulating
            # would drop detail the plane does not have
            if loops:
                inner = np.setdiff1d(np.unique(F[fids]), np.concatenate([lv for lv, _ in loops]))
                if kept[inner].any():
                    loops = []
            if loops:
                try:
                    T = _triangulate_plane_patch(fits[rid].eq, Vn, loops, F[fids], V)
                except Exception:  # noqa: BLE001
                    T = None
                n_plane += 1
                n_plane_fallback += T is None
        if T is None:
            T = F[fids]
        out_faces.append(T)
        out_patch.append(np.full(len(T), pid, np.int64))
        if pid % 2000 == 0:
            progress(0.8 + 0.15 * pid / max(len(region_of), 1), "Re-triangulating planes")
    Fo = np.vstack(out_faces)
    patch = np.concatenate(out_patch)
    # drop triangles that collapsed (all three corners on one point after placement)
    e1 = Vn[Fo[:, 1]] - Vn[Fo[:, 0]]
    e2 = Vn[Fo[:, 2]] - Vn[Fo[:, 0]]
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    good = area2 > 1e-14 * s.diag ** 2
    Fo, patch = Fo[good], patch[good]
    # one vertex copy per patch: smooth shading within a surface, sharp between surfaces
    key = patch[:, None] * len(V) + Fo
    uk, inv = np.unique(key.ravel(), return_inverse=True)
    Fs = inv.reshape(-1, 3)
    Vs = Vn[uk % len(V)]
    fn = np.cross(Vs[Fs[:, 1]] - Vs[Fs[:, 0]], Vs[Fs[:, 2]] - Vs[Fs[:, 0]])  # area-weighted
    Ns = np.zeros_like(Vs)
    for k in range(3):
        np.add.at(Ns, Fs[:, k], fn)
    Ns /= np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-300)
    # how much of the surface comes from equations / data-fitted splines / measurements:
    # a triangle is on its surface when all three corners are (within a tenth of the tolerance)
    area = 0.5 * np.linalg.norm(fn, axis=1)
    vreg = np.zeros(len(Vs), np.int64)
    vreg[Fs.ravel()] = np.repeat(region_of[patch], 3)
    on = np.zeros(len(Vs), bool)
    r_order = np.argsort(vreg, kind="stable")
    r_at = np.searchsorted(vreg[r_order], np.arange(len(s.regions) + 1))
    for rid in range(len(s.regions)):
        vv = r_order[r_at[rid]:r_at[rid + 1]]
        if len(vv):
            Q, _ = lift(fits[rid].eq, Vs[vv], np.inf)
            on[vv] = np.linalg.norm(Q - Vs[vv], axis=1) <= 0.1 * tol
    measured = ~on[Fs].all(axis=1)
    data_fit = np.array([bool(fits[r.id].cls.get("data_fit")) for r in s.regions])[region_of[patch]]
    tot = max(float(area.sum()), 1e-300)
    coverage = {"equations": float(area[~measured & ~data_fit].sum() / tot * 100),
                "splines": float(area[~measured & data_fit].sum() / tot * 100),
                "measured": float(area[measured].sum() / tot * 100)}
    stats = {"coverage_pct": coverage,
             "triangles": int(len(Fs)), "input_triangles": int(len(F)), "patches": int(len(region_of)),
             "planes_retriangulated": int(n_plane - n_plane_fallback), "plane_fallbacks": int(n_plane_fallback),
             "vertices_kept_measured": int(kept.sum()), "vertices": int(len(V)),
             "regions_kept_measured": int((lim <= 0).sum()),
             "collapsed_triangles_dropped": int((~good).sum())}
    return Vs, Fs, Ns, region_of[patch], stats


UNIT_TO_METRE = {"mm": 0.001, "cm": 0.01, "m": 1.0, "inch": 0.0254}


def export_glb(path, Vs, Fs, Ns, units: str = "m", face_region=None, colors=None):
    """Write the mesh as binary glTF (glTF lengths are metres)."""
    scale = UNIT_TO_METRE.get(units, 1.0)
    mesh = trimesh.Trimesh(Vs * scale, Fs, process=False)
    mesh.vertex_normals = Ns
    if colors is not None:
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)
    scene = trimesh.Scene(mesh)
    data = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)
