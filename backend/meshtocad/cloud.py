"""Mesh import and dense point-cloud sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

SUPPORTED = (".stl", ".obj", ".ply", ".off", ".glb", ".gltf", ".fbx")


def fix_winding(mesh: trimesh.Trimesh) -> int:
    """Make triangle winding consistent per connected body, closed bodies outward.

    Neighbouring triangles must run their shared edge in opposite directions, or a
    body's border cannot be traced (and a closed part reads as open). Breadth-first
    over the face graph (scipy, no networkx), each face takes its parent's
    orientation corrected by whether their shared edge agrees. Returns the number
    of faces flipped.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    F = mesh.faces.copy()
    n = len(F)
    adj = mesh.face_adjacency
    if n == 0 or len(adj) == 0:
        return 0
    edges = mesh.face_adjacency_edges  # the shared vertex pair of each adjacent face pair

    def runs_forward(f, u, v):
        tri = F[f]
        nxt = np.roll(tri, -1, axis=1)
        return ((tri == u[:, None]) & (nxt == v[:, None])).any(axis=1)

    u, v = edges[:, 0], edges[:, 1]
    agree = runs_forward(adj[:, 0], u, v) != runs_forward(adj[:, 1], u, v)  # consistent pair
    w = np.where(agree, 1.0, 2.0)  # weight encodes whether crossing flips orientation
    G = csr_matrix((np.r_[w, w], (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), shape=(n, n))
    ncomp, comp = connected_components(G, directed=False)
    # one search tree per component; each face's flip is the XOR of the edge flags
    # on its path to the root (pointer doubling, vectorised)
    pred = np.arange(n)
    for c in range(ncomp):
        root = int(np.argmax(comp == c))
        _, p = breadth_first_order(G, root, directed=False, return_predecessors=True)
        sel = (p >= 0) & (comp == c)
        pred[sel] = p[sel]
    key = np.r_[adj[:, 0] * n + adj[:, 1], adj[:, 1] * n + adj[:, 0]]
    kw = np.r_[~agree, ~agree]
    order = np.argsort(key)
    q = pred * n + np.arange(n)
    pos = np.clip(np.searchsorted(key[order], q), 0, len(key) - 1)
    rel = np.where(pred == np.arange(n), False, kw[order[pos]])
    flip, anc = rel.copy(), pred.copy()  # flip[f]: XOR of flags from f up to (not incl.) anc[f]
    for _ in range(64):
        if (anc == anc[anc]).all():  # every face now points at its root
            break
        flip = flip ^ flip[anc]
        anc = anc[anc]
    F[flip] = F[flip][:, ::-1]
    # closed bodies point outward (positive volume)
    V = mesh.vertices
    for c in range(ncomp):
        fc = np.where(comp == c)[0]
        tri = V[F[fc]]
        vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
        e = np.sort(F[fc][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        if (cnt == 2).all() and vol < 0:
            F[fc] = F[fc][:, ::-1]
            flip[fc] = ~flip[fc]
    mesh.faces = F
    return int(flip.sum())


def _part_name(name: str, k: int) -> str:
    """A STEP-safe part name (ASCII, no quotes)."""
    clean = "".join(c if (c.isalnum() or c in "-_. ") else "_" for c in str(name or "")).strip()
    return clean[:60] or f"part_{k + 1}"


def split_nonmanifold(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Unweld vertices along edges shared by three or more triangles, so that every edge
    joins two triangles at most.

    Two solids exported as one welded mesh that touch along a face (each meshing the face
    with its own copy of every triangle, reversed) or a stray reversed copy of a triangle
    give edges with four triangles. trimesh's face_adjacency skips those edges, so the
    triangles there lose their neighbours: a shared disc fell apart into dozens of
    two-triangle bodies and one-triangle faces. Across such an edge each triangle keeps
    the neighbour that runs the edge the other way (consistent winding), never its own
    reversed copy, preferring the smoothest continuation; vertices are then copied per
    fan of triangles still joined around them. A sheet meshed from both sides (a body
    squashed to zero thickness) comes apart into its two sides. Triangle order is kept.
    Returns the mesh itself when every edge is manifold."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    F = np.asarray(mesh.faces, np.int64)
    n = len(F)
    if n == 0:
        return mesh
    nv = int(F.max()) + 1
    a_ = np.r_[F[:, 0], F[:, 1], F[:, 2]]  # edge j of triangle t runs corner j -> j+1
    b_ = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    t_ = np.tile(np.arange(n), 3)
    c_ = np.repeat(np.arange(3), n)
    key = np.minimum(a_, b_) * nv + np.maximum(a_, b_)
    o = np.argsort(key, kind="stable")
    ks = key[o]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    cnt = np.diff(np.r_[starts, len(ks)])
    if not (cnt > 2).any():
        return mesh
    N = np.asarray(mesh.face_normals, float)
    tri_key = np.sort(F, axis=1)
    wa, wb = [], []  # wedges (triangle*3 + corner) to join

    def link(i, j):  # half-edge slots i, j in a_/b_/t_/c_ index space
        ti, tj, ci, cj = t_[i], t_[j], c_[i], c_[j]
        # corner ci of ti is vertex a_[i]; find that vertex's corner in tj
        for vi, wi in ((a_[i], ti * 3 + ci), (b_[i], ti * 3 + (ci + 1) % 3)):
            cj_v = int(np.flatnonzero(F[tj] == vi)[0])
            wa.append(wi)
            wb.append(tj * 3 + cj_v)

    # vertices on no crowded edge keep all their wedges together (no change there), so
    # only edges touching a crowded vertex need linking one by one
    crowded = np.zeros(nv, bool)
    many = np.repeat(cnt > 2, cnt)
    crowded[a_[o][many]] = True
    crowded[b_[o][many]] = True
    s_first = o[starts]
    near = (cnt > 2) | ((cnt == 2) & (crowded[a_[s_first]] | crowded[b_[s_first]]))
    for s0, k in zip(starts[near], cnt[near]):
        idx = o[s0:s0 + k]
        if k == 2:
            if not (tri_key[t_[idx[0]]] == tri_key[t_[idx[1]]]).all():
                link(idx[0], idx[1])
            continue
        if k < 2:
            continue
        cand = []
        for x in range(k):
            for y in range(x + 1, k):
                i, j = idx[x], idx[y]
                if a_[i] == a_[j]:  # same direction: inconsistent winding
                    continue
                if (tri_key[t_[i]] == tri_key[t_[j]]).all():  # its own reversed copy
                    continue
                cand.append((-float(N[t_[i]] @ N[t_[j]]), int(i), int(j)))
        used = set()
        for _, i, j in sorted(cand):
            if i in used or j in used:
                continue
            used.update((i, j))
            link(i, j)
    W = np.arange(3 * n)
    wv = F[W // 3, W % 3]
    keep = ~crowded[wv]
    first = np.full(nv, -1, np.int64)
    first[wv[keep][::-1]] = W[keep][::-1]
    ra = np.r_[np.asarray(wa, np.int64), W[keep]]
    rb = np.r_[np.asarray(wb, np.int64), first[wv[keep]]]
    g = csr_matrix((np.ones(len(ra)), (ra, rb)), shape=(3 * n, 3 * n))
    k, comp = connected_components(g, directed=False)
    if k == len(np.unique(F)):
        return mesh
    newV = np.zeros((k, 3))
    newV[comp] = np.asarray(mesh.vertices, float)[wv]
    out = trimesh.Trimesh(newV, comp.reshape(n, 3), process=False)
    out.metadata.update(mesh.metadata)
    return out


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """The file's triangles as one mesh, keeping its parts apart.

    Scene formats (glTF/GLB, OBJ groups) are assemblies: each placed object is a
    part. Vertices are welded within a part, never across parts, so touching parts
    stay separate bodies. mesh.metadata["face_part"] gives each face's part and
    mesh.metadata["part_names"] their names.
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type {path.suffix}. Use one of: {', '.join(SUPPORTED)}")
    raw = []
    if path.suffix.lower() == ".fbx":
        # binary FBX (CAD and DCC exports): our own reader, in millimetres
        from .fbx import read as read_fbx
        V, F, part, pnames = read_fbx(path)
        for k, nm in enumerate(pnames):
            Fk = F[part == k]
            used = np.unique(Fk)
            remap = np.full(len(V), -1, np.int64)
            remap[used] = np.arange(len(used))
            raw.append((nm, trimesh.Trimesh(V[used], remap[Fk], process=False)))
        loaded = None
    else:
        loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        for node in loaded.graph.nodes_geometry:
            T, geom = loaded.graph[node]
            g = loaded.geometry.get(geom)
            if isinstance(g, trimesh.Trimesh) and len(g.faces):
                m = trimesh.Trimesh(g.vertices.copy(), g.faces.copy(), process=False)
                m.apply_transform(T)
                raw.append((node if node != geom else geom, m))
    elif isinstance(loaded, trimesh.Trimesh):
        raw.append((path.stem, loaded))
    parts, names = [], []
    for k, (name, m) in enumerate(raw):
        m = trimesh.Trimesh(m.vertices, m.faces, process=True)
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
        if len(m.faces):
            parts.append(m)
            names.append(_part_name(name, k))
    if not parts:
        raise ValueError("File contains no triangle geometry")
    offs = np.cumsum([0] + [len(m.vertices) for m in parts])
    V = np.vstack([m.vertices for m in parts])
    F = np.vstack([m.faces + o for m, o in zip(parts, offs[:-1])])
    mesh = split_nonmanifold(trimesh.Trimesh(V, F, process=False))
    try:
        fix_winding(mesh)
    except Exception:  # winding repair is best-effort
        pass
    mesh.metadata["face_part"] = np.repeat(np.arange(len(parts)), [len(m.faces) for m in parts])
    mesh.metadata["part_names"] = names
    return mesh


def mesh_is_cad(mesh: trimesh.Trimesh) -> bool:
    """A tessellation exported from CAD (vertices exactly on the surfaces).

    CAD tessellators split flat faces, and the quads of curved strips, into
    triangles that are exactly coplanar; scans and sculpted meshes have almost
    no exactly coplanar neighbours (CAD exports measured 14-39 %, scans 0 %).
    """
    ang = mesh.face_adjacency_angles
    return len(ang) > 0 and float((ang < 1e-4).mean()) > 0.05


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    lo, hi = mesh.bounds
    return {
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "bbox_min": lo,
        "bbox_max": hi,
        "diag": float(np.linalg.norm(hi - lo)),
        "area": float(mesh.area),
        "watertight": bool(mesh.is_watertight),
    }


def facet_sag(mesh: trimesh.Trimesh) -> float:
    """How far flat triangles sit from the smooth surface they approximate.

    For two faces meeting at a small angle a (a tessellated curve), a triangle of
    height h across the curve deviates by about h * a / 8 from the true arc. The
    90th percentile over smooth edges is a good tolerance floor for coarse exports.
    """
    ang = mesh.face_adjacency_angles
    smooth = (ang > np.radians(0.5)) & (ang < np.radians(30))
    if not smooth.any():
        return 0.0
    pairs = mesh.face_adjacency[smooth]
    edges = mesh.face_adjacency_edges[smooth]
    elen = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    area = mesh.area_faces[pairs].mean(axis=1)
    height = 2 * area / np.maximum(elen, 1e-12)
    return float(np.percentile(height * ang[smooth] / 8, 90))


def sample_cloud(mesh: trimesh.Trimesh, n_points: int, seed: int = 0):
    """Evenly spaced surface samples with face normals and source face index."""
    pts, fidx = trimesh.sample.sample_surface_even(mesh, int(n_points), seed=seed)
    if len(pts) < 0.7 * n_points:
        # Even sampling rejects close points; top up with area-weighted samples.
        extra, efidx = trimesh.sample.sample_surface(mesh, int(n_points - len(pts)), seed=seed + 1)
        pts = np.vstack([pts, extra])
        fidx = np.concatenate([fidx, efidx])
    normals = mesh.face_normals[fidx]
    return np.asarray(pts, float), np.asarray(normals, float), np.asarray(fidx, np.int64)


def mesh_is_noisy(mesh: trimesh.Trimesh) -> tuple[bool, float, float]:
    """Is this a noisy scan (vs a clean model with fine detail)?

    Scan noise shows at the triangle level: hardly any neighbouring triangles
    are exactly coplanar and the rest jitter by random angles. Clean models -
    even very detailed ones - have many exactly flat neighbour pairs. (Comparing
    point normals with neighbourhood normals cannot tell noise from detail that
    is finer than the sample spacing.)
    """
    ang = np.degrees(mesh.face_adjacency_angles)
    if len(ang) == 0:
        return False, 1.0, 0.0
    w = mesh.area_faces[mesh.face_adjacency].mean(axis=1)
    flat = float(w[ang < 0.5].sum() / max(w.sum(), 1e-300))
    jitter = float(np.median(ang[ang >= 0.5])) if (ang >= 0.5).any() else 0.0
    return flat < 0.10 and jitter > 5.0, flat, jitter


def refine_normals(P: np.ndarray, N: np.ndarray, k: int = 16, force: bool | None = None):
    """Use neighbourhood (PCA) normals when the mesh's own face normals are noisy.

    On a scan, tiny noisy triangles give normals tens of degrees off the true
    surface, which shatters segmentation. Clean CAD meshes keep their exact face
    normals (crisp sharp edges). Returns (normals, used_pca, median_deviation_deg).
    """
    from scipy.spatial import cKDTree

    _, idx = cKDTree(P).query(P, k=k)
    nb = P[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    _, V = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
    pca = V[:, :, 0]
    flip = np.einsum("ij,ij->i", pca, N) < 0
    pca[flip] *= -1
    dev = float(np.median(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", pca, N), -1, 1)))))
    use = dev > 6.0 if force is None else force
    return (pca, True, dev) if use else (N, False, dev)

