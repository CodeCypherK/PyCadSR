"""Surfaces for a curve-built face layout: trimmed cubic B-spline faces sharing B-spline edges.

fit_brep(V, F, L, mirror) takes a mesh cut exactly along its face borders (labels L per
triangle). With mirror=True the part is mirror-symmetric about the plane y = 0 and only the
kept half (y >= 0) is built (the other half is its exact mirror image, made by the writer).
1. (mirror) the mesh is cut exactly at y = 0; crumbs join their neighbours; a face piece with
   a big hole (a handle's tunnel) is cut in two at the hole outline's two sharpest corners
2. every piece is flattened (LSCM, then ARAP) and gets one cubic B-spline surface: a coarse,
   fair surface first, whose foot points are a smooth parametrization (parameter correction),
   then finer knots until the worst vertex is within TOL (least squares, border points weigh
   more, a little third-derivative fairness of length ELL)
3. borders between corners become cubic B-spline edge curves sampled on the surfaces, shared
   by both faces, each face with its own pcurve on the same parameters
Returns a JSON-able B-rep: vertices, surfaces, edges (+pcurves, tolerance), faces (loops).
All lengths in mm.
"""
import json
import time

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags, identity
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu, spsolve

PLANE, OPEN = -1, -2


def fit_brep(V, F, L, mirror=True, H=5.0, ALPHA=1e-5, TOL=0.1, ELL=0.35, TOL_P=0.03, TOL_MAX=0.25, PC_IT=2,
             log=print, curves=None, split=False, SIMPLIFY=True, SIMP_P99=0.05, SIMP_MAX=0.2, SIMP_B=0.05, EDGE_DEV=0.02, SIMP_H0=2.0):
    t0 = time.perf_counter()
    THETA, SMAX, CURVE_MM, ELL0 = 0.25, 1.0, 1.0, 3.0
    V = np.asarray(V, float).copy()
    F = np.asarray(F, np.int64)
    L = np.asarray(L, np.int64).copy()
    # ---------------------------------------------------------------- 1. exact cut at y = 0
    # (split: a part that is not symmetric is cut there too and both halves are built, so bands and
    # tubes that wrap round come out as halves, as they do on a mirrored part)
    if mirror or split:
        V[np.abs(V[:, 1]) < 1e-6, 1] = 0.0
        s = V[:, 1]
        tv = s[F]
        strad = (tv.min(1) < 0) & (tv.max(1) > 0)
        nF, nL, extra, cache = [F[~strad]], [L[~strad]], [], {}
        nv = len(V)
        for t in np.where(strad)[0]:
            f, g = F[t], s[F[t]]
            for sign in (1.0, -1.0):
                poly = []
                for a in range(3):
                    b = (a + 1) % 3
                    ga, gb = sign * g[a], sign * g[b]
                    if ga >= 0:
                        poly.append(int(f[a]))
                    if (ga > 0 > gb) or (ga < 0 < gb):
                        key = (min(f[a], f[b]), max(f[a], f[b]))
                        if key not in cache:
                            tt = ga / (ga - gb)
                            p = V[f[a]] + tt * (V[f[b]] - V[f[a]])
                            p[1] = 0.0
                            cache[key] = nv
                            extra.append(p)
                            nv += 1
                        poly.append(cache[key])
                for q in range(1, len(poly) - 1):
                    nF.append(np.array([[poly[0], poly[q], poly[q + 1]]]))
                    nL.append(np.array([L[t]]))
        V = np.vstack([V, np.array(extra).reshape(-1, 3)])
        F = np.vstack(nF)
        L = np.concatenate(nL)
        C = V[F].mean(1)
        if mirror:
            keep = C[:, 1] > 1e-9
            F, L = F[keep], L[keep]
        else:
            L = np.where(C[:, 1] < 0, L + L.max() + 1, L)
        used, inv = np.unique(F, return_inverse=True)
        V, F = V[used], inv.reshape(-1, 3)
    n_t, n_v = len(F), len(V)
    log(f"surfaces: {n_t} triangles, {n_v} vertices" + (" on the kept half" if mirror else ""))

    # ---------------------------------------------------------------- 2. pieces
    def topo(V, F):
        n_t = len(F)
        he = np.c_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]].reshape(-1, 2)  # half-edges, 3 per triangle
        he_t = np.repeat(np.arange(n_t), 3)
        ek, e_of_he = np.unique(np.sort(he, axis=1), axis=0, return_inverse=True)
        e_of_he = e_of_he.ravel()
        cnt = np.bincount(e_of_he, minlength=len(ek))
        et = np.full((len(ek), 2), -1, np.int64)
        order = np.argsort(e_of_he, kind="stable")
        first = np.r_[0, np.cumsum(cnt)[:-1]]
        et[:, 0] = he_t[order[first]]
        two = cnt >= 2
        et[two, 1] = he_t[order[first[two] + 1]]
        area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        return dict(he=he, he_t=he_t, ek=ek, e_of_he=e_of_he, et=et, inner=et[:, 1] >= 0, area=area, n_t=n_t,
                    nonman=int((cnt > 2).sum()))


    def pieces_of(L, T, walls=None):
        et, inner, n_t = T["et"], T["inner"], T["n_t"]
        same = inner & (L[et[:, 0]] == L[np.maximum(et[:, 1], 0)])
        if walls is not None:
            same &= ~walls
        a, b = et[same, 0], et[same, 1]
        return connected_components(csr_matrix((np.ones(len(a)), (a, b)), shape=(n_t, n_t)), directed=False)[1]


    def merge_crumbs(L, T):
        et = T["et"]
        for _ in range(20):  # crumbs (stray triangles, tiny pieces) join their main neighbour
            pid = pieces_of(L, T)
            pa = np.bincount(pid, weights=T["area"])
            pn = np.bincount(pid)
            bad = np.where((pa < 2.0) | (pn < 30))[0]
            changed = False
            for p in bad:
                tt = np.where(pid == p)[0]
                es = np.isin(et[:, 0], tt) | np.isin(et[:, 1], tt)
                nb = np.r_[et[es, 0], et[es, 1]]
                nb = nb[(nb >= 0)]
                nb = nb[pid[nb] != p]
                if len(nb):
                    L[tt] = np.bincount(L[nb]).argmax()
                    changed = True
            if not changed:
                break
        return L


    def split_plane(V, F, L, mask, nrm, d0):
        """Conforming split of the triangles in mask (and their neighbours across crossed edges) by the
        plane nrm.x = d0."""
        s = V @ nrm - d0
        s[np.abs(s) < 1e-9] = 0.0
        nvv = len(V)
        tk = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=2)  # (n, 3, 2)
        tkey = tk[:, :, 0] * nvv + tk[:, :, 1]
        tv = s[F]
        strad = mask & (tv.min(1) < 0) & (tv.max(1) > 0)
        ce = tk[strad].reshape(-1, 2)
        ce = np.unique(ce[s[ce[:, 0]] * s[ce[:, 1]] < 0], axis=0)
        if not len(ce):
            return V, F, L
        ckey = ce[:, 0] * nvv + ce[:, 1]
        o = np.argsort(ckey)
        ce, ckey = ce[o], ckey[o]
        hit = np.isin(tkey, ckey).any(1)
        ta = s[ce[:, 0]] / (s[ce[:, 0]] - s[ce[:, 1]])
        newp = V[ce[:, 0]] + ta[:, None] * (V[ce[:, 1]] - V[ce[:, 0]])
        pf, pl = [], []
        for t in np.where(hit)[0]:
            f, g = F[t], s[F[t]]
            if not strad[t]:  # a neighbour: only its shared crossed edges get the point (no new crossings)
                poly, first_pt = [], None
                for a_ in range(3):
                    b_ = (a_ + 1) % 3
                    poly.append(int(f[a_]))
                    kk = min(f[a_], f[b_]) * nvv + max(f[a_], f[b_])
                    j = int(np.searchsorted(ckey, kk))
                    if j < len(ckey) and ckey[j] == kk:
                        poly.append(nvv + j)
                        if first_pt is None:
                            first_pt = len(poly) - 1
                poly = poly[first_pt:] + poly[:first_pt]
                for q in range(1, len(poly) - 1):
                    pf.append((poly[0], poly[q], poly[q + 1]))
                    pl.append(L[t])
                continue
            for sign in (1.0, -1.0):
                poly = []
                for a_ in range(3):
                    b_ = (a_ + 1) % 3
                    ga, gb = sign * g[a_], sign * g[b_]
                    if ga >= 0:
                        poly.append(int(f[a_]))
                    if (ga > 0 > gb) or (ga < 0 < gb):
                        kk = min(f[a_], f[b_]) * nvv + max(f[a_], f[b_])
                        poly.append(nvv + int(np.searchsorted(ckey, kk)))
                for q in range(1, len(poly) - 1):
                    pf.append((poly[0], poly[q], poly[q + 1]))
                    pl.append(L[t])
        V2 = np.vstack([V, newp])
        F2 = np.vstack([F[~hit], np.array(pf, np.int64).reshape(-1, 3)])
        L2 = np.concatenate([L[~hit], np.array(pl, np.int64)])
        return V2, F2, L2


    def boundary_loops(p, pid, T):
        """Ordered boundary loops (vertex lists) of piece p."""
        et, ek = T["et"], T["ek"]
        s0 = pid[et[:, 0]]
        s1 = np.where(T["inner"], pid[np.maximum(et[:, 1], 0)], -9)
        eb = ek[(s0 == p) != (s1 == p)]
        nb = {}
        for i, j in eb:
            nb.setdefault(int(i), []).append(int(j))
            nb.setdefault(int(j), []).append(int(i))
        seen, loops = set(), []
        for v0 in nb:
            if v0 in seen:
                continue
            loop, prev, cur = [v0], None, v0
            seen.add(v0)
            while True:
                nxt = [w for w in nb[cur] if w != prev and w not in seen]
                if not nxt:
                    break
                w = nxt[0]
                loop.append(w)
                seen.add(w)
                prev, cur = cur, w
            loops.append(loop)
        return loops


    def hole_cuts(V, P):
        """Two half-planes through the hole's axis at the outline's two sharpest corners."""
        c = P.mean(0)
        ax = np.linalg.eigh(np.cov((P - c).T))[1][:, 0]
        Pc = np.vstack([P, P[:1]])
        u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]
        m = max(16, int(u[-1] / 1.0))
        uu = np.linspace(0, u[-1], m, endpoint=False)
        R = np.c_[[np.interp(uu, u, Pc[:, k]) for k in range(3)]].T
        w = max(2, int(round(4.0 / (u[-1] / m))))
        d1 = R - np.roll(R, w, axis=0)
        d2 = np.roll(R, -w, axis=0) - R
        cosang = (d1 * d2).sum(1) / np.maximum(np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1), 1e-12)
        turn = np.arccos(np.clip(cosang, -1, 1))
        i1 = int(np.argmax(turn))
        cd = np.minimum(np.abs(np.arange(m) - i1), m - np.abs(np.arange(m) - i1))
        i2 = int(np.argmax(np.where(cd > m // 4, turn, -1)))
        cuts = []
        for q in (R[i1], R[i2]):
            r = (q - c) - ((q - c) @ ax) * ax
            r /= np.linalg.norm(r)
            ncut = np.cross(ax, r)
            ncut /= np.linalg.norm(ncut)
            cuts.append((c, r, ncut, float(ncut @ q), q))
        return cuts


    def split_annuli(V, F, L):
        """A piece with a hole is cut in two by two half-planes through the hole's axis, at the hole
        outline's two sharpest corners (the handle tunnel -> bar + the surface under it)."""
        for _ in range(6):
            T = topo(V, F)
            pid = pieces_of(L, T)
            target = None
            for p in range(pid.max() + 1):
                loops = [lp for lp in boundary_loops(p, pid, T) if len(lp) > 8]
                if len(loops) < 2:
                    continue
                lens = [np.linalg.norm(np.diff(V[lp + lp[:1]], axis=0), axis=1).sum() for lp in loops]
                if min(lens) < 20.0:
                    continue
                # a face facing one way (a front around an opening) keeps its hole: it flattens as it
                # is; a band or tube (wrapping round) is cut in two
                tp = np.where(pid == p)[0]
                nt = np.cross(V[F[tp, 1]] - V[F[tp, 0]], V[F[tp, 2]] - V[F[tp, 0]])
                at = np.linalg.norm(nt, axis=1)
                m_ = nt.sum(0)
                m_ /= max(np.linalg.norm(m_), 1e-12)
                facing = float(at[(nt @ m_) > 0.5 * at].sum() / max(at.sum(), 1e-12))
                if facing >= 0.99 and min(lens) < 0.5 * max(lens):
                    continue
                target = (p, loops)
                break
            if target is None:
                break
            p, loops = target
            lab_p = int(np.bincount(L[pid == p]).argmax())
            lens = [np.linalg.norm(np.diff(V[lp + lp[:1]], axis=0), axis=1).sum() for lp in loops]
            hole = loops[int(np.argmin(lens))]
            cuts = hole_cuts(V, V[hole])
            keep0 = (V.copy(), F.copy(), L.copy(), pid, T)
            for c, r, ncut, d0, q in cuts:
                Tc = V[F].mean(1)
                mask = (L == lab_p) & (((Tc - c) @ r) > 0)
                V, F, L = split_plane(V, F, L, mask, ncut, d0)
            onp = np.zeros(len(V), bool)
            for c, r, ncut, d0, q in cuts:
                onp |= (np.abs(V @ ncut - d0) < 1e-6) & (((V - c) @ r) > -1e-6)
            T = topo(V, F)
            walls = onp[T["ek"][:, 0]] & onp[T["ek"][:, 1]]
            comp = pieces_of(L, T, walls)
            sel = L == lab_p
            cs = np.unique(comp[sel])
            if len(cs) < 2:
                log(f"  piece with a hole (label {lab_p}): the cuts did not divide it")
                break
            big = sorted(cs, key=lambda x: -T["area"][comp == x].sum())
            nl = L.max() + 1
            for k_, cc in enumerate(big[1:]):
                L[(comp == cc) & sel] = nl + k_
            # every piece the plane cuts leave must have one border loop; else the face is cut along the
            # two shortest paths between its loops instead (on opposite sides)
            new = [lab_p] + [nl + k_ for k_ in range(len(big) - 1)]
            T2 = topo(V, F)
            pid2 = pieces_of(L, T2)
            ok_ = all(
                len([lp for lp in boundary_loops(q, pid2, T2) if len(lp) > 8]) == 1
                for q in np.unique(pid2[np.isin(L, new)]))
            if not ok_:
                V, F, L, pid0, T0 = keep0
                res = path_cut(V, F, L, p, pid0, T0, loops)
                if res is None:
                    log(f"  piece with a hole (label {lab_p}): could not be cut cleanly")
                    break
                L = res
                log(f"  piece with a hole (label {lab_p}): cut along its shortest paths into 2 faces")
                continue
            log(f"  piece with a hole (label {lab_p}): cut at the hole corners into {len(cs)} faces "
                  f"(corners {np.round(cuts[0][4], 1)}, {np.round(cuts[1][4], 1)})")
        return V, F, L

    def path_cut(V, F, L, p, pid, T, loops):
        """Piece p (two border loops) cut in two along the shortest path between its loops and the
        shortest one starting opposite it on the inner loop; new labels, or None."""
        et, ek, inner = T["et"], T["ek"], T["inner"]
        in_p = (pid[et[:, 0]] == p) | (inner & (pid[np.maximum(et[:, 1], 0)] == p))
        E = ek[in_p]
        w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) + 1e-9
        nv_ = len(V)
        G = csr_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(nv_, nv_))
        lens = [np.linalg.norm(np.diff(V[lp + lp[:1]], axis=0), axis=1).sum() for lp in loops]
        A_ = np.array(loops[int(np.argmax(lens))])
        B_ = list(loops[int(np.argmin(lens))])

        def walk(pred, a):
            path = [int(a)]
            while pred[path[-1]] >= 0:
                path.append(int(pred[path[-1]]))
            return path
        d1, pr1, src1 = dijkstra(G, indices=B_, min_only=True, return_predecessors=True)
        if not np.isfinite(d1[A_]).any():
            return None
        a1 = A_[int(np.argmin(d1[A_]))]
        path1 = walk(pr1, a1)
        b1 = path1[-1]
        Bc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(V[B_ + B_[:1]], axis=0), axis=1))]
        i1 = B_.index(b1)
        tgt = (Bc[i1] + 0.5 * Bc[-1]) % Bc[-1]
        b2 = B_[int(np.argmin(np.abs(Bc[:-1] - tgt)))]
        keep = np.ones(nv_, bool)
        keep[path1] = False
        D = diags(keep.astype(float))
        G2 = (D @ G @ D).tocsr()
        d2, pr2 = dijkstra(G2, indices=b2, return_predecessors=True)
        far = np.linalg.norm(V[A_] - V[a1], axis=1) > 0.25 * np.sqrt(max(lens) * min(lens) / np.pi)
        cand = A_[far & np.isfinite(d2[A_])]
        if not len(cand):
            return None
        a2 = cand[int(np.argmin(d2[cand]))]
        path2 = walk(pr2, a2)
        key = {(int(i), int(j)): k for k, (i, j) in enumerate(ek)}
        walls = np.zeros(len(ek), bool)
        for path in (path1, path2):
            for i, j in zip(path[:-1], path[1:]):
                k = key.get((min(i, j), max(i, j)))
                if k is not None:
                    walls[k] = True
        comp = pieces_of(L, T, walls)
        tp = pid == p
        cs = np.unique(comp[tp])
        if len(cs) != 2:
            return None
        L = L.copy()
        small = min(cs, key=lambda x: T["area"][comp == x].sum())
        L[tp & (comp == small)] = L.max() + 1
        return L


    def drop_junk(V, F, L):
        T = topo(V, F)
        L = merge_crumbs(L, T)
        pid = pieces_of(L, T)
        pa = np.bincount(pid, weights=T["area"])
        bad = np.isin(pid, np.where(pa < 2.0)[0])
        if bad.any():
            log(f"  removed {int(bad.sum())} stray sliver triangles ({pa[pa < 2.0].sum():.2f} mm2)")
            F, L = F[~bad], L[~bad]
            used, inv = np.unique(F, return_inverse=True)
            V, F = V[used], inv.reshape(-1, 3)
        return V, F, L


    T0 = topo(V, F)
    log(f"  {T0['nonman']} non-manifold edges")
    V, F, L = drop_junk(V, F, L)
    V, F, L = split_annuli(V, F, L)
    V, F, L = drop_junk(V, F, L)
    T0 = topo(V, F)
    L = merge_crumbs(L, T0)
    he, he_t, ek, e_of_he, et, inner = T0["he"], T0["he_t"], T0["ek"], T0["e_of_he"], T0["et"], T0["inner"]
    tri_area = T0["area"]
    n_t, n_v, n_e = len(F), len(V), len(ek)
    pid = pieces_of(L, T0)
    _, pid = np.unique(pid, return_inverse=True)
    n_p = pid.max() + 1
    log(f"{n_p} face pieces on the kept half")

    # ---------------------------------------------------------------- 3. borders, corners, chains
    side = np.full((n_e, 2), OPEN, np.int64)  # piece on each side of an edge
    side[:, 0] = pid[et[:, 0]]
    side[inner, 1] = pid[et[inner, 1]]
    onplane = (V[ek[:, 0], 1] == 0) & (V[ek[:, 1], 1] == 0) & bool(mirror)
    side[~inner & onplane, 1] = PLANE
    border = side[:, 0] != side[:, 1]
    vsides = {}
    for e in np.where(border)[0]:
        for v in ek[e]:
            vsides.setdefault(int(v), set()).update(int(x) for x in side[e])
    corner = {v for v, ss in vsides.items() if len(ss) >= 3}
    pair = np.sort(side, axis=1)[:, ::-1]  # (piece, other) with other possibly PLANE/OPEN
    by_pair = {}
    for e in np.where(border)[0]:
        by_pair.setdefault((int(pair[e, 0]), int(pair[e, 1])), []).append(e)
    chains = []
    for pr, es in by_pair.items():
        keys = [(int(ek[e, 0]), int(ek[e, 1])) for e in es]
        nb = {}
        for i, j in keys:
            nb.setdefault(i, []).append(j)
            nb.setdefault(j, []).append(i)
        usedE = set()
        while len(usedE) < len(keys):
            rest = [k for k in keys if k not in usedE]
            verts = {v for k in rest for v in k}
            starts = [v for v in verts if v in corner or len(nb[v]) != 2]
            if not starts:  # a closed loop with no corner on it: give it one
                corner.add(rest[0][0])
                starts = [rest[0][0]]
            progressed = False
            for s0 in starts:
                for nx in nb[s0]:
                    e0 = (min(s0, nx), max(s0, nx))
                    if e0 in usedE:
                        continue
                    path = [s0, nx]
                    usedE.add(e0)
                    progressed = True
                    while path[-1] not in corner and len(nb[path[-1]]) == 2:
                        a_, b_ = nb[path[-1]]
                        q = b_ if a_ == path[-2] else a_
                        e1 = (min(path[-1], q), max(path[-1], q))
                        if e1 in usedE:
                            break
                        usedE.add(e1)
                        path.append(q)
                    chains.append({"pair": pr, "path": path})
            if not progressed:
                break
    log(f"{len(chains)} edges, {len(corner)} corners")

    # oriented half-edges per piece (winding order, face on the left)
    he_dir = {}
    for h_ in range(len(he)):
        e = e_of_he[h_]
        if border[e]:
            he_dir[(int(he[h_, 0]), int(he[h_, 1]))] = int(pid[he_t[h_]])


    # ---------------------------------------------------------------- 4. flattening
    def local_frames(X, T, M=None):
        p0, p1, p2 = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
        e1 = p1 - p0
        l1 = np.linalg.norm(e1, axis=1)
        nrm = np.cross(p1 - p0, p2 - p0)
        a2 = np.linalg.norm(nrm, axis=1)
        good = (a2 > 1e-10) & (l1 > 1e-9)
        e1 = e1 / np.maximum(l1, 1e-12)[:, None]
        nrm = nrm / np.maximum(a2, 1e-12)[:, None]
        e2 = np.cross(nrm, e1)
        x = np.zeros((len(T), 3))
        y = np.zeros((len(T), 3))
        x[:, 1] = l1
        x[:, 2] = ((p2 - p0) * e1).sum(1)
        y[:, 2] = ((p2 - p0) * e2).sum(1)
        if M is not None:
            xy = np.einsum("mab,mkb->mka", M, np.stack([x, y], 2))
            x, y = xy[:, :, 0], xy[:, :, 1]
            a2 = a2 * np.abs(np.linalg.det(M))
        return x, y, a2, good


    def tri_curvature(X, T, N, smooth=2):
        """Per-triangle shape operator (Rusinkiewicz) in 3D, lightly smoothed over neighbours."""
        p0, p1, p2 = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
        e1 = p1 - p0
        e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
        nr = np.cross(p1 - p0, p2 - p0)
        ar = 0.5 * np.linalg.norm(nr, axis=1)
        nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-12)
        e2 = np.cross(nr, e1)
        Es = [p2 - p1, p0 - p2, p1 - p0]
        Ds = [N[T[:, 2]] - N[T[:, 1]], N[T[:, 0]] - N[T[:, 2]], N[T[:, 1]] - N[T[:, 0]]]
        m = len(T)
        A = np.zeros((m, 6, 3))
        r = np.zeros((m, 6))
        for k in range(3):
            eu, ev = (Es[k] * e1).sum(1), (Es[k] * e2).sum(1)
            A[:, 2 * k, 0], A[:, 2 * k, 1] = eu, ev
            A[:, 2 * k + 1, 1], A[:, 2 * k + 1, 2] = eu, ev
            r[:, 2 * k], r[:, 2 * k + 1] = (Ds[k] * e1).sum(1), (Ds[k] * e2).sum(1)
        AtA = np.einsum("mki,mkj->mij", A, A) + 1e-12 * np.eye(3)
        abc = np.linalg.solve(AtA, np.einsum("mki,mk->mi", A, r)[:, :, None])[:, :, 0]
        S3 = (abc[:, 0, None, None] * np.einsum("mi,mj->mij", e1, e1)
              + abc[:, 1, None, None] * (np.einsum("mi,mj->mij", e1, e2) + np.einsum("mi,mj->mij", e2, e1))
              + abc[:, 2, None, None] * np.einsum("mi,mj->mij", e2, e2))
        # neighbours by shared edge
        E = np.sort(np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
        key = E[:, 0] * (len(X) + 1) + E[:, 1]
        o = np.argsort(key)
        ks, ts = key[o], np.repeat(np.arange(m), 3)[o]
        same = ks[1:] == ks[:-1]
        a_, b_ = ts[:-1][same], ts[1:][same]
        W = csr_matrix((np.r_[ar[b_], ar[a_]], (np.r_[a_, b_], np.r_[b_, a_])), shape=(m, m)) + diags(ar)
        deg = np.asarray(W.sum(1)).ravel()
        flat = S3.reshape(m, 9)
        for _ in range(smooth):
            flat = (W @ flat) / deg[:, None]
        S3 = flat.reshape(m, 3, 3)
        S2 = np.stack([np.stack([np.einsum("mi,mij,mj->m", e1, S3, e1), np.einsum("mi,mij,mj->m", e1, S3, e2)], 1),
                       np.stack([np.einsum("mi,mij,mj->m", e2, S3, e1), np.einsum("mi,mij,mj->m", e2, S3, e2)], 1)], 1)
        S2 = 0.5 * (S2 + np.transpose(S2, (0, 2, 1)))
        ev, evec = np.linalg.eigh(S2)
        k = np.argmax(np.abs(ev), axis=1)
        kmax = np.abs(ev[np.arange(m), k])
        d = evec[np.arange(m), :, k]
        return kmax, d


    def umbrella(T, n):
        i = np.r_[T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]]
        j = np.r_[T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]]
        A = csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
        A.data[:] = 1.0
        return (diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()


    def lscm(X, T, bnd, M=None):
        n = len(X)
        x, y, a2, good = local_frames(X, T, M)
        Wr = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
        Wi = np.stack([y[:, 2] - y[:, 1], y[:, 0] - y[:, 2], y[:, 1] - y[:, 0]], 1)
        sc = np.where(good, 1.0 / np.sqrt(np.maximum(a2, 1e-12)), 0.0)[:, None]
        Wr, Wi = Wr * sc, Wi * sc
        m = len(T)
        r = np.arange(m)
        rows = np.r_[np.repeat(r, 6), np.repeat(r + m, 6)]
        cu = T
        cv = T + n
        cols = np.r_[np.c_[cu, cv].ravel(), np.c_[cu, cv].ravel()]
        vals = np.r_[np.c_[Wr, -Wi].ravel(), np.c_[Wi, Wr].ravel()]
        A = csr_matrix((vals, (rows, cols)), shape=(2 * m, 2 * n))
        # pin the two boundary vertices furthest apart
        B = X[bnd]
        sub = bnd[:: max(1, len(bnd) // 400)]
        D = np.linalg.norm(X[sub][:, None] - X[sub][None], axis=2)
        i, j = np.unravel_index(D.argmax(), D.shape)
        a, b = sub[i], sub[j]
        d = D[i, j]
        fixed = np.array([a, b, a + n, b + n])
        fv = np.array([0.0, d, 0.0, 0.0])
        free = np.setdiff1d(np.arange(2 * n), fixed)
        Af, Ax = A[:, free], A[:, fixed]
        Lu = umbrella(T, n)
        Z = csr_matrix((n, n))
        Lu2 = bmat([[Lu, Z], [Z, Lu]]).tocsr()
        N = (Af.T @ Af + 1e-4 * Lu2[free][:, free]).tocsc() + 1e-9 * identity(len(free), format="csc")
        rhs = -(Af.T @ (Ax @ fv))
        sol = spsolve(N, rhs)
        z = np.zeros(2 * n)
        z[free] = sol
        z[fixed] = fv
        return np.c_[z[:n], z[n:]]


    def arap(X, T, uv, iters=12, M=None):
        n = len(X)
        x, y, a2, good = local_frames(X, T, M)
        P = np.stack([x, y], 2)  # (m, 3, 2) local coords
        # cotangent weights per triangle edge (i->j opposite to k), clamped
        cot = np.zeros((len(T), 3))
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            u_ = P[:, i] - P[:, k]
            w_ = P[:, j] - P[:, k]
            cr = u_[:, 0] * w_[:, 1] - u_[:, 1] * w_[:, 0]
            dt = (u_ * w_).sum(1)
            cot[:, k] = np.clip(dt / np.maximum(np.abs(cr), 1e-12), 1e-3, 1e3)
        cot[~good] = 0.0
        I_, J_, W_ = [], [], []
        for k in range(3):
            i, j = T[:, (k + 1) % 3], T[:, (k + 2) % 3]
            I_ += [i, j, i, j]
            J_ += [j, i, i, j]
            W_ += [-cot[:, k], -cot[:, k], cot[:, k], cot[:, k]]
        Lm = coo_matrix((np.concatenate(W_), (np.concatenate(I_), np.concatenate(J_))), shape=(n, n)).tocsr()
        Lm = Lm + 1e-3 * umbrella(T, n) + 1e-8 * identity(n)
        Lm = Lm.tolil()
        Lm[0, :] = 0
        Lm[0, 0] = 1.0
        lu = splu(Lm.tocsc())
        for _ in range(iters):
            # local: best rotation per triangle
            S = np.zeros((len(T), 2, 2))
            for k in range(3):
                i, j = (k + 1) % 3, (k + 2) % 3
                du = uv[T[:, i]] - uv[T[:, j]]
                dx = P[:, i] - P[:, j]
                S += cot[:, k, None, None] * du[:, :, None] * dx[:, None, :]
            U_, _, Vt = np.linalg.svd(S)
            R = U_ @ Vt
            neg = np.linalg.det(R) < 0
            U_[neg, :, 1] *= -1
            R[neg] = U_[neg] @ Vt[neg]
            b = np.zeros((n, 2))
            for k in range(3):
                i, j = (k + 1) % 3, (k + 2) % 3
                dx = P[:, i] - P[:, j]
                r_ = np.einsum("mab,mb->ma", R, dx) * cot[:, k, None]
                np.add.at(b, T[:, i], r_)
                np.add.at(b, T[:, j], -r_)
            b[0] = uv[0]
            uv = np.c_[lu.solve(b[:, 0]), lu.solve(b[:, 1])]
        return uv


    def flips(uv, T):
        a = uv[T[:, 1]] - uv[T[:, 0]]
        b = uv[T[:, 2]] - uv[T[:, 0]]
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


    # ---------------------------------------------------------------- 5. B-spline fitting
    def clamped(lo, hi, nspan, k=3):
        return np.r_[[lo] * k, np.linspace(lo, hi, nspan + 1), [hi] * k]


    def design(x, t, k=3):
        return BSpline.design_matrix(np.clip(x, t[k], t[-k - 1] - 1e-12), t, k).tocsr()


    def tensor(uv, tu, tv):
        Bu, Bv = design(uv[:, 0], tu), design(uv[:, 1], tv)
        nu, nv_ = len(tu) - 4, len(tv) - 4
        N = len(uv)
        iu, wu = Bu.indices.reshape(N, 4), Bu.data.reshape(N, 4)
        iv, wv = Bv.indices.reshape(N, 4), Bv.data.reshape(N, 4)
        cols = (iu[:, :, None] * nv_ + iv[:, None, :]).reshape(N, 16)
        vals = (wu[:, :, None] * wv[:, None, :]).reshape(N, 16)
        return csr_matrix((vals.ravel(), (np.repeat(np.arange(N), 16), cols.ravel())), shape=(N, nu * nv_))


    def bending(nu, nv_):
        idx = np.arange(nu * nv_).reshape(nu, nv_)
        rows, cols, vals = [], [], []
        r = 0

        def add(ids, ws):
            nonlocal r
            m = ids[0].size
            for id_, w_ in zip(ids, ws):
                rows.append(np.arange(r, r + m))
                cols.append(id_.ravel())
                vals.append(np.full(m, w_))
            r += m
        add([idx[:-2], idx[1:-1], idx[2:]], [1, -2, 1])
        add([idx[:, :-2], idx[:, 1:-1], idx[:, 2:]], [1, -2, 1])
        s2 = np.sqrt(2.0)
        add([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]], [s2, -s2, -s2, s2])
        D = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nu * nv_))
        return D


    def jerk(nu, nv_):
        """Third differences of the control net (curvature variation): uuu, uuv, uvv, vvv."""
        idx = np.arange(nu * nv_).reshape(nu, nv_)
        rows, cols, vals = [], [], []
        r = 0

        def add(ids, ws):
            nonlocal r
            m = ids[0].size
            for id_, w_ in zip(ids, ws):
                rows.append(np.arange(r, r + m))
                cols.append(id_.ravel())
                vals.append(np.full(m, w_))
            r += m
        c = [1, -3, 3, -1]
        add([idx[k:nu - 3 + k] for k in range(4)], c)
        add([idx[:, k:nv_ - 3 + k] for k in range(4)], c)
        s3 = np.sqrt(3.0)
        # uuv: second difference in u of a first difference in v
        add([idx[a:nu - 2 + a, b:nv_ - 1 + b] for a in range(3) for b in range(2)],
            [s3 * x * y for x in (1, -2, 1) for y in (-1, 1)])
        add([idx[b:nu - 1 + b, a:nv_ - 2 + a] for a in range(3) for b in range(2)],
            [s3 * x * y for x in (1, -2, 1) for y in (-1, 1)])
        D = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nu * nv_))
        return D


    def fit_surface(X, uv, w, H=H, area_uv=None, ell=None, knots=None):
        lo, hi = uv.min(0), uv.max(0)
        span = hi - lo
        if knots is None:
            nsu = max(3, int(np.ceil(span[0] / H)) + 2)
            nsv = max(3, int(np.ceil(span[1] / H)) + 2)
            hu, hv = span[0] / (nsu - 2), span[1] / (nsv - 2)
            tu = clamped(lo[0] - hu, hi[0] + hu, nsu)
            tv = clamped(lo[1] - hv, hi[1] + hv, nsv)
        else:  # given (non-uniform) knots
            tu, tv = knots
            nsu, nsv = len(tu) - 7, len(tv) - 7
            hu, hv = (tu[-4] - tu[3]) / nsu, (tv[-4] - tv[3]) / nsv
        nu, nv_ = nsu + 3, nsv + 3
        M = tensor(uv, tu, tv)
        W = diags(w)
        # fairness in physical units: thin-plate energy with smoothing length ELL (mm), so finer knots
        # never mean a wigglier surface; rho = weighted data per unit area
        if area_uv is None:
            area_uv = span[0] * span[1]
        rho = w.sum() / max(area_uv, 1e-9)
        h2 = hu * hv
        lam3 = rho * (ELL if ell is None else ell) ** 6 / h2 ** 2
        lam2 = ALPHA * w.sum() / (nu * nv_)
        D2, D3 = bending(nu, nv_), jerk(nu, nv_)
        A = (M.T @ W @ M + lam3 * (D3.T @ D3) + lam2 * (D2.T @ D2)
             + 1e-10 * identity(nu * nv_)).tocsc()
        lu = splu(A)
        rhs = M.T @ (W @ X)
        P = np.c_[[lu.solve(rhs[:, k]) for k in range(3)]].T
        res = np.linalg.norm(M @ P - X, axis=1)
        return {"tu": tu, "tv": tv, "nu": nu, "nv": nv_, "P": P}, res


    def surf_eval(S, uv):
        return tensor(uv, S["tu"], S["tv"]) @ S["P"]


    def foot_points(S, Xt, uv0, iters=6, eps=1e-4):
        """Parameters of the closest points of Xt on S (Gauss-Newton from uv0)."""
        uv = uv0.copy()
        lo = np.array([S["tu"][3], S["tv"][3]]) + 1e-9
        hi = np.array([S["tu"][-4], S["tv"][-4]]) - 1e-9
        for _ in range(iters):
            r = surf_eval(S, uv) - Xt
            du = (surf_eval(S, uv + [eps, 0]) - surf_eval(S, uv - [eps, 0])) / (2 * eps)
            dv = (surf_eval(S, uv + [0, eps]) - surf_eval(S, uv - [0, eps])) / (2 * eps)
            a11, a12, a22 = (du * du).sum(1), (du * dv).sum(1), (dv * dv).sum(1)
            b1, b2 = (du * r).sum(1), (dv * r).sum(1)
            det = np.maximum(a11 * a22 - a12 * a12, 1e-18)
            step = np.c_[(a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det]
            sn = np.linalg.norm(step, axis=1, keepdims=True)
            step *= np.minimum(1.0, 2.0 / np.maximum(sn, 1e-12))  # damped
            uv = np.clip(uv - step, lo, hi)
        return uv


    def surf_normal(S, uv, eps=1e-3):
        du = surf_eval(S, uv + [eps, 0]) - surf_eval(S, uv - [eps, 0])
        dv = surf_eval(S, uv + [0, eps]) - surf_eval(S, uv - [0, eps])
        n = np.cross(du, dv)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


    def fit_curve(t, Q, knots=None, k=3, fixed_ends=True, lam=1e-6):
        m = len(Q)
        if knots is None:
            L_ = np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()
            k = min(3, m - 1)
            nc = int(np.clip(round(L_ / CURVE_MM) + k + 1, k + 1, max(k + 1, m)))
            knots = np.r_[[0.0] * k, np.linspace(0, 1, nc - k + 1), [1.0] * k]
        k = len(knots) - len(np.unique(knots)) - 1
        k = int(round((len(knots) - (len(np.unique(knots)))) / 2))
        nc = len(knots) - k - 1
        B = BSpline.design_matrix(np.clip(t, 0, 1 - 1e-12), knots, k).toarray()
        B[-1] = 0
        B[-1, -1] = 1.0
        D = np.diff(np.eye(nc), 2, axis=0) if nc > 2 else np.zeros((0, nc))
        if fixed_ends:
            Pin = np.zeros((nc, Q.shape[1]))
            Pin[0], Pin[-1] = Q[0], Q[-1]
            fr = np.arange(1, nc - 1)
            if len(fr):
                Bf = B[:, fr]
                Af = np.vstack([Bf, np.sqrt(lam) * D[:, fr]])
                rhs = np.vstack([Q - B[:, [0, -1]] @ Pin[[0, -1]], -np.sqrt(lam) * D[:, [0, -1]] @ Pin[[0, -1]]])
                Pin[fr] = np.linalg.lstsq(Af, rhs, rcond=None)[0]
            return knots, Pin
        return knots, np.linalg.lstsq(np.vstack([B, np.sqrt(lam) * D]), np.vstack([Q, np.zeros((len(D), Q.shape[1]))]),
                                      rcond=None)[0]


    def curve_eval(knots, P, t):
        k = int(round((len(knots) - len(np.unique(knots))) / 2))
        return BSpline(knots, P, k)(np.clip(t, 0, 1))


    # ---------------------------------------------------------------- 6. per piece
    vnorm = np.zeros_like(V)
    crs = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for q in range(3):
        np.add.at(vnorm, F[:, q], crs)
    vnorm /= np.maximum(np.linalg.norm(vnorm, axis=1, keepdims=True), 1e-300)
    bverts = set(int(v) for e in np.where(border)[0] for v in ek[e])
    surfaces, puv, stats = [], [], []
    for p in range(n_p):
        T = F[pid == p]
        gv, Tl = np.unique(T, return_inverse=True)
        Tl = Tl.reshape(-1, 3)
        X = V[gv]
        isb = np.array([int(v) in bverts for v in gv])
        t1 = time.perf_counter()
        # metric: stretch across strong curvature so the knots crowd onto crests and beads
        kmax, kd = tri_curvature(X, Tl, vnorm[gv])
        sfac = np.clip(kmax * H / THETA, 1.0, SMAX)
        Mt = np.eye(2)[None] + (sfac - 1.0)[:, None, None] * np.einsum("ma,mb->mab", kd, kd)
        uv = lscm(X, Tl, np.where(isb)[0], Mt)
        sa = flips(uv, Tl)
        if sa.sum() < 0:
            uv[:, 1] *= -1
        _, _, a2m, _ = local_frames(X, Tl, Mt)
        a3 = 0.5 * a2m.sum()
        uv *= np.sqrt(a3 / max(0.5 * np.abs(flips(uv, Tl)).sum(), 1e-12))
        uv = arap(X, Tl, uv, M=Mt)
        sa = flips(uv, Tl)
        if sa.sum() < 0:
            uv[:, 1] *= -1
            sa = -sa
        nflip = int((sa < -1e-12).sum())
        # principal axes of the flattening, so the patch rectangle is tight
        c0 = uv.mean(0)
        ev, evec = np.linalg.eigh(np.cov((uv - c0).T))
        R2 = evec[:, ::-1]
        if np.linalg.det(R2) < 0:
            R2[:, 1] *= -1
        uv = (uv - c0) @ R2
        w = np.where(isb, 30.0, 1.0)
        # parameter correction: a coarse, fair surface first; the points' foot points on it are a smooth
        # parametrization (the flattening is only piecewise linear), the fine fit then works on those
        a_uv = 0.5 * np.abs(flips(uv, Tl)).sum()
        for H0 in (3 * H, 1.6 * H):
            S0, _ = fit_surface(X, uv, w, H0, area_uv=a_uv, ell=ELL0)
            uv_new = foot_points(S0, X, uv)
            sh = np.linalg.norm(uv_new - uv, axis=1)
            lo0 = np.array([S0["tu"][3], S0["tv"][3]])
            hi0 = np.array([S0["tu"][-4], S0["tv"][-4]])
            atb = ((uv_new - lo0) < 1e-6).any(1) | ((hi0 - uv_new) < 1e-6).any(1)
            # a border point keeps its place when it would move far or hit the patch edge; the
            # correction fades in over the first rings inside the border
            keep_old = atb | (sh > 3.0) | (isb & (sh > 0.5))
            wgt = np.where(keep_old, 0.0, 1.0)
            nbr_ = umbrella(Tl, len(X))
            deg_ = np.asarray(nbr_.diagonal()).ravel()
            A_ = (diags(deg_) - nbr_)
            wgt[isb] = 0.0
            for _ in range(3):
                wgt = np.minimum(wgt, (A_ @ wgt + wgt) / (deg_ + 1))
            uv_new = uv + wgt[:, None] * (uv_new - uv)
            fl = flips(uv_new, Tl)
            if (fl < -1e-12).sum() <= max(nflip, 0) + 2:
                uv = uv_new
                a_uv = 0.5 * np.abs(fl).sum()
        nbr_ = umbrella(Tl, len(X))
        deg_ = np.asarray(nbr_.diagonal()).ravel()
        A_ = (diags(deg_) - nbr_)

        def correct(S, uv):
            """Foot points of the vertices on S as their new parameters (border points stay, the
            correction fades in over the first rings; rejected if it would fold the flattening)."""
            uv_new = foot_points(S, X, uv)
            sh = np.linalg.norm(uv_new - uv, axis=1)
            lo0 = np.array([S["tu"][3], S["tv"][3]])
            hi0 = np.array([S["tu"][-4], S["tv"][-4]])
            atb = ((uv_new - lo0) < 1e-6).any(1) | ((hi0 - uv_new) < 1e-6).any(1)
            wgt = np.where(atb | (sh > 3.0), 0.0, 1.0)
            wgt[isb] = 0.0
            for _ in range(3):
                wgt = np.minimum(wgt, (A_ @ wgt + wgt) / (deg_ + 1))
            uv_new = uv + wgt[:, None] * (uv_new - uv)
            if (flips(uv_new, Tl) < -1e-12).sum() <= max(nflip, 0) + 2:
                return uv_new
            return uv

        for Hk in (H, H / 1.6, H / 2.5, H / 4.0):  # finer knots until the worst point is within TOL
            S, res = fit_surface(X, uv, w, Hk, area_uv=a_uv)
            for _ in range(PC_IT):  # parameter correction: the residual becomes the distance to the surface
                uv_c = correct(S, uv)
                if uv_c is uv:
                    break
                uv = uv_c
                S, res = fit_surface(X, uv, w, Hk, area_uv=a_uv)
            if PC_IT:
                res = np.linalg.norm(surf_eval(S, foot_points(S, X, uv)) - X, axis=1)
            if res.max() <= TOL or S["nu"] * S["nv"] > 40000 or (
                    TOL_P and np.percentile(res, 99.5) <= TOL_P and res.max() <= TOL_MAX):
                break
        if SIMPLIFY:
            # simplification: as few control points as hold the shape. On the corrected parameters, a
            # coarse grid first; then knot rows / columns only in the spans where points are still off
            # (each step splits in u or in v, whichever gains more per added control point)
            n0 = S["nu"] * S["nv"]
            # limits: the stated ones, or a little above what the full fit reaches where that is worse
            tp = max(SIMP_P99, 1.2 * float(np.percentile(res, 99)))
            tm = max(SIMP_MAX, 1.2 * float(res.max()))
            tb = max(SIMP_B, 1.2 * float(res[isb].max()))

            def judge(S_):
                r_ = np.linalg.norm(surf_eval(S_, foot_points(S_, X, uv)) - X, axis=1)
                bad_ = (r_ > tp) | (isb & (r_ > tb))
                ok_ = np.percentile(r_, 99) <= tp and r_.max() <= tm and r_[isb].max() <= tb
                return r_, bad_, ok_

            def split(t_, x_, hmin):
                ku = np.unique(t_[3:-3])
                si = np.clip(np.searchsorted(ku, x_, side="right") - 1, 0, len(ku) - 2)
                cnt = np.bincount(si, minlength=len(ku) - 1)
                wide = np.diff(ku) > 1.5 * hmin
                sel = np.where((cnt >= 2) & wide)[0]
                if not len(sel):
                    return None
                return np.sort(np.r_[t_, 0.5 * (ku[sel] + ku[sel + 1])])

            Sa, _ = fit_surface(X, uv, w, SIMP_H0 * H, area_uv=a_uv)
            ra, bada, oka = judge(Sa)
            hmin = H / 4.0
            for _ in range(24):
                if oka or Sa["nu"] * Sa["nv"] >= n0:
                    break
                uvb = foot_points(Sa, X[bada], uv[bada])
                best_ = None
                for axis in (0, 1):
                    tnew = split(Sa["tu"] if axis == 0 else Sa["tv"], uvb[:, axis], hmin)
                    if tnew is None:
                        continue
                    kn = (tnew, Sa["tv"]) if axis == 0 else (Sa["tu"], tnew)
                    Sb, _ = fit_surface(X, uv, w, area_uv=a_uv, knots=kn)
                    rb, badb, okb = judge(Sb)
                    gain = (bada.sum() - badb.sum()) / max(Sb["nu"] * Sb["nv"] - Sa["nu"] * Sa["nv"], 1)
                    if best_ is None or okb > best_[0] or (okb == best_[0] and gain > best_[1]):
                        best_ = (okb, gain, Sb, rb, badb)
                if best_ is None:
                    break
                _, _, Sa, ra, bada = best_
                oka = best_[0]
            if oka and Sa["nu"] * Sa["nv"] < n0:
                S, res = Sa, ra
                log(f"  piece {p}: simplified {n0} -> {S['nu'] * S['nv']} control points")
        ns = surf_normal(S, uv)
        agree = float(np.mean((ns * vnorm[gv]).sum(1) > 0))
        # distortion: 3D edge length / uv edge length
        e3 = np.linalg.norm(X[Tl[:, 0]] - X[Tl[:, 1]], axis=1)
        e2 = np.linalg.norm(uv[Tl[:, 0]] - uv[Tl[:, 1]], axis=1)
        rat = e3[e2 > 1e-9] / e2[e2 > 1e-9]
        st = {"piece": p, "label": int(np.bincount(L[pid == p]).argmax()), "tris": len(T), "area": round(float(a3), 1),
              "ctrl": [S["nu"], S["nv"]], "rms": round(float(np.sqrt(np.mean(res ** 2))), 4),
              "p99": round(float(np.percentile(res, 99)), 4), "max": round(float(res.max()), 4),
              "border_max": round(float(res[isb].max()), 4), "flips": nflip, "normal_agree": round(agree, 3),
              "scale_p5_p95": [round(float(np.percentile(rat, 5)), 2), round(float(np.percentile(rat, 95)), 2)],
              "s": round(time.perf_counter() - t1, 1), "worst_at": np.round(X[int(np.argmax(res))], 1).tolist()}
        log("surface " + json.dumps(st))
        stats.append(st)
        surfaces.append(S)
        puv.append(dict(zip(gv.tolist(), uv)))

    # ---------------------------------------------------------------- 7. edges and pcurves
    for ch in chains:
        corner.add(ch["path"][0])
        corner.add(ch["path"][-1])
    cidx = {v: i for i, v in enumerate(sorted(corner))}
    def clean_path(path, step=0.2, spike=45.0):
        """Drop points closer than `step` to the last kept one and sharp hooks next to the ends."""
        P = V[path]
        keep = [0]
        for i in range(1, len(path) - 1):
            if np.linalg.norm(P[i] - P[keep[-1]]) >= step:
                keep.append(i)
        if len(keep) > 1 and np.linalg.norm(P[-1] - P[keep[-1]]) < step:
            keep.pop()
        keep.append(len(path) - 1)
        for _ in range(4):
            if len(keep) < 4:
                break
            changed = False
            for side in (1, -2):
                k = side if side > 0 else len(keep) + side
                a_, b_, c_ = P[keep[k - 1]], P[keep[k]], P[keep[k + 1]]
                d1, d2 = b_ - a_, c_ - b_
                cs = d1 @ d2 / max(np.linalg.norm(d1) * np.linalg.norm(d2), 1e-12)
                if np.degrees(np.arccos(np.clip(cs, -1, 1))) > spike:
                    keep.pop(k)
                    changed = True
                    break
            if not changed:
                break
        return [path[i] for i in keep]


    def surf_jac(S, uv, eps=1e-4):
        du = (surf_eval(S, uv + [eps, 0]) - surf_eval(S, uv - [eps, 0])) / (2 * eps)
        dv = (surf_eval(S, uv + [0, eps]) - surf_eval(S, uv - [0, eps])) / (2 * eps)
        return du, dv


    def project(S, Xt, uv0, iters=8):
        """Closest points of Xt on surface S (damped Gauss-Newton from uv0)."""
        return foot_points(S, Xt, uv0, iters)


    def project_old(S, Xt, uv0, iters=6):
        uv = uv0.copy()
        lo = np.array([S["tu"][3], S["tv"][3]])
        hi = np.array([S["tu"][-4], S["tv"][-4]]) - 1e-9
        for _ in range(iters):
            r = surf_eval(S, uv) - Xt
            du, dv = surf_jac(S, uv)
            a11, a12, a22 = (du * du).sum(1), (du * dv).sum(1), (dv * dv).sum(1)
            b1, b2 = (du * r).sum(1), (dv * r).sum(1)
            det = np.maximum(a11 * a22 - a12 * a12, 1e-18)
            uv = uv - np.c_[(a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det]
            uv = np.clip(uv, lo, hi)
        return uv


    def densify(path, step=0.5):
        """Sample a border on the surface of its first face: linear in that face's flattening between
        the border's mesh points (a long mesh edge is a chord, the surface is not)."""
        P = V[path]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        k = np.maximum(1, np.ceil(seg / step).astype(int))
        idx = np.r_[np.repeat(np.arange(len(seg)), k), len(seg) - 1]
        fr = np.r_[np.concatenate([np.arange(kk) / kk for kk in k]), 1.0]
        return idx, fr


    edges = []
    def path_len(pth):
        return float(np.linalg.norm(np.diff(V[pth], axis=0), axis=1).sum())


    chains = [ch for ch in chains if path_len(clean_path(ch["path"])) > 0.05
              and not (ch["path"][0] == ch["path"][-1] and path_len(ch["path"]) < 2.0)]
    def build_edge(path, a, b, mode):
        """3D curve + pcurves for one border. mode 'surface': samples on the surfaces between the mesh
        points (better on long mesh edges); mode 'vertices': the border's mesh points only."""
        if mode == "surface":
            idx, fr = densify(path)
        else:
            idx, fr = np.arange(len(path)), np.zeros(len(path))
        nxt = np.minimum(idx + 1, len(path) - 1)
        Ua0 = np.array([puv[a][v] for v in path])
        Ua = Ua0[idx] + fr[:, None] * (Ua0[nxt] - Ua0[idx])
        Ub = None
        if b >= 0:
            Ub0 = np.array([puv[b][v] for v in path])
            Ub = Ub0[idx] + fr[:, None] * (Ub0[nxt] - Ub0[idx])
        if mode == "surface":
            Q = surf_eval(surfaces[a], Ua)
            if b >= 0:
                Qb = surf_eval(surfaces[b], Ub)
                Ub_p = project(surfaces[b], Q, Ub)
                Ua_p = project(surfaces[a], Qb, Ua)
                ga = np.linalg.norm(Q - surf_eval(surfaces[b], Ub_p), axis=1)
                gb = np.linalg.norm(Qb - surf_eval(surfaces[a], Ua_p), axis=1)
                use_b = gb < ga
                Q = np.where(use_b[:, None], 0.5 * (Qb + surf_eval(surfaces[a], Ua_p)),
                             0.5 * (Q + surf_eval(surfaces[b], Ub_p)))
                Ua = project(surfaces[a], Q, np.where(use_b[:, None], Ua_p, Ua))
                Ub = project(surfaces[b], Q, np.where(use_b[:, None], Ub, Ub_p))
        else:
            Q = V[path].copy()
        Q[0], Q[-1] = V[path[0]], V[path[-1]]
        Ua[0], Ua[-1] = Ua0[0], Ua0[-1]
        if Ub is not None:
            Ub[0], Ub[-1] = Ub0[0], Ub0[-1]
        return finish_edge(Q, Ua, Ub, a, b, mode)

    def finish_edge(Q, Ua, Ub, a, b, mode):
        if b == PLANE:
            Q[:, 1] = 0.0
        # the ends' parameters: the corners' foot points on each surface (a corner's flattening position
        # need not map onto it once parameter correction has refitted the surface)
        Ua[[0, -1]] = project(surfaces[a], Q[[0, -1]], Ua[[0, -1]])
        if Ub is not None and b >= 0:
            Ub[[0, -1]] = project(surfaces[b], Q[[0, -1]], Ub[[0, -1]])
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        t = np.r_[0, np.cumsum(seg)] / max(seg.sum(), 1e-12)
        tt = np.linspace(0, 1, 400)

        def build(kn):
            knots, P3 = fit_curve(t, Q, knots=kn)
            c3 = curve_eval(knots, P3, tt)
            pcs, tol = {}, 0.0
            for pc, U in ((a, Ua), (b, Ub)):
                if pc < 0:
                    continue
                _, P2 = fit_curve(t, U, knots=knots)
                pcs[pc] = P2
                tol = max(tol, float(np.linalg.norm(surf_eval(surfaces[pc], curve_eval(knots, P2, tt)) - c3, axis=1).max()))
            dev = float(np.linalg.norm(curve_eval(knots, P3, t) - Q, axis=1).max())
            return dict(knots=knots, P=P3, pcurves=pcs, tol=tol, len=float(seg.sum()), dev=dev, mode=mode)
        full = build(None)
        if SIMPLIFY:  # simplification: the fewest control points that hold the curve and its faces
            nmax = len(full["P"])
            nc = max(5, int(np.ceil(seg.sum() / 25.0)) + 3)
            while nc < nmax and nc <= len(Q):
                cand = build(np.r_[[0.0] * 3, np.linspace(0, 1, nc - 2), [1.0] * 3])
                if cand["dev"] <= max(EDGE_DEV, full["dev"]) and cand["tol"] <= max(1.5 * full["tol"], full["tol"] + 0.01):
                    return cand
                nc = int(nc * 1.5) + 1
        return full


    # the design curves, densified, for edges between two faces
    if curves:
        dpts, dcid, dsv = [], [], []
        for ci_, c in enumerate(curves):
            Q = np.asarray(c["points"], float)
            if c.get("closed"):
                Q = np.vstack([Q, Q[:1]])
            sq = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
            ss = np.linspace(0, sq[-1], max(4, int(sq[-1] / 0.2) + 1))
            dpts.append(np.c_[[np.interp(ss, sq, Q[:, k]) for k in range(3)]].T)
            dcid.append(np.full(len(ss), ci_))
            dsv.append(ss)
        dpts, dcid, dsv = np.vstack(dpts), np.concatenate(dcid), np.concatenate(dsv)
        from scipy.spatial import cKDTree as _KD
        dtree = _KD(dpts)

    def design_edge(path, a, b):
        """The edge along its design curve: the curve between the corners, brought onto both faces
        (alternating projection: on a crease it meets the two surfaces' intersection), or None."""
        P = V[path]
        d, j = dtree.query(P)
        ci = dcid[j]
        vals, cnt = np.unique(ci[d < 2.5], return_counts=True)
        if not len(vals) or cnt.max() < 0.8 * len(P):
            return None
        c_ = int(vals[np.argmax(cnt)])
        sel = np.where(dcid == c_)[0]
        Qc, sc = dpts[sel], dsv[sel]
        Lc_ = sc[-1]
        closed = bool(curves[c_].get("closed"))
        _, jj = _KD(Qc).query(P)
        sp = sc[jj]
        if closed:  # direction and wrap from the path's own order
            dsp = np.diff(sp)
            dsp = (dsp + 0.5 * Lc_) % Lc_ - 0.5 * Lc_
            sp = sp[0] + np.r_[0, np.cumsum(dsp)]
        if abs(sp[-1] - sp[0]) < 0.5:
            return None
        ss = np.linspace(sp[0], sp[-1], max(8, int(abs(sp[-1] - sp[0]) / 0.25) + 1))
        sw = ss % Lc_ if closed else np.clip(ss, 0, Lc_)
        Q = np.c_[[np.interp(sw, sc, Qc[:, k]) for k in range(3)]].T
        # ends: exactly the B-rep corners, the difference faded in over 3 mm
        sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        for end, dist in ((0, sl), (-1, sl[-1] - sl)):
            delta = V[path[end]] - Q[end]
            t_ = np.clip(1 - dist / max(min(3.0, 0.4 * sl[-1]), 1e-9), 0, 1)
            Q = Q + (t_ * t_ * (3 - 2 * t_))[:, None] * delta
        # starting parameters from the nearest border vertex
        _, jv = _KD(P).query(Q)
        Ua = np.array([puv[a][path[k]] for k in jv])
        Ub = np.array([puv[b][path[k]] for k in jv])
        for _ in range(5):
            Ua = project(surfaces[a], Q, Ua)
            Qa = surf_eval(surfaces[a], Ua)
            Ub = project(surfaces[b], Qa, Ub)
            Qb = surf_eval(surfaces[b], Ub)
            Q = 0.5 * (Qa + Qb)
        Q[0], Q[-1] = V[path[0]], V[path[-1]]
        Ua[0], Ua[-1] = puv[a][path[0]], puv[a][path[-1]]
        Ub[0], Ub[-1] = puv[b][path[0]], puv[b][path[-1]]
        return finish_edge(Q, Ua, Ub, a, b, "design")

    n_design = 0
    for ch in chains:
        path = clean_path(ch["path"])
        a, b = ch["pair"]
        cands = [build_edge(path, a, b, m) for m in ("surface", "vertices")]
        best = min(cands, key=lambda c: c["tol"])
        if curves and b >= 0:
            de = design_edge(path, a, b)
            if de is not None and de["tol"] <= max(2.0 * best["tol"], 0.05):
                best = de
                n_design += 1
        # orientation for each piece: forward if (start -> next) runs with that piece's winding
        o0, o1 = ch["path"][0], ch["path"][1]
        fw = {pc: he_dir.get((o0, o1)) == pc for pc in (a, b) if pc >= 0}
        edges.append({"v0": cidx[path[0]], "v1": cidx[path[-1]], "pair": [a, b], "forward": fw, **best})
    log(f"edges from surface samples: {sum(e['mode'] == 'surface' for e in edges)} of {len(edges)}, "
        f"from the design curves: {n_design}")
    for e in sorted(edges, key=lambda e: -e["tol"])[:4]:
        if e["tol"] > 0.25:
            log(f"  edge {e['pair']} {e['mode']} {e['len']:.1f} mm: {e['tol']:.3f} mm off its faces (from {np.round(V[sorted(corner)[e['v0']]], 1)})")
    log(f"edge tolerance: median {np.median([e['tol'] for e in edges]):.4f}  max {max(e['tol'] for e in edges):.4f} mm;"
          f" curve vs border max {max(e['dev'] for e in edges):.4f} mm")

    # loops per piece: chain the oriented edges end to start
    faces = []
    loop_problems = 0
    for p in range(n_p):
        mine = [(i, e["forward"][p]) for i, e in enumerate(edges) if p in e["forward"]]
        start = {}
        for i, f_ in mine:
            e = edges[i]
            s_, t_ = (e["v0"], e["v1"]) if f_ else (e["v1"], e["v0"])
            start.setdefault(s_, []).append((i, f_, t_))
        left = set(range(len(mine)))
        loops = []
        pos = {(i, f_): k for k, (i, f_) in enumerate(mine)}
        while left:
            k0 = min(left)
            i, f_ = mine[k0]
            loop = []
            e = edges[i]
            s0 = e["v0"] if f_ else e["v1"]
            cur = (i, f_)
            while True:
                left.discard(pos[cur])
                loop.append(cur)
                e = edges[cur[0]]
                end = e["v1"] if cur[1] else e["v0"]
                if end == s0:
                    break
                nxt = [c for c in start.get(end, []) if pos[(c[0], c[1])] in left]
                if not nxt:
                    log(f"  piece {p}: loop does not close at corner {end}")
                    loop_problems += 1
                    break
                cur = (nxt[0][0], nxt[0][1])
            loops.append(loop)

        def uv_area(loop):
            pts = []
            for i, f_ in loop:
                e = edges[i]
                c2 = curve_eval(e["knots"], e["pcurves"][p], np.linspace(0, 1, 50))
                pts.append(c2 if f_ else c2[::-1])
            Pp = np.vstack(pts)
            return 0.5 * np.sum(Pp[:, 0] * np.roll(Pp[:, 1], -1) - np.roll(Pp[:, 0], -1) * Pp[:, 1])
        ar = [uv_area(lp) for lp in loops]
        o = np.argsort(-np.array(ar))
        faces.append({"surface": p, "loops": [loops[k] for k in o], "uv_areas": [round(float(ar[k]), 1) for k in o],
                      "label": stats[p]["label"]})
        if ar[o[0]] <= 0 or any(a > 0 for a in np.array(ar)[o[1:]]):
            loop_problems += 1
        if len(loops) != 1 or ar[o[0]] <= 0:
            log(f"  piece {p}: {len(loops)} loops, uv areas {[round(float(a), 1) for a in ar]}")
            for lp in loops:
                log("      " + str([(i, edges[i]["pair"], round(edges[i]["len"], 2)) for i, _ in lp]))

    J = {"units": "mm", "mirror": {"normal": [0, 1, 0], "d": 0.0} if mirror else None,
         "vertices": [V[v].tolist() for v in sorted(corner)],
         "surfaces": [{"udeg": 3, "vdeg": 3, "uknots": S["tu"].tolist(), "vknots": S["tv"].tolist(),
                       "nu": S["nu"], "nv": S["nv"], "ctrl": S["P"].reshape(S["nu"], S["nv"], 3).tolist()}
                      for S in surfaces],
         "edges": [{"v0": e["v0"], "v1": e["v1"], "pair": e["pair"], "knots": e["knots"].tolist(),
                    "ctrl": e["P"].tolist(), "pcurves": {str(k): v.tolist() for k, v in e["pcurves"].items()},
                    "tol": e["tol"], "len": e["len"]} for e in edges],
         "faces": [{"surface": f["surface"], "label": f["label"], "loops": [[[int(i), bool(fw)] for i, fw in lp]
                                                                            for lp in f["loops"]]} for f in faces],
         "stats": stats, "seconds": round(time.perf_counter() - t0, 1), "loop_problems": loop_problems,
         "max_fit": max(s_["max"] for s_ in stats), "max_edge_tol": max(e["tol"] for e in edges)}
    log(f"surfaces: {n_p} faces on the kept half, {len(edges)} edges, {time.perf_counter() - t0:.0f} s")
    return J
