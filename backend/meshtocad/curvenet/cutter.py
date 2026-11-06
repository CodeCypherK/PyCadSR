"""Cut a mesh exactly along a curve network: the faces' borders ARE the curves.

Each vertex takes the signed side of its nearest curve; a mesh edge whose ends are on
opposite sides of a curve (near it, not past an open end) is a wall; the vertex groups
between walls are the faces. Every crossed mesh edge then gets ONE crossing point on the
curve, shared by both its triangles, and the triangles are split there (a conforming cut:
no cracks). Tiny or thin pieces join the neighbour they share most border with.
All lengths in mm.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .shape import vertex_normals


def _dense(Q, step=0.5, closed=False):
    if closed:
        Q = np.vstack([Q, Q[:1]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    ss = np.linspace(0, s[-1], max(3, int(s[-1] / step) + 1))
    R = np.c_[[np.interp(ss, s, Q[:, k]) for k in range(3)]].T
    return R[:-1] if closed else R


def border_edges(Fn, Ln):
    E = np.r_[Fn[:, [0, 1]], Fn[:, [1, 2]], Fn[:, [2, 0]]]
    lb = np.r_[Ln, Ln, Ln]
    k = np.sort(E, axis=1)
    o = np.lexsort((k[:, 1], k[:, 0]))
    k, lb = k[o], lb[o]
    same = (k[1:] == k[:-1]).all(1) & (lb[1:] != lb[:-1])
    return k[:-1][same], lb[:-1][same], lb[1:][same]


def _corners(curves, tol=0.3):
    """Design corners: where three or more open curves end (ends within `tol` mm are one corner).
    Each: position, and the incident curves as (curve index, points running out of the corner)."""
    ends = [(ci, k, np.asarray(c["points"], float)[k]) for ci, c in enumerate(curves)
            if not c.get("closed") and not c.get("name", "").startswith("link") for k in (0, -1)]
    if not ends:
        return []
    P = np.array([e[2] for e in ends])
    tr = cKDTree(P)
    grp = np.full(len(P), -1)
    out = []
    for i in range(len(P)):
        if grp[i] >= 0:
            continue
        mem = [j for j in tr.query_ball_point(P[i], tol) if grp[j] < 0]
        grp[mem] = len(out)
        out.append({"p": P[mem].mean(0),
                    "inc": [(ends[j][0], np.asarray(curves[ends[j][0]]["points"], float)[::1 if ends[j][1] == 0 else -1])
                            for j in mem]})
    return [c for c in out if len(c["inc"]) >= 3]


def _sector_labels(V, VN, curves, corners, vlab, R_MAX=6.0):
    """Around a design corner the faces are the sectors between the curves leaving it: a vertex within
    R mm of the corner takes the face between the two curves it lies between (each curve's direction
    taken at the vertex's own distance, so bent curves are followed). The layout's labels wander a
    few mm there, which would put the B-rep corner off the curves' corner (a notch in the edges)."""
    vlab = vlab.copy()
    best = np.full(len(V), np.inf)
    tree = cKDTree(V)
    for cn in corners:
        c0 = cn["p"]
        arcs = [np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))] for _, Q in cn["inc"]]
        R = min(R_MAX, 0.5 * min(a[-1] for a in arcs))
        if R < 1.0:
            continue
        idx = np.array(tree.query_ball_point(c0, R), np.int64)
        if not len(idx):
            continue
        n = VN[idx].sum(0)
        n /= max(np.linalg.norm(n), 1e-12)
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        D = V[idx] - c0
        r = np.linalg.norm(D, axis=1)
        ap = np.arctan2(D @ e2, D @ e1)
        ang = []
        for (ci, Q), s in zip(cn["inc"], arcs):
            rr = np.clip(r, min(0.5, s[-1]), s[-1])
            q = np.c_[[np.interp(rr, s, Q[:, k]) for k in range(3)]].T - c0
            ang.append(np.arctan2(q @ e2, q @ e1))
        rel = (np.array(ang) - ap[None, :]) % (2 * np.pi)  # curves x vertices, counter-clockwise from the vertex
        nxt, prv = rel.argmin(0), rel.argmax(0)
        pairs = [set(int(x) for x in curves[ci]["faces"]) for ci, _ in cn["inc"]]
        around = set().union(*pairs)
        for m, v in enumerate(idx):
            if r[m] >= best[v] or int(vlab[v]) not in around:
                continue
            com = pairs[nxt[m]] & pairs[prv[m]]
            if len(com) == 1:
                vlab[v] = next(iter(com))
                best[v] = r[m]
    return vlab


def _closest_on_tri(p, A, B, C):
    """Point of triangle ABC nearest to p (barycentric clamp onto the triangle)."""
    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return A
    bp = p - B
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return B
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return A + d1 / (d1 - d3) * ab
    cp = p - C
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return C
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return A + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return B + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (C - B)
    den = 1.0 / (va + vb + vc)
    return A + ab * vb * den + ac * vc * den


def _labels_by_side(V, F, E, curves, allp, cid, tang, NQ, lab, VN=None):
    """Vertex labels from the layout's faces: each vertex takes its triangles' majority face; a vertex
    within 4 mm of a curve (and not past an open end of it) whose face is one of the curve's two faces
    takes the face on its side of the nearest such curve."""
    nv = len(V)
    pairs = np.c_[F.ravel(), np.repeat(lab, 3)]
    u, c = np.unique(pairs, axis=0, return_counts=True)
    o = np.lexsort((-c, u[:, 0]))
    u = u[o]
    first = np.r_[True, u[1:, 0] != u[:-1, 0]]
    vlab0 = np.full(nv, -1, np.int64)
    vlab0[u[first, 0]] = u[first, 1]
    vlab = vlab0.copy()
    best_d = np.full(nv, np.inf)
    for ci_, cv in enumerate(curves):
        a_, b_ = (int(x) for x in cv["faces"])
        sel_p = np.where(cid == ci_)[0]
        Q, tq, nq = allp[sel_p], tang[sel_p], NQ[sel_p]
        d, j = cKDTree(Q).query(V, distance_upper_bound=4.0)
        cand = np.isfinite(d) & ((vlab0 == a_) | (vlab0 == b_)) & (d < best_d)
        if not cv.get("closed"):
            for end, t_end in ((0, -tq[0]), (len(Q) - 1, tq[-1])):
                jj = np.minimum(j, len(Q) - 1)
                cand &= ~((jj == end) & (((V - Q[end]) @ t_end) > 0))
        idx = np.where(cand)[0]
        if not len(idx):
            continue
        jj = j[idx]
        side = ((V[idx] - Q[jj]) * np.cross(tq[jj], nq[jj])).sum(1) > 0
        # which face is on the + side: where most of the curve's a-vertices are
        on_a = vlab0[idx] == a_
        plus_a = (side & on_a).sum() + (~side & ~on_a).sum() >= (side & ~on_a).sum() + (~side & on_a).sum()
        pa, pb = (a_, b_) if plus_a else (b_, a_)
        vlab[idx] = np.where(side, pa, pb)
        best_d[idx] = d[idx]
    if VN is not None:
        vlab = _sector_labels(V, VN, curves, _corners(curves), vlab)
    # stray groups (a face's vertices cut off from the rest of it) join the neighbouring face
    ea, eb = E[:, 0], E[:, 1]
    same = vlab[ea] == vlab[eb]
    _, comp = connected_components(csr_matrix((np.ones(int(same.sum())), (ea[same], eb[same])), shape=(nv, nv)),
                                   directed=False)
    size = np.bincount(comp)
    lab_main = {}
    order = np.argsort(-size)
    rep_v = np.full(len(size), -1)
    rep_v[comp] = np.arange(nv)
    for cc in order:
        l_ = int(vlab[rep_v[cc]])
        if l_ not in lab_main:
            lab_main[l_] = cc
    stray = np.array([lab_main.get(int(vlab[rep_v[cc]])) != cc for cc in range(len(size))])
    bad = stray[comp]
    for _ in range(60):
        if not bad.any():
            break
        m_ = bad[ea] != bad[eb]
        src = np.where(bad[ea[m_]], eb[m_], ea[m_])
        dst = np.where(bad[ea[m_]], ea[m_], eb[m_])
        vlab[dst] = vlab[src]
        bad[dst] = False
    return vlab


def cut_along(V, F, curves, log=print, lab=None):
    nv = len(V)
    VN = vertex_normals(V, F)
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1), axis=0)
    curves = [dict(c, points=_dense(np.asarray(c["points"], float), closed=c.get("closed", False))) for c in curves]
    allp = np.vstack([c["points"] for c in curves])
    cid = np.concatenate([np.full(len(c["points"]), i) for i, c in enumerate(curves)])
    tang = np.vstack([np.gradient(np.vstack([c["points"][-1:], c["points"], c["points"][:1]]), axis=0)[1:-1]
                      if c.get("closed") else np.gradient(c["points"], axis=0) for c in curves])
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    ctree = cKDTree(allp)
    # the surface normal AT each curve point (mean of the nearby vertex normals): the side test is the
    # plane through the curve spanned by its tangent and this normal, so it stays consistent in creases
    # and steps, where the vertex normals on the two walls differ
    _, jn = cKDTree(V).query(allp, k=10)
    NQ = VN[jn].sum(1)
    NQ /= np.maximum(np.linalg.norm(NQ, axis=1, keepdims=True), 1e-12)
    ea, eb = E[:, 0], E[:, 1]
    if lab is not None:
        vlab = _labels_by_side(V, F, E, curves, allp, cid, tang, NQ, np.asarray(lab, np.int64), VN=VN)
    corners = _corners(curves)
    ctree_c = cKDTree(np.array([c["p"] for c in corners])) if corners else None
    wall = np.zeros(len(E), bool)
    for ci_, c in (enumerate(curves) if lab is None else ()):
        Q = c["points"]
        tq = tang[cid == ci_]
        nq = NQ[cid == ci_]
        tr = cKDTree(Q)
        da, ja = tr.query(V[ea])
        db, jb = tr.query(V[eb])
        near_e = (da < 4.0) & (db < 4.0)
        if not near_e.any():
            continue
        sa = ((V[ea] - Q[ja]) * np.cross(tq[ja], nq[ja])).sum(1)
        sb = ((V[eb] - Q[jb]) * np.cross(tq[jb], nq[jb])).sum(1)
        cross_e = near_e & ((sa > 0) != (sb > 0))
        if not c.get("closed"):  # past an open end the curve does not divide anything
            for end, t_end in ((0, -tq[0]), (len(Q) - 1, tq[-1])):
                beyond = (((V[ea] - Q[end]) @ t_end) > 0) & (((V[eb] - Q[end]) @ t_end) > 0)
                cross_e &= ~(beyond & ((ja == end) | (jb == end)))
        wall |= cross_e
    if lab is None:
        ok = ~wall
        _, vlab = connected_components(csr_matrix((np.ones(int(ok.sum())), (ea[ok], eb[ok])), shape=(nv, nv)),
                                       directed=False)
        cnt = np.bincount(vlab)
        small = cnt[vlab] < 30  # stray vertices pinned between curves join their neighbours
    else:
        small = np.zeros(nv, bool)
    for _ in range(20):
        if not small.any():
            break
        nbl = np.full(nv, -1)
        for i, j in E[(small[ea] != small[eb])]:
            if small[i] and not small[j]:
                nbl[i] = vlab[j]
            elif small[j] and not small[i]:
                nbl[j] = vlab[i]
        fix = small & (nbl >= 0)
        vlab[fix] = nbl[fix]
        small &= ~fix
    _, vlab = np.unique(vlab, return_inverse=True)

    lab3 = vlab[F]
    mixed = ~((lab3[:, 0] == lab3[:, 1]) & (lab3[:, 1] == lab3[:, 2]))
    Em = np.sort(np.r_[F[mixed][:, [0, 1]], F[mixed][:, [1, 2]], F[mixed][:, [2, 0]]], axis=1)
    Em = np.unique(Em[vlab[Em[:, 0]] != vlab[Em[:, 1]]], axis=0)
    _, jm = ctree.query(0.5 * (V[Em[:, 0]] + V[Em[:, 1]]))
    cm = cid[jm]
    if lab is not None:  # the curve between exactly these two faces, when there is one
        pair_of = {}
        for i, c in enumerate(curves):
            pair_of.setdefault(tuple(sorted(int(x) for x in c["faces"])), []).append(i)
        la, lb_ = vlab[Em[:, 0]], vlab[Em[:, 1]]
        mid = 0.5 * (V[Em[:, 0]] + V[Em[:, 1]])
        for key, cis in pair_of.items():
            sel = np.where((np.minimum(la, lb_) == key[0]) & (np.maximum(la, lb_) == key[1]))[0]
            if not len(sel):
                continue
            pts_c = np.where(np.isin(cid, cis))[0]
            _, q = cKDTree(allp[pts_c]).query(mid[sel])
            cm[sel] = cid[pts_c[q]]
    tpar = np.full(len(Em), 0.5)
    for ci_ in np.unique(cm):
        sel = np.where(cm == ci_)[0]
        pts_c = np.where(cid == ci_)[0]
        sub = cKDTree(allp[pts_c])
        si = []
        for col in (0, 1):
            vv = Em[sel, col]
            _, q = sub.query(V[vv])
            q = pts_c[q]
            si.append(((V[vv] - allp[q]) * np.cross(tang[q], NQ[q])).sum(1))
        sa, sb = si
        opp = (sa > 0) != (sb > 0)
        tt = np.where(opp, sa / np.where(opp, sa - sb, 1.0), 0.5)
        tpar[sel] = np.clip(tt, 0.02, 0.98)
    ekey = {(int(i), int(j)): k for k, (i, j) in enumerate(Em)}
    extra = list(V[Em[:, 0]] + tpar[:, None] * (V[Em[:, 1]] - V[Em[:, 0]]))
    nvv = nv + len(Em)
    pf, pl = [], []
    for t in np.where(mixed)[0]:
        f = [int(x) for x in F[t]]
        lb = [int(vlab[x]) for x in f]
        pnt = {}
        for k in range(3):
            i, j = f[k], f[(k + 1) % 3]
            if lb[k] != lb[(k + 1) % 3]:
                pnt[k] = nv + ekey[(min(i, j), max(i, j))]
        centre = None
        if len(set(lb)) == 3:
            centre = nvv
            cpt = V[f].mean(0)
            if ctree_c is not None:  # the triple point as near the design corner as the triangle allows
                dc, jc = ctree_c.query(cpt)
                if dc < 3.0:
                    cpt = _closest_on_tri(corners[jc]["p"], V[f[0]], V[f[1]], V[f[2]])
            extra.append(cpt)
            nvv += 1
        for l_ in set(lb):
            poly = []
            for k in range(3):
                if lb[k] == l_:
                    poly.append(f[k])
                if k in pnt and (lb[k] == l_ or lb[(k + 1) % 3] == l_):
                    poly.append(pnt[k])
                    if centre is not None and lb[k] == l_:
                        poly.append(centre)
            for q in range(1, len(poly) - 1):
                pf.append((poly[0], poly[q], poly[q + 1]))
                pl.append(l_)
    Vn = np.vstack([V, np.array(extra).reshape(-1, 3)])
    Fn = np.vstack([F[~mixed], np.array(pf, np.int64).reshape(-1, 3)])
    Ln = np.concatenate([vlab[F[~mixed, 0]], np.array(pl, np.int64)])

    _, Ln = np.unique(Ln, return_inverse=True)
    tot = 0.5 * np.linalg.norm(np.cross(Vn[Fn[:, 1]] - Vn[Fn[:, 0]], Vn[Fn[:, 2]] - Vn[Fn[:, 0]]), axis=1)
    for _ in range(10):
        be, la, lb2 = border_edges(Fn, Ln)
        el = np.linalg.norm(Vn[be[:, 0]] - Vn[be[:, 1]], axis=1)
        K2 = Ln.max() + 1
        ar2 = np.bincount(Ln, weights=tot, minlength=K2)
        per = np.bincount(la, weights=el, minlength=K2) + np.bincount(lb2, weights=el, minlength=K2)
        bad = [k for k in np.argsort(ar2) if ar2[k] < 5e-4 * tot.sum() or 2 * ar2[k] / max(per[k], 1e-9) < 2.0]
        if not bad:
            break
        k = bad[0]
        sh = {}
        for a_, b_, l_ in zip(la, lb2, el):
            if a_ == k:
                sh[b_] = sh.get(b_, 0) + l_
            elif b_ == k:
                sh[a_] = sh.get(a_, 0) + l_
        if not sh:
            break
        Ln[Ln == k] = max(sh, key=sh.get)
        _, Ln = np.unique(Ln, return_inverse=True)
    log(f"curve network: mesh cut into {Ln.max() + 1} faces")
    return Vn, Fn, Ln
