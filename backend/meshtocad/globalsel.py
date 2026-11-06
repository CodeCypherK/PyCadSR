"""Global face selection: choose every face of the part at once.

The greedy passes decide region by region (absorb this sliver, merge that pair), each
with its own threshold, so the same kind of feature ends up lumped in one place and
kept apart in another. Here one energy over the whole mesh decides:

    E = sum over triangles   of how badly its label's surface holds its vertices
      + sum over mesh edges  between two labels of the edge's length
      + a small cost per surface used

Lengths and areas are measured in units of the export tolerance `delta`, so a thin
feature (a 0.1 mm step on a 100 mm part) costs more as a separate face (its long
border) than folded into its neighbour (small area x small deviation), while a wide
one is kept: one size-aware trade-off instead of many local rules. A triangle may
only take a surface that holds its vertices within `loose`; FREE (freeform) takes
anything at a price per area, so freeform is what no simple surface explains.

Solved by alpha-expansion with label costs (graph cuts, scipy's max flow), with the
surfaces refitted to their triangles between rounds.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow

from .segmentation import fit_plane

EXACT = ("plane", "cylinder", "cone", "sphere", "torus")
CAP_INF = 10 ** 8
SCALE = 200.0  # cost units -> integer capacities


class _Label:
    __slots__ = ("kind", "prm", "tris", "cost", "home", "own")

    def __init__(self, kind, prm, home):
        self.kind, self.prm, self.home = kind, prm, home  # home: greedy regions it may reach
        self.tris = np.zeros(0, np.int64)
        self.cost = np.zeros(0)
        self.own = None  # the greedy face it came from: those triangles may always stay


def _data_cost(tm, kind, prm, faces, tol, loose, delta, dev_w, greedy=None):
    """Cost of each triangle on this surface (inf where it may not go). Between tol and
    loose a triangle may only fold in if it roughly faces along the surface, unless
    its whole greedy face folds in (a 0.1 mm step riser, a fillet thinner than the
    tolerance): a ring of wall slivers taken into an end cap folds the cap."""
    P = tm.V[tm.F[faces]].reshape(-1, 3)
    from .segmentation import residual
    d, _ = residual(kind, prm, P, np.zeros_like(P))
    d = d.reshape(-1, 3)
    dmax = d.max(axis=1)
    drms = np.sqrt((d ** 2).mean(axis=1))
    over = np.maximum(drms - tol, 0.0) / delta
    c = dev_w * tm.A[faces] / delta ** 2 * over ** 2 + 0.01 * np.minimum(dmax / tol, 10.0)
    bad = ~(dmax <= loose)
    soft = (dmax > tol) & ~bad
    if soft.any() and greedy is not None:
        if kind == "plane":
            facing = np.abs(tm.FN[faces] @ np.asarray(prm["normal"], float)) >= np.cos(np.radians(45))
        else:
            facing = tm.normal_ok(kind, prm, faces, np.cos(np.radians(35)))
        g = greedy[faces]
        # greedy faces lying wholly within loose of this surface
        gu, inv = np.unique(g, return_inverse=True)
        whole = np.ones(len(gu), bool)
        np.logical_and.at(whole, inv, ~bad)
        cnt = np.bincount(greedy, minlength=int(gu.max()) + 1)[gu]
        here = np.bincount(inv, minlength=len(gu))
        whole &= here == cnt  # every triangle of that face is in reach
        bad |= soft & ~facing & ~whole[inv]
    if kind != "plane":  # vertices on a curved surface, triangle across it (a cap on a rim)
        bad |= ~tm.normal_ok(kind, prm, faces, np.cos(np.radians(35)))
    c[bad] = np.inf
    return c


def _spread_seeds(tm, faces, covered, budget):
    """Seed triangles for a big freeform face, spread over it: each next seed is the triangle
    farthest from the seeds so far among those no surface proposed here holds yet (`covered`
    is read as the caller fills it in). A random draw left whole surfaces unseeded by luck -
    a 35 mm fillet on one draw, two failing faces on another - and the result is the same
    whatever else changed in the model."""
    cen = tm.C[faces]
    dmin = np.full(len(faces), np.inf)
    cur = int(np.argmin(np.linalg.norm(cen - cen.mean(axis=0), axis=1)))
    for _ in range(budget):
        yield int(faces[cur])
        dmin = np.minimum(dmin, np.linalg.norm(cen - cen[cur], axis=1))
        cand = np.where(covered[faces], -1.0, dmin)
        cur = int(np.argmax(cand))
        if cand[cur] <= 0:
            return


def _propose_curved(tm, regions, near, labels, tol, seeds_per_region):
    """Curved surfaces seen from small neighbourhoods inside freeform faces.

    The greedy seeds look at rings of 12-40 triangles; on a rim of long thin triangles
    such a ring reaches round the corner into the next fillet and nothing fits, so
    three straight cylindrical fillets and their corner blend stay one freeform face.
    Here a triangle and its neighbours inside the face (one ring, then two) propose
    surfaces, each kept if it holds a few of the face's triangles exactly; which of
    them are used is left to the global choice."""
    from .meshseg import _hypotheses, _pair_cylinders, _quick_refit, _refine, _simplest, _too_coarse, _valid
    cos_min = np.cos(np.radians(35))
    if not hasattr(tm, "on_curve"):  # set by mesh_segment: triangles touching a curved edge
        rows = np.repeat(np.arange(len(tm.F)), np.diff(tm.indptr))
        tm.on_curve = np.zeros(len(tm.F), bool)
        tm.on_curve[rows[tm.angle > np.radians(0.3)]] = True
    seen = set()
    n = len(tm.F)
    inreg = np.zeros(n, bool)
    for g, (kind, prm, faces) in enumerate(regions):
        faces = np.asarray(faces)
        if kind in EXACT or len(faces) < 3:
            continue
        inreg[:] = False
        inreg[faces] = True
        covered = np.zeros(n, bool)  # held by a surface already proposed here
        seeds = faces if len(faces) <= seeds_per_region else _spread_seeds(tm, faces, covered, seeds_per_region)
        for t in seeds:
            if covered[t]:
                continue
            ring = np.array([t])
            for _ in range(2):
                nb = tm.Gall.indices[np.concatenate([np.arange(tm.Gall.indptr[x], tm.Gall.indptr[x + 1]) for x in ring])]
                ring = np.unique(np.r_[ring, nb[inreg[nb]]])
                if len(ring) < 3:
                    continue
                hyps = _hypotheses(tm, ring) + _pair_cylinders(tm, int(t), ring)
                for kind2, prm2 in hyps:
                    if not _valid(kind2, prm2, tm.diag):
                        continue
                    try:  # closed-form refit first; the slow geometric one only for a likely surface
                        prm2 = _quick_refit(tm, kind2, prm2, ring)
                        if not _valid(kind2, prm2, tm.diag):
                            continue
                        d0 = tm.vert_residual(kind2, prm2, ring)
                        if not np.isfinite(d0).all() or int((d0 <= 5 * tol).sum()) < 3:
                            continue
                        prm2 = _refine(kind2, prm2, tm.verts(ring), tol)
                    except Exception:
                        continue
                    if not _valid(kind2, prm2, tm.diag):
                        continue
                    ok = (tm.vert_residual(kind2, prm2, faces) <= tol) & tm.normal_ok(kind2, prm2, faces, cos_min)
                    held = faces[ok]
                    if len(held) < 3 or t not in held:
                        continue
                    if _too_coarse(tm, kind2, prm2, held):
                        continue
                    kind3, prm3 = _simplest(tm, kind2, prm2, held, tol, cos_min)
                    key = (kind3, hash(held.tobytes()))
                    if key in seen:
                        continue
                    seen.add(key)
                    covered[held] = True
                    labels.append(_Label(kind3, prm3, near[g]))


def global_select(tm, regions, label, tol, loose, log=None, rounds=2, sweeps=3, h=2.0, free_w=0.02, dev_w=4.0, propose=1.0, seeds_per_region=40, planes=1.0, margin=4.0, minimal=1.0,
                  free_c=0.5, crumb=4):
    log = log or (lambda *a: None)
    n = len(tm.F)
    delta = loose
    # a triangle may sit off its surface by at most margin x tol: inside the writer's
    # own tolerance (8 tol), or the face it joins cannot be built within it
    loose = min(loose, margin * tol)
    from .meshseg import _refine
    # ------------------------------------------------ greedy regions and their neighbours
    nreg = len(regions)
    reg_faces = [np.asarray(f, np.int64) for _, _, f in regions]
    G = tm.Gall.tocoo()
    a, b = G.row, G.col
    m = (label[a] != label[b]) & (a < b)
    RA = csr_matrix((np.ones(2 * m.sum()), (np.r_[label[a[m]], label[b[m]]], np.r_[label[b[m]], label[a[m]]])),
                    shape=(nreg, nreg)).tocsr()
    near = [np.r_[g, RA.indices[RA.indptr[g]:RA.indptr[g + 1]]] for g in range(nreg)]
    # ------------------------------------------------ edge costs: length / delta
    E = np.asarray(tm.Gall.nonzero()).T
    E = E[E[:, 0] < E[:, 1]]
    # the shared edge of two adjacent triangles; a doubled triangle shares all three
    # corners with its twin, so take the first two shared corners of each pair
    fa, fb = tm.F[E[:, 0]], tm.F[E[:, 1]]
    shared = (fa[:, :, None] == fb[:, None, :]).any(axis=2)
    two = np.take_along_axis(fa, np.argsort(~shared, axis=1, kind="stable")[:, :2], axis=1)
    elen = np.linalg.norm(tm.V[two[:, 0]] - tm.V[two[:, 1]], axis=1) * (shared.sum(1) >= 2)
    W = csr_matrix((np.r_[elen, elen] / delta, (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    W.sort_indices()
    free_cost = free_w * tm.A / delta ** 2 + free_c
    # ------------------------------------------------ labels: 0 = FREE, then surfaces
    labels = [None]
    L = np.zeros(n, np.int64)
    for g, (kind, prm, faces) in enumerate(regions):
        if kind in EXACT:
            labels.append(_Label(kind, prm, near[g]))
            labels[-1].own = np.asarray(faces, np.int64)
            L[faces] = len(labels) - 1
    n_greedy = len(labels) - 1
    # new plane candidates: coplanar groups of triangles their surface does not hold
    # within tol (a lumped step), and flat groups inside freeform (a big wall among slivers)
    inn = np.zeros(n, bool)
    for g, (kind, prm, faces) in enumerate(regions):
        faces = np.asarray(faces)
        if kind not in EXACT:
            inn[faces] = True
        else:
            inn[faces[tm.vert_residual(kind, prm, faces) > tol]] = True
    flat = tm.angle <= np.radians(0.3)
    rows = np.repeat(np.arange(n), np.diff(tm.indptr))
    k = flat & inn[rows] & inn[tm.indices] & (label[rows] == label[tm.indices])
    Gc = csr_matrix((np.ones(int(k.sum())), (rows[k], tm.indices[k])), shape=(n, n))
    _, comp = connected_components(Gc, directed=False)
    cand = np.where(inn)[0]
    cc = comp[cand]
    order_c = np.argsort(cc, kind="stable")
    bnd = np.r_[0, np.cumsum(np.bincount(cc)[np.unique(cc)])]
    seen = set()
    for i in range(len(bnd) - 1 if planes else 0):
        ff = cand[order_c[bnd[i]:bnd[i + 1]]]
        # a plane only pays when what freeform would cost exceeds its own border
        # (one-facet groups on a spring or a sculpted face never do)
        if (regions[int(label[ff[0]])][0] not in EXACT and
                free_w * tm.A[ff].sum() / delta ** 2 + free_c * len(ff) <= h + 2 * np.sqrt(np.pi * tm.A[ff].sum()) / delta):
            continue
        pl = fit_plane(tm.verts(ff), tm.FN[ff])
        if not (tm.vert_residual("plane", pl, ff) <= tol).all():
            continue
        # a group its own exact surface still holds within the margin is mesh noise, not
        # a step: as its own plane it only chops a noisy wall into pieces (the encoder's
        # walls: 372 pieces at 1-4 tol). A real step sits tens of tol off (receptacle).
        gk, gp, _ = regions[int(label[ff[0]])]
        if gk in EXACT and (tm.vert_residual(gk, gp, ff) <= loose).all():
            continue
        nrm = np.asarray(pl["normal"], float)
        nrm = nrm if nrm[np.argmax(np.abs(nrm))] > 0 else -nrm
        key = (int(label[ff[0]]),) + tuple(np.round(nrm * 2000).astype(int)) + (int(round(float(nrm @ pl["point"]) / tol)),)
        if key in seen:  # the same plane twice in one region
            continue
        seen.add(key)
        labels.append(_Label("plane", pl, near[int(label[ff[0]])]))
    n_planes = len(labels) - 1 - n_greedy
    if propose:
        _propose_curved(tm, regions, near, labels, tol, seeds_per_region)
    log(f"global selection: {n_greedy} greedy surfaces + {n_planes} new plane candidates + "
        f"{len(labels) - 1 - n_greedy - n_planes} curved candidates from inside freeform faces")

    # only what the greedy passes got wrong may move: freeform triangles, and triangles
    # their own surface does not hold within tol (lumped). A face the greedy passes got
    # exactly right stays as it is (rearranging correct faces only gives the writer
    # new, untested arrangements: coplanar faces joined through slivers, swapped rounds)
    movable = np.ones(n, bool)
    if minimal:
        for g, (kind, prm, faces) in enumerate(regions):
            if kind in EXACT:
                faces = np.asarray(faces)
                movable[faces[tm.vert_residual(kind, prm, faces) <= tol]] = False

    def evaluate(li):
        lab = labels[li]
        faces = np.unique(np.concatenate([reg_faces[g] for g in lab.home] + ([lab.own] if lab.own is not None else [])))
        c = _data_cost(tm, lab.kind, lab.prm, faces, tol, loose, delta, dev_w, label)
        if lab.own is not None:
            # where the greedy passes put a triangle it may stay (they fold slivers in on
            # purpose), at half what freeform would cost: never pushed out into freeform
            pos = np.searchsorted(faces, lab.own)
            stuck = ~np.isfinite(c[pos])
            c[pos[stuck]] = 0.5 * free_cost[lab.own[stuck]]
            others = np.ones(len(faces), bool)
            others[pos] = False
            c[others & ~movable[faces]] = np.inf
        else:
            c[~movable[faces]] = np.inf
        ok = np.isfinite(c)
        lab.tris, lab.cost = faces[ok], c[ok]

    for li in range(1, len(labels)):
        evaluate(li)
    # current data cost of every triangle
    curD = free_cost.copy()
    for li in range(1, n_greedy + 1):
        lab = labels[li]
        own = L[lab.tris] == li
        curD[lab.tris[own]] = lab.cost[own]
    # a greedy triangle its own surface no longer holds within loose goes freeform
    held = np.zeros(n, bool)
    for li in range(1, n_greedy + 1):
        lab = labels[li]
        held[lab.tris[L[lab.tris] == li]] = True
    lost = (L > 0) & ~held
    L[lost] = 0
    curD[lost] = free_cost[lost]

    def energy():
        e = curD.sum()
        Wc = W.tocoo()
        e += 0.5 * Wc.data[L[Wc.row] != L[Wc.col]].sum()
        e += h * len(np.unique(L[L > 0]))
        return float(e)

    log(f"global selection: start energy {energy():.1f}, freeform triangles {int((L == 0).sum())}")
    counts = np.bincount(L, minlength=len(labels))

    def expand(al):
        nonlocal counts
        lab = labels[al]
        m = L[lab.tris] != al
        S = lab.tris[m]
        if len(S) == 0:
            return 0
        c1 = lab.cost[m] + 0.01  # a tie keeps the current label (no churn)
        c0 = curD[S].copy()
        ns = len(S)
        idx = np.full(n, -1, np.int64)
        idx[S] = np.arange(ns)
        # edges at S
        st, en = W.indptr[S], W.indptr[S + 1]
        cnt = en - st
        rowi = np.repeat(np.arange(ns), cnt)
        pos = np.repeat(st - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(int(cnt.sum()))
        q = W.indices[pos]
        w = W.data[pos]
        Lp, Lq = L[S[rowi]], L[q]
        qi = idx[q]
        fixed = qi < 0
        # neighbour keeps its label
        np.add.at(c0, rowi[fixed], w[fixed] * (Lp[fixed] != Lq[fixed]))
        np.add.at(c1, rowi[fixed], w[fixed] * (Lq[fixed] != al))
        # both may switch (each edge once)
        pair = (~fixed) & (rowi < qi)
        pi, pj, pw = rowi[pair], qi[pair], w[pair]
        dif = (Lp[pair] != Lq[pair]).astype(float)
        np.add.at(c1, pi, pw * (1 - dif))
        np.add.at(c0, pj, pw)
        cap_pq = pw * (2 - dif)
        # graph: nodes 0..ns-1, source ns, sink ns+1, aux after
        src, snk = ns, ns + 1
        rr, cc_, vv = [], [], []
        mn = np.minimum(c0, c1)
        u0, u1 = c0 - mn, c1 - mn
        u0 = np.where(np.isfinite(u0), u0, CAP_INF / SCALE)
        u1 = np.where(np.isfinite(u1), u1, CAP_INF / SCALE)
        rr.append(np.full(ns, src)); cc_.append(np.arange(ns)); vv.append(u1)   # cut if x=1
        rr.append(np.arange(ns)); cc_.append(np.full(ns, snk)); vv.append(u0)   # cut if x=0
        rr.append(pi); cc_.append(pj); vv.append(cap_pq)
        nxt = ns + 2
        if counts[al] == 0:  # opening a new surface costs h
            y = nxt
            nxt += 1
            rr.append(np.array([src])); cc_.append(np.array([y])); vv.append(np.array([h]))
            rr.append(np.full(ns, y)); cc_.append(np.arange(ns)); vv.append(np.full(ns, CAP_INF / SCALE))
        # a surface whose triangles all switch is saved
        lb_s = L[S]
        ul, uc = np.unique(lb_s[lb_s > 0], return_counts=True)
        for beta, k in zip(ul, uc):
            if k == counts[beta]:
                z = nxt
                nxt += 1
                mem = np.where(lb_s == beta)[0]
                rr.append(np.array([z])); cc_.append(np.array([snk])); vv.append(np.array([h]))
                rr.append(mem); cc_.append(np.full(len(mem), z)); vv.append(np.full(len(mem), CAP_INF / SCALE))
        R = np.concatenate(rr)
        C = np.concatenate(cc_)
        V = np.minimum(np.round(np.concatenate(vv) * SCALE), CAP_INF).astype(np.int64)
        keep = V > 0
        Gf = csr_matrix((V[keep].astype(np.int32), (R[keep], C[keep])), shape=(nxt, nxt))
        Gf.sum_duplicates()
        Gf.data = np.minimum(Gf.data, CAP_INF).astype(np.int32)
        res = maximum_flow(Gf, src, snk)
        Fl = res.flow if hasattr(res, "flow") else res.residual
        Rg = (Gf - Fl).tocsr()  # residual capacities (reverse edges carry the flow)
        Rg.data = np.where(Rg.data > 0, 1, 0)
        Rg.eliminate_zeros()
        reach = breadth_first_order(Rg, src, directed=True, return_predecessors=False)
        side0 = np.zeros(nxt, bool)
        side0[reach] = True
        sw = np.where(~side0[:ns])[0]  # sink side: switch to alpha
        if len(sw) == 0:
            return 0
        old = L[S[sw]]
        counts -= np.bincount(old, minlength=len(counts))
        counts[al] += len(sw)
        L[S[sw]] = al
        curD[S[sw]] = lab.cost[m][sw]
        return len(sw)

    for rd in range(rounds):
        for sp in range(sweeps):
            moved = 0
            order = np.arange(1, len(labels))
            for al in order:
                moved += expand(al)
            log(f"global selection: round {rd + 1} sweep {sp + 1}: {moved} triangles moved, energy {energy():.1f}, "
                f"surfaces {int((counts[1:] > 0).sum())}, freeform triangles {int(counts[0])}")
            if moved == 0:
                break
        if rd == rounds - 1:
            break
        # refit each used surface on the triangles it holds within tol, re-evaluate
        for li in range(1, len(labels)):
            if counts[li] == 0:
                continue
            lab = labels[li]
            mine = np.where(L == li)[0]
            d = tm.vert_residual(lab.kind, lab.prm, mine)
            inl = mine[d <= tol]
            if len(inl) >= 2:
                if lab.kind == "plane":
                    pl = fit_plane(tm.verts(inl), tm.FN[inl])
                else:
                    pl = _refine(lab.kind, lab.prm, tm.verts(inl), tol)
                if pl is not None:
                    new = tm.vert_residual(lab.kind, pl, inl)
                    if new.max() <= max(tol, d[d <= tol].max()):
                        lab.prm = pl
            # it may now reach the greedy regions its triangles came from and their neighbours
            homes = np.unique(label[mine])
            lab.home = np.unique(np.concatenate([near[g] for g in homes]))
            evaluate(li)
        curD = free_cost.copy()
        for li in range(1, len(labels)):
            if counts[li] == 0:
                continue
            lab = labels[li]
            pos = np.full(n, -1)
            pos[lab.tris] = np.arange(len(lab.tris))
            mine = np.where(L == li)[0]
            ok = pos[mine] >= 0
            curD[mine[ok]] = lab.cost[pos[mine[ok]]]
            L[mine[~ok]] = 0
            curD[mine[~ok]] = free_cost[mine[~ok]]
        counts = np.bincount(L, minlength=len(labels))
    # ------------------------------------------------ stray pieces go back
    # The energy pays once per surface, but the export pays per face: one surface can
    # collect isolated slivers elsewhere at almost no cost (a few microns of border),
    # and each would be its own face. A piece of a surface not joined to its own greedy
    # face (or, for a new surface, to its main body) goes back where the greedy passes
    # had it. Repeated, since a returned piece can leave another one stranded.
    # ------------------------------------------------ no freeform crumbs
    # Taking most of a small freeform face into a neighbouring surface can leave a crumb
    # of two or three triangles that nothing holds: a sliver face against that surface
    # which the writer and the STEP readers can't stitch (KFL08 lost a whole body this
    # way). Such a freeform face goes back whole, as the greedy passes had it.
    if crumb:
        Gt = tm.Gall.tocoo()
        k = (L[Gt.row] == 0) & (L[Gt.col] == 0) & (label[Gt.row] == label[Gt.col])
        _, cc = connected_components(csr_matrix((np.ones(int(k.sum())), (Gt.row[k], Gt.col[k])), shape=(n, n)), directed=False)
        size = np.bincount(cc)
        for g, (kind, prm, faces) in enumerate(regions):
            if kind in EXACT:
                continue
            faces = np.asarray(faces, np.int64)
            left = faces[L[faces] == 0]
            if 0 < len(left) < len(faces) and size[cc[left]].min() < crumb:
                L[faces] = 0
    greedy_L = np.zeros(n, np.int64)
    for li in range(1, n_greedy + 1):
        greedy_L[labels[li].own] = li
    Ga0 = tm.Gall.tocoo()
    for _ in range(4):
        same0 = L[Ga0.row] == L[Ga0.col]
        Gp = csr_matrix((np.ones(int(same0.sum())), (Ga0.row[same0], Ga0.col[same0])), shape=(n, n))
        _, comp0 = connected_components(Gp, directed=False)
        back = np.zeros(n, bool)
        for li in np.unique(L[L > 0]):
            mine = np.where(L == li)[0]
            cc = comp0[mine]
            uc, cnt = np.unique(cc, return_counts=True)
            if len(uc) == 1:
                continue
            lab = labels[li]
            if lab.own is not None:
                keep = np.unique(comp0[lab.own][L[lab.own] == li])
                if len(keep) == 0:
                    keep = uc[[np.argmax(cnt)]]
            else:
                keep = uc[cnt >= max(3, 0.25 * cnt.max())]
            back[mine[~np.isin(cc, keep)]] = True
        if not back.any():
            break
        L[back] = greedy_L[back]
    log(f"global selection: stray pieces returned")
    # ------------------------------------------------ faces: connected pieces of each label
    out, new_label = [], np.full(n, -1, np.int64)
    Ga = tm.Gall.tocoo()
    same = L[Ga.row] == L[Ga.col]
    # freeform keeps the greedy face it was in (those passes group small freeform pieces
    # on purpose, across sharp edges too); only what moved to a surface leaves it
    ex = same & ((L[Ga.row] > 0) | (label[Ga.row] == label[Ga.col]))
    Gc = csr_matrix((np.ones(int(ex.sum())), (Ga.row[ex], Ga.col[ex])), shape=(n, n))
    _, comp = connected_components(Gc, directed=False)
    # Pieces that were one greedy region stay one region, in the greedy order, with the
    # greedy equation when nothing about them changed: the writer depends on region order
    # and grouping, so an untouched part must reach it exactly as without this step (the
    # camera went from 0 to 5 failing faces on a pure renumbering).
    ncomp = int(comp.max()) + 1
    first = np.full(ncomp, n, np.int64)
    np.minimum.at(first, comp, np.arange(n))
    lab_c = L[first]
    key = {}
    root = np.arange(ncomp)
    origin = np.zeros(ncomp)
    for c in range(ncomp):
        li = int(lab_c[c])
        if li > 0 and labels[li].own is not None:
            k = ("own", li)
            origin[c] = label[labels[li].own[0]]
        elif li == 0:
            k = ("free", int(label[first[c]]))
            origin[c] = label[first[c]]
        else:
            k = ("new", c)
            origin[c] = label[first[c]] + 0.5
        root[c] = key.setdefault(k, c)
    groups = {}
    for t, c in enumerate(comp):
        groups.setdefault(int(root[c]), []).append(t)
    for c in sorted(groups, key=lambda c: (origin[c], first[c])):
        faces = np.asarray(groups[c], np.int64)
        li = int(lab_c[c])
        new_label[faces] = len(out)
        g = int(label[faces[0]])
        greedy_faces = reg_faces[g]
        untouched = len(faces) == len(greedy_faces) and (label[faces] == g).all()
        if untouched and ((li == 0 and regions[g][0] not in EXACT) or
                          (li > 0 and labels[li].own is not None and label[labels[li].own[0]] == g)):
            out.append((regions[g][0], regions[g][1], faces))
        elif li == 0:
            # a flat freeform piece is a plane, unless the greedy passes already had it as
            # freeform: they group small flat crumbs into freeform faces on purpose, and
            # as one-triangle planes they become slivers the STEP readers can't stitch
            was_free = all(regions[int(h)][0] not in EXACT for h in np.unique(label[faces]))
            pl = fit_plane(tm.verts(faces), tm.FN[faces])
            if not was_free and (tm.vert_residual("plane", pl, faces) <= tol).all():
                out.append(("plane", pl, faces))
            else:
                out.append(("freeform", {}, faces))
        else:
            out.append((labels[li].kind, labels[li].prm, faces))
    log(f"global selection: {nreg} faces -> {len(out)} faces, freeform triangles {int((L == 0).sum())}")
    return out, new_label
