"""Face layout from the shape map: faces = the calm islands; borders = the crests between them,
smoothed into clean curves that meet exactly at the corners.

1. islands: connected flat or wide-tube areas (shape classes, no narrow curved strip) bigger
   than MIN_FRAC of the surface; a neck's base circle and its finish rings are hard cuts
2. the bands go to the islands by flooding in order of rising curvature from every island at
   once: neighbouring fronts meet on the crest; two faces whose border has no crest (radius of the
   curvature along it above MERGE_R) are one face
3. a symmetric part (tri_mirror) gets a symmetric layout: the kept half decides
4. each border stretch between corners becomes a smoothing spline with its ends pinned
   (SMOOTH mm of wander); where a stretch runs around a neck axis at one height and radius it is
   that ring, an exact circle arc, blended into the spline where it leaves the ring; corners on a ring
   move onto it; stretches shorter than 2 mm are straight links (curves always meet at the corners)
All lengths in mm.
"""
import heapq

import numpy as np
import trimesh
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


DEFAULTS = dict(R_FLAT=60.0, R_ROUND=24.0, MIN_FRAC=0.002, SMOOTH=2.5, CALM_R=40.0, NARROW_R=50.0, TOPHAT_MM=14.0,
                MERGE_R=30.0, CORNER_MM=8.0, STUB_MM=12.0, NECK_BAND=3.25, TUNNEL_MM=8.0, TUNNELS=False, USE_MODEL=True, DUMP=None, DUMP_ONLY=False, DUMP_ALL=False, CYL_R=12.0, COLUMN_MM=4.0, POCKET_R=400.0)
SMM = 30.0
RING_TOL = 2.0  # mm: border points this close to a ring (height and radius) are on it
RING_MIN = 10.0  # mm: shortest stretch of a border that becomes a ring arc
STROKE_MIN = 5.0  # mm: a border shorter than this is never part of a stroke (no reliable direction)
THROUGH_COS = 0.8  # two borders whose directions at a corner are this close to opposite continue each other
BLEND = 15.0  # mm: a ring arc blends into its border's spline (and a moved corner fades) over this length


BORDER_FEATURES = ["log_r_med", "log_r_p25", "log_r_p75", "narrow_frac", "prominence", "background", "log_len",
                   "log_amin", "log_amax", "sign_min", "sign_max", "turn_deg", "h_med", "contrast"]


def _inside(Q, poly):
    """Points Q (n, 2) inside the closed polygon poly (m, 2) (ray casting)."""
    x, y = Q[:, :1], Q[:, 1:2]
    x1, y1 = poly[None, :, 0], poly[None, :, 1]
    x2, y2 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 == y1, 1e-30, y2 - y1) + x1)
    return c.sum(1) % 2 == 1


def _frame2(pn):
    pn = np.asarray(pn, float) / np.linalg.norm(pn)
    e1 = np.cross(pn, [1.0, 0, 0] if abs(pn[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(pn, e1)


def plane_holes(V, F, pn=(0, 1.0, 0), pp=(0, 0, 0), off=0.013):
    """Holes through the part at a plane (normal pn through pp): the inner loops of the part's section
    there that a line along pn passes through without meeting the part (a handle's opening, not a
    hollow's inner wall), as polygons in the plane's (e1, e2) frame."""
    pn, pp = np.asarray(pn, float) / np.linalg.norm(pn), np.asarray(pp, float)
    e1, e2 = _frame2(pn)
    m = trimesh.Trimesh(V, F, process=False)
    seg = trimesh.intersections.mesh_plane(m, pn, pp + off * pn)
    if not len(seg):
        return []
    P = (seg.reshape(-1, 3) - pp) @ np.c_[e1, e2]
    _, inv = np.unique(np.round(P / 1e-4).astype(np.int64), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1, 2)
    nn = int(inv.max()) + 1
    pts = np.zeros((nn, 2))
    pts[inv.ravel()] = P
    nb = [[] for _ in range(nn)]
    for a, b in inv:
        if a != b:
            nb[a].append(int(b))
            nb[b].append(int(a))
    seen = np.zeros(nn, bool)
    loops = []
    for s0 in range(nn):
        if seen[s0] or len(nb[s0]) != 2:
            continue
        lp, prev, cur = [s0], s0, nb[s0][0]
        seen[s0] = True
        while cur != s0 and not seen[cur] and len(nb[cur]) == 2:
            lp.append(cur)
            seen[cur] = True
            prev, cur = cur, (nb[cur][0] if nb[cur][0] != prev else nb[cur][1])
        if cur == s0 and len(lp) >= 12:
            loops.append(pts[lp])
    # a hole: a closed loop with the rest of the section around it on all sides (the outline itself may
    # be open where the part is, at its mouths)
    holes = []
    for L in loops:
        area = 0.5 * abs(np.sum(L[:, 0] * np.roll(L[:, 1], -1) - np.roll(L[:, 0], -1) * L[:, 1]))
        if area < 50.0:
            continue
        lo_, hi_ = L.min(0), L.max(0)
        other = ~(np.all((P >= lo_ - 1e-6) & (P <= hi_ + 1e-6), axis=1))
        if not other.any():
            continue
        Po = P[other]
        if not (np.all(Po.min(0) < lo_ - 1.0) and np.all(Po.max(0) > hi_ + 1.0)):
            continue
        # through-hole: lines along the plane's normal from points inside the loop meet nothing
        gx, gy = np.meshgrid(np.linspace(lo_[0], hi_[0], 9)[1:-1], np.linspace(lo_[1], hi_[1], 9)[1:-1])
        Q2 = np.c_[gx.ravel(), gy.ravel()]
        Q2 = Q2[_inside(Q2, L)][:12]
        if not len(Q2):
            continue
        O = pp + Q2[:, :1] * e1 + Q2[:, 1:] * e2
        try:
            free = ~m.ray.intersects_any(np.vstack([O, O]), np.vstack([np.tile(pn, (len(O), 1)), np.tile(-pn, (len(O), 1))]))
        except Exception:
            free = np.ones(2 * len(O), bool)
        if free.mean() >= 0.75:
            holes.append(L)
    return holes


def neck_borders(nk, min_band):
    """A neck's border rings, top down: its crest rings, skipping one closer than min_band (mm) below the
    last kept one (a thinner band is a detail of its ring); its base (where the round slices end) only
    when there is a crest at it, and then nothing within min_band above it."""
    circ = sorted(nk["circles"], key=lambda c: -c["h"])
    base = next(c for c in circ if c["base"])
    keep = []
    for c in circ:
        if c is base:
            continue
        if keep and keep[-1]["h"] - c["h"] < min_band:
            continue
        keep.append(c)
    if any(base["h"] < c["h"] <= base["h"] + 1.5 for c in circ if c is not base) or not keep:
        keep = [c for c in keep if c["h"] - base["h"] >= min_band] + [base]
    return keep


def island_curves(V, F, feat, necks=(), tri_mirror=None, half=None, log=print, **prm):
    P_ = dict(DEFAULTS, **prm)
    mir_ = tri_mirror is not None  # a mirrored part: corners on the plane y = 0 stay on it
    R_FLAT, R_ROUND, MIN_FRAC, SMOOTH = P_["R_FLAT"], P_["R_ROUND"], P_["MIN_FRAC"], P_["SMOOTH"]
    CALM_R, NARROW_R, TOPHAT_MM, MERGE_R = P_["CALM_R"], P_["NARROW_R"], P_["TOPHAT_MM"], P_["MERGE_R"]
    CYL_R, COLUMN_MM, POCKET_R, CORNER_MM = P_["CYL_R"], P_["COLUMN_MM"], P_["POCKET_R"], P_["CORNER_MM"]
    NECK_BAND, STUB_MM, TUNNEL_MM = P_["NECK_BAND"], P_["STUB_MM"], P_["TUNNEL_MM"]
    TUNNELS, USE_MODEL, DUMP, DUMP_ONLY, DUMP_ALL = P_["TUNNELS"], P_["USE_MODEL"], P_["DUMP"], P_["DUMP_ONLY"], P_["DUMP_ALL"]
    mesh = trimesh.Trimesh(V, F, process=False)
    n = len(F)
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    A = 0.5 * np.linalg.norm(cr, axis=1)
    ft = feat[F].mean(1)
    k1, k2 = np.sinh(ft[:, 0]) / SMM, np.sinh(ft[:, 1]) / SMM
    if np.sum(A * (k1 + k2)) < 0:  # a bottle or jug is convex overall: normals pointing in flip the signs
        k1, k2 = -k2, -k1
    si = (2 / np.pi) * np.arctan2(k1 + k2, k1 - k2)
    kabs = np.maximum(np.abs(k1), np.abs(k2))
    R = 1.0 / np.maximum(kabs, 1e-9)
    cls = np.full(n, 4)
    cls[(si > 0.25) & (R < R_FLAT)] = 1
    cls[(si > 0.25) & (R < R_ROUND)] = 2
    cls[(si < -0.25) & (R < R_FLAT)] = 3
    cls[R >= R_FLAT] = 0
    adj = np.asarray(mesh.face_adjacency, np.int64)
    G = csr_matrix((np.ones(2 * len(adj)), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), shape=(n, n))
    G.sort_indices()
    ip, ix = G.indptr, G.indices
    for _ in range(6):
        cnt = np.zeros((n, 5))
        np.add.at(cnt, (np.repeat(np.arange(n), np.diff(ip)), cls[ix]), 1.0)
        cnt[np.arange(n), cls] += 1.5
        cls = cnt.argmax(1)

    # ---- 1. islands
    Cc = V[F].mean(1)
    step = float(np.median(np.linalg.norm(Cc[adj[:, 0]] - Cc[adj[:, 1]], axis=1)))
    its = max(1, int(round(TOPHAT_MM / step)))
    o_ = kabs.copy()
    for _ in range(its):
        o_ = np.minimum(np.minimum.reduceat(o_[ix], ip[:-1]), o_)
    for _ in range(its):
        o_ = np.maximum(np.maximum.reduceat(o_[ix], ip[:-1]), o_)
    narrow = (kabs - o_) > 1.0 / NARROW_R
    kmin = np.minimum(np.abs(k1), np.abs(k2))
    cyl = (cls == 2) & (kmin < 0.3 * kabs)
    # calm: gently curved (below 1/CALM_R: flat, domed, tube-like or a shallow pocket); never a narrow
    # crest band (a crest separates faces); below a neck, its column (tube-like down to CYL_R)
    calm = ~narrow & (kabs < 1.0 / CALM_R) & ((cls == 0) | (cls == 1) | (cls == 3) | cyl)
    tube = ~narrow & cyl & (kabs < 1.0 / CYL_R)
    neck_zone = np.full(n, -1)
    neck_base = []  # per neck with a base border: (its base region, the neck's index)
    column = np.full(n, -1)
    band = np.full(n, -1)
    for ni, nk in enumerate(necks):
        o, a = nk["origin"], nk["axis"]
        rmax = max(c["r"] for c in nk["circles"])
        base = next(c for c in nk["circles"] if c["base"])
        keep = neck_borders(nk, NECK_BAND)
        low = keep[-1]
        hv = (V - o) @ a
        hlo, hhi = hv[F].min(1), hv[F].max(1)
        hC = (Cc - o) @ a
        # radius about the rings' own centres (a finish's platform can sit off the mouth's axis), taken
        # at each height between its rings
        hs_c = np.array([c["h"] for c in nk["circles"]])
        oc_ = np.array([(c["centre"] - o) - ((c["centre"] - o) @ a) * a for c in nk["circles"]])
        srt = np.argsort(hs_c)
        off = np.c_[[np.interp(hC, hs_c[srt], oc_[srt, k]) for k in range(3)]].T
        rad = np.linalg.norm((Cc - o) - hC[:, None] * a - off, axis=1)
        zone = (hC > low["h"]) & (rad < rmax + 4)
        strips = np.zeros(n, bool)
        for c in keep:  # the neck's borders: its crest rings (and its base, where that is a crest too)
            near = (rad < rmax + 12) & narrow if c is base else (np.abs(rad - c["r"]) < 3.0)
            strips |= (hlo <= c["h"] + 0.6) & (hhi >= c["h"] - 0.6) & near
        if low is base:
            # the last band goes down to the crest under it, which can sink below the base circle where
            # the part is not round (a platform's wall running into a handle): it takes what lies just
            # under the base and is reached without crossing a crest
            ext = (hC > low["h"] - 4.0) & (hC <= low["h"]) & (rad < rmax + 6)
            S_ = (zone | ext) & ~narrow & ~strips
            sm = S_[adj[:, 0]] & S_[adj[:, 1]]
            _, cc_ = connected_components(csr_matrix((np.ones(int(sm.sum())), (adj[sm, 0], adj[sm, 1])), shape=(n, n)),
                                          directed=False)
            reach = np.isin(cc_, np.unique(cc_[zone & S_]))
            zone |= ext & reach & S_
        calm &= ~strips
        calm |= zone & ~strips  # a neck's bands are its faces (its rings are the borders)
        neck_zone[zone] = ni
        hk = np.array([c["h"] for c in keep if c is not base])
        band[zone] = 100 * ni + np.sum(hC[zone][:, None] < hk[None, :], axis=1)
        col = (hC < low["h"] - 1.0) & (rad < base["r"] + COLUMN_MM)  # the neck's column, under its last ring
        calm |= ~narrow & col & ~zone
        column[col & ~zone] = ni
        if low is base:
            neck_base.append(((hC > low["h"] - 4.0) & (hC < low["h"] + 1.0) & (rad < rmax + 6), ni))
    # a hole through the part at its mirror plane (a handle's opening): its tunnel is a face of its own,
    # bounded by its opening (where the surface turns from facing along the hole to facing across it),
    # even where the front blends smoothly into it
    tunnel = np.full(n, -1)
    nrm_ = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-30)
    if tri_mirror is not None:
        planes_ = [(np.array([0, 1.0, 0]), np.zeros(3))]
    else:  # no mirror: the principal planes through the area centroid
        c0_ = (Cc * A[:, None]).sum(0) / A.sum()
        _, U_ = np.linalg.eigh(((Cc - c0_) * A[:, None]).T @ (Cc - c0_))
        planes_ = [(U_[:, k], c0_) for k in range(3)]
    n_tun = 0
    walls_ = []  # each hole's wall where it crosses its plane
    for pn_, pp_ in planes_:
        e1_, e2_ = _frame2(pn_)
        for poly in plane_holes(V, F, pn_, pp_):
            d_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1))]
            dd = np.linspace(0, d_[-1], max(50, int(d_[-1] / 0.5)))
            pd = np.c_[np.interp(dd, d_, np.r_[poly[:, 0], poly[0, 0]]), np.interp(dd, d_, np.r_[poly[:, 1], poly[0, 1]])]
            xz = (Cc - pp_) @ np.c_[e1_, e2_]
            lo_, hi2 = poly.min(0) - TUNNEL_MM, poly.max(0) + TUNNEL_MM
            box = np.all((xz > lo_) & (xz < hi2), axis=1)
            cand = np.zeros(n, bool)
            ib = np.where(box)[0]
            near_ = cKDTree(pd).query(xz[ib])[0] < TUNNEL_MM
            cand[ib] = (near_ | _inside(xz[ib], poly)) & (np.abs(nrm_[ib] @ pn_) < 0.5) & (tunnel[ib] < 0)
            sm = cand[adj[:, 0]] & cand[adj[:, 1]]
            _, cc_ = connected_components(csr_matrix((np.ones(int(sm.sum())), (adj[sm, 0], adj[sm, 1])), shape=(n, n)),
                                          directed=False)
            mid_ = cand & (np.abs((Cc - pp_) @ pn_) < 1.5)
            t_ = cand & np.isin(cc_, np.unique(cc_[mid_]))
            if A[t_].sum() <= 100.0:
                continue
            walls_.append(t_ & (np.abs((Cc - pp_) @ pn_) < 3.0))
            if TUNNELS:
                tunnel[t_] = n_tun
                n_tun += 1
                calm |= t_
                log(f"curve network: a hole through the part: its tunnel is a face ({A[t_].sum():.0f} mm2)")
    # islands: calm, connected, of one kind: a shallow pocket (concave) is not the convex wall around it,
    # a neck's column is not the body under it, each band of a neck is its own face
    Hm = 0.5 * (k1 + k2)
    kind = np.where(Hm < -1.0 / POCKET_R, 1, 0) + 2 * (column + 1) + 1000 * (band + 1) + 10 ** 6 * (tunnel + 1)
    kind[tunnel >= 0] = 10 ** 6 * (tunnel[tunnel >= 0] + 1)
    same = calm[adj[:, 0]] & calm[adj[:, 1]] & (kind[adj[:, 0]] == kind[adj[:, 1]])
    _, comp = connected_components(csr_matrix((np.ones(int(same.sum())), (adj[same, 0], adj[same, 1])), shape=(n, n)),
                                   directed=False)
    total = A.sum()
    ca = np.bincount(comp[calm], weights=A[calm], minlength=comp.max() + 1)
    rep_t = np.full(len(ca), -1)
    rep_t[comp[calm]] = np.where(calm)[0]
    neck_comp = (rep_t >= 0) & (band[np.maximum(rep_t, 0)] >= 0)
    big = np.where((ca >= MIN_FRAC * total) | (neck_comp & (ca >= 30.0)))[0]
    lab = np.full(n, -1)
    for i, c_ in enumerate(big):
        lab[(comp == c_) & calm] = i
    log(f"curve network: {len(big)} calm islands")

    # ---- 2. flood the bands in order of rising curvature (a neck base is a hard border)
    ks = kabs.copy()
    isl_neck = {}
    for l_ in np.unique(lab[lab >= 0]):
        vals, cnts = np.unique(neck_zone[lab == l_], return_counts=True)
        isl_neck[int(l_)] = int(vals[np.argmax(cnts)])

    def flood(lab, constrained):
        heap = []
        for t in np.where(lab >= 0)[0]:
            for q in ix[ip[t]:ip[t + 1]]:
                if lab[q] < 0 and (not constrained or neck_zone[q] == isl_neck.get(int(lab[t]), -1)):
                    heap.append((ks[q], int(q), int(lab[t])))
        heapq.heapify(heap)
        while heap:
            _, t, l_ = heapq.heappop(heap)
            if lab[t] >= 0:
                continue
            lab[t] = l_
            for q in ix[ip[t]:ip[t + 1]]:
                if lab[q] < 0 and (not constrained or neck_zone[q] == isl_neck.get(l_, -1)):
                    heapq.heappush(heap, (ks[q], int(q), l_))
        return lab
    lab = flood(lab, True)
    lab = flood(lab, False)
    lab[lab < 0] = 0

    def absorb_crumbs(lab):
        """Pieces under 0.2 % of the surface join the neighbouring face they touch most."""
        same = lab[adj[:, 0]] == lab[adj[:, 1]]
        _, pc = connected_components(csr_matrix((np.ones(int(same.sum())), (adj[same, 0], adj[same, 1])),
                                                shape=(n, n)), directed=False)
        pa = np.bincount(pc, weights=A)
        for p in np.where(pa < 0.002 * total)[0]:
            f = np.where(pc == p)[0]
            nb = lab[ix[np.concatenate([np.arange(ip[t], ip[t + 1]) for t in f])]]
            nb = nb[nb != lab[f[0]]]
            if len(nb):
                lab[f] = np.bincount(nb).argmax()
        return lab
    lab = absorb_crumbs(lab)
    def face_sign(l_):
        """+1 convex, -1 concave (area-weighted mean curvature of the face's calmer half), 0 unclear."""
        m_ = lab == l_
        hm = (k1[m_] + k2[m_]) * A[m_]
        h_ = hm.sum() / max(A[m_].sum(), 1e-12)
        return 0 if abs(h_) < 1.0 / 400 else (1 if h_ > 0 else -1)

    def tun_of(l_):
        v_, c_ = np.unique(tunnel[lab == l_], return_counts=True)
        return int(v_[np.argmax(c_)])

    # ---- faces meet on crests: two faces whose shared border is no design line are one face; the
    # weakest such border goes first. The decision is a small model learned from true CAD faces and the
    # user's layouts (curvenet/border_model.json, see BORDER_FEATURES); without it: no crest (median
    # radius of the curvature along the border above MERGE_R) and not a pocket's rim
    from .learned import probability as _prob
    learned = USE_MODEL and _prob("border_model", [0.0] * len(BORDER_FEATURES)) is not None
    aex = np.asarray(mesh.face_adjacency_edges, np.int64)
    elen = np.linalg.norm(V[aex[:, 0]] - V[aex[:, 1]], axis=1)
    total_a = float(A.sum())

    def band(tt, l_, rings=3):
        """Triangles of face l_ within a few rings of the triangles tt."""
        cur = np.unique(tt[lab[tt] == l_])
        seen_ = set(cur.tolist())
        for _ in range(rings):
            if not len(cur):
                break
            nb = np.unique(ix[np.concatenate([np.arange(ip[t], ip[t + 1]) for t in cur])])
            nb = nb[lab[nb] == l_]
            cur = np.array([t for t in nb if t not in seen_], np.int64)
            seen_.update(cur.tolist())
        return np.array(sorted(seen_), np.int64)

    calm_med = {}

    def border_feats(a_, b_, e):
        t0, t1 = adj[e, 0], adj[e, 1]
        kk_ = np.maximum(kabs[t0], kabs[t1])

        def lr(k):
            return float(np.log10(1.0 / max(float(k), 1e-4)))
        pa, pb = A[lab == a_].sum() / total_a, A[lab == b_].sum() / total_a
        ba, bb = band(np.r_[t0, t1], a_), band(np.r_[t0, t1], b_)
        na = (nrm_[ba] * A[ba, None]).sum(0)
        nb2 = (nrm_[bb] * A[bb, None]).sum(0)
        turn = np.degrees(np.arccos(np.clip(na @ nb2 / max(np.linalg.norm(na) * np.linalg.norm(nb2), 1e-12), -1, 1)))
        for l_ in (a_, b_):
            if l_ not in calm_med:
                ml = (lab == l_) & ~narrow
                calm_med[l_] = float(np.median(kabs[ml])) if ml.any() else float(np.median(kabs[lab == l_]))
        sa, sb = face_sign(a_), face_sign(b_)
        return [lr(np.median(kk_)), lr(np.percentile(kk_, 75)), lr(np.percentile(kk_, 25)),
                float(np.mean(narrow[t0] | narrow[t1])),
                100.0 * float(np.median(np.maximum(kabs[t0] - o_[t0], kabs[t1] - o_[t1]))),
                100.0 * float(np.median(np.minimum(o_[t0], o_[t1]))),
                float(np.log10(max(elen[e].sum(), 1e-3))),
                float(np.log10(max(min(pa, pb), 1e-9))), float(np.log10(max(max(pa, pb), 1e-9))),
                float(min(sa, sb)), float(max(sa, sb)), float(turn),
                100.0 * float(np.median(np.r_[Hm[t0], Hm[t1]])),
                float(np.log10(max(np.median(kk_), 1e-6) / max(min(calm_med[a_], calm_med[b_]), 1e-6)))]

    if DUMP is not None:
        DUMP.append({"lab0": lab.copy(), "calm": calm.copy()})
    first = True
    rnd = 0
    while True:
        bm = lab[adj[:, 0]] != lab[adj[:, 1]]
        if not bm.any():
            break
        pr_ = np.sort(np.c_[lab[adj[bm, 0]], lab[adj[bm, 1]]], axis=1)
        kk = np.maximum(kabs[adj[bm, 0]], kabs[adj[bm, 1]])
        keys, inv = np.unique(pr_, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        best = None
        bm_idx = np.where(bm)[0]
        for q, (a_, b_) in enumerate(keys):
            m_ = inv == q
            if m_.sum() < 10:
                continue
            if tun_of(a_) != tun_of(b_):
                continue  # a tunnel's opening is a hard border
            na_, nb_ = isl_neck.get(int(a_), -1), isl_neck.get(int(b_), -1)
            if na_ >= 0 or nb_ >= 0:
                # a neck's borders are its rings: hard cuts. A border running across the heights is no
                # ring (a band that leaves the neck's zone where the part is not round goes on as
                # another face): that one may go like any other
                if na_ >= 0 and nb_ >= 0 and na_ != nb_:
                    continue
                nk_ = necks[max(na_, nb_)]
                cb = 0.5 * (Cc[adj[bm, 0][m_]] + Cc[adj[bm, 1][m_]])
                hb = (cb - nk_["origin"]) @ nk_["axis"]
                if np.percentile(hb, 90) - np.percentile(hb, 10) < 3.0:
                    continue
            r_ = 1.0 / max(float(np.median(kk[m_])), 1e-9)
            dump_ = DUMP is not None and (first or DUMP_ALL)
            if learned or dump_:
                x = border_feats(int(a_), int(b_), bm_idx[m_])
                if dump_:
                    DUMP.append({"a": int(a_), "b": int(b_), "x": x, "n": int(m_.sum()), "round": rnd})
            if learned:
                pk = _prob("border_model", x)
                if pk < 0.5 and (best is None or (-pk, r_) > best[0]):
                    best = ((-pk, r_), int(a_), int(b_))  # equal scores: the weakest crest first
                continue
            if face_sign(a_) * face_sign(b_) < 0:
                continue  # a pocket and its surroundings: the border is the inflection, a design line
            if r_ > MERGE_R and (best is None or r_ > best[0]):
                best = (r_, int(a_), int(b_))
        first = False
        if best is None:
            break
        r_, a_, b_ = best
        keep_, gone = (a_, b_) if A[lab == a_].sum() >= A[lab == b_].sum() else (b_, a_)
        lab[lab == gone] = keep_
        isl_neck[keep_] = max(isl_neck.get(keep_, -1), isl_neck.get(gone, -1))
        if DUMP is not None and DUMP_ALL:
            DUMP.append({"merge": (int(keep_), int(gone)), "round": rnd})
        rnd += 1
        if learned:
            calm_med.pop(gone, None)
            calm_med.pop(keep_, None)
            log(f"curve network: two faces merged (learned: no design line, p {-r_[0]:.2f})")
        else:
            log(f"curve network: two faces merged (no crest between them: radius {r_:.0f} mm)")
    # no stub borders: two faces touching along less than STUB_MM (a wedge tip at a ring, two borders
    # crossing) do not touch; near that stretch the smaller face's triangles go to its other neighbour
    ae_ = np.asarray(mesh.face_adjacency_edges, np.int64)
    for _ in range(4):
        bm = lab[adj[:, 0]] != lab[adj[:, 1]]
        if not bm.any():
            break
        pr_ = np.sort(np.c_[lab[adj[bm, 0]], lab[adj[bm, 1]]], axis=1)
        keys, inv = np.unique(pr_, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        mid_ = 0.5 * (V[ae_[bm, 0]] + V[ae_[bm, 1]])
        ext = np.array([np.linalg.norm(np.ptp(mid_[inv == q], axis=0)) for q in range(len(keys))])
        changed = False
        area_l = np.bincount(lab, weights=A)
        for q in np.where(ext < STUB_MM)[0]:
            a_, b_ = (int(x) for x in keys[q])
            small, other = (a_, b_) if area_l[a_] <= area_l[b_] else (b_, a_)
            ed = np.where(bm)[0][inv == q]
            seed = np.unique(adj[ed].ravel())
            seed = seed[lab[seed] == small]
            if not len(seed):
                continue
            near = np.where((lab == small) & (np.linalg.norm(Cc - Cc[seed].mean(0), axis=1) < STUB_MM))[0]
            for t in near:
                nb = lab[ix[ip[t]:ip[t + 1]]]
                nb = nb[(nb != small) & (nb != other)]
                if len(nb):
                    lab[t] = np.bincount(nb).argmax()
                    changed = True
        if not changed:
            break
    lab = absorb_crumbs(lab)
    tm = tri_mirror
    if tm is not None and len(tm) == n:
        h = int(half)
        lab[h:] = lab[np.asarray(tm[h:], np.int64)]
        same = lab[adj[:, 0]] == lab[adj[:, 1]]
        _, lab = connected_components(csr_matrix((np.ones(int(same.sum())), (adj[same, 0], adj[same, 1])),
                                                 shape=(n, n)), directed=False)
    # a face lying in a neck's base region (a strip of the wall under the base circle, between two
    # crests a mm or two apart) is part of the neck's last band, not a face of its own
    for reg, ni in neck_base:
        def neck_of(l_):
            v_, c_ = np.unique(neck_zone[lab == l_], return_counts=True)
            return int(v_[np.argmax(c_)])
        for l_ in np.unique(lab[reg]):
            m_ = lab == l_
            if neck_of(l_) not in (-1, ni) or A[m_ & reg].sum() < 0.9 * A[m_].sum():
                continue
            nb = lab[ix[np.concatenate([np.arange(ip[t], ip[t + 1]) for t in np.where(m_)[0]])]]
            nb = nb[nb != l_]
            nb = nb[np.array([neck_of(x) == ni for x in nb], bool)] if len(nb) else nb
            if len(nb):
                lab[m_] = np.bincount(nb).argmax()
                log("curve network: a strip under a neck's base joined its last band")
    _, lab = np.unique(lab, return_inverse=True)
    if walls_ and not TUNNELS:
        wrapped = False
        for w_ in walls_:
            ar_ = np.bincount(lab[w_], weights=A[w_])
            if ar_.max() >= 0.8 * ar_.sum():
                wrapped = True
        if wrapped:
            log("curve network: one face wraps round a hole through the part: its tunnel becomes a face")
            if DUMP is not None:
                DUMP.clear()
            return island_curves(V, F, feat, necks, tri_mirror, half, log, **dict(prm, TUNNELS=True))
    log(f"curve network: {lab.max() + 1} faces")
    if DUMP is not None:
        DUMP.append({"lab1": lab.copy()})
        if DUMP_ONLY:
            return [], lab

    # ---- 4. border stretches between corners -> clean curves
    ae = np.sort(np.asarray(mesh.face_adjacency_edges, np.int64), axis=1)
    m = lab[adj[:, 0]] != lab[adj[:, 1]]
    be = ae[m]
    pr = np.sort(np.c_[lab[adj[m, 0]], lab[adj[m, 1]]], axis=1)
    vf = {}
    for (i, j), (a_, b_) in zip(be, pr):
        for v in (i, j):
            vf.setdefault(v, set()).update((a_, b_))
    eu = np.sort(mesh.edges, axis=1)
    bnd = trimesh.grouping.group_rows(eu, require_count=1)
    bnd_v = set(np.unique(eu[bnd]).tolist())
    corner = {v for v, s in vf.items() if len(s) >= 3 or v in bnd_v}
    by_pair = {}
    for (i, j), p_ in zip(be, map(tuple, pr)):
        by_pair.setdefault(p_, []).append((i, j))
    chains = []
    for p_, edges in by_pair.items():
        nb = {}
        for i, j in edges:
            nb.setdefault(i, []).append(j)
            nb.setdefault(j, []).append(i)
        used = set()
        starts = [v for v in nb if len(nb[v]) != 2 or v in corner]
        loops_only = not starts
        if loops_only:
            starts = [next(iter(nb))]
        for s0 in starts:
            for nx in nb[s0]:
                e = (min(s0, nx), max(s0, nx))
                if e in used:
                    continue
                path = [s0, nx]
                used.add(e)
                while len(nb[path[-1]]) == 2 and (path[-1] not in corner or loops_only) and path[-1] != s0:
                    a_, b_ = nb[path[-1]]
                    q = b_ if a_ == path[-2] else a_
                    e = (min(path[-1], q), max(path[-1], q))
                    if e in used:
                        break
                    used.add(e)
                    path.append(q)
                chains.append((p_, path, path[-1] == path[0]))

    cr_ = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn_ = np.zeros_like(V)
    for q_ in range(3):
        np.add.at(vn_, F[:, q_], cr_)
    vn_ /= np.maximum(np.linalg.norm(vn_, axis=1, keepdims=True), 1e-30)
    vtree = cKDTree(V)

    def onto_surface(Q, keep_ends=True, closed=False, sm=3.0):
        """Curve points moved onto the surface (along the local normal, to the tangent planes of the
        nearest vertices): a smoothing spline cuts across humps and rounded edges by up to its wander.
        The correction is smoothed along the curve over `sm` mm, so a curve on a crest does not pick up
        the facets' jitter."""
        Q = np.asarray(Q, float).copy()
        e0, e1 = Q[0].copy(), Q[-1].copy()
        step = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).mean()) if len(Q) > 1 else 1.0
        sig = sm / max(step, 1e-6)
        for _ in range(3):
            d, j = vtree.query(Q, k=6)
            w = 1.0 / (d + 0.05)
            w /= w.sum(1, keepdims=True)
            nn = (w[:, :, None] * vn_[j]).sum(1)
            off = (w * ((Q[:, None, :] - V[j]) * vn_[j]).sum(2)).sum(1)
            if len(Q) > 4 and sig > 0.5:
                mode = "wrap" if closed else "nearest"
                off = gaussian_filter1d(off, sig, mode=mode)
                nn = gaussian_filter1d(nn, sig, axis=0, mode=mode)
            nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
            Q -= off[:, None] * nn
        if keep_ends:
            Q[0], Q[-1] = e0, e1
        return Q

    def smooth_tck(P, closed, pin=True):
        """Smoothing spline through a chain: (tck, L) or None (parameter = arc length / L)."""
        u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        keep = np.r_[True, np.diff(u) > 1e-6]
        Pk = P[keep]
        L = u[-1]
        if len(Pk) < 6 or L <= 0:
            return None
        w = np.ones(len(Pk))
        if not closed and pin:
            w[[0, -1]] = 1e4
        try:
            tck, _ = splprep(Pk.T, u=u[keep] / L, w=w, s=len(Pk) * min(SMOOTH, max(0.6, 0.05 * L)) ** 2, k=3, per=1 if closed else 0)
        except Exception:
            return None
        return tck, L

    def smooth_chain(P, closed, pin=True):
        r = smooth_tck(P, closed, pin)
        if r is None:
            return P
        tck, L = r
        Q = np.array(splev(np.linspace(0, 1, max(12, int(L / 0.75))), tck)).T
        if not closed and pin:
            Q[0], Q[-1] = P[0], P[-1]
        return Q

    def ring_of(P, closed, L):
        """A ring around a neck axis that explains most of this chain: (ring, on-mask) or None.
        ring: dict(o, a, e1, e2, h, r); the chain's points within RING_TOL of the circle are 'on'."""
        best = None
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        wl = np.r_[seg, 0.0] * 0.5 + np.r_[0.0, seg] * 0.5  # length each point stands for
        axes_ = []  # each neck's axis through its mouth and through each ring centre off it
        for nk in necks:
            for c in [None] + list(nk["circles"]):
                oc = nk["origin"] if c is None else c["centre"] - c["h"] * nk["axis"]
                if not any(q is nk and np.linalg.norm(oc - p) < 0.5 for q, p in axes_):
                    axes_.append((nk, oc))
        for nk, o in axes_:
            a = nk["axis"]
            d = P - o
            hq = d @ a
            rv = d - hq[:, None] * a
            rq = np.linalg.norm(rv, axis=1)
            near = (rq < nk["r"].max() + 10) & (hq > nk["h"].min() - 10) & (hq < 2)
            if near.mean() < 0.5:
                continue
            h0, r0 = np.median(hq[near]), np.median(rq[near])
            for _ in range(6):
                on = near & (np.abs(hq - h0) < 1.5) & (np.abs(rq - r0) < 2.5)
                if on.sum() < 5:
                    break
                h0, r0 = float(np.median(hq[on])), float(np.median(rq[on]))
            for c in nk["circles"]:  # a ring crest found on the profile is exact
                if abs(c["h"] - h0) < 1.0 and abs(c["r"] - r0) < 1.0 and np.linalg.norm(c["centre"] - c["h"] * a - o) < 1.0:
                    h0, r0 = c["h"], c["r"]
                    break
            # a border chain zigzags across its crest by a triangle: judge 'on' by a running median
            dev = np.maximum(np.abs(hq - h0), np.abs(rq - r0))
            k_ = 5
            dpad = np.r_[dev[-k_:], dev, dev[:k_]] if closed else np.r_[[dev[0]] * k_, dev, [dev[-1]] * k_]
            dev_s = np.median(np.lib.stride_tricks.sliding_window_view(dpad, 2 * k_ + 1), axis=1)
            on = dev_s < RING_TOL
            ang = np.unwrap(np.arctan2(rv @ nk["e2"], rv @ nk["e1"]))
            # the longest run of 'on' points (cyclic for a closed chain)
            m = len(P)
            idx = np.arange(m)
            if closed and on.all():
                run = idx
            else:
                o2 = np.r_[on, on] if closed else on
                runs, i = [], 0
                while i < len(o2):
                    if o2[i]:
                        j = i
                        while j + 1 < len(o2) and o2[j + 1] and j + 1 - i < m - 1:
                            j += 1
                        runs.append((i, j))
                        i = j + 1
                    else:
                        i += 1
                if not runs:
                    continue
                i0, i1 = max(runs, key=lambda r_: wl[np.arange(r_[0], r_[1] + 1) % m].sum())
                run = np.arange(i0, i1 + 1) % m
            rl = wl[run].sum()
            span = np.degrees(abs(np.unwrap(ang[run])[-1] - np.unwrap(ang[run])[0])) if len(run) > 1 else 0.0
            if rl < max(0.35 * L, RING_MIN) or (span < 60 and not (closed and len(run) == m)):
                continue
            if best is None or rl > best[2]:
                mask = np.zeros(m, bool)
                mask[run] = True
                best = (dict(o=o, a=a, e1=nk["e1"], e2=nk["e2"], h=h0, r=r0), mask, rl)
        return None if best is None else best[:2]

    def on_ring(rg, X):
        """Nearest points of the ring circle to X."""
        d = np.atleast_2d(X) - rg["o"]
        hq = d @ rg["a"]
        rv = d - hq[:, None] * rg["a"]
        rv /= np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-12)
        return rg["o"] + rg["h"] * rg["a"] + rg["r"] * rv

    def ring_curve(P, closed, rg, mask, L):
        """Exact circle where the chain is on the ring, the chain's smoothing spline elsewhere, blended
        smoothly over BLEND mm (the spline gives the angle along the ring)."""
        n_out = max(24, int(L / 0.5))
        if closed and mask.mean() > 0.97:
            th = np.linspace(-np.pi, np.pi, 721)[:-1]
            return rg["o"] + rg["h"] * rg["a"] + rg["r"] * (np.cos(th)[:, None] * rg["e1"] + np.sin(th)[:, None] * rg["e2"])
        r = smooth_tck(P, closed)
        uu = np.linspace(0, 1, n_out, endpoint=not closed)
        if r is None:
            S = np.c_[[np.interp(uu, np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] / L, P[:, k])
                       for k in range(3)]].T
        else:
            S = np.array(splev(uu, r[0])).T
        if not closed:
            S[0], S[-1] = P[0], P[-1]
        S = onto_surface(S, keep_ends=not closed, closed=closed)
        C = on_ring(rg, S)
        u_p = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] / L
        on_u = u_p[mask]
        # distance (in mm along the chain) from each output sample to the nearest 'on' point
        dd = np.abs(uu[:, None] - on_u[None, :])
        if closed:
            dd = np.minimum(dd, 1 - dd)
        dist = dd.min(1) * L
        t = np.clip(1 - dist / BLEND, 0, 1)
        w = t * t * (3 - 2 * t)
        Q = w[:, None] * C + (1 - w[:, None]) * S
        if not closed:
            for e_, k_ in ((0, 0), (-1, -1)):
                if w[k_] < 1:
                    Q[k_] = P[e_]
        return Q

    # corners a few mm apart (joined by a short stretch) are one corner; the short stretch goes
    par = {}

    def find(v):
        while par.get(v, v) != v:
            par[v] = par.get(par[v], par[v])
            v = par[v]
        return v
    for p_, path, closed in chains:
        if not closed and path[0] != path[-1] and (
                np.linalg.norm(np.diff(V[path], axis=0), axis=1).sum() < CORNER_MM or
                (np.linalg.norm(V[path[0]] - V[path[-1]]) < CORNER_MM
                 and np.linalg.norm(np.diff(V[path], axis=0), axis=1).sum() < 3 * CORNER_MM)):
            a_, b_ = find(path[0]), find(path[-1])
            if a_ != b_:
                par[b_] = a_
    grp = {}
    for v in set(par) | {find(v) for v in par}:
        grp.setdefault(find(v), []).append(v)

    curves = []
    for p_, path, closed in chains:
        P = V[path]
        L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
        if not closed and find(path[0]) == find(path[-1]) and path[0] != path[-1]:
            continue  # inside a merged corner
        if L < 2.0:
            if not closed and np.linalg.norm(P[-1] - P[0]) > 1e-9:  # a short link between two corners
                curves.append({"name": f"link {p_[0]}|{p_[1]}", "points": np.vstack([P[0], P[-1]]), "closed": False,
                               "faces": p_, "ends": (path[0], path[-1])})
            continue
        rr = ring_of(P, closed, L)
        if rr is not None:
            rg, mask = rr
            Q = ring_curve(P, closed, rg, mask, L)
            curves.append({"name": f"neck ring r={rg['r']:.2f}" + ("" if mask.all() else " (part)"), "points": Q,
                           "closed": closed, "faces": p_, "ring": rg, "ends": None if closed else (path[0], path[-1]),
                           "end_on": None if closed else (bool(mask[0]), bool(mask[-1]))})
            continue
        curves.append({"name": f"border {p_[0]}|{p_[1]}", "points": smooth_chain(P, closed, pin=False), "closed": closed,
                       "faces": p_, "ends": None if closed else (path[0], path[-1]), "raw": P})

    # corners: on a ring, onto the ring; elsewhere where the (free-ended) curves meeting there end, on
    # average; every curve bends onto its corners over BLEND mm (no hooks from the jagged label border)
    cpos = {}
    for c in curves:
        if c.get("ring") is not None and c["ends"] is not None:
            for k_, v in enumerate(c["ends"]):
                if c["end_on"][k_] and find(v) not in cpos:
                    cpos[find(v)] = on_ring(c["ring"], V[v])[0]
                    if mir_ and abs(V[v][1]) < 1e-9:
                        cpos[find(v)][1] = 0.0
    # T- and X-junctions: two borders that continue each other through a corner (nearly opposite
    # directions) are one stroke, fitted as one smoothing spline; the corner sits on it, so the through
    # line stays smooth and the other border ends on it
    def end_dir(Q, k_):
        s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        sp = min(6.0, 0.4 * s_[-1])
        sp = sp if k_ == 0 else s_[-1] - sp
        d = np.array([np.interp(sp, s_, Q[:, j]) for j in range(3)]) - Q[0 if k_ == 0 else -1]
        return d / max(np.linalg.norm(d), 1e-12)
    inc = {}
    for ci, c in enumerate(curves):
        if c["ends"] is not None and not c["name"].startswith("link"):
            for k_, v in enumerate(c["ends"]):
                inc.setdefault(find(v), []).append((ci, k_))
    link_ = {}
    for v, lst in inc.items():
        if v in cpos or len(lst) < 3:
            continue
        cand = [e for e in lst if curves[e[0]].get("raw") is not None and curves[e[0]].get("ring") is None
                and np.linalg.norm(np.diff(np.asarray(curves[e[0]]["points"], float), axis=0), axis=1).sum() >= STROKE_MIN]
        dirs = {e: end_dir(np.asarray(curves[e[0]]["points"], float), e[1]) for e in cand}
        pairs = sorted((float(dirs[a] @ dirs[b]), a, b) for i, a in enumerate(cand) for b in cand[i + 1:] if a[0] != b[0])
        used = set()
        for cs, a, b in pairs:
            if cs > -THROUGH_COS:
                break
            if a not in used and b not in used:
                link_[a], link_[b] = b, a
                used |= {a, b}
    seen, strokes = set(), []
    for ci in range(len(curves)):
        if ci in seen or (ci, 0) not in link_ and (ci, 1) not in link_:
            continue
        cur, kin, back = ci, 0, {ci}  # walk back to the stroke's start
        while (cur, kin) in link_:
            nb, kb = link_[(cur, kin)]
            if nb in back:
                break
            back.add(nb)
            cur, kin = nb, 1 - kb
        seq, c_, ent = [], cur, kin
        while True:
            seq.append((c_, ent))
            seen.add(c_)
            if (c_, 1 - ent) not in link_:
                break
            nb, kb = link_[(c_, 1 - ent)]
            if nb in seen:
                break
            c_, ent = nb, kb
        if len(seq) < 2:
            continue
        parts, last = [], []
        for c_, ent in seq:
            P = np.asarray(curves[c_]["raw"], float)
            P = P if ent == 0 else P[::-1]
            if parts and np.linalg.norm(P[0] - parts[-1][-1]) < 1e-9:
                P = P[1:]
            parts.append(P)
            last.append(sum(len(q) for q in parts) - 1)
        Pall = np.vstack(parts)
        s_all = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pall, axis=0), axis=1))]
        r_ = smooth_tck(Pall, False, pin=False)
        if r_ is None:
            continue
        tck, L_ = r_
        u_ = np.r_[0.0, s_all[last]] / max(s_all[-1], 1e-12)
        strokes.append([seq, tck, L_, u_])
        for c_, _ in seq:
            curves[c_]["stroke"] = True
    # where each junction sits on its stroke: a side border meets the stroke where its own course runs into
    # it (not where the layout's labels happen to meet, a few mm off: the side border would hook back to
    # reach that); two strokes crossing meet where they cross
    joints = {}
    for si, (seq, tck, L_, u_) in enumerate(strokes):
        for i in range(1, len(seq)):
            c_, ent = seq[i]
            joints.setdefault(find(curves[c_]["ends"][ent]), []).append((si, i))
    in_stroke = {c_ for s in strokes for c_, _ in s[0]}

    def window(si, i, half=15.0):
        seq, tck, L_, u_ = strokes[si]
        lo = max(u_[i - 1] + 1.0 / L_, u_[i] - half / L_)
        hi = min(u_[i + 1] - 1.0 / L_, u_[i] + half / L_)
        if hi <= lo:
            return None
        uu = np.linspace(lo, hi, max(40, int((hi - lo) * L_ / 0.1)))
        return uu, np.array(splev(uu, tck)).T
    for v, js in joints.items():
        if v in cpos:
            continue
        if len(js) >= 2:  # strokes crossing: their closest points
            w1, w2 = window(*js[0]), window(*js[1])
            if w1 is None or w2 is None:
                continue
            D = np.linalg.norm(w1[1][:, None, :] - w2[1][None, :, :], axis=2)
            a, b = np.unravel_index(np.argmin(D), D.shape)
            strokes[js[0][0]][3][js[0][1]] = w1[0][a]
            strokes[js[1][0]][3][js[1][1]] = w2[0][b]
            cpos[v] = 0.5 * (w1[1][a] + w2[1][b])
        else:
            si, i = js[0]
            br = [e for e in inc.get(v, []) if e[0] not in in_stroke and curves[e[0]].get("ring") is None
                  and curves[e[0]].get("raw") is not None]
            w = window(si, i)
            if not br or w is None:
                continue
            Qb = np.asarray(curves[br[0][0]]["points"], float)
            pE = Qb[0 if br[0][1] == 0 else -1]
            dE = -end_dir(Qb, br[0][1])  # running into the junction
            uu, S = w
            tS = np.gradient(S, axis=0)
            tS /= np.maximum(np.linalg.norm(tS, axis=1, keepdims=True), 1e-12)
            W = S - pE
            t = W @ dE
            perp = np.linalg.norm(W - t[:, None] * dE, axis=1)
            ok = (t > -3.0) & (t < 15.0) & (np.abs((tS * dE).sum(1)) < 0.9)
            k_ = int(np.argmin(np.where(ok, perp, np.inf))) if ok.any() else int(np.argmin(np.linalg.norm(W, axis=1)))
            strokes[si][3][i] = uu[k_]
            cpos[v] = S[k_].copy()
        if mir_ and any(abs(V[w_][1]) < 1e-9 for w_ in grp.get(v, [v])):
            cpos[v][1] = 0.0
    for seq, tck, L_, u_ in strokes:
        for i, (c_, ent) in enumerate(seq):
            uu = np.linspace(u_[i], u_[i + 1], max(12, int((u_[i + 1] - u_[i]) * L_ / 0.75)))
            Q = np.array(splev(uu, tck)).T
            curves[c_]["points"] = Q if ent == 0 else Q[::-1]
            if i > 0:  # the junction between this piece and the previous one: on the stroke
                v = find(curves[c_]["ends"][ent])
                if v not in cpos:
                    cpos[v] = Q[0].copy()
                    if mir_ and any(abs(V[w][1]) < 1e-9 for w in grp.get(v, [v])):
                        cpos[v][1] = 0.0
    ends_at = {}
    for c in curves:
        if c["ends"] is not None and c.get("ring") is None and not c["name"].startswith("link"):
            Q = np.asarray(c["points"], float)
            for k_, v in enumerate(c["ends"]):
                ends_at.setdefault(find(v), []).append(Q[0 if k_ == 0 else -1])
    for v, pts in ends_at.items():
        if v not in cpos:
            cp_ = np.mean(pts, axis=0) if len(pts) > 1 else np.asarray(pts[0], float)
            vv = grp.get(v, [v])
            q_ = V[vv].mean(0)
            nq = vn_[vv].sum(0)
            nq /= max(np.linalg.norm(nq), 1e-30)
            cp_ = cp_ - ((cp_ - q_) @ nq) * nq  # onto the surface (the corner's tangent plane)
            if mir_ and any(abs(V[u][1]) < 1e-9 for u in vv):
                cp_[1] = 0.0
            cpos[v] = cp_
    ring_c = {find(v) for c in curves if c.get("ring") is not None and c["ends"] is not None
              for k_, v in enumerate(c["ends"]) if c["end_on"][k_]}
    for v in cpos:  # every corner off a ring exactly on the surface
        if v not in ring_c:
            on_plane = mir_ and abs(cpos[v][1]) < 1e-9
            cpos[v] = onto_surface(np.atleast_2d(cpos[v]), keep_ends=False)[0]
            if on_plane:
                cpos[v][1] = 0.0
    for c in curves:
        if c["ends"] is None:
            continue
        if c.get("raw") is not None and len(c["raw"]) >= 6 and not c.get("stroke"):
            # refit through the final corners: the raw border without its last 2.5 mm at each end (the
            # label border wanders there), pinned at the corners
            P = np.asarray(c["raw"], float)
            a_, b_ = cpos.get(find(c["ends"][0]), P[0]), cpos.get(find(c["ends"][1]), P[-1])
            Lr = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
            ex = min(5.0, 0.3 * Lr)
            keep = (np.linalg.norm(P - a_, axis=1) > ex) & (np.linalg.norm(P - b_, axis=1) > ex)
            Pm = np.vstack([a_, P[keep], b_])
            if len(Pm) >= 6:
                Q = smooth_chain(Pm, False, pin=True)
                c["points"] = Q
                continue
        Q = np.asarray(c["points"], float).copy()
        s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        Lc = s_[-1]
        for k_, v in enumerate(c["ends"]):
            v = find(v)
            if v not in cpos:
                continue
            delta = cpos[v] - Q[0 if k_ == 0 else -1]
            if np.linalg.norm(delta) < 1e-12:
                continue
            dist = s_ if k_ == 0 else Lc - s_
            t = np.clip(1 - dist / max(min(BLEND, 0.45 * Lc), 1e-9), 0, 1)
            Q += (t * t * (3 - 2 * t))[:, None] * delta
        c["points"] = Q
    for c in curves:
        if c.get("ring") is None and len(c["points"]) > 2:
            c["points"] = onto_surface(c["points"], keep_ends=not c["closed"], closed=c["closed"])

    if tm is not None and len(tm) == n:  # exact mirror (plane y = 0): kept-side curves copied, crossing ones averaged
        mir = lambda Q: Q * np.r_[1.0, -1.0, 1.0]
        twin = np.zeros(lab.max() + 1, np.int64)
        twin[lab] = lab[np.asarray(tm, np.int64)]

        def resample(Q, m, closed):
            Qc = np.vstack([Q, Q[:1]]) if closed else Q
            u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Qc, axis=0), axis=1))]
            t = np.linspace(0, u[-1], m + (1 if closed else 0))
            R_ = np.c_[[np.interp(t, u, Qc[:, k]) for k in range(3)]].T
            return R_[:-1] if closed else R_

        sym = []
        for c in curves:
            Q = c["points"]
            if Q[:, 1].min() > -0.05:
                sym.append(c)
                if Q[:, 1].max() > 0.05:
                    sym.append({**c, "points": mir(Q)[::-1].copy(), "faces": tuple(sorted(twin[list(c["faces"])])),
                                "name": c["name"] + " (mirror)"})
            elif Q[:, 1].max() > 0.05:
                if c["closed"]:
                    M = mir(Q)[::-1]
                    M = np.roll(M, -int(np.argmin(np.linalg.norm(M - Q[0], axis=1))), axis=0)
                    Qs = 0.5 * (resample(Q, len(Q), True) + resample(M, len(Q), True))
                else:
                    Qs = 0.5 * (resample(Q, len(Q), False) + resample(mir(Q)[::-1], len(Q), False))
                sym.append({**c, "points": Qs})
        curves = sym
    for c in curves:
        c.pop("ring", None), c.pop("ends", None), c.pop("end_on", None), c.pop("raw", None), c.pop("stroke", None)
    log(f"curve network: {len(curves)} border curves ({sum('ring' in c['name'] for c in curves)} on neck rings)")
    return curves, lab
