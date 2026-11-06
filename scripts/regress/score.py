"""Score one model's segmentation against its true CAD faces (lower is better).

Terms (all per model, so every model counts the same):
  incomplete  area share of true faces lying outside our biggest face on them
  impure      area share of our faces lying on other true surfaces (B-spline faces of the
              source count as one class: one freeform face over two spline faces is fine)
  wrong_type  area share of exact true faces whose face of ours has another type
              (a torus covered by cylinders, an exact face left as freeform)
  extra       extra faces of ours per true face: any face of ours, however small, that is a
              fragment of a true face another face of ours mainly covers (specks, tiles)
  mixed       faces of ours per true face that straddle two true surfaces (second >= 10%)
  jagged      saw-tooth spots per 100 border edges
"""
import numpy as np

WEIGHTS = {"incomplete": 1.0, "impure": 2.0, "wrong_type": 1.0, "extra": 0.3, "mixed": 1.0, "jagged": 0.02}
ANALYTIC = ("plane", "cylinder", "cone", "sphere", "torus")


def tri_area(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def _group_sum(keys, w):
    """Unique keys and the sum of w per key."""
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.bincount(inv, weights=w)


def score(data, label, kinds, jag=None):
    """data: the prep npz (dict-like); label: our face per triangle; kinds: our face types;
    jag: (saw-tooth spots, border edges) or None. -> dict of terms and 'score'."""
    V, F = data["V"], data["F"]
    A = tri_area(V, F)
    tot = float(A.sum()) or 1.0
    T0 = np.asarray(data["tface"], np.int64)
    ftype = np.asarray(data["face_type"])
    fsurf = np.asarray(data["face_surf"], np.int64)
    analytic = np.isin(ftype, ANALYTIC)
    # each side of a true face counts as its own face: a zero-thickness sheet in the source
    # is one face there but two (front and back) in any solid we build
    side = np.asarray(data["tside"], np.int64) if "tside" in data else np.ones(len(T0), np.int64)
    T = 2 * T0 + (side < 0)
    L = np.asarray(label, np.int64)
    K = np.asarray(kinds)
    nL = int(L.max()) + 1
    nT = int(T.max()) + 1
    # true face x our face areas and triangle counts
    key = T * nL + L
    u, a = _group_sum(key, A)
    _, c = _group_sum(key, np.ones_like(A))
    tf, rf = u // nL, u % nL
    face_area = np.bincount(T, weights=A, minlength=nT)
    face_tris = np.bincount(T, minlength=nT)
    # incomplete: per true face, area outside its biggest piece
    best = np.zeros(nT)
    np.maximum.at(best, tf, a)
    real = face_tris >= 3
    incomplete = float((face_area[real] - best[real]).sum()) / tot
    # extra faces per true face: every face of ours, however small, whose main true face is
    # also the main true face of another face of ours is a fragment of it (a speck, a tile).
    # Border triangles a tangent neighbour took do not count: they are not faces of their own
    o = np.lexsort((-a, rf))
    first_of = np.r_[True, rf[o][1:] != rf[o][:-1]]
    dom = tf[o][first_of]  # main true face of each face of ours
    pieces = np.bincount(dom, minlength=nT)
    nreal = max(int(real.sum()), 1)
    extra = float(np.maximum(pieces - 1, 0).sum()) / nreal
    # impure: classes are true surfaces for exact true faces, one class for all spline faces
    cls = np.where(analytic[T0], fsurf[T0], -1)
    key2 = (cls + 1) * nL + L
    u2, a2 = _group_sum(key2, A)
    r2 = u2 % nL
    reg_area = np.bincount(L, weights=A, minlength=nL)
    first = np.zeros(nL)
    np.maximum.at(first, r2, a2)
    impure = float((reg_area - first).sum()) / tot
    # mixed: our faces whose second class holds >= 10% of the face and >= 3 triangles
    _, c2 = _group_sum(key2, np.ones_like(A))
    order = np.lexsort((-a2, r2))
    rs, as_, cs = r2[order], a2[order], c2[order]
    starts = np.r_[0, np.flatnonzero(np.diff(rs)) + 1]
    second_a = np.zeros(nL)
    second_c = np.zeros(nL)
    has2 = np.r_[np.diff(starts) > 1, len(rs) - starts[-1] > 1] if len(starts) else np.zeros(0, bool)
    idx2 = starts[has2] + 1
    second_a[rs[idx2]] = as_[idx2]
    second_c[rs[idx2]] = cs[idx2]
    mixed = float(((second_a >= 0.1 * reg_area) & (second_c >= 3)).sum()) / nreal
    # wrong type on exact true faces
    tt = ftype[T0]
    ours = K[L]
    wrong = analytic[T0] & (ours != tt)
    wrong_type = float(A[wrong].sum()) / tot
    jagged = 0.0
    if jag is not None and jag[1] > 0:
        jagged = 100.0 * jag[0] / jag[1]
    terms = {"incomplete": incomplete, "impure": impure, "wrong_type": wrong_type, "extra": extra,
             "mixed": mixed, "jagged": jagged}
    terms["score"] = float(sum(WEIGHTS[k] * v for k, v in terms.items()))
    terms["faces"] = nL
    terms["true_faces"] = nreal
    return terms
