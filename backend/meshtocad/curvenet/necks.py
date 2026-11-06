"""Necks and finishes of bottles and jugs: rotational features found from the part's openings.

Every open boundary loop of the mesh that is a circle is a neck's mouth; its axis points out
of the part. Slicing perpendicular to the axis from the mouth inwards, each slice loop around
the axis is fitted with a circle; the neck is the stack of round slices (it ends where the
slice stops being a circle: the shoulder). Along it, the profile r(h) bends where the finish
has rings: peaks of the profile curvature above KAPPA (1/mm) are ring crests, each an exact
circle on the axis, and the neck's last round slice is its base circle.
All lengths in mm.
"""
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _fit_circle(P):
    A = np.c_[2 * P, np.ones(len(P))]
    s, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    c = s[:2]
    r = np.sqrt(max(s[2] + c @ c, 1e-12))
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - r) ** 2)))


def _frame(a):
    a = a / np.linalg.norm(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, t)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def _boundary_loops(mesh):
    e = np.sort(mesh.edges, axis=1)
    g = trimesh.grouping.group_rows(e, require_count=1)
    be = e[g]
    if not len(be):
        return []
    n = len(mesh.vertices)
    k, lab = connected_components(csr_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(n, n)), directed=False)
    vs = np.unique(be)
    return [vs[lab[vs] == c] for c in np.unique(lab[vs])]


def find_necks(V, F, step=0.5, kappa=0.1, log=print):
    """Each neck: dict(origin, axis, e1, e2, h (slice heights along the axis, mouth first),
    r, circles=[dict(h, centre (3D), r, base)]) ."""
    mesh = trimesh.Trimesh(V, F, process=False)
    diag = float(np.linalg.norm(np.ptp(V, axis=0)))
    ctr = V.mean(0)
    necks = []
    for lp in _boundary_loops(mesh):
        P = V[lp]
        if len(P) < 16:
            continue
        c0 = P.mean(0)
        ev, evec = np.linalg.eigh(np.cov((P - c0).T))
        a = evec[:, 0]
        if (c0 - ctr) @ a < 0:
            a = -a  # out of the part
        e1, e2 = _frame(a)
        c2, r, rms = _fit_circle(np.c_[(P - c0) @ e1, (P - c0) @ e2])
        if rms > 0.02 * r + 0.15 or r > 0.25 * diag or abs((P - c0) @ a).max() > 0.05 * r + 0.2:
            continue  # not a round flat mouth
        o = c0 + c2[0] * e1 + c2[1] * e2
        hs, cs, rs = [], [], []
        h = -0.2
        miss = 0
        while h > -0.45 * diag:
            seg = trimesh.intersections.mesh_plane(mesh, a, o + h * a)
            if not len(seg):
                break
            Q = seg.reshape(-1, 3)
            key = np.round(Q / 1e-4).astype(np.int64)
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.reshape(-1, 2)
            nn = inv.max() + 1
            k, lab = connected_components(csr_matrix((np.ones(len(inv)), (inv[:, 0], inv[:, 1])), shape=(nn, nn)),
                                          directed=False)
            pts = np.zeros((nn, 3))
            pts[inv.ravel()] = Q
            found = None
            for c in range(k):
                L = pts[lab == c]
                if len(L) < 12:
                    continue
                q2 = np.c_[(L - o) @ e1, (L - o) @ e2]
                cc, rr, rm = _fit_circle(q2)
                if np.linalg.norm(cc) < 0.5 * r + 2 and rr < 3 * r and rm < 0.02 * rr + 0.15:
                    found = (cc, rr)
            if found is None:
                miss += 1
                if miss > 4:
                    break  # the slices are no longer round: the shoulder
                h -= step
                continue
            miss = 0
            hs.append(h)
            cs.append(found[0])
            rs.append(found[1])
            h -= step
        if len(hs) * step < 8:
            continue
        hs, cs, rs = np.array(hs), np.array(cs), np.array(rs)
        rr = gaussian_filter1d(rs, 2.0)
        dr = np.gradient(rr, -step)
        d2 = np.gradient(dr, -step)
        kap = np.abs(d2) / (1 + dr ** 2) ** 1.5
        peaks = [j for j in range(2, len(kap) - 2) if kap[j] >= kappa and kap[j] == kap[max(0, j - 2):j + 3].max()]
        circles = []
        for j in peaks + [len(hs) - 1]:
            centre = o + cs[j][0] * e1 + cs[j][1] * e2 + hs[j] * a
            circles.append({"h": float(hs[j]), "centre": centre, "r": float(rs[j]), "base": j == len(hs) - 1})
        log(f"neck at {np.round(o, 1)}: axis {np.round(a, 3)}, {len(hs) * step:.0f} mm long, "
            f"r {rs.min():.1f}..{rs.max():.1f}, {len(peaks)} ring crests")
        necks.append({"origin": o, "axis": a, "e1": e1, "e2": e2, "h": hs, "r": rs, "circles": circles})
    return necks


def circle_points(nk, c, n=720):
    th = np.linspace(-np.pi, np.pi, n + 1)[:-1]
    return c["centre"] + c["r"] * (np.cos(th)[:, None] * nk["e1"] + np.sin(th)[:, None] * nk["e2"])
