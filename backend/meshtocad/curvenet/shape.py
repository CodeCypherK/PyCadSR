"""Shape map: the two principal curvatures at every vertex, as smoothed asinh features.

feat[:, 0] = asinh(k1 S), feat[:, 1] = asinh(k2 S) with k1 >= k2 (1/mm, convex > 0) and
S = SMM: flat (0, 0), a 30 mm tube (0.9, 0), a 10 mm blend (1.8, 0), a dome (a, a), a saddle
(a, -b). Curvatures from a quadric over the 24 nearest vertices.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

SMM = 30.0


def vertex_normals(V, F):
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], cr)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)


def principal_features(V, F, smooth=4, k=24):
    VN = vertex_normals(V, F)
    _, J = cKDTree(V).query(V, k=k)
    K1 = np.zeros(len(V))
    K2 = np.zeros(len(V))
    for s in range(0, len(V), 20000):
        j, n_ = J[s:s + 20000], VN[s:s + 20000]
        u = np.cross(n_, np.where(np.abs(n_[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        w = np.cross(n_, u)
        D = V[j] - V[s:s + 20000, None]
        ok = (VN[j] * n_[:, None]).sum(2) > 0.5
        x, y, z = (D * u[:, None]).sum(2), (D * w[:, None]).sum(2), (D * n_[:, None]).sum(2)
        M_ = np.stack([x * x, x * y, y * y, x, y], 2) * ok[..., None]
        G = np.einsum("nki,nkj->nij", M_, M_) + 1e-12 * np.eye(5)
        rhs = np.einsum("nki,nk->ni", M_, z * ok)[..., None]
        try:
            c = np.linalg.solve(G, rhs)[..., 0]
        except np.linalg.LinAlgError:
            # neighbours on a line (a CAD export's sliver fans) leave a system singular: a ridge scaled
            # to each system makes the unknown direction read as flat instead of failing
            tr = np.trace(G, axis1=1, axis2=2)
            c = np.linalg.solve(G + (1e-9 * tr + 1e-12)[:, None, None] * np.eye(5), rhs)[..., 0]
        H = -np.stack([np.stack([2 * c[:, 0], c[:, 1]], 1), np.stack([c[:, 1], 2 * c[:, 2]], 1)], 1)
        ev = np.linalg.eigvalsh(H)
        K2[s:s + 20000], K1[s:s + 20000] = ev[:, 0], ev[:, 1]
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1), axis=0)
    n = len(V)
    Adj = csr_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    dv = np.asarray(Adj.sum(1)).ravel()
    feat = np.c_[np.arcsinh(K1 * SMM), np.arcsinh(K2 * SMM)]
    for _ in range(smooth):
        feat = 0.5 * feat + 0.5 * (Adj @ feat) / np.maximum(dv, 1)[:, None]
    return feat
