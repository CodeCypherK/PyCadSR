"""Segment models with one backend and score each against its true CAD faces.
    runtime python -I seg_worker.py BACKEND OUT.jsonl LABEL_DIR|- NAME...
Writes one JSON line per model; with a LABEL_DIR also saves each model's faces
(label per triangle + face kinds) as LABEL_DIR/NAME.npz for later analysis."""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

backend, out_path, label_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, backend)
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

import meshtocad.meshseg as ms  # noqa: E402
from score import score  # noqa: E402

try:
    from meshtocad.cloud import split_nonmanifold  # noqa: E402
except ImportError:  # backends before round 27
    split_nonmanifold = None


def bodies(mesh):
    """Bodies as the app finds them on load (server.cmd_load_mesh)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    adj = mesh.face_adjacency
    n = len(mesh.faces)
    k, lab = connected_components(csr_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n)), directed=False)
    return lab if k > 1 else None

DATA = Path(os.environ.get("REGRESS_DATA", Path(__file__).resolve().parents[2] / ".cache/r30/data"))
out = open(out_path, "a")
for name in sys.argv[4:]:
    t0 = time.time()
    try:
        d = dict(np.load(DATA / f"{name}.npz", allow_pickle=False))
        mesh = trimesh.Trimesh(d["V"], d["F"], process=False)
        groups = d["groups"] if len(d["groups"]) else None
        if split_nonmanifold is not None:
            # the stored mesh came from an older loader: repeat this backend's load repair
            # (triangle order is kept, so the true faces still line up)
            m2 = split_nonmanifold(mesh)
            if m2 is not mesh:
                mesh = m2
                groups = bodies(mesh)
        label, regions, tol = ms.mesh_segment(mesh, 30.0, groups, sag=float(d["sag"]))
        seg_s = time.time() - t0
        kinds = [k for k, _, _ in regions]
        tm = ms.TriMesh(mesh, 30.0, groups)
        acr = ms._across(tm, tm.G)
        acr_all = ms._across(tm, tm.Gall)
        teeth, _ = ms._teeth(acr, label, ms._corner_angles(tm))
        spikes = ms._spikes(tm, ms._vertex_fans(tm), acr, acr_all, label, ms._corner_angles(tm))
        coo = tm.Gall.tocoo()
        border = int((label[coo.row] != label[coo.col]).sum()) // 2
        r = score(d, label, kinds, jag=(len(teeth) + len(spikes), border))
        h = hashlib.md5(np.asarray(label, np.int64).tobytes() + "|".join(kinds).encode()).hexdigest()[:16]
        r.update(name=name, seg_s=round(seg_s, 2), tol=float(tol), hash=h)
        if label_dir != "-":
            np.savez_compressed(Path(label_dir) / f"{name}.npz", label=np.asarray(label, np.int32),
                                kinds=np.asarray(kinds))
    except Exception as e:  # one broken model must not stop the run
        r = {"name": name, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    out.write(json.dumps(r) + "\n")
    out.flush()
