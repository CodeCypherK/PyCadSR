"""The mesh as the desktop app hands it to the curve network (loaded as the app loads it; the curve
network finds the mirror plane itself unless the load step found one), with the mm scale.
Saved as .npz (V, F, mirror, scale) that mine.py reads in place of a mesh file, so approved layouts can be
learned on exactly what the app builds from.

    runtime/python/python.exe -I scripts/curvenet_train/app_mesh.py BACKEND MESH OUT.npz"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    backend, mesh, out = sys.argv[1:4]
    sys.path.insert(0, backend)
    from meshtocad import protocol
    from meshtocad.curvenet import mm_scale
    from meshtocad.server import App

    protocol.set_hook(lambda m: None)
    app = App(Path(tempfile.mkdtemp()))
    app.cmd_load_mesh({"path": mesh})
    s = app.s
    mir = None if s.mirror is None else {"normal": [float(x) for x in s.mirror["normal"]], "d": float(s.mirror["d"])}
    scale = mm_scale(s.mesh_path, s.diag)
    np.savez_compressed(out, V=np.asarray(s.mesh.vertices, float), F=np.asarray(s.mesh.faces, np.int64),
                        mirror=json.dumps(mir), scale=scale)
    print(f"{len(s.mesh.faces)} triangles, mirror {mir}, {scale:g} mm per unit")


if __name__ == "__main__":
    main()
