"""Dev-only: side-by-side render, true CAD faces vs found surfaces, coloured by surface type.

    runtime/python/python.exe scripts/render_compare.py SESSION.pkl TRUTH_DIR OUT.png
        [--hide-largest N] [--view x,y,z] [--size 900]

Left: the source CAD tessellation coloured by each face's true surface type.
Right: our segmentation of the mesh the app received, coloured by the type found,
with region borders drawn. A small painter's-algorithm renderer (numpy + PIL).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

COLORS = {"plane": (120, 150, 190), "cylinder": (230, 140, 60), "cone": (225, 200, 70), "sphere": (90, 185, 110),
          "torus": (160, 110, 210), "bspline": (215, 80, 90), "freeform": (215, 80, 90), "other": (150, 150, 150)}
LABEL = {"plane": "plane", "cylinder": "cylinder", "cone": "cone", "sphere": "sphere", "torus": "torus",
         "bspline": "B-spline / freeform"}


def render(tris, cols, view, size, edges=None, box=None):
    """tris (n,3,3), cols (n,3) uint8 -> PIL image. edges: (m,2,3) segments drawn on top."""
    d = view / np.linalg.norm(view)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(up, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    P = tris.reshape(-1, 3)
    lo, hi = box if box is not None else (P.min(0), P.max(0))
    c = 0.5 * (lo + hi)
    corners = np.array([[a, b, e] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for e in (lo[2], hi[2])])
    span = max(np.ptp((corners - c) @ x), np.ptp((corners - c) @ y))
    s = 0.9 * size / span
    u = ((tris - c) @ x) * s + size / 2
    v = size / 2 - ((tris - c) @ y) * s
    depth = ((tris - c) @ d).mean(axis=1)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    light = np.array([0.4, -0.3, 0.86])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)
    rgb = np.clip(cols * shade[:, None], 0, 255).astype(np.uint8)
    img = Image.new("RGB", (size, size), (24, 24, 26))
    dr = ImageDraw.Draw(img)
    for k in np.argsort(depth):  # far to near (depth grows toward the viewer)
        dr.polygon([(u[k, 0], v[k, 0]), (u[k, 1], v[k, 1]), (u[k, 2], v[k, 2])], fill=tuple(rgb[k]))
    if edges is not None and len(edges):
        eu = ((edges - c) @ x) * s + size / 2
        ev = size / 2 - ((edges - c) @ y) * s
        ed = ((edges - c) @ d).mean(axis=1)
        # only edges in front of nearby surface: crude, draw the nearer half
        keep = ed > np.percentile(depth, 35)
        for k in np.where(keep)[0]:
            dr.line([(eu[k, 0], ev[k, 0]), (eu[k, 1], ev[k, 1])], fill=(20, 20, 20), width=1)
    return img


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = lambda k, dflt: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else dflt
    sess, tdir, out = Path(args[0]), Path(args[1]), Path(args[2])
    hide = int(opt("--hide-largest", "0"))
    view = np.array([float(x) for x in opt("--view", "1,-1.2,0.8").split(",")])
    size = int(opt("--size", "900"))
    s = pickle.loads(sess.read_bytes())
    faces = json.loads((tdir / "truth_faces.json").read_text())
    tf = np.load(tdir / "truth_faces.npz")
    truth = json.loads((tdir / "truth.json").read_text())
    mesh = s.mesh
    scale = truth["diag_mm"] / float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    # snap to the nearest unit factor (OpenCASCADE's bounding box is padded, so the ratio is only rough)
    scale = min((1.0, 10.0, 25.4, 1000.0), key=lambda k: abs(np.log(scale / k)))
    # hide the largest parts (an engine block) to show the inside
    part_area = np.bincount(s.face_part, weights=mesh.area_faces)
    hidden = set(np.argsort(-part_area)[:hide].tolist())
    keep_ours = ~np.isin(s.face_part, list(hidden))
    # truth parts are numbered by solid order; ours carry the GLB node names part_<k>
    def truth_index(p):
        name = s.part_names[p] if p < len(s.part_names) else ""
        return int(name.split("_")[-1]) - 1 if name.startswith("part_") else -1
    hidden_t = {truth_index(p) for p in hidden}
    tri_part = np.array([faces[f]["part"] for f in tf["face"]])
    keep_t = ~np.isin(tri_part, list(hidden_t))
    T = tf["corners"][keep_t]
    Tc = np.array([COLORS.get(faces[f]["type"], COLORS["other"]) for f in tf["face"][keep_t]], float)
    O = mesh.triangles[keep_ours] * scale
    kinds = np.array([r.prim for r in s.regions])
    Oc = np.array([COLORS.get(k, COLORS["other"]) for k in kinds[s.face_labels[keep_ours]]], float)
    # region borders
    adj = mesh.face_adjacency
    fl = s.face_labels
    border = (fl[adj[:, 0]] != fl[adj[:, 1]]) & keep_ours[adj[:, 0]] & keep_ours[adj[:, 1]]
    # region borders: darken the triangles along them (respects what is in front)
    # each region a slightly different brightness of its type colour, so borders show
    rng = np.random.default_rng(3)
    jitter = 0.78 + 0.44 * rng.random(len(s.regions))
    Oc *= jitter[s.face_labels[keep_ours]][:, None]
    E = None
    box = (np.minimum(T.reshape(-1, 3).min(0), O.reshape(-1, 3).min(0)),
           np.maximum(T.reshape(-1, 3).max(0), O.reshape(-1, 3).max(0)))
    left = render(T, Tc, view, size, box=box)
    right = render(O, Oc, view, size, box=box)
    W = Image.new("RGB", (2 * size, size + 70), (24, 24, 26))
    W.paste(left, (0, 0))
    W.paste(right, (size, 0))
    dr = ImageDraw.Draw(W)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()
    n_true = len(faces)
    dr.text((14, 10), f"Source CAD: {n_true} faces (true surface types)", fill=(230, 230, 230), font=font)
    dr.text((size + 14, 10), f"MeshToCad from the mesh: {len(s.regions)} surfaces (found types; shades = separate surfaces)",
            fill=(230, 230, 230), font=font)
    xk = 14
    for k in ("plane", "cylinder", "cone", "sphere", "torus", "bspline"):
        dr.rectangle([xk, size + 25, xk + 18, size + 43], fill=COLORS[k])
        dr.text((xk + 24, size + 25), LABEL[k], fill=(220, 220, 220), font=small)
        xk += 30 + int(dr.textlength(LABEL[k], font=small)) + 18
    W.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
