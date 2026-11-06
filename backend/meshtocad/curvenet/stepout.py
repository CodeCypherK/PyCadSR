"""STEP (AP214) writer for a curve-network B-rep (curvenet.brepfit): trimmed B-spline faces
sharing B-spline edges, each edge with its pcurves; a mirror-symmetric B-rep gets its other
half here (exact mirror image; edges on the plane are shared by a face and its mirror).
"""
from pathlib import Path

import numpy as np

from ..cad import UNITS, Step, fr, knots_mult

PLANE, OPEN = -1, -2
MIR = np.array([1.0, -1.0, 1.0])


def _deg(knots):
    k = np.round(np.asarray(knots, float), 12)
    return int(np.sum(k == k[0])) - 1


def write_step(J, path: Path, name: str, units: str = "mm", to_part=None) -> dict:
    """to_part(P) maps module coordinates (mm, mirror plane y = 0) to the part's coordinates in
    `units` (affine: B-splines map by their control points)."""
    to_part = to_part or (lambda P: np.asarray(P, float))
    st = Step()

    def g9(x):  # 9 significant digits: nanometres on a part this size, a quarter less file
        x = float(x)
        if abs(x) < 1e-14:
            return "0."
        t = f"{x:.9G}"
        if "E" in t:
            m, e = t.split("E")
            return (m if "." in m else m + ".") + "E" + e
        return t if "." in t else t + "."
    st.point = lambda p: st.add(f"CARTESIAN_POINT('',({g9(p[0])},{g9(p[1])},{g9(p[2])}))")
    ctx2d = st.add("( GEOMETRIC_REPRESENTATION_CONTEXT(2) PARAMETRIC_REPRESENTATION_CONTEXT() "
                   "REPRESENTATION_CONTEXT('2D SPACE','') )")
    mirror = J.get("mirror") is not None
    sides = (0, 1) if mirror else (0,)
    VX = np.asarray(J["vertices"], float)
    on_plane = np.abs(VX[:, 1]) < 1e-6 if mirror else np.zeros(len(VX), bool)

    def tr(P, side):
        P = np.asarray(P, float)
        return to_part(P * MIR if side else P)

    # surfaces
    surf = {}
    for side in sides:
        for p, S in enumerate(J["surfaces"]):
            C = tr(np.asarray(S["ctrl"], float).reshape(-1, 3), side).reshape(S["nu"], S["nv"], 3)
            surf[(side, p)] = st.bspline_surface(S["udeg"], S["vdeg"], C, np.asarray(S["uknots"]),
                                                 np.asarray(S["vknots"]))
    # vertices
    vid = {}
    for side in sides:
        for i, p in enumerate(VX):
            if side and on_plane[i]:
                vid[(1, i)] = vid[(0, i)]
                continue
            vid[(side, i)] = st.add(f"VERTEX_POINT('',#{st.point(tr(p, side))})")

    def pcurve(sid, knots, P2):
        u, m = knots_mult(np.asarray(knots, float))
        pts = [st.add(f"CARTESIAN_POINT('',({g9(q[0])},{g9(q[1])}))") for q in P2]
        bs = st.add(f"B_SPLINE_CURVE_WITH_KNOTS('',{_deg(knots)},{st.ref(pts)},.UNSPECIFIED.,.F.,.F.,"
                    f"({','.join(map(str, m))}),({','.join(fr(x) for x in u)}),.UNSPECIFIED.)")
        rep_ = st.add(f"DEFINITIONAL_REPRESENTATION('',(#{bs}),#{ctx2d})")
        return st.add(f"PCURVE('',#{sid},#{rep_})")

    # edges: kept half, mirror copies (plane edges shared by a face and its mirror)
    eid = {}
    for i, e in enumerate(J["edges"]):
        a, b = e["pair"]
        for side in sides:
            if side and b == PLANE:
                eid[(1, i)] = eid[(0, i)]
                continue
            knots = np.asarray(e["knots"], float)
            c3 = st.bspline_curve(_deg(knots), tr(e["ctrl"], side), knots)
            faces_on = [a] + ([b] if b >= 0 else [])
            pcs = [pcurve(surf[(side, q)], knots, e["pcurves"][str(q)]) for q in faces_on]
            if b == PLANE:  # also on the mirror face, same parameters
                pcs.append(pcurve(surf[(1, a)], knots, e["pcurves"][str(a)]))
            sc = st.add(f"SURFACE_CURVE('',#{c3},{st.ref(pcs)},.PCURVE_S1.)")
            eid[(side, i)] = st.add(f"EDGE_CURVE('',#{vid[(side, e['v0'])]},#{vid[(side, e['v1'])]},#{sc},.T.)")
    # faces: the kept ones as they are; the mirror ones with reversed sense (a mirror flips the
    # surface normal, and turns the loops around)
    faces = []
    for side in sides:
        for k, f in enumerate(J["faces"]):
            bounds = []
            for li, loop in enumerate(f["loops"]):
                oes = [st.add(f"ORIENTED_EDGE('',*,*,#{eid[(side, ei)]},{'.T.' if fw else '.F.'})") for ei, fw in loop]
                lp = st.add(f"EDGE_LOOP('',{st.ref(oes)})")
                kind = "FACE_OUTER_BOUND" if li == 0 else "FACE_BOUND"
                bounds.append(st.add(f"{kind}('',#{lp},{'.F.' if side else '.T.'})"))
            nm = f"F{f['label']}" + ("m" if side else "")
            faces.append(st.add(f"ADVANCED_FACE('{nm}',{st.ref(bounds)},#{surf[(side, f['surface'])]},"
                                f"{'.F.' if side else '.T.'})"))
    is_open = any(e["pair"][1] == OPEN for e in J["edges"])
    tol = max([e["tol"] for e in J["edges"]] + [1e-4])
    length = st.add(UNITS.get(units, UNITS["mm"]))
    angle = st.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    solid = st.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    mm_per = {"mm": 1.0, "cm": 10.0, "m": 1000.0}.get(units, 1.0)
    unc = st.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({fr(tol / mm_per)}),#{length},"
                 f"'distance_accuracy_value','confusion accuracy')")
    ctx = st.add(f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) "
                 f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid})) "
                 f"REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )")
    if is_open:
        sh = st.add(f"OPEN_SHELL('',{st.ref(faces)})")
        item = st.add(f"SHELL_BASED_SURFACE_MODEL('{name}',(#{sh}))")
        rep_type = "MANIFOLD_SURFACE_SHAPE_REPRESENTATION"
    else:
        sh = st.add(f"CLOSED_SHELL('',{st.ref(faces)})")
        item = st.add(f"MANIFOLD_SOLID_BREP('{name}',#{sh})")
        rep_type = "ADVANCED_BREP_SHAPE_REPRESENTATION"
    origin = st.axis2([0, 0, 0], [0, 0, 1], [1, 0, 0])
    rep = st.add(f"{rep_type}('{name}',{st.ref([origin, item])},#{ctx})")
    app = st.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    st.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
    pctx = st.add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    pdc = st.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    prod = st.add(f"PRODUCT('{name}','{name}','',(#{pctx}))")
    st.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{prod}))")
    pdf = st.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    pd = st.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdc})")
    pds = st.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
    st.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    st.write(Path(path), name)
    return {"faces": len(faces), "closed": not is_open, "bodies": 1, "parts": 1,
            "uncertainty": float(tol / mm_per), "closed_parts": int(not is_open), "open_bodies": int(is_open),
            "warnings": []}
