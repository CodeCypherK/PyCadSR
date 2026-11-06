"""Backend entry point:  ``python -m meshtocad.server``

Reads JSON commands (one per line) on stdin, writes events with ``protocol.emit``.
Long jobs run on a single job thread; ``stop``/``skip``/``ping`` are handled
immediately by the reader thread so the UI can interrupt a running search.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from . import __version__
from . import protocol as proto
from .protocol import b64_f32, b64_u32, emit, log

CREATE_NO_WINDOW = 0x08000000
# A part designed symmetric built as one half and its mirror image, for every part: off until the
# cut at the plane stops adding faces on some parts (a 2x2 pin header got 6 extra, 2 invalid).
# Mostly freeform parts get it anyway, inside curvenet.
MIRROR_SYMMETRY = False
CURVENET = True  # mostly freeform parts: faces built from curves on the shape's crests (curvenet)


# ===================================================================== worker

class WorkerHandle:
    """Owns the Julia/PySR worker process and its message pipe."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.proc: subprocess.Popen | None = None
        self.conn = None
        self.state = "stopped"
        self.info: dict = {}
        self.events: queue.Queue = queue.Queue()
        self._send_lock = threading.Lock()

    def _set(self, state: str, msg: str, **extra) -> None:
        self.state = state
        self.info.update(extra)
        emit("julia_status", state=state, msg=msg, **self.info)

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        from multiprocessing.connection import Client, Listener

        family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        address = (rf"\\.\pipe\meshtocad-{os.getpid()}-{secrets.token_hex(6)}" if os.name == "nt"
                   else str(self.workdir / f"sr-{secrets.token_hex(6)}.sock"))
        key = secrets.token_bytes(16)
        listener = Listener(address, family=family, authkey=key)
        self.conn = None
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "meshtocad.sr_worker", address, key.hex()],
            stdin=subprocess.DEVNULL, stdout=2, stderr=2, cwd=str(self.workdir),
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._set("loading", "Launching symbolic regression worker")

        def watchdog():  # unblock accept() if the worker dies before connecting
            while self.conn is None:
                if self.proc.poll() is not None:
                    try:
                        Client(address, family=family, authkey=key).close()
                    except Exception:
                        pass
                    return
                time.sleep(0.2)

        threading.Thread(target=watchdog, daemon=True).start()
        threading.Thread(target=self._accept_and_read, args=(listener,), daemon=True).start()

    def _accept_and_read(self, listener) -> None:
        try:
            conn = listener.accept()
        except Exception as exc:
            self._set("error", f"Worker connection failed: {exc!r}")
            return
        finally:
            listener.close()
        if self.proc.poll() is not None:
            self._set("error", f"Worker exited during startup (code {self.proc.returncode}). See log.")
            return
        self.conn = conn
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            if msg.get("ev") == "status":
                extra = {k: v for k, v in msg.items() if k not in ("ev", "state", "msg")}
                self._set(msg["state"], msg["msg"], **extra)
            else:
                self.events.put(msg)
        code = self.proc.wait()
        self.conn = None
        self._set("error" if code else "stopped", f"Worker exited (code {code})")
        self.events.put({"ev": "worker_exit", "code": code})

    def send(self, msg: dict) -> None:
        if self.conn is None:
            raise RuntimeError("Symbolic regression worker is not connected")
        with self._send_lock:
            self.conn.send(msg)

    def wait_ready(self, ctl, timeout: float = 1800) -> None:
        if self.state in ("stopped", "error"):
            self.start()
        t0 = time.time()
        while self.state != "ready":
            if self.state == "error":
                raise RuntimeError("Symbolic regression worker failed to start; see log")
            if ctl.stop_all:
                raise InterruptedError
            if time.time() - t0 > timeout:
                raise TimeoutError("Timed out waiting for Julia")
            time.sleep(0.2)

    def drain(self) -> None:
        while not self.events.empty():
            self.events.get_nowait()

    def shutdown(self) -> None:
        try:
            if self.conn is not None:
                self.send({"cmd": "shutdown"})
        except Exception:
            pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ===================================================================== session

class Control:
    stop_all = False
    skip = False


class Session:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.reset_mesh()

    def reset_mesh(self):
        self.mesh = None
        self.mesh_path = None
        self.diag = 1.0
        self.face_part = None
        self.part_names = []
        self.mirror = None
        self.reset_cloud()

    def reset_cloud(self):
        self.P = self.N = self.fidx = None
        self.reset_regions()

    def reset_regions(self):
        self.curvenet = None
        self.labels = None
        self.face_labels = None
        self.regions = []
        self.frames = {}
        self.results: dict[int, dict] = {}
        self.adjacency = []
        self.latex_cache: dict = {}

    def write_bin(self, name: str, arr: np.ndarray, dtype: str) -> str:
        path = self.workdir / name
        np.ascontiguousarray(arr, dtype=dtype).tofile(path)
        return str(path)


def region_json(s: Session, r) -> dict:
    prm = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.params.items()}
    fl = s.face_labels
    if fl is not None and getattr(s, "_area_cache_key", None) is not fl:
        s._area_by_label = np.bincount(fl, weights=s.mesh.area_faces)
        s._area_cache_key = fl
    area = float(s._area_by_label[r.id]) if fl is not None and r.id < len(s._area_by_label) else None
    res = s.results.get(r.id, {})
    return {
        "id": r.id, "prim": r.prim, "n_points": int(len(r.idx)), "params": prm,
        "rms": r.rms, "area": area, "frame": s.frames[r.id].to_json(),
        "status": res.get("status", "pending"), "world": res.get("world"),
        "classification": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in (res.get("classification") or {}).items()} or None,
        "exact": bool(getattr(r, "exact", False)),
        "feature": (getattr(s, "features", None) or {}).get(r.id),
    }


# ===================================================================== app

class App:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.s = Session(workdir)
        self.ctl = Control()
        self.worker = WorkerHandle(workdir)
        self.jobs: queue.Queue = queue.Queue()

    # ------------------------------------------------------------ plumbing
    def reader(self) -> None:
        for line in proto.commands():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log(f"bad command line: {line[:200]}", "error")
                continue
            cmd = msg.get("cmd")
            if cmd == "stop":
                self.ctl.stop_all = True
                log("Stop requested")
            elif cmd == "skip":
                self.ctl.skip = True
                log("Skip requested")
            elif cmd == "ping":
                emit("pong", id=msg.get("id"))
            elif cmd == "debug_stacks":  # diagnostics: where is every thread right now?
                import faulthandler
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            else:
                self.jobs.put(msg)
        self.jobs.put({"cmd": "shutdown"})

    def run(self) -> None:
        emit("hello", version=__version__, python=sys.version.split()[0], workdir=str(self.workdir),
             pid=os.getpid())
        self.worker.start()
        threading.Thread(target=self.reader, daemon=True).start()
        while True:
            msg = self.jobs.get()
            cmd = msg.get("cmd")
            if cmd == "shutdown":
                break
            handler = getattr(self, f"cmd_{cmd}", None)
            if handler is None:
                emit("error", cmd=cmd, msg=f"Unknown command {cmd!r}")
                continue
            emit("busy", busy=True, cmd=cmd)
            try:
                handler(msg.get("args") or {})
            except InterruptedError:
                log(f"{cmd} interrupted")
            except Exception as exc:
                traceback.print_exc()
                emit("error", cmd=cmd, msg=str(exc) or repr(exc), trace=traceback.format_exc())
            finally:
                emit("busy", busy=False, cmd=cmd)
        self.worker.shutdown()

    def progress(self, stage: str, frac: float, msg: str) -> None:
        emit("progress", stage=stage, frac=frac, msg=msg)

    # ------------------------------------------------------------ commands
    def cmd_restart_worker(self, args):
        self.worker.shutdown()
        self.worker.start()

    def cmd_load_mesh(self, args):
        from .cloud import load_mesh, mesh_stats

        path = args["path"]
        self.progress("mesh", 0.1, f"Loading {Path(path).name}")
        mesh = load_mesh(path)
        mirror = None
        if MIRROR_SYMMETRY and args.get("symmetry", "auto") != "off" and len(mesh.metadata.get("part_names") or []) <= 1:
            # a part designed symmetric: one half and its exact mirror image
            from .symmetry import find_mirror, symmetrize
            self.progress("mesh", 0.4, "Looking for mirror symmetry")
            m = find_mirror(mesh.vertices, mesh.faces, log=log)
            if m is not None:
                diag = float(np.linalg.norm(np.ptp(np.asarray(mesh.vertices), axis=0))) or 1.0
                mesh, mirror = symmetrize(mesh, m, tol=3e-5 * diag, log=log)
        s = self.s
        s.reset_mesh()
        s.mesh, s.mesh_path = mesh, path
        s.mirror = mirror
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        n = len(mesh.faces)
        adj = mesh.face_adjacency
        s.n_bodies, s.face_body = connected_components(
            csr_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n)), directed=False)
        # Parts: the file's own objects (an assembly), or else each separate body.
        # Bodies never cross parts (vertices are only welded within a part).
        fp = mesh.metadata.get("face_part")
        names = mesh.metadata.get("part_names") or []
        if fp is None or len(names) <= 1:
            s.face_part = s.face_body.copy()
            stem = Path(path).stem
            s.part_names = [stem] if s.n_bodies == 1 else [f"{stem}_{i + 1}" for i in range(s.n_bodies)]
        else:
            s.face_part, s.part_names = np.asarray(fp), list(names)
        st = mesh_stats(mesh)
        st["bodies"] = int(s.n_bodies)
        if mirror is not None:
            st["mirror"] = {"normal": [float(x) for x in mirror["normal"]], "d": float(mirror["d"]),
                            "p50": float(mirror["p50"]), "max": float(mirror["max"])}
        st["parts"] = len(s.part_names)
        from .cloud import mesh_is_cad
        st["cad"] = bool(mesh_is_cad(mesh))  # a CAD export: segment its triangles, no sampling needed
        # glTF stores metres by specification and FBX is read in millimetres; other
        # formats carry no unit
        suffix = Path(path).suffix.lower()
        st["units_hint"] = "m" if suffix in (".glb", ".gltf") else ("mm" if suffix == ".fbx" else None)
        s.diag = st["diag"] or 1.0
        files = {
            "positions": s.write_bin("mesh_pos.bin", mesh.vertices, "<f4"),
            "indices": s.write_bin("mesh_idx.bin", mesh.faces, "<u4"),
        }
        emit("mesh_loaded", path=path, name=Path(path).name, stats=st, files=files)

    def cmd_sample(self, args):
        from .cloud import mesh_is_noisy, refine_normals, sample_cloud

        s = self.s
        if s.mesh is None:
            raise RuntimeError("Load a mesh first")
        n = int(args.get("n_points", 60000))
        self.progress("cloud", 0.1, f"Sampling {n:,} points")
        P, N, fidx = sample_cloud(s.mesh, n, seed=int(args.get("seed", 0)))
        noisy, flat, jitter = mesh_is_noisy(s.mesh)
        if noisy:
            N, _, dev = refine_normals(P, N, force=True)
            log(f"noisy mesh (only {100 * flat:.0f} % coplanar neighbour triangles, {jitter:.1f} deg jitter): "
                f"using neighbourhood normals")
        s.reset_cloud()
        s.P, s.N, s.fidx = P, N, fidx
        files = {
            "positions": s.write_bin("cloud_pos.bin", P, "<f4"),
            "normals": s.write_bin("cloud_nrm.bin", N, "<f4"),
        }
        emit("cloud_ready", n_points=int(len(P)), files=files)

    def cmd_segment(self, args):
        from .cloud import facet_sag
        from .segmentation import (ALL_KINDS, SegParams, Segmenter, clean_face_islands, face_labels,
                                   fit_face_labels, region_adjacency, smooth_face_labels)
        from .surfaces import frame_for_region

        s = self.s
        if s.mesh is None:
            raise RuntimeError("Open a mesh first")
        strategy = args.get("strategy", "auto")
        if strategy == "auto":
            # A clean mesh (a CAD export) is segmented on its triangles; a noisy one
            # (a scan) through the point cloud.
            from .cloud import mesh_is_cad, mesh_is_noisy
            # CAD unless it clearly is a scan: noisy normals and no exactly coplanar
            # neighbours (densely meshed curved CAD parts have few coplanar pairs)
            strategy = "cad" if mesh_is_cad(s.mesh) or not mesh_is_noisy(s.mesh)[0] else "points"
            log(f"strategy: {'CAD mesh (segmenting its triangles)' if strategy == 'cad' else 'point cloud (scan or sculpted mesh)'}")
        if strategy != "cad":
            # The input is always a mesh: its vertices are the points (a CAD mesh's
            # vertices are split per region inside segment_cad instead).
            self.use_vertices()
        prm = SegParams(
            angle_deg=float(args.get("angle_deg", 30)),
            tol=float(args.get("tol", 0.0)),
            sag=facet_sag(s.mesh),
            normal_deg=float(args.get("normal_deg", 20)),
            min_frac=float(args.get("min_frac", SegParams.min_frac)),
            quality=float(args.get("quality", SegParams.quality)),
            kinds=tuple(k for k in args.get("kinds", ALL_KINDS) if k in ALL_KINDS),
        )
        t0 = time.perf_counter()
        if strategy == "cad":
            return self.segment_cad(prm, t0)
        if strategy == "coarse":
            return self.segment_coarse_to_fine(prm, len(s.P), t0)
        groups = s.face_body[s.fidx] if getattr(s, "n_bodies", 1) > 1 else None
        seg = Segmenter(s.P, s.N, prm, progress=lambda f, m: self.progress("segment", f, m), groups=groups)
        regions, labels = seg.run()
        s.reset_regions()
        s.regions, s.labels = regions, labels
        fl = face_labels(len(s.mesh.faces), s.fidx, labels, s.mesh.triangles_center, s.P)
        if len(s.mesh.faces) > len(s.P):  # dense (scan-like) mesh: labels from sparse samples zigzag
            fl = smooth_face_labels(s.mesh.face_adjacency, fl)
        else:  # coarse (CAD) mesh: decide border triangles by which surface they lie on
            fl = fit_face_labels(s.mesh, fl, regions, seg.eps)
        s.face_labels = clean_face_islands(s.mesh.face_adjacency, fl)
        s.adjacency = region_adjacency(labels, seg)
        s.seg_eps, s.noise = seg.eps, seg.noise
        s.strategy = "points"
        log(f"segmentation tolerance {seg.eps:.4g} (noise {seg.noise:.3g}, facet sag {prm.sag:.3g}, "
            f"{100 * seg.eps / s.diag:.2f} % of size)")
        for r in regions:
            s.frames[r.id] = frame_for_region(r, s.P, s.N)
        files = {
            "labels": s.write_bin("labels.bin", labels, "<u4"),
            "face_labels": s.write_bin("face_labels.bin", s.face_labels, "<u4"),
        }
        emit("segments_ready", regions=[region_json(s, r) for r in regions], adjacency=s.adjacency,
             files=files, t=time.perf_counter() - t0, tolerance=seg.eps)

    def use_vertices(self, cap: int = 400_000) -> None:
        """Points = the mesh's own vertices, with their normals (a very dense scan is thinned)."""
        from .cloud import mesh_is_noisy, refine_normals

        s = self.s
        mesh = s.mesh
        V = np.asarray(mesh.vertices, float)
        N = np.asarray(mesh.vertex_normals, float)
        vf = mesh.vertex_faces
        fidx = np.where(vf[:, 0] >= 0, vf[:, 0], 0).astype(np.int64)
        keep = np.arange(len(V))
        if len(V) > cap:
            keep = np.sort(np.random.default_rng(0).choice(len(V), cap, replace=False))
        P, N, fidx = V[keep], N[keep], fidx[keep]
        if mesh_is_noisy(mesh)[0]:
            N = refine_normals(P, N, force=True)[0]  # (normals, used, deviation)
        s.reset_cloud()
        s.P, s.N, s.fidx = P, N, fidx
        emit("cloud_ready", n_points=int(len(P)), files={
            "positions": s.write_bin("cloud_pos.bin", P, "<f4"),
            "normals": s.write_bin("cloud_nrm.bin", N, "<f4")})

    def segment_cad(self, prm, t0: float) -> None:
        """A CAD export: segment its triangles (meshseg.py); each region's points are
        its own mesh vertices, which lie exactly on the true surface."""
        import sympy as sp

        from .meshseg import mesh_segment
        from .segmentation import Region
        from .surfaces import Equation, frame_for_region

        s = self.s
        mesh = s.mesh
        prog = lambda f, m: self.progress("segment", f, m)
        groups = s.face_body if getattr(s, "n_bodies", 1) > 1 else None
        if getattr(s, "mirror", None) is not None:
            from .symmetry import mirror_segment
            fl, regs, tol = mirror_segment(mesh, s.mirror, prm.angle_deg, groups, sag=prm.sag, progress=prog, log=log)
        else:
            fl, regs, tol = mesh_segment(mesh, prm.angle_deg, groups, sag=prm.sag, progress=prog)
        cn = None
        if CURVENET and groups is None:
            from . import curvenet
            share = curvenet.freeform_share(regs, mesh)
            if share >= curvenet.FREEFORM_SHARE:
                log(f"{share:.0%} of the surface is freeform: building its faces from curves")
                try:
                    cn = curvenet.build(mesh.vertices, mesh.faces, getattr(s, "mirror", None),
                                        curvenet.mm_scale(s.mesh_path, s.diag), log=log,
                                        progress=lambda f, m: prog(0.3 + 0.6 * f, m))
                    lab = np.asarray(cn["labels"], np.int64)
                    regs = [("freeform", {}, np.where(lab == k)[0]) for k in range(int(lab.max()) + 1)]
                    fl = lab
                except Exception as exc:  # the segmentation still exports
                    log(f"faces from curves failed ({exc}); keeping the segmentation", "warn")
                    cn = None
        # points: every region's vertices (a border vertex appears once per region it bounds),
        # with the region's own normal there (mean of its faces at that vertex)
        V, F, FN = mesh.vertices, mesh.faces, mesh.face_normals
        Ps, Ns, fids, labs = [], [], [], []
        for k, (_, _, faces) in enumerate(regs):
            vs = F[faces].ravel()
            fr = np.repeat(faces, 3)
            uv, inv = np.unique(vs, return_inverse=True)
            nsum = np.zeros((len(uv), 3))
            np.add.at(nsum, inv, FN[fr])
            nsum /= np.maximum(np.linalg.norm(nsum, axis=1, keepdims=True), 1e-300)
            first = np.zeros(len(uv), np.int64)
            first[inv[::-1]] = fr[::-1]
            Ps.append(V[uv])
            Ns.append(nsum)
            fids.append(first)
            labs.append(np.full(len(uv), k, np.int64))
        s.reset_regions()
        s.curvenet = cn
        s.P, s.N, s.fidx = np.vstack(Ps), np.vstack(Ns), np.concatenate(fids)
        labels = np.concatenate(labs)
        emit("cloud_ready", n_points=int(len(s.P)), files={
            "positions": s.write_bin("cloud_pos.bin", s.P, "<f4"),
            "normals": s.write_bin("cloud_nrm.bin", s.N, "<f4")})
        regions = []
        for k, (kind, p, faces) in enumerate(regs):
            idx = np.where(labels == k)[0]
            rms = 0.0
            if kind != "freeform" and len(idx):
                from .segmentation import residual
                d, _ = residual(kind, p, s.P[idx], s.N[idx])
                rms = float(np.sqrt(np.mean(d ** 2)))
            regions.append(Region(k, idx, kind, p, rms, exact=(kind == "plane")))
        s.regions, s.labels, s.face_labels = regions, labels, fl
        adj = mesh.face_adjacency
        la, lb = fl[adj[:, 0]], fl[adj[:, 1]]
        m = la != lb
        s.adjacency = [tuple(map(int, q)) for q in np.unique(np.sort(np.c_[la[m], lb[m]], axis=1), axis=0)]
        s.seg_eps, s.noise = tol, 0.0
        s.strategy = "cad"
        prog(0.95, "Frames for every region")
        from .cad import fallback_equation
        from .surfaces import classify_region, torus_equation

        n_exact = 0
        for r in regions:
            fr = s.frames[r.id] = frame_for_region(r, s.P, s.N)
            if r.exact:  # an exact plane: w = 0 in its own frame
                n = r.params["normal"]
                s.results[r.id] = {"status": "exact", "eq": Equation(sp.Integer(0), fr), "hof": [],
                                   "history": [], "selected": None, "rms": r.rms,
                                   "classification": {"type": "plane", "offset": 0.0},
                                   "world": plane_text(n, float(n @ r.params["point"])), "world_latex": None}
                n_exact += 1
            elif r.prim in ("cylinder", "cone", "sphere", "torus") and r.rms <= 2 * tol and len(r.idx) >= 3:
                # The fit already is the math: r = R for a cylinder, r = a h + b for a
                # cone, ... exact on the vertices. Only freeform faces need a search.
                X, y = fr.to_param(s.P[r.idx])
                eq = (torus_equation(r, fr, s.P[r.idx]) if r.prim == "torus" and fr.kind == "cylindrical"
                      else fallback_equation(r.prim, fr, X, y))
                rms_eq = float(np.sqrt(np.nanmean(((eq(X) - y) * fr.scale) ** 2)))
                if np.isfinite(rms_eq) and rms_eq <= 2 * tol:
                    world, world_tex = eq.local_world()
                    s.results[r.id] = {"status": "exact", "eq": eq, "hof": [], "history": [], "selected": None,
                                       "rms": rms_eq, "classification": classify_region(eq, rms_eq, r, s.diag),
                                       "world": world, "world_latex": world_tex}
                    r.exact = True
                    n_exact += 1
                elif r.prim == "torus" and fr.kind == "cylindrical":
                    # A torus over the top of its tube (a pipe elbow) is no function
                    # r = f(h); the torus fit itself is exact on the vertices: keep it
                    from .surfaces import ExactTorus
                    p = r.params
                    eq = ExactTorus(fr, p["center"], p["axis"], p["R"], p["r"])
                    s.results[r.id] = {"status": "exact", "eq": eq, "hof": [], "history": [], "selected": None,
                                       "rms": r.rms, "classification": {"type": "torus", "R": float(p["R"]),
                                                                         "r": float(p["r"])},
                                       "world": torus_text(p), "world_latex": None}
                    r.exact = True
                    n_exact += 1
        # The vertex tolerance came from a local noise estimate, which a coarse mesh of
        # curved faces inflates: a quadratic cannot follow a cylinder over two rings of
        # 10-degree strips, so a PVC fitting whose cylinders fit their vertices to 1e-10
        # got 2e-4 of its size and lost 8 % of its area to loose edges. When the exact
        # surfaces fit far closer, the rest of the pipeline (export) works to that
        # precision instead, never below the usual floor; the exact faces stay exact.
        ex_rms = [r.rms for r in regions if r.prim in ("plane", "cylinder", "cone", "sphere", "torus")
                  and r.exact and len(r.idx) >= 3]
        if ex_rms:
            s.seg_eps = min(tol, max(3e-5 * s.diag, 2 * float(max(ex_rms))))
        # Thread flanks: a freeform face that is a profile swept along a helix has its
        # exact surface already (the export writes it), and the list shows it as one
        from .helix import HelixSurface, fit_helix
        tol_x = max(20 * s.seg_eps, 1e-6 * s.diag)
        for r in regions:
            if r.prim == "freeform" and "helix_lead" in r.params and not r.exact:
                # found whole when segmenting (an auger, a worm): its own profile curve
                hx = HelixSurface.from_params(s.frames[r.id], r.params)
                world, world_tex = hx.local_world()
                s.results[r.id] = {"status": "exact", "eq": hx, "hof": [], "history": [], "selected": None,
                                   "rms": hx.rms, "classification": {"type": "helix", "lead": float(hx.L)},
                                   "world": world, "world_latex": world_tex}
                r.exact = True
                n_exact += 1
                continue
            if r.prim != "freeform" or r.exact or "tube_r" in r.params or len(r.idx) < 30:
                continue
            try:
                hx = fit_helix(s.P[r.idx], s.N[r.idx], s.frames[r.id], tol_x)
            except Exception:
                hx = None
            if hx is None or hx.rms > 0.002 * tol_x:
                continue
            world, world_tex = hx.local_world()
            s.results[r.id] = {"status": "exact", "eq": hx, "hof": [], "history": [], "selected": None,
                               "rms": hx.rms, "classification": {"type": "helix", "lead": float(hx.L)},
                               "world": world, "world_latex": world_tex}
            r.exact = True
            n_exact += 1
        # Features from the exact surfaces: a cylinder whose surface faces its axis is a
        # hole, one facing away a boss or shaft (normals are outward after the winding fix).
        s.features = {}
        for r in regions:
            if r.prim != "cylinder" or len(r.idx) < 3:
                continue
            p = r.params
            a = np.asarray(p["axis"], float)
            a = a / (np.linalg.norm(a) or 1.0)
            v = s.P[r.idx] - p["point"]
            h = v @ a
            rad = v - np.outer(h, a)
            rn = rad / np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-300)
            inward = float(np.mean(np.einsum("ij,ij->i", s.N[r.idx], rn))) < 0
            s.features[r.id] = {"kind": "hole" if inward else "boss", "diameter": float(2 * p["radius"]),
                                "length": float(np.ptp(h))}
        n_holes = sum(1 for f in s.features.values() if f["kind"] == "hole")
        if s.features:
            sizes = sorted({round(f["diameter"], 6) for f in s.features.values() if f["kind"] == "hole"})
            log(f"features: {n_holes} holes ({len(sizes)} distinct diameters), "
                f"{len(s.features) - n_holes} bosses/shafts")
        kinds = {}
        for r in regions:
            kinds[r.prim] = kinds.get(r.prim, 0) + 1
        log(f"CAD mesh: {len(regions)} surfaces {kinds} from {len(F):,} triangles in "
            f"{time.perf_counter() - t0:.1f} s (vertex tolerance {tol:.3g}); {n_exact} have their exact "
            f"equation already, {len(regions) - n_exact} need discovery")
        files = {"labels": s.write_bin("labels.bin", labels, "<u4"),
                 "face_labels": s.write_bin("face_labels.bin", fl, "<u4")}
        emit("segments_ready", regions=[region_json(s, r) for r in regions], adjacency=s.adjacency,
             files=files, t=time.perf_counter() - t0, tolerance=tol, strategy="cad")

    @staticmethod
    def _split_freeform(seg, regions, axes):
        """Freeform detail that is neither a height field nor a surface of revolution
        (a sculpted relief, an ornament) is split by which side of the overall box it
        faces, so each piece gets a bounded height-field equation."""
        from . import hierarchy as H
        from .segmentation import Region
        from .surfaces import _freeform_kind, revolution_axis

        nbrs = np.where(seg.knn_d < 4 * seg.spacing, seg.knn_idx, np.arange(seg.n)[:, None])
        min_pts = max(60, seg.min_pts // 4)
        out, n_split, n_pieces = [], 0, 0
        for r in regions:
            if r.prim == "freeform" and "normal" not in r.params and len(r.idx) >= 2 * min_pts:
                Pr, Nr = seg.P[r.idx], seg.N[r.idx]
                ext = float(np.linalg.norm(Pr.max(0) - Pr.min(0))) or 1.0
                if _freeform_kind(Nr) != "height" and revolution_axis(Pr, Nr, ext) is None:
                    parts = H.cube_map_split(seg.N, nbrs, r.idx, axes, min_pts, P=seg.P)
                    out += [Region(0, part, "freeform", {"normal": d}) for part, d in parts]
                    n_split += 1
                    n_pieces += len(parts)
                    continue
            out.append(r)
        if n_split:
            log(f"split {n_split} freeform regions into {n_pieces} pieces, each facing one side of the box")
        return out

    def segment_coarse_to_fine(self, prm, budget: int, t0: float) -> None:
        """Whole shape -> directions -> planes -> details (see hierarchy.py)."""
        from . import hierarchy as H
        from .segmentation import Region, Segmenter, clean_face_islands, face_labels, fit_face_labels
        from .surfaces import Equation, frame_for_region
        import sympy as sp

        s = self.s
        mesh = s.mesh
        prog = lambda f, m: self.progress("segment", f, m)
        h = H.build(mesh, prm.sag, progress=prog)
        s.hierarchy = h
        V0, F0 = H.box_mesh(h)
        emit("refine_level", level=0, name="Overall shape", expression=h.expression, latex=h.latex,
             summary=f"box {2 * h.half[0]:.4g} x {2 * h.half[1]:.4g} x {2 * h.half[2]:.4g}",
             preview={"positions": b64_f32(V0), "indices": b64_u32(F0)})
        A = mesh.area_faces
        total = float(A.sum()) or 1.0
        cov1 = 1 - A[h.level_labels[1] == len(h.directions)].sum() / total
        emit("refine_level", level=1, name="Dominant directions", count=len(h.directions),
             summary=f"{len(h.directions)} directions cover {100 * cov1:.1f} % of the surface",
             files={"face_labels": s.write_bin("level1.bin", h.level_labels[1], "<u4")})
        planar = h.plane_of_face >= 0
        cov2 = A[planar].sum() / total
        big = sum(1 for p in h.planes if p.area > 1e-4 * total)
        emit("refine_level", level=2, name="Planes at each depth", count=len(h.planes),
             summary=f"{len(h.planes)} exact planes ({big} major) explain {100 * cov2:.1f} % of the surface",
             files={"face_labels": s.write_bin("level2.bin", h.level_labels[2], "<u4")})

        # re-sample: a few points per plane (they are exact already), dense on the details
        prog(0.35, "Level 3: sampling the remaining detail densely")
        rng = np.random.default_rng(0)
        a_plane, a_left = float(A[planar].sum()), float(A[~planar].sum())
        n_left = int(budget * 4 * a_left / max(a_plane + 4 * a_left, 1e-300))
        n_plane = max(budget - n_left, 3 * len(h.planes))
        pf = np.where(planar)[0]
        cnt = H.allocate(A[pf], n_plane, None, rng)
        # every plane keeps at least 3 samples (on its largest triangle)
        per_plane = np.bincount(h.plane_of_face[pf], weights=cnt, minlength=len(h.planes))
        for i in np.where(per_plane < 3)[0]:
            f = h.planes[i].faces
            j = f[np.argmax(A[f])]
            cnt[np.searchsorted(pf, j)] += int(3 - per_plane[i])
        P1, f1 = H.sample_faces(mesh, pf, cnt, rng)
        lf = np.where(~planar)[0]
        P2, f2 = H.sample_faces(mesh, lf, H.allocate(A[lf], n_left, None, rng), rng) if len(lf) else \
            (np.zeros((0, 3)), np.zeros(0, np.int64))
        P = np.vstack([P1, P2])
        fidx = np.r_[f1, f2].astype(np.int64)
        N = mesh.face_normals[fidx]
        s.P, s.N, s.fidx = P, N, fidx
        emit("cloud_ready", n_points=int(len(P)), files={
            "positions": s.write_bin("cloud_pos.bin", P, "<f4"),
            "normals": s.write_bin("cloud_nrm.bin", N, "<f4")})

        # regions: exact planes first ...
        regions: list[Region] = []
        labels = np.full(len(P), -1, np.int64)
        pid = h.plane_of_face[fidx]
        order = np.argsort(pid, kind="stable")
        starts = np.searchsorted(pid[order], np.arange(len(h.planes) + 1))
        plane_region = np.full(len(h.planes), -1, np.int64)
        for i, p in enumerate(h.planes):
            idx = np.sort(order[starts[i]:starts[i + 1]])
            if len(idx) == 0:
                continue
            rid = len(regions)
            regions.append(Region(rid, idx, "plane", {"point": p.normal * p.offset, "normal": p.normal},
                                  rms=0.0, exact=True))
            labels[idx] = rid
            plane_region[i] = rid
        n_planes = len(regions)
        # ... then the detail, segmented by the point method. Planes stay candidates
        # there: faces tilted away from every dominant direction are still flat.
        left = np.where(pid < 0)[0]
        seg_eps = max(1e-4 * s.diag, 3 * prm.sag)
        if len(left) >= 30:
            prog(0.45, f"Level 3: primitives in the remaining detail ({len(left):,} points)")
            groups = s.face_body[fidx[left]] if getattr(s, "n_bodies", 1) > 1 else None
            seg = Segmenter(P[left], N[left], prm, progress=lambda f, m: prog(0.45 + 0.5 * f, m), groups=groups)
            r_left, l_left = seg.run()
            seg_eps = seg.eps
            r_left = self._split_freeform(seg, r_left, h.frame)
            for r in r_left:
                rid = len(regions)
                regions.append(Region(rid, left[r.idx], r.prim, r.params, r.rms))
                labels[left[r.idx]] = rid
        emit("refine_level", level=3, name="Details", count=len(regions) - n_planes,
             summary=f"{len(regions) - n_planes} detail regions (pipes, rounded edges, freeform) "
                     f"on {100 * (1 - cov2):.1f} % of the surface")

        # face labels: planes exactly from the mesh, details by their samples
        fl = np.full(len(mesh.faces), -1, np.int64)
        fl[planar] = plane_region[h.plane_of_face[planar]]
        if len(left) and (fl < 0).any():
            fd = face_labels(len(mesh.faces), fidx[left], labels[left], mesh.triangles_center, P[left])
            fl[fl < 0] = fd[fl < 0]
            fl = fit_face_labels(mesh, fl, regions, seg_eps, fixed=planar)  # planes stay exact
        fl[fl < 0] = 0
        s.reset_regions()
        s.hierarchy = h
        s.regions, s.labels = regions, labels
        s.face_labels = fl
        adj = mesh.face_adjacency
        la, lb = fl[adj[:, 0]], fl[adj[:, 1]]
        m = la != lb
        s.adjacency = [tuple(map(int, p)) for p in np.unique(np.sort(np.c_[la[m], lb[m]], axis=1), axis=0)]
        s.seg_eps, s.noise = seg_eps, 0.0
        prog(0.97, "Frames for every region")
        for r in regions:
            s.frames[r.id] = frame_for_region(r, s.P, s.N)
            if r.exact:  # the equation of an exact plane is known: w = 0 in its own frame
                eq = Equation(sp.Integer(0), s.frames[r.id])
                s.results[r.id] = {"status": "exact", "eq": eq, "hof": [], "history": [], "selected": None,
                                   "rms": 0.0, "classification": {"type": "plane", "offset": 0.0},
                                   "world": plane_text(r.params["normal"], float(r.params["normal"] @ r.params["point"])),
                                   "world_latex": None}
        files = {"labels": s.write_bin("labels.bin", labels, "<u4"),
                 "face_labels": s.write_bin("face_labels.bin", s.face_labels, "<u4")}
        log(f"coarse-to-fine: {n_planes} exact planes + {len(regions) - n_planes} detail regions "
            f"in {time.perf_counter() - t0:.1f} s")
        emit("segments_ready", regions=[region_json(s, r) for r in regions], adjacency=s.adjacency,
             files=files, t=time.perf_counter() - t0, tolerance=seg_eps, strategy="coarse")

    def cmd_set_region_kind(self, args):
        from .surfaces import frame_for_region

        s = self.s
        rid, kind = int(args["region"]), args["kind"]
        s.frames[rid] = frame_for_region(s.regions[rid], s.P, s.N, kind)
        s.results.pop(rid, None)
        s.latex_cache = {k: v for k, v in s.latex_cache.items() if k[0] != rid}
        emit("region_updated", region=region_json(s, s.regions[rid]))

    def cmd_discover(self, args):
        s = self.s
        if not s.regions:
            raise RuntimeError("Identify surface regions first")
        self.ctl.stop_all = False
        self.ctl.skip = False
        ids = args.get("regions") or [r.id for r in s.regions
                                      if (s.results.get(r.id) or {}).get("status") != "exact"]
        settings = args.get("settings") or {}
        emit("discover_start", regions=ids)
        self.progress("discover", 0.0, "Waiting for the symbolic regression engine")
        self.worker.wait_ready(self.ctl)
        self.worker.drain()
        for k, rid in enumerate(ids):
            if self.ctl.stop_all:
                break
            self.progress("discover", k / len(ids), f"Region {rid} ({k + 1}/{len(ids)})")
            if len(s.regions[int(rid)].idx) < 30:
                emit("sr_done", region=int(rid), reason="too few points", status="skipped",
                     summary={"status": "skipped"})
                continue
            self.discover_region(int(rid), settings)
        emit("discover_done", stopped=self.ctl.stop_all)

    def discover_region(self, rid: int, settings: dict) -> None:
        from .surfaces import seed_guesses

        s = self.s
        region = s.regions[rid]
        frame = s.frames[rid]
        X_all, y_all = frame.to_param(s.P[region.idx])
        ok = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all)
        rng = np.random.default_rng((int(region.idx.min()) if len(region.idx) else 0, len(region.idx)))
        pool = np.where(ok)[0]
        max_points = int(settings.get("max_points", 600))
        sub = rng.choice(pool, size=min(len(pool), max_points), replace=False)
        tol = float(settings.get("tolerance", 0.004))
        iters = int(settings.get("iterations", 40))
        job_id = uuid.uuid4().hex[:8]
        # Never demand more than the data can give: the target error is floored at a
        # half the segmentation tolerance (mesh noise / facet sag) and at the
        # primitive's own fit, otherwise small fillets never "converge" and the
        # search wanders into wiggly overfits.
        floor = 0.5 * getattr(s, "seg_eps", 0.0)  # ~2x the noise level on scans
        target = max(tol, floor / frame.scale) ** 2
        settings = dict(settings)
        if region.prim == "freeform":  # nothing to seed from: search harder and longer
            settings["populations"] = 2 * int(settings.get("populations", 15))
            iters *= 2
        unary = list(settings.get("unary_operators") or ["sin", "cos", "sqrt", "square"])
        guesses = seed_guesses(region, frame, X_all[sub], y_all[sub], unary) if settings.get("seed", True) else []
        self.ctl.skip = False
        self.worker.send({
            "cmd": "fit", "job_id": job_id, "X": X_all[sub], "y": y_all[sub],
            "var_names": frame.var_names, "settings": settings, "max_iterations": iters,
            "min_iterations": int(settings.get("min_iterations", 2)), "target_loss": target,
            "output_directory": str(self.workdir / "sr"), "run_id": f"region{rid}_{job_id}",
            "guesses": guesses,
        })
        res = {"status": "running", "hof": [], "history": [], "selected": None, "eq": None,
               "target_loss": target, "job_id": job_id}
        s.results[rid] = res
        emit("sr_start", region=rid, job_id=job_id, n_train=int(len(sub)), n_points=int(len(region.idx)),
             max_iterations=iters, target_loss=target, frame=frame.to_json(), prim=region.prim, guesses=guesses)
        stop_sent = False
        while True:
            if (self.ctl.stop_all or self.ctl.skip) and not stop_sent:
                self.worker.send({"cmd": "stop", "job_id": job_id,
                                  "reason": "stopped" if self.ctl.stop_all else "skipped"})
                stop_sent = True
            try:
                msg = self.worker.events.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg.get("ev") == "worker_exit":
                res["status"] = "error"
                raise RuntimeError("Symbolic regression worker crashed; see log")
            if msg.get("job_id") != job_id:
                continue
            if msg["ev"] == "iter":
                hof = msg["hof"]
                if not hof:
                    continue
                res["hof"] = hof
                sel = self.pick_equation(rid, hof, target, X_all, y_all)
                manual = res.get("manual_complexity")
                if manual is not None:  # the user picked an equation; keep following that complexity
                    sel = next((i for i, h in enumerate(hof) if h["complexity"] == manual), sel)
                res["selected"] = sel
                best = min(h["loss"] for h in hof)
                res["history"].append({"iteration": msg["iteration"], "best_loss": best,
                                       "selected_loss": hof[sel]["loss"], "dt": msg["dt"],
                                       "elapsed": msg["elapsed"]})
                payload = self.evaluate(rid, hof, sel, X_all, y_all)
                emit("sr_iter", region=rid, iteration=msg["iteration"], max_iterations=iters,
                     dt=msg["dt"], elapsed=msg["elapsed"], best_loss=best, history=res["history"][-1],
                     hof=[{k: h[k] for k in ("complexity", "loss", "score", "equation")} | {"latex": h.get("latex")}
                          for h in hof],
                     selected=sel, **payload)
            elif msg["ev"] == "done":
                res["status"] = msg["reason"] if msg["reason"] != "max_iterations" else "finished"
                if res["hof"] and res["selected"] is not None:
                    payload = self.evaluate(rid, res["hof"], res["selected"], X_all, y_all, refine=True)
                    emit("sr_selected", region=rid, selected=res["selected"], summary=self.summary(rid), **payload)
                if msg.get("error"):
                    log(f"Region {rid}: {msg['error']}", "error")
                emit("sr_done", region=rid, reason=msg["reason"], status=res["status"],
                     elapsed=msg.get("elapsed"), summary=self.summary(rid))
                break

    def pick_equation(self, rid: int, hof: list[dict], target: float, X_all, y_all) -> int:
        """choose_equation(), judged on all region points and the whole domain.

        PySR scores on a training subsample. An expression like cos(0.48/theta) can
        look fine there and still blow up between samples, which wrecks the CAD
        surface built from it. Losses are recomputed on (a large sample of) every
        point, and expressions that are not finite across the domain are skipped.
        """
        from .surfaces import Equation, param_domain

        frame = self.s.frames[rid]
        cache = self.s.results[rid].setdefault("_checks", {})
        ok = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all)
        Xe, ye = X_all[ok], y_all[ok]
        if len(ye) > 8000:
            k = np.random.default_rng(1).choice(len(ye), 8000, replace=False)
            Xe, ye = Xe[k], ye[k]
        # centres of grid cells that contain data: probes between samples without
        # stepping past the region's edge (where e.g. a torus profile is undefined)
        ga, gb, _, _ = param_domain(frame, Xe, target_cells=900)
        ia = np.clip(np.searchsorted(ga, Xe[:, 0]) - 1, 0, len(ga) - 2)
        ib = np.clip(np.searchsorted(gb, Xe[:, 1]) - 1, 0, len(gb) - 2)
        cid = ia * (len(gb) + 1) + ib
        uniq, inv, cnt = np.unique(cid, return_inverse=True, return_counts=True)
        # centroid of the samples in each cell: always inside the data, unlike the
        # geometric centre of an edge cell
        grid = np.c_[np.bincount(inv, Xe[:, 0]) / cnt, np.bincount(inv, Xe[:, 1]) / cnt]
        checked = []
        for h in hof:
            key = h["sympy"]
            if key not in cache:
                try:
                    eq = Equation(key, frame)
                    pred = eq(Xe)
                    fin = np.isfinite(pred)
                    # tolerate a sliver of samples just past a natural domain edge (a
                    # torus profile ends at its tube); reject real singularities
                    gv = eq(grid)
                    span = float(np.ptp(ye)) + float(np.abs(ye).max()) + 1e-9
                    ok = (fin.mean() >= 0.99 and np.isfinite(gv).mean() >= 0.97
                          and float(np.nanmax(np.abs(gv - np.median(ye)))) <= 10 * span)
                    loss = float(np.mean((pred[fin] - ye[fin]) ** 2)) if fin.any() else float("inf")
                    cache[key] = (bool(ok and np.isfinite(loss)), loss)
                except Exception:
                    cache[key] = (False, float("inf"))
            checked.append({**h, "loss": cache[key][1]} if cache[key][0] else None)
        good = [(i, h) for i, h in enumerate(checked) if h is not None]
        if not good:
            return choose_equation(hof, target)
        sub = [h for _, h in good]
        return good[choose_equation(sub, target)][0]

    def evaluate(self, rid: int, hof: list[dict], sel: int, X_all=None, y_all=None, refine: bool = False) -> dict:
        from .surfaces import Equation, classify_region, polish, preview_mesh

        s = self.s
        frame = s.frames[rid]
        if X_all is None:
            X_all, y_all = frame.to_param(s.P[s.regions[rid].idx])
        cache = self.s.latex_cache
        for h in hof:
            key = (rid, h["sympy"])
            if key not in cache:
                try:
                    cache[key] = Equation(h["sympy"], frame).latex()
                except Exception:
                    cache[key] = None
            h["latex"] = cache[key]
        eq = Equation(hof[sel]["sympy"], frame)
        if refine:
            eq = polish(eq, X_all, y_all)
        s.results[rid]["eq"] = eq
        s.results[rid]["refined"] = refine
        pred = eq(X_all)
        err = (pred - y_all) * frame.scale
        fin = np.isfinite(err)
        rms = float(np.sqrt(np.mean(err[fin] ** 2))) if fin.any() else float("nan")
        V, F = preview_mesh(eq, X_all)
        world_txt, world_tex = eq.local_world()
        cls = classify_region(eq, rms, s.regions[rid], s.diag)
        s.results[rid].update(rms=rms, classification=cls, world=world_txt, world_latex=world_tex)
        return {
            "rms": rms, "rms_rel": rms / s.diag,
            "max_err": float(np.max(np.abs(err[fin]))) if fin.any() else None,
            "residuals": b64_f32(np.where(fin, err, np.nan)),
            "preview": {"positions": b64_f32(V), "indices": b64_u32(F)},
            "latex": eq.latex(), "world": world_txt, "world_latex": world_tex,
            "classification": cls, "refined": refine,
        }

    def summary(self, rid: int) -> dict:
        r = self.s.results.get(rid, {})
        return {k: r.get(k) for k in ("status", "rms", "classification", "world", "world_latex", "selected")}

    def cmd_select_equation(self, args):
        s = self.s
        rid, idx = int(args["region"]), int(args["index"])
        res = s.results.get(rid)
        if not res or not res.get("hof"):
            raise RuntimeError("No equations for this region yet")
        res["manual_complexity"] = res["hof"][idx]["complexity"]
        res["selected"] = idx
        payload = self.evaluate(rid, res["hof"], idx, refine=res.get("status") != "running")
        emit("sr_selected", region=rid, selected=idx, summary=self.summary(rid), **payload)

    def cmd_export(self, args):
        from .cad import export_all

        s = self.s
        if not s.regions:
            raise RuntimeError("Identify surface regions before exporting")
        missing = [r.id for r in s.regions if (s.results.get(r.id) or {}).get("eq") is None]
        if missing:
            log(f"{len(missing)} region(s) have no discovered equation; using least-squares stand-ins", "warn")
        out_dir = Path(args["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        result = export_all(s, out_dir, args.get("name") or "meshtocad",
                            args.get("formats") or ["step", "stl", "json"], units=args.get("units", "mm"),
                            progress=lambda f, m: self.progress("export", f, m))
        emit("exported", out_dir=str(out_dir), **result)


def plane_text(n: np.ndarray, d: float) -> str:
    """Readable world equation of a plane: 'z = 2.7487' or '0.923 y - 0.386 z = 1.2'."""
    k = int(np.argmax(np.abs(n)))
    if abs(n[k]) > 1 - 1e-9:
        return f"{'xyz'[k]} = {d * np.sign(n[k]):.6g}"
    terms = [f"{c:+.4g} {v}" for c, v in zip(n, "xyz") if abs(c) > 1e-6]
    return (" ".join(terms) + f" = {d:.6g}").lstrip("+")


def torus_text(p: dict) -> str:
    """Readable torus: '(ρ - 0.15)² + h² = 0.075²', ρ and h about its axis through its centre."""
    c = np.asarray(p["center"], float)
    a = np.asarray(p["axis"], float)
    a = a / (np.linalg.norm(a) or 1.0)
    fmt3 = lambda v: "(" + ", ".join(f"{x:.6g}" for x in v) + ")"
    return f"(ρ - {float(p['R']):.6g})² + h² = {float(p['r']):.6g}²  · centre {fmt3(c)}, axis {fmt3(a)}"


def choose_equation(hof: list[dict], target_loss: float) -> int:
    """Simplest equation within tolerance, else PySR's 'best' rule."""
    within = [i for i, h in enumerate(hof) if h["loss"] <= target_loss]
    if within:
        return min(within, key=lambda i: hof[i]["complexity"])
    best = min(h["loss"] for h in hof)
    cands = [i for i, h in enumerate(hof) if h["loss"] <= 1.5 * best]
    return max(cands, key=lambda i: hof[i]["score"])


def main() -> None:
    proto.init()
    workdir = Path(os.environ.get("MESHTOCAD_WORKDIR") or (Path.cwd() / "work"))
    workdir.mkdir(parents=True, exist_ok=True)
    App(workdir).run()


if __name__ == "__main__":
    main()
