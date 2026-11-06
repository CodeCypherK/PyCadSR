"""Symbolic-regression worker process (owns Julia via PySR).

Started by the coordinator as ``python -m meshtocad.sr_worker <pipe> <authkey>``.
Running Julia in its own process keeps the coordinator responsive while the
search runs, and lets us stop a search between iterations.

Messages are pickled dicts over a multiprocessing connection:

    worker -> coordinator:  {"ev": "status", "state": ..., "msg": ...}
                            {"ev": "iter", "job_id", "iteration", "dt", "hof": [...]}
                            {"ev": "done", "job_id", "reason", "error"?}
    coordinator -> worker:  {"cmd": "fit", "job_id", "X", "y", "var_names", "settings", ...}
                            {"cmd": "stop", "job_id"} | {"cmd": "shutdown"}
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from multiprocessing.connection import Client

import numpy as np

DEFAULT_SETTINGS = {
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos", "sqrt", "square"],
    "maxsize": 20,
    "populations": 15,
    "population_size": 33,
    "ncycles_per_iteration": 120,
    "parsimony": 0.0,
}

TRIG = ("sin", "cos", "tan")


def _nested(unary: list[str]) -> dict | None:
    nested: dict = {}
    trig = [u for u in unary if u in TRIG]
    for t in trig:
        nested[t] = {tt: 0 for tt in trig}
    for f in ("sqrt", "exp", "log", "square"):
        if f in unary:
            nested[f] = {f: 1 if f == "square" else 0}
    return nested or None


def make_model(settings: dict, output_directory: str | None, run_id: str | None, guesses=None):
    from pysr import PySRRegressor

    s = {**DEFAULT_SETTINGS, **(settings or {})}
    unary = list(s["unary_operators"])
    return PySRRegressor(
        niterations=1,
        warm_start=True,
        binary_operators=list(s["binary_operators"]),
        unary_operators=unary,
        nested_constraints=_nested(unary),
        maxsize=int(s["maxsize"]),
        populations=int(s["populations"]),
        population_size=int(s["population_size"]),
        ncycles_per_iteration=int(s["ncycles_per_iteration"]),
        parsimony=float(s["parsimony"]),
        model_selection="best",
        progress=False,
        verbosity=0,
        precision=32,
        parallelism="multithreading",
        output_directory=output_directory,
        run_id=run_id,
        guesses=list(guesses) if guesses else None,
    )


def hall_of_fame(model) -> list[dict]:
    eqs = model.equations_
    if eqs is None:
        return []
    out = []
    for row in eqs.itertuples():
        out.append({
            "complexity": int(row.complexity),
            "loss": float(row.loss),
            "score": float(row.score),
            "equation": str(row.equation),
            "sympy": str(row.sympy_format),
        })
    return out


class Worker:
    def __init__(self, conn):
        self.conn = conn

    def send(self, msg: dict) -> None:
        self.conn.send(msg)

    def status(self, state: str, msg: str, **extra) -> None:
        self.send({"ev": "status", "state": state, "msg": msg, **extra})

    def boot(self) -> None:
        t0 = time.perf_counter()
        self.status("loading", "Starting Julia runtime")
        import pysr  # noqa: F401  (initialises Julia + SymbolicRegression)
        from juliacall import Main as jl

        nthreads = int(jl.seval("Threads.nthreads()"))
        self.status("compiling", "Compiling the search engine (takes a moment)",
                    julia=str(jl.VERSION), threads=nthreads, t=time.perf_counter() - t0)
        t1 = time.perf_counter()
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (64, 2))
        y = 1.0 + 0.5 * X[:, 0]
        m = make_model({"populations": 4, "population_size": 20, "ncycles_per_iteration": 20}, None, None)
        m.fit(X, y, variable_names=["u", "v"])
        hall_of_fame(m)
        self.status("ready", "Symbolic regression engine ready", julia=str(jl.VERSION), threads=nthreads,
                    t=time.perf_counter() - t0, compile_t=time.perf_counter() - t1)

    def fit(self, job: dict) -> None:
        jid = job["job_id"]
        X = np.asarray(job["X"], float)
        y = np.asarray(job["y"], float)
        max_iter = int(job.get("max_iterations", 40))
        min_iter = int(job.get("min_iterations", 2))
        target = float(job.get("target_loss", 0.0))
        guesses = job.get("guesses") or None
        model = make_model(job.get("settings"), job.get("output_directory"), job.get("run_id"), guesses)
        reason = "max_iterations"
        t_start = time.perf_counter()
        for it in range(1, max_iter + 1):
            t0 = time.perf_counter()
            try:
                model.fit(X, y, variable_names=list(job["var_names"]))
            except Exception as exc:
                if it != 1 or not guesses:
                    raise
                # a seed the parser rejects must not cost the search: start unseeded
                print(f"[sr_worker] seed rejected ({exc!r}); continuing without it", file=sys.stderr)
                guesses = None
                model = make_model(job.get("settings"), job.get("output_directory"), job.get("run_id"))
                model.fit(X, y, variable_names=list(job["var_names"]))
            hof = hall_of_fame(model)
            self.send({"ev": "iter", "job_id": jid, "iteration": it, "dt": time.perf_counter() - t0,
                       "elapsed": time.perf_counter() - t_start, "hof": hof})
            stop = None
            while self.conn.poll():
                msg = self.conn.recv()
                cmd = msg.get("cmd")
                if cmd == "shutdown":
                    raise SystemExit(0)
                if cmd == "stop" and msg.get("job_id") in (None, jid):
                    stop = msg.get("reason", "stopped")
            if stop:
                reason = stop
                break
            if it >= min_iter and hof and min(h["loss"] for h in hof) <= target:
                reason = "converged"
                break
        self.send({"ev": "done", "job_id": jid, "reason": reason, "elapsed": time.perf_counter() - t_start})

    def serve(self) -> None:
        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                return
            cmd = msg.get("cmd")
            if cmd == "shutdown":
                return
            if cmd == "fit":
                try:
                    self.fit(msg)
                except SystemExit:
                    return
                except Exception as exc:  # report and keep serving
                    traceback.print_exc()
                    self.send({"ev": "done", "job_id": msg.get("job_id"), "reason": "error", "error": repr(exc)})


def main() -> None:
    address, authkey = sys.argv[1], bytes.fromhex(sys.argv[2])
    conn = Client(address, family="AF_PIPE" if os.name == "nt" else "AF_UNIX", authkey=authkey)
    w = Worker(conn)
    try:
        w.boot()
    except Exception as exc:
        traceback.print_exc()
        w.status("error", f"Julia failed to start: {exc!r}")
        return
    w.serve()


if __name__ == "__main__":
    main()
