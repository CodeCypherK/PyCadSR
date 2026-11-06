"""Line protocol between the Python backend and the Tauri shell.

Every message to the UI is one line on the *original* stdout:  ``@@MTC@@{json}``.
At startup we move the real stdout to a private file descriptor and point fd 1
at stderr, so anything else that prints (Julia, numpy warnings, libraries) ends
up in the log stream instead of corrupting the protocol.
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import threading
from typing import Any

import numpy as np

SENTINEL = "@@MTC@@"
_lock = threading.Lock()
_out = None
_hook = None


def set_hook(fn) -> None:
    """Route events to a callback instead of stdout (used by the self-test)."""
    global _hook
    _hook = fn


_in = None


def init() -> None:
    """Take private copies of stdin/stdout for the protocol.

    stdout: fd 1 is pointed at stderr so stray prints land in the log.
    stdin: commands are read from a private duplicate and fd 0 is pointed at NUL.
    On Windows a blocking read pending on the stdin pipe stalls any DLL whose C
    runtime probes the standard handles while loading (SciPy's OpenBLAS does),
    which froze the first mesh import until the next command arrived.
    """
    global _out, _in
    if _out is not None:
        return
    fd = os.dup(1)
    _out = os.fdopen(fd, "w", encoding="utf-8", newline="\n", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    _in = os.fdopen(os.dup(0), "r", encoding="utf-8")
    nul = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nul, 0)
    os.close(nul)
    sys.stdin = open(os.devnull, "r")


def commands():
    """Iterator over command lines from the UI."""
    return _in if _in is not None else sys.stdin


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def emit(type_: str, **data: Any) -> None:
    msg = _clean({"type": type_, **data})
    if _hook is not None:
        _hook(msg)
        return
    line = SENTINEL + json.dumps(msg, separators=(",", ":"), allow_nan=False) + "\n"
    with _lock:
        if _out is None:  # used outside the app (selftest): print readable summary
            sys.stderr.write(f"[event] {type_}\n")
            return
        _out.write(line)
        _out.flush()


def log(msg: str, level: str = "info") -> None:
    emit("log", level=level, msg=msg)


def b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def b64_u32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<u4").tobytes()).decode("ascii")
