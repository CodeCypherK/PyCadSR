"""Build the self-contained runtime that ships inside the MeshToCad installer.

Produces ``<repo>/runtime`` containing:

    runtime/python/        portable CPython (python-build-standalone) + all pip packages
    runtime/julia/         portable Julia
    runtime/julia_depot/   Julia packages (SymbolicRegression, PythonCall, ...) precompiled
    runtime/julia_env/     the Julia project juliacall/PySR use

Nothing here is installed system-wide; the end user never runs this script.
Run it on the build machine with any Python >= 3.9:

    python scripts/build_runtime.py            # build (skips finished stages)
    python scripts/build_runtime.py --clean    # wipe runtime/ and rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
CACHE = ROOT / ".cache" / "downloads"

PY_RELEASE = "20261003"
PY_VERSION = "3.12.15"
PY_ASSET = f"cpython-{PY_VERSION}+{PY_RELEASE}-x86_64-pc-windows-msvc-install_only.tar.gz"
PY_BASE = f"https://github.com/astral-sh/python-build-standalone/releases/download/{PY_RELEASE}"

JULIA_VERSION = "1.12.7"
JULIA_VERSIONS_JSON = "https://julialang-s3.julialang.org/bin/versions.json"

PY_DIR = RUNTIME / "python"
PY_EXE = PY_DIR / "python.exe"
JULIA_DIR = RUNTIME / "julia"
JULIA_EXE = JULIA_DIR / "bin" / "julia.exe"
DEPOT = RUNTIME / "julia_depot"
JULIA_ENV = RUNTIME / "julia_env"


def log(msg: str) -> None:
    print(f"[build_runtime {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_sha is None or sha256(dest) == expected_sha):
        log(f"cached {dest.name}")
        return dest
    log(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)
    if expected_sha is not None:
        got = sha256(dest)
        if got != expected_sha:
            dest.unlink()
            raise SystemExit(f"checksum mismatch for {dest.name}: {got} != {expected_sha}")
    return dest


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def julia_depot_path(user_depot: Path | None = None) -> str:
    """Depot search list. Julia's own depots must stay in the list: they hold the
    shipped (relocatable) stdlib caches. Without them every stdlib is recompiled
    into our depot with absolute paths and the bundle stops being relocatable."""
    parts = [user_depot] if user_depot else []
    parts += [DEPOT, JULIA_DIR / "local" / "share" / "julia", JULIA_DIR / "share" / "julia"]
    return ";".join(str(x) for x in parts)


def runtime_env(build: bool) -> dict[str, str]:
    """Environment the bundled Python/Julia run with (mirrors the Tauri launcher)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "JULIA", "CONDA", "VIRTUAL_ENV"))}
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        PYTHON_JULIAPKG_PROJECT=str(JULIA_ENV),
        PYTHON_JULIAPKG_EXE=str(JULIA_EXE),
        PYTHON_JULIACALL_HANDLE_SIGNALS="yes",
        PYTHON_JULIACALL_THREADS="auto",
        JULIA_DEPOT_PATH=julia_depot_path(),
    )
    if not build:
        env["PYTHON_JULIAPKG_OFFLINE"] = "yes"
        env["JULIA_PKG_OFFLINE"] = "true"
    # Keep the system PATH from leaking another python/julia into the build.
    env["PATH"] = os.pathsep.join([str(PY_DIR), str(PY_DIR / "Scripts"), str(JULIA_DIR / "bin"),
                                   os.environ.get("SystemRoot", r"C:\Windows") + r"\System32",
                                   os.environ.get("SystemRoot", r"C:\Windows")])
    return env


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=env, cwd=cwd)


# ---------------------------------------------------------------- stages

def stage_python() -> None:
    if PY_EXE.exists():
        log("python: already present")
        return
    sums = fetch_text(f"{PY_BASE}/SHA256SUMS")
    expected = next((line.split()[0] for line in sums.splitlines() if line.strip().endswith(PY_ASSET)), None)
    if expected is None:
        raise SystemExit(f"{PY_ASSET} not listed in SHA256SUMS")
    archive = download(f"{PY_BASE}/{PY_ASSET.replace('+', '%2B')}", CACHE / PY_ASSET, expected)
    log("extracting python")
    with tarfile.open(archive) as tf:
        tf.extractall(RUNTIME, filter="data") if sys.version_info >= (3, 12) else tf.extractall(RUNTIME)
    assert PY_EXE.exists(), "python.exe missing after extraction"


def stage_pip() -> None:
    marker = PY_DIR / ".meshtocad_pip_done"
    req = ROOT / "backend" / "requirements.txt"
    stamp = sha256(req) + sha256(ROOT / "PySR" / "pyproject.toml")
    if marker.exists() and marker.read_text() == stamp:
        log("pip: up to date")
        return
    env = runtime_env(build=True)
    run([PY_EXE, "-m", "pip", "install", "--no-warn-script-location", "--upgrade", "pip"], env=env)
    run([PY_EXE, "-m", "pip", "install", "--no-warn-script-location", "-r", req], env=env)
    # PySR from the local clone (non-editable so it is copied into site-packages).
    run([PY_EXE, "-m", "pip", "install", "--no-warn-script-location", ROOT / "PySR"], env=env)
    marker.write_text(stamp)


def stage_julia() -> None:
    if JULIA_EXE.exists():
        log("julia: already present")
        return
    versions = json.loads(fetch_text(JULIA_VERSIONS_JSON))
    files = versions[JULIA_VERSION]["files"]
    f = next(f for f in files if f["os"] == "winnt" and f["arch"] == "x86_64"
             and f["kind"] == "archive" and f["extension"] == "zip")
    archive = download(f["url"], CACHE / Path(f["url"]).name, f["sha256"])
    log("extracting julia")
    tmp = RUNTIME / "_julia_extract"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp)
    inner = next(tmp.iterdir())
    shutil.rmtree(JULIA_DIR, ignore_errors=True)
    inner.rename(JULIA_DIR)
    shutil.rmtree(tmp, ignore_errors=True)
    assert JULIA_EXE.exists()


def stage_julia_packages() -> None:
    """Let juliapkg resolve + install SymbolicRegression/PythonCall into our depot."""
    marker = DEPOT / ".meshtocad_pkgs_done"
    if marker.exists():
        log("julia packages: already installed")
        return
    DEPOT.mkdir(parents=True, exist_ok=True)
    JULIA_ENV.mkdir(parents=True, exist_ok=True)
    env = runtime_env(build=True)
    env["JULIA_PKG_PRECOMPILE_AUTO"] = "1"
    code = "import pysr, juliacall; from juliacall import Main as jl; print('julia', jl.VERSION)"
    run([PY_EXE, "-c", code], env=env)
    marker.write_text("ok")


def stage_patch() -> None:
    """Make PythonCall relocatable.

    PythonCall 0.9.36 bakes its install folder into the precompile cache
    (``const ROOT_DIR = dirname(dirname(@__DIR__))``) and uses it at start-up to
    find ``pysrc/juliacall/init.jl``. Once the runtime is installed somewhere
    else that path is wrong and Julia fails to start. Compute it at runtime via
    ``pkgdir`` instead. Must run before ``precompile``.
    """
    changed = False
    for core in (DEPOT / "packages" / "PythonCall").glob("*/src/Core"):
        if "ROOT_DIR" in (core / "Core.jl").read_text(encoding="utf-8") and os.name == "nt":
            # Pkg write-protects package trees with ACLs on Windows; reset them for this package.
            subprocess.run(["icacls", str(core.parents[1]), "/reset", "/T", "/Q"], check=True)
        edits = {
            core / "Core.jl": [("const ROOT_DIR = dirname(dirname(@__DIR__))",
                                "root_dir() = (d = Base.pkgdir(parentmodule(@__MODULE__)); "
                                "d === nothing ? dirname(dirname(@__DIR__)) : d)  # patched by MeshToCad")],
            core / "stdlib.jl": [("joinpath(ROOT_DIR,", "joinpath(root_dir(),")],
            core / "juliacall.jl": [("joinpath(ROOT_DIR,", "joinpath(root_dir(),")],
        }
        for path, reps in edits.items():
            text = path.read_text(encoding="utf-8")
            new = text
            for old, rep in reps:
                new = new.replace(old, rep)
            if new != text:
                os.chmod(path, 0o644)
                path.write_text(new, encoding="utf-8")
                changed = True
        if "ROOT_DIR" in "".join(p.read_text(encoding="utf-8") for p in edits):
            raise SystemExit(f"PythonCall patch incomplete in {core}")
    if changed:
        (DEPOT / "compiled" / ".meshtocad_precompiled").unlink(missing_ok=True)
    log("patch: PythonCall made relocatable" if changed else "patch: already applied")


def stage_precompile() -> None:
    """(Re)build the depot's package caches with the correct depot list."""
    marker = DEPOT / "compiled" / ".meshtocad_precompiled"
    if marker.exists():
        log("precompile: up to date")
        return
    shutil.rmtree(DEPOT / "compiled", ignore_errors=True)
    env = runtime_env(build=False)
    code = ("import time; t=time.time(); import pysr; from juliacall import Main as jl; "
            "print('precompiled in %.0fs' % (time.time()-t))")
    run([PY_EXE, "-c", code], env=env)
    marker.write_text("ok")


def stage_relocation_check() -> None:
    """Move the runtime, start Julia from the new place, and fail if anything recompiles."""
    moved = ROOT / ".cache" / "reloc_check" / "runtime"
    user = ROOT / ".cache" / "reloc_check" / "user_depot"
    shutil.rmtree(user, ignore_errors=True)
    user.mkdir(parents=True)
    moved.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME.rename(moved)
    try:
        env = runtime_env(build=False)
        old = str(RUNTIME)
        for k, v in list(env.items()):
            if old in v:
                env[k] = v.replace(old, str(moved))
        env["JULIA_DEPOT_PATH"] = str(user) + ";" + env["JULIA_DEPOT_PATH"]
        env["PATH"] = env["PATH"].replace(old, str(moved))
        code = ("import time; t=time.time(); import pysr; from juliacall import Main as jl; "
                "print('relocated import in %.0fs' % (time.time()-t))")
        run([moved / "python" / "python.exe", "-c", code], env=env)
    finally:
        moved.rename(RUNTIME)
    compiled = list((user / "compiled").rglob("*.ji")) if (user / "compiled").exists() else []
    if compiled:
        raise SystemExit(f"relocation check FAILED: {len(compiled)} packages recompiled, e.g. "
                         + ", ".join(sorted({c.parent.name for c in compiled})[:8]))
    log("relocation check passed: nothing recompiled after moving the runtime")


def stage_warmup() -> None:
    """Run a real search once so every pkgimage is built and verify offline startup."""
    env = runtime_env(build=False)
    run([PY_EXE, "-m", "meshtocad.selftest"], env=env | {"PYTHONPATH": str(ROOT / "backend")})


def _rm(p: Path) -> int:
    if not p.exists():
        return 0
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    return size


def stage_trim() -> None:
    """Remove files the app never needs at runtime to shrink the installer."""
    freed = 0
    site = PY_DIR / "Lib" / "site-packages"
    for pattern in ("*/tests", "*/test", "*/*/tests", "*/*/*/tests"):
        for d in site.glob(pattern):
            # pysr.test is imported by nothing at runtime; numpy/scipy tests are big.
            if d.is_dir():
                freed += _rm(d)
    for d in [PY_DIR / "Lib" / "test", PY_DIR / "Lib" / "idlelib", PY_DIR / "Lib" / "tkinter",
              PY_DIR / "tcl", PY_DIR / "Lib" / "turtledemo", PY_DIR / "include"]:
        freed += _rm(d)
    for d in [DEPOT / "logs", DEPOT / "clones", DEPOT / "registries", JULIA_DIR / "share" / "doc",
              JULIA_DIR / "share" / "julia" / "test"]:
        freed += _rm(d)
    for pkg in (DEPOT / "packages").glob("*/*"):
        for sub in ("docs", "test", "benchmark", "examples"):
            freed += _rm(pkg / sub)
    for d in PY_DIR.rglob("__pycache__"):
        freed += _rm(d)
    log(f"trim: freed {freed / 1e6:.0f} MB")
    # Precompile .pyc so the read-only install never needs to write bytecode.
    run([PY_EXE, "-m", "compileall", "-q", "-j", "0", PY_DIR / "Lib"], env=runtime_env(build=False))


def stage_backend() -> None:
    """Copy the meshtocad backend package into the runtime."""
    dst = RUNTIME / "backend"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(ROOT / "backend" / "meshtocad", dst / "meshtocad",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    log("backend copied")


def report() -> None:
    def size(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0
    for p in (PY_DIR, JULIA_DIR, DEPOT, JULIA_ENV, RUNTIME / "backend"):
        log(f"{p.relative_to(ROOT)}: {size(p):,.0f} MB")


STAGES = {
    "python": stage_python,
    "pip": stage_pip,
    "julia": stage_julia,
    "julia_packages": stage_julia_packages,
    "patch": stage_patch,
    "precompile": stage_precompile,
    "backend": stage_backend,
    "warmup": stage_warmup,
    "trim": stage_trim,
    "relocation_check": stage_relocation_check,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="delete runtime/ first")
    ap.add_argument("--only", nargs="*", choices=list(STAGES), help="run only these stages")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(STAGES))
    args = ap.parse_args()
    if args.clean:
        log("removing runtime/")
        shutil.rmtree(RUNTIME, ignore_errors=True)
    RUNTIME.mkdir(exist_ok=True)
    for name, fn in STAGES.items():
        if args.only and name not in args.only:
            continue
        if name in args.skip:
            continue
        log(f"== stage {name}")
        fn()
    report()
    log("done")


if __name__ == "__main__":
    main()
