"""Remove the runtime copy Tauri staged in target/<profile>/ by a previous build.

Julia's package manager write-protects package files (with ACLs on Windows), so a
second `tauri build` cannot overwrite the staged copy ("Access is denied"). This
only touches build output under app/src-tauri/target.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "app" / "src-tauri" / "target"


def main() -> None:
    for profile in ("release", "debug"):
        staged = TARGET / profile / "runtime"
        if not staged.exists():
            continue
        if os.name == "nt":
            subprocess.run(["icacls", str(staged), "/reset", "/T", "/Q"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def make_writable(func, path, _exc):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(staged, onerror=make_writable)
        print(f"removed staged runtime {staged}")


if __name__ == "__main__":
    main()
