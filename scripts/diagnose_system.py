"""Print read-only system details relevant to reproducible experiments."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load non-secret project defaults without replacing shell environment variables.
load_dotenv(Path.cwd() / ".env", override=False)

import psutil


def gibibytes(value: int | float) -> str:
    return f"{value / 1024**3:.2f} GiB"


def main() -> None:
    project_root = Path.cwd().resolve()
    project_disk = shutil.disk_usage(project_root)
    shm_disk = shutil.disk_usage("/dev/shm")
    memory = psutil.virtual_memory()

    print(f"Project root: {project_root}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Logical CPU: {psutil.cpu_count(logical=True)}")
    print(f"RAM total: {gibibytes(memory.total)}")
    print(f"RAM available: {gibibytes(memory.available)}")
    print(f"Project disk total: {gibibytes(project_disk.total)}")
    print(f"Project disk free: {gibibytes(project_disk.free)}")
    print(f"/dev/shm total: {gibibytes(shm_disk.total)}")
    print(f"/dev/shm free: {gibibytes(shm_disk.free)}")

    for variable in ("UV_CACHE_DIR", "TORCH_HOME", "YOLO_CONFIG_DIR"):
        print(f"{variable}: {os.environ.get(variable, '<unset>')}")


if __name__ == "__main__":
    main()
