"""Read-only NVIDIA device diagnostics; intentionally does not import PyTorch."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Load CUDA visibility defaults without replacing shell environment variables.
load_dotenv(Path.cwd() / ".env", override=False)


DEVICE_NODES = (
    "/dev/nvidia0",
    "/dev/nvidia1",
    "/dev/nvidia2",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
)


def report_command(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        print("Command not found.")
        return
    except subprocess.TimeoutExpired:
        print("Command timed out after 15 seconds.")
        return

    print(f"Return code: {result.returncode}")
    if result.stdout.strip():
        print("stdout:")
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print("stderr:")
        print(result.stderr.rstrip())
    if not result.stdout.strip() and not result.stderr.strip():
        print("No output.")


def main() -> None:
    print(f"nvidia-smi path: {shutil.which('nvidia-smi') or '<not found>'}")
    print("\nNVIDIA device nodes:")
    for node in DEVICE_NODES:
        path = Path(node)
        print(f"{node}: {'present' if path.exists() else 'missing'}")

    for variable in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        print(f"{variable}: {os.environ.get(variable, '<unset>')}")

    report_command(["nvidia-smi"])
    report_command(["nvidia-smi", "-L"])


if __name__ == "__main__":
    main()
