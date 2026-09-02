"""Verify that the installed PyTorch build can execute a CUDA operation."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load project defaults before importing torch, while preserving shell exports.
load_dotenv(Path.cwd() / ".env", override=False)

import torch
import torchvision


def gibibytes(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not cuda_available:
        print("CUDA verification failed: no CUDA device is available to this process.", file=sys.stderr)
        return 1

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"Device logical index: {index}")
        print(f"  Name: {properties.name}")
        print(f"  Total VRAM: {gibibytes(properties.total_memory)}")
        print(f"  Compute capability: {properties.major}.{properties.minor}")

    try:
        device = torch.device("cuda:0")
        left = torch.randn((256, 256), device=device)
        right = torch.randn((256, 256), device=device)
        result = left @ right
        torch.cuda.synchronize()
        print(f"CUDA tensor sanity test: success (result shape={tuple(result.shape)})")
    except Exception as error:
        print(f"CUDA tensor sanity test failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
