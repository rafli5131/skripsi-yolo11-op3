"""Run a one-epoch FP32 YOLO11n smoke test on one visible CUDA device."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# CUDA_VISIBLE_DEVICES from the project .env must be present before torch imports.
load_dotenv(Path.cwd() / ".env", override=False)

import numpy as np
import torch
import torchvision
import yaml


def project_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Run this script from the project root (directory containing pyproject.toml).")
    return root


def run_nvidia_smi() -> None:
    if not shutil_which("nvidia-smi"):
        print("nvidia-smi not found; continuing with PyTorch CUDA checks.")
        return
    try:
        result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"nvidia-smi unavailable: {error}; continuing with PyTorch CUDA checks.")
        return
    print("nvidia-smi before smoke training:")
    print(result.stdout.rstrip() or result.stderr.rstrip() or f"return code: {result.returncode}")


def shutil_which(command: str) -> str | None:
    from shutil import which

    return which(command)


def require_cuda() -> tuple[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Set CUDA_VISIBLE_DEVICES before starting Python and retry from SSH host.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No visible CUDA device was found.")
    properties = torch.cuda.get_device_properties(0)
    return properties.name, properties.total_memory / 1024**3


def write_manifest(
    path: Path, config: dict, gpu_name: str, elapsed_seconds: float, ultralytics_version: str
) -> None:
    training = config["training"]
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_name": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ultralytics_version": ultralytics_version,
        "numpy_version": np.__version__,
        "dataset_workspace": config["dataset"]["workspace"],
        "dataset_project": config["dataset"]["project"],
        "dataset_version": config["dataset"]["version"],
        "class_mapping": config["classes"],
        "imgsz": config["model"]["imgsz"],
        "batch": training["batch"],
        "workers": training["workers"],
        "seed": config["project"]["seed"],
        "optimizer": training["optimizer"],
        "lr0": training["lr0"],
        "momentum": training["momentum"],
        "weight_decay": training["weight_decay"],
        "elapsed_seconds": elapsed_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    root = project_root()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache" / "ultralytics"))
    from ultralytics import YOLO, __version__ as ultralytics_version, settings

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("WARNING: CUDA_VISIBLE_DEVICES is not set. No physical GPU was selected explicitly.", file=sys.stderr)

    gpu_name, total_vram = require_cuda()
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"Logical cuda:0: {gpu_name}")
    print(f"Logical cuda:0 total VRAM: {total_vram:.2f} GiB")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    run_nvidia_smi()

    config = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    data_yaml = root / "data" / "dataset" / "data.yaml"
    if not data_yaml.is_file():
        raise RuntimeError(f"Normalized dataset YAML not found: {data_yaml}. Run download_dataset.py first.")

    settings.update({"wandb": False})
    training = config["training"]
    model = YOLO(config["model"]["pretrained"])
    start = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=1,
        fraction=0.10,
        imgsz=config["model"]["imgsz"],
        batch=training["batch"],
        workers=training["workers"],
        device=training["device"],
        seed=config["project"]["seed"],
        optimizer=training["optimizer"],
        lr0=training["lr0"],
        momentum=training["momentum"],
        weight_decay=training["weight_decay"],
        amp=training["amp"],
        val=False,
        save=False,
        plots=False,
        project=str(root / "experiments" / "debug"),
        name="fp32_smoke",
        exist_ok=True,
        verbose=True,
    )
    elapsed_seconds = time.perf_counter() - start
    manifest_path = root / "experiments" / "debug" / "fp32_smoke" / "manifest.json"
    write_manifest(manifest_path, config, gpu_name, elapsed_seconds, ultralytics_version)

    print(f"Smoke training elapsed: {elapsed_seconds:.2f} seconds ({elapsed_seconds / 60:.2f} minutes)")
    print(f"workers: {training['workers']}")
    print(f"batch: {training['batch']}")
    print(f"imgsz: {config['model']['imgsz']}")
    print(f"visible GPU: {gpu_name}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Manifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FP32 smoke test failed: {error}", file=sys.stderr)
        raise SystemExit(1)
