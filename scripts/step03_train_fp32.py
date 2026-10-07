"""Train or resume the final single-GPU YOLO11n FP32 baseline experiment."""

from __future__ import annotations

import argparse
import json
import numbers
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# CUDA_VISIBLE_DEVICES and optional W&B values may be supplied by root .env.
# Shell exports remain authoritative because override=False.
load_dotenv(Path.cwd() / ".env", override=False)

import numpy as np
import torch
import torchvision
import yaml


RUN_NAME = "fp32_baseline"
DEFAULT_WANDB_PROJECT = "skripsi-yolo11-qat-openvino"
DEFAULT_RESUME_SENTINEL = "__default_last_checkpoint__"


def project_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Run this script from the project root (directory containing pyproject.toml).")
    return root


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=DEFAULT_RESUME_SENTINEL,
        default=None,
        metavar="CHECKPOINT",
        help="Resume from CHECKPOINT, or from this run's weights/last.pt when no path is supplied.",
    )
    parser.add_argument("--enable-wandb", action="store_true", help="Enable opt-in Weights & Biases logging.")
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT,
        help="W&B project used with --enable-wandb (defaults to WANDB_PROJECT when set).",
    )
    parser.add_argument(
        "--wandb-name",
        default=os.environ.get("WANDB_NAME") or None,
        help="Optional W&B run name (defaults to WANDB_NAME when set).",
    )
    return parser.parse_args()


def require_cuda() -> tuple[str, float]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable. Select a GPU with CUDA_VISIBLE_DEVICES before starting Python.")
    if torch.cuda.device_count() != 1:
        print(
            f"WARNING: {torch.cuda.device_count()} CUDA devices are visible; using logical device 0 only.",
            file=sys.stderr,
        )
    properties = torch.cuda.get_device_properties(0)
    return properties.name, properties.total_memory / 1024**3


def resolve_resume_checkpoint(argument: str | None, run_dir: Path, root: Path) -> Path | None:
    if argument is None:
        return None
    checkpoint = run_dir / "weights" / "last.pt" if argument == DEFAULT_RESUME_SENTINEL else Path(argument)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    return checkpoint


def serializable_metrics(trainer: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for source in (getattr(trainer, "metrics", {}), getattr(trainer, "lr", {})):
        for key, value in source.items():
            if isinstance(value, numbers.Real):
                values[str(key)] = float(value)
    values["epoch"] = int(trainer.epoch) + 1
    return values


def initialize_wandb(enabled: bool, project: str, name: str | None, config: dict[str, Any]):
    if not enabled:
        return None, None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B support requires the project dependency 'wandb'. Run uv sync first.") from error

    run = wandb.init(project=project, name=name, config=config, job_type="train")

    def on_fit_epoch_end(trainer: Any) -> None:
        try:
            run.log(serializable_metrics(trainer), step=int(trainer.epoch) + 1)
        except Exception as error:
            print(f"WARNING: W&B metric logging failed: {error}", file=sys.stderr)

    return run, on_fit_epoch_end


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    arguments = parse_arguments()
    root = project_root()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache" / "ultralytics"))
    from ultralytics import YOLO, __version__ as ultralytics_version, settings

    config = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    training = config["training"]
    if training["cache"] is not False:
        raise RuntimeError("Final FP32 baseline requires training.cache=false.")

    data_yaml = root / "data" / "dataset" / "data.yaml"
    if not data_yaml.is_file():
        raise RuntimeError(f"Normalized dataset YAML not found: {data_yaml}. Run dataset download and audit first.")
    run_dir = root / "experiments" / "fp32" / RUN_NAME
    checkpoint = resolve_resume_checkpoint(arguments.resume, run_dir, root)
    if checkpoint is None and (run_dir / "weights" / "last.pt").exists():
        raise RuntimeError(
            f"An existing checkpoint was found at {run_dir / 'weights' / 'last.pt'}. "
            "Use --resume to continue it instead of starting over."
        )

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("WARNING: CUDA_VISIBLE_DEVICES is unset; no physical GPU was selected explicitly.", file=sys.stderr)
    gpu_name, total_vram = require_cuda()
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"Logical cuda:0: {gpu_name} ({total_vram:.2f} GiB)")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"Resume checkpoint: {checkpoint if checkpoint else '<new baseline run>'}")
    print(f"W&B logging: {'enabled' if arguments.enable_wandb else 'disabled'}")

    # Keep Ultralytics' optional integrations disabled; W&B below is explicit and opt-in.
    settings.update({"wandb": False})
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics_version": ultralytics_version,
        "numpy_version": np.__version__,
        "gpu_name": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset_workspace": config["dataset"]["workspace"],
        "dataset_project": config["dataset"]["project"],
        "dataset_version": config["dataset"]["version"],
        "class_mapping": config["classes"],
        "checkpoint_source": str(checkpoint) if checkpoint else config["model"]["pretrained"],
        "resume": checkpoint is not None,
        "wandb_enabled": arguments.enable_wandb,
        "wandb_project": arguments.wandb_project if arguments.enable_wandb else None,
        "training": {
            "epochs": training["epochs"],
            "patience": training["patience"],
            "batch": training["batch"],
            "workers": training["workers"],
            "cache": training["cache"],
            "device": training["device"],
            "imgsz": config["model"]["imgsz"],
            "amp": training["amp"],
            "seed": config["project"]["seed"],
            "optimizer": training["optimizer"],
            "lr0": training["lr0"],
            "momentum": training["momentum"],
            "weight_decay": training["weight_decay"],
        },
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    model = YOLO(str(checkpoint) if checkpoint else config["model"]["pretrained"])
    wandb_run, wandb_callback = initialize_wandb(
        arguments.enable_wandb,
        arguments.wandb_project,
        arguments.wandb_name or RUN_NAME,
        manifest,
    )
    if wandb_callback:
        model.add_callback("on_fit_epoch_end", wandb_callback)

    train_arguments = {
        "data": str(data_yaml),
        "epochs": training["epochs"],
        "patience": training["patience"],
        "batch": training["batch"],
        "workers": training["workers"],
        "cache": training["cache"],
        "device": training["device"],
        "imgsz": config["model"]["imgsz"],
        "amp": training["amp"],
        "seed": config["project"]["seed"],
        "optimizer": training["optimizer"],
        "lr0": training["lr0"],
        "momentum": training["momentum"],
        "weight_decay": training["weight_decay"],
        "save": True,
        "save_period": -1,
        "plots": True,
        "project": str(root / "experiments" / "fp32"),
        "name": RUN_NAME,
        "exist_ok": True,
        "verbose": True,
    }
    if checkpoint:
        train_arguments["resume"] = True

    try:
        model.train(**train_arguments)
    except Exception:
        manifest["status"] = "failed"
        write_manifest(manifest_path, manifest)
        raise
    else:
        manifest["status"] = "completed"
        write_manifest(manifest_path, manifest)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(f"FP32 baseline completed. Checkpoints: {run_dir / 'weights'}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FP32 baseline training failed: {error}", file=sys.stderr)
        raise SystemExit(1)
