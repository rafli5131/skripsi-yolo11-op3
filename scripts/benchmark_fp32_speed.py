"""Benchmark one FP32 YOLO11n train epoch across batch and RAM-cache configurations."""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# CUDA_VISIBLE_DEVICES from the project .env must be present before torch imports.
load_dotenv(Path.cwd() / ".env", override=False)

import psutil
import torch
import yaml
from PIL import Image


BATCH_SIZES = (16, 32, 64, 128)
CACHE_OPTIONS: tuple[bool | str, ...] = (False, "ram")
FRACTION = 0.25
RAM_CACHE_AVAILABLE_FRACTION_LIMIT = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def project_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Run this script from the project root (directory containing pyproject.toml).")
    return root


def cache_label(cache: bool | str) -> str:
    return "ram" if cache == "ram" else "false"


def estimate_ram_cache(train_images_dir: Path) -> tuple[int, int]:
    """Estimate decoded uint8 BGR image memory for the same sorted fraction used by this benchmark."""
    images = sorted(
        path for path in train_images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No supported training images found in {train_images_dir}.")
    selected = images[: max(1, round(len(images) * FRACTION))]
    decoded_bytes = 0
    for image_path in selected:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as error:
            raise RuntimeError(f"Could not read image dimensions for cache estimate: {image_path}: {error}") from error
        decoded_bytes += width * height * 3
    return len(selected), decoded_bytes


def gibibytes(value: int) -> float:
    return value / 1024**3


def validate_ram_cache_argument() -> None:
    """Verify the installed Ultralytics parser accepts the requested cache mode before training."""
    from ultralytics.cfg import get_cfg

    if get_cfg(overrides={"cache": "ram"}).cache != "ram":
        raise RuntimeError("Installed Ultralytics does not accept cache='ram'.")


def make_epoch_timer() -> tuple[dict[str, Any], Any, Any]:
    timing: dict[str, Any] = {}

    def on_train_epoch_start(trainer: Any) -> None:
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        timing["started_at"] = time.perf_counter()

    def on_train_epoch_end(trainer: Any) -> None:
        torch.cuda.synchronize(0)
        started_at = timing.get("started_at")
        if started_at is None:
            raise RuntimeError("Training epoch start callback was not invoked.")
        timing["train_epoch_seconds"] = time.perf_counter() - started_at
        timing["train_batches"] = len(trainer.train_loader)
        timing["images_processed"] = len(trainer.train_loader.dataset)
        timing["actual_batch"] = trainer.batch_size
        timing["peak_gpu_memory_allocated_gb"] = gibibytes(torch.cuda.max_memory_allocated(0))
        timing["peak_gpu_memory_reserved_gb"] = gibibytes(torch.cuda.max_memory_reserved(0))

    return timing, on_train_epoch_start, on_train_epoch_end


def is_cuda_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() or isinstance(error, torch.cuda.OutOfMemoryError)


def print_table(results: list[dict[str, Any]]) -> None:
    print("\nbatch | cache | train_seconds | img/s | it/s | peak_vram_gb | status")
    print("-" * 84)
    for result in results:
        train_seconds = result.get("train_epoch_seconds")
        images_per_second = result.get("images_per_second")
        iterations_per_second = result.get("iterations_per_second")
        peak_vram = result.get("peak_gpu_memory_allocated_gb")
        if train_seconds is not None:
            print(
                f"{result['batch']:>5} | {result['cache']:<5} | {train_seconds:.3f}"
                f" | {images_per_second:.2f} | {iterations_per_second:.2f} | {peak_vram:.2f} | {result['status']}"
            )
        else:
            print(f"{result['batch']:>5} | {result['cache']:<5} | {'-':>13} | {'-':>5} | {'-':>4} | {'-':>12} | {result['status']}")


def main() -> int:
    root = project_root()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache" / "ultralytics"))
    from ultralytics import YOLO, __version__ as ultralytics_version, settings

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable. Run this benchmark only from the GPU-enabled SSH host.")
    if torch.cuda.device_count() != 1:
        print(
            f"WARNING: {torch.cuda.device_count()} CUDA devices are visible. The benchmark will use logical device 0 only.",
            file=sys.stderr,
        )
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("WARNING: CUDA_VISIBLE_DEVICES is unset; no physical GPU was selected explicitly.", file=sys.stderr)

    config = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    data_yaml = root / "data" / "dataset" / "data.yaml"
    if not data_yaml.is_file():
        raise RuntimeError(f"Normalized dataset YAML not found: {data_yaml}. Run download_dataset.py first.")
    data_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    train_images_dir = Path(data_config.get("train", ""))
    if not train_images_dir.is_absolute() or not train_images_dir.is_dir():
        raise RuntimeError("Normalized data.yaml train path must be an existing absolute image directory.")

    validate_ram_cache_argument()
    selected_images, estimated_cache_bytes = estimate_ram_cache(train_images_dir)
    available_ram_bytes = psutil.virtual_memory().available
    ram_cache_allowed = estimated_cache_bytes <= available_ram_bytes * RAM_CACHE_AVAILABLE_FRACTION_LIMIT
    print(f"RAM available before cache benchmark: {gibibytes(available_ram_bytes):.2f} GiB")
    print(
        f"Estimated decoded RAM cache for {selected_images} training images (fraction={FRACTION}): "
        f"{gibibytes(estimated_cache_bytes):.2f} GiB"
    )
    print(f"RAM cache safety limit: {RAM_CACHE_AVAILABLE_FRACTION_LIMIT:.0%} of available RAM")
    print(f"cache='ram' enabled: {ram_cache_allowed}")

    settings.update({"wandb": False})
    training = config["training"]
    gpu_name = torch.cuda.get_device_name(0)
    results: list[dict[str, Any]] = []

    for cache in CACHE_OPTIONS:
        if cache == "ram" and not ram_cache_allowed:
            for batch in BATCH_SIZES:
                results.append(
                    {
                        "batch": batch,
                        "cache": cache_label(cache),
                        "status": "skipped_insufficient_ram_for_cache",
                    }
                )
            continue

        for batch in BATCH_SIZES:
            label = cache_label(cache)
            print(f"\nBenchmarking batch={batch}, cache={label}...")
            result: dict[str, Any] = {"batch": batch, "cache": label, "status": "error"}
            timing, on_train_epoch_start, on_train_epoch_end = make_epoch_timer()
            try:
                model = YOLO(config["model"]["pretrained"])
                model.add_callback("on_train_epoch_start", on_train_epoch_start)
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                model.train(
                    data=str(data_yaml),
                    epochs=1,
                    fraction=FRACTION,
                    imgsz=config["model"]["imgsz"],
                    batch=batch,
                    workers=training["workers"],
                    device=training["device"],
                    seed=config["project"]["seed"],
                    optimizer=training["optimizer"],
                    lr0=training["lr0"],
                    momentum=training["momentum"],
                    weight_decay=training["weight_decay"],
                    amp=training["amp"],
                    cache=cache,
                    val=False,
                    save=False,
                    plots=False,
                    project=str(root / "experiments" / "debug"),
                    name=f"fp32_speed_b{batch}_cache_{label}",
                    exist_ok=True,
                    verbose=True,
                )
                result.update(timing)
                seconds = result.get("train_epoch_seconds")
                images = result.get("images_processed")
                batches = result.get("train_batches")
                if not all(isinstance(value, (int, float)) and value > 0 for value in (seconds, images, batches)):
                    raise RuntimeError("Epoch timing callbacks did not produce valid throughput inputs.")
                result["images_per_second"] = images / seconds
                result["iterations_per_second"] = batches / seconds
                actual_batch = result.get("actual_batch")
                result["status"] = "success" if actual_batch == batch else f"auto_reduced_to_{actual_batch}"
            except Exception as error:
                result["status"] = "oom" if is_cuda_oom(error) else "error"
                result["error"] = str(error)
                print(f"Benchmark batch={batch}, cache={label} failed: {error}", file=sys.stderr)
                if is_cuda_oom(error):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(0)
                    print("CUDA OOM recorded; CUDA cache cleared before the next configuration.", file=sys.stderr)
            finally:
                del timing
                gc.collect()
                torch.cuda.empty_cache()
            results.append(result)

    stable = [result for result in results if result["status"] == "success"]
    fastest = max(stable, key=lambda result: result["images_per_second"]) if stable else None
    print_table(results)
    if fastest:
        print(
            "\nFASTEST STABLE CONFIGURATION: "
            f"batch={fastest['batch']}, cache={fastest['cache']}, "
            f"{fastest['images_per_second']:.2f} img/s"
        )
    else:
        print("\nFASTEST STABLE CONFIGURATION: none")

    output_path = root / "experiments" / "debug" / "fp32_speed_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics_version": ultralytics_version,
        "gpu_name": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fraction": FRACTION,
        "cache_estimate": {
            "selected_images": selected_images,
            "estimated_decoded_gib": gibibytes(estimated_cache_bytes),
            "available_ram_gib": gibibytes(available_ram_bytes),
            "available_ram_fraction_limit": RAM_CACHE_AVAILABLE_FRACTION_LIMIT,
            "ram_cache_allowed": ram_cache_allowed,
        },
        "benchmark": {
            "imgsz": config["model"]["imgsz"],
            "workers": training["workers"],
            "device": training["device"],
            "seed": config["project"]["seed"],
            "optimizer": training["optimizer"],
            "lr0": training["lr0"],
            "momentum": training["momentum"],
            "weight_decay": training["weight_decay"],
            "amp": training["amp"],
            "val": False,
        },
        "results": results,
        "fastest_stable_configuration": fastest,
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Benchmark JSON written: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FP32 speed benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1)
