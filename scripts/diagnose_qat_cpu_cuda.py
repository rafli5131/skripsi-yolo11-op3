"""Diagnose CPU/CUDA numerical equivalence of the restored final NNCF QAT model.

This diagnostic performs no OpenVINO conversion, PTQ, calibration, training,
strip(), or TEST access. It compares raw [1, 7, 8400] YOLO outputs only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path.cwd().resolve()
load_dotenv(ROOT / ".env", override=False)

import nncf
import torch
import torchvision
import ultralytics

from export_openvino import (
    EXPECTED_QUANTIZERS,
    FP32_CHECKPOINT,
    INPUT_SHAPE,
    QAT_CHECKPOINT,
    ExportStop,
    preprocess_deployment_image,
    resolve_val_images,
    restore_qat_with_controller,
)
from probe_qat_compatibility import require_cuda, require_project_root
from smoke_train_qat import quantizers, quantizers_enabled, sha256_file


OUTPUT_PATH = Path("artifacts/openvino/qat_cpu_cuda_diagnostic.json")
VAL_COUNT = 32
SEED = 42


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def image_paths(val_images: Path) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png"}
    paths = sorted(path for path in val_images.iterdir() if path.is_file() and path.suffix.lower() in allowed)
    if len(paths) < VAL_COUNT:
        raise ExportStop(f"Need exactly {VAL_COUNT} VAL images, found only {len(paths)} at {val_images}")
    return paths[:VAL_COUNT]


def stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(tensor.device),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
    }


def aggregate_tensor_stats(tensors: list[torch.Tensor]) -> dict[str, Any]:
    total = 0
    total_sum = 0.0
    total_square_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for tensor in tensors:
        value = tensor.detach().cpu().double()
        total += value.numel()
        total_sum += float(value.sum().item())
        total_square_sum += float((value * value).sum().item())
        minimum = min(minimum, float(value.min().item()))
        maximum = max(maximum, float(value.max().item()))
    mean = total_sum / total
    variance = max(total_square_sum / total - mean * mean, 0.0)
    return {
        "tensor_count": len(tensors),
        "per_tensor_shape": list(tensors[0].shape),
        "dtype": str(tensors[0].dtype),
        "value_count": total,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": variance**0.5,
    }


def raw_output(model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        output = model(input_tensor)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ExportStop(f"Expected raw YOLO tensor output, got {type(output).__name__}")
    if tuple(output.shape) != (1, 7, 8400):
        raise ExportStop(f"Expected raw output [1, 7, 8400], got {list(output.shape)}")
    return output.detach().float().cpu()


def compare(cuda_output: torch.Tensor, cpu_output: torch.Tensor) -> dict[str, Any]:
    difference = (cuda_output - cpu_output).abs()
    reference_magnitude = float(cuda_output.abs().mean().item())
    mae = float(difference.mean().item())
    cuda_flat = cuda_output.reshape(-1)
    cpu_flat = cpu_output.reshape(-1)
    cosine = float(torch.nn.functional.cosine_similarity(cuda_flat, cpu_flat, dim=0).item())
    relative_mae = mae / max(reference_magnitude, 1e-12)
    return {
        "output_shape": list(cuda_output.shape),
        "mean_absolute_error": mae,
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_reference_magnitude": reference_magnitude,
        "relative_MAE": relative_mae,
        "cosine_similarity": cosine,
        "passes_relative_MAE_le_1_percent": relative_mae <= 0.01,
    }


def quantizer_state(model: torch.nn.Module) -> dict[str, int]:
    items = quantizers(model)
    return {
        "quantizer_count": len(items),
        "enabled_quantizer_count": sum(item.is_enabled_quantization() for item in items),
    }


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ExportStop("Cannot calculate percentile from no VAL comparisons.")
    index = (len(sorted_values) - 1) * fraction
    low, high = int(index), min(int(index) + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def val_summary(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted((report["relative_MAE"], filename) for filename, report in comparisons.items())
    values = [value for value, _ in ranked]
    median = percentile(values, 0.5)
    median_nearest = min(ranked, key=lambda item: abs(item[0] - median))
    return {
        "minimum_relative_MAE": values[0],
        "median_relative_MAE": median,
        "mean_relative_MAE": sum(values) / len(values),
        "p95_relative_MAE": percentile(values, 0.95),
        "maximum_relative_MAE": values[-1],
        "count_relative_MAE_le_1_percent": sum(value <= 0.01 for value in values),
        "count_relative_MAE_gt_1_percent": sum(value > 0.01 for value in values),
        "minimum_filename": ranked[0][1],
        "median_nearest_filename": median_nearest[1],
        "maximum_filename": ranked[-1][1],
    }


def write_json(payload: dict[str, Any]) -> None:
    destination = ROOT / OUTPUT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parse_args()
    report: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "started"}
    try:
        require_project_root(ROOT)
        device, gpu_name = require_cuda()
        if torch.cuda.device_count() != 1:
            raise ExportStop(f"Exactly one visible GPU is required, found {torch.cuda.device_count()}.")
        fp32_path = (ROOT / FP32_CHECKPOINT).resolve()
        qat_path = (ROOT / QAT_CHECKPOINT).resolve()
        if not fp32_path.is_file() or not qat_path.is_file():
            raise ExportStop("Required fixed FP32 or QAT checkpoint is missing.")

        val_images = resolve_val_images()
        selected_paths = image_paths(val_images)
        val_tensors = [preprocess_deployment_image(path, device) for path in selected_paths]
        original_synthetic = torch.linspace(
            -1.0, 1.0, steps=int(torch.tensor(INPUT_SHAPE).prod()), dtype=torch.float32, device=device
        ).reshape(INPUT_SHAPE)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        in_domain_synthetic = torch.rand(INPUT_SHAPE, generator=generator, dtype=torch.float32).to(device)

        report.update(
            {
                "versions": {
                    "Python": platform.python_version(),
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "ultralytics": ultralytics.__version__,
                    "nncf": nncf.__version__,
                },
                "GPU": {"logical device": "cuda:0", "name": gpu_name},
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "source QAT checkpoint": str(qat_path),
                "QAT SHA256": sha256_file(qat_path),
                "test split": "not loaded or used",
                "original synthetic generation method": (
                    "torch.linspace(-1.0, 1.0, steps=1*3*640*640, dtype=torch.float32).reshape([1,3,640,640]).to(cuda:0)"
                ),
                "original synthetic statistics": stats(original_synthetic),
                "in-domain synthetic generation method": "torch.Generator(device='cpu').manual_seed(42); torch.rand([1,3,640,640], dtype=torch.float32).to(cuda:0)",
                "in-domain synthetic statistics": stats(in_domain_synthetic),
                "VAL selection": {"count": VAL_COUNT, "method": "first 32 filenames in deterministic lexical sort"},
                "VAL preprocessing": "shared export_openvino.preprocess_deployment_image: cv2 decode -> LetterBox(640, auto=False, stride=32) -> BGR to RGB -> NCHW float32 / 255",
                "VAL tensor aggregate statistics": aggregate_tensor_stats(val_tensors),
                "VAL filenames": [path.name for path in selected_paths],
            }
        )

        _controller, model, restored = restore_qat_with_controller(qat_path, fp32_path, device)
        report["NNCF restoration"] = restored
        report["quantizers before CUDA inference"] = quantizer_state(model)
        if report["quantizers before CUDA inference"] != {
            "quantizer_count": EXPECTED_QUANTIZERS,
            "enabled_quantizer_count": EXPECTED_QUANTIZERS,
        }:
            raise ExportStop("Unexpected CUDA quantizer state.")

        cuda_outputs = {
            "original_synthetic": raw_output(model, original_synthetic),
            "in_domain_synthetic": raw_output(model, in_domain_synthetic),
            **{path.name: raw_output(model, tensor) for path, tensor in zip(selected_paths, val_tensors)},
        }
        model.cpu().eval()
        report["quantizers after CPU move"] = quantizer_state(model)
        if report["quantizers after CPU move"] != {
            "quantizer_count": EXPECTED_QUANTIZERS,
            "enabled_quantizer_count": EXPECTED_QUANTIZERS,
        }:
            raise ExportStop("Unexpected CPU quantizer state.")

        cpu_inputs = {
            "original_synthetic": original_synthetic.cpu(),
            "in_domain_synthetic": in_domain_synthetic.cpu(),
            **{path.name: tensor.cpu() for path, tensor in zip(selected_paths, val_tensors)},
        }
        comparisons = {name: compare(cuda_outputs[name], raw_output(model, tensor)) for name, tensor in cpu_inputs.items()}
        val_comparisons = {path.name: comparisons[path.name] for path in selected_paths}
        report["comparisons"] = comparisons
        report["VAL relative-MAE summary"] = val_summary(val_comparisons)

        old_fails = not comparisons["original_synthetic"]["passes_relative_MAE_le_1_percent"]
        domain_passes = comparisons["in_domain_synthetic"]["passes_relative_MAE_le_1_percent"]
        val_fails = report["VAL relative-MAE summary"]["count_relative_MAE_gt_1_percent"]
        if old_fails and domain_passes and val_fails == 0:
            report["classification"] = "Mismatch is likely input-distribution-specific."
        else:
            report["classification"] = "CPU/CUDA QAT backend equivalence remains unresolved."
        report["status"] = "passed"
        print(report["classification"])
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        print(f"QAT CPU/CUDA diagnostic failed: {report['error']}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        write_json(report)
        print(f"Report: {ROOT / OUTPUT_PATH}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
