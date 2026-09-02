"""Phase 7G: low-resource isolation of NNCF/Ultralytics QAT export state.

This diagnostic runs exactly one deterministic VAL image through fresh QAT
model instances on the single visible CUDA device.  It does not export ONNX,
convert OpenVINO, move the model to CPU, train, calibrate, or read TEST.

It answers one narrow question: does normal native QAT inference change when
Ultralytics ``Detect.export`` and/or NNCF ``prepare_for_export()`` is applied?
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
import ultralytics

from export_openvino import (
    EXPECTED_QUANTIZERS,
    FP32_CHECKPOINT,
    QAT_CHECKPOINT,
    ExportStop,
    enable_ultralytics_detect_export,
    preprocess_deployment_image,
    resolve_val_images,
    restore_qat_with_controller,
)
from probe_qat_compatibility import require_cuda, require_project_root
from smoke_train_qat import quantizers, quantizers_enabled, sha256_file


OUTPUT_PATH = Path("artifacts/openvino/qat_export_state_diagnostic.json")
EXPECTED_STATE_ENTRIES = 1323
RELATIVE_MAE_LIMIT = 0.01


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def first_val_image() -> Path:
    directory = resolve_val_images()
    images = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ExportStop(f"No VAL images found: {directory}")
    return images[0]


def raw_prediction(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        output = model(tensor)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 7, 8400):
        raise ExportStop(f"Expected native raw output [1,7,8400], got {type(output).__name__} {getattr(output, 'shape', None)}")
    if not torch.isfinite(output).all():
        raise ExportStop("Model output contains non-finite values.")
    return output.detach().float().cpu().contiguous()


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    difference = (reference - candidate).abs()
    reference_mean = float(reference.abs().mean().item())
    mae = float(difference.mean().item())
    relative = mae / max(reference_mean, 1e-12)
    return {
        "status": relative <= RELATIVE_MAE_LIMIT,
        "mean absolute difference": mae,
        "mean absolute reference magnitude": reference_mean,
        "relative MAE": relative,
        "maximum absolute difference": float(difference.max().item()),
        "cosine similarity": float(torch.nn.functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0).item()),
        "maximum allowed relative MAE": RELATIVE_MAE_LIMIT,
    }


def model_state(model: torch.nn.Module) -> dict[str, Any]:
    items = quantizers(model)
    return {
        "model mode": "eval" if not model.training else "train",
        "quantizer count": len(items),
        "enabled quantizer count": sum(item.is_enabled_quantization() for item in items),
        "quantizers valid": len(items) == EXPECTED_QUANTIZERS and quantizers_enabled(items),
    }


def fresh_model(qat_path: Path, fp32_path: Path, device: torch.device) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    controller, model, restored = restore_qat_with_controller(qat_path, fp32_path, device)
    if restored["loaded state entries"] != EXPECTED_STATE_ENTRIES:
        raise ExportStop(f"Expected {EXPECTED_STATE_ENTRIES} restored NNCF entries, got {restored['loaded state entries']}.")
    if not model_state(model)["quantizers valid"]:
        raise ExportStop("Fresh QAT instance has invalid quantizers.")
    return controller, model, restored


def main() -> int:
    parse_args()
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "scope": "One deterministic VAL image; QAT CUDA forward only.",
        "not performed": [
            "ONNX export",
            "OpenVINO conversion or inference",
            "CPU model transfer",
            "PTQ or calibration",
            "training",
            "TEST loading",
        ],
    }
    try:
        require_project_root(ROOT)
        fp32_path = (ROOT / FP32_CHECKPOINT).resolve()
        qat_path = (ROOT / QAT_CHECKPOINT).resolve()
        if not fp32_path.is_file() or not qat_path.is_file():
            raise ExportStop("Required fixed checkpoint is missing.")
        device, gpu_name = require_cuda()
        if torch.cuda.device_count() != 1:
            raise ExportStop(f"Exactly one visible GPU is required, found {torch.cuda.device_count()}.")
        image_path = first_val_image()
        image = preprocess_deployment_image(image_path, device)
        report.update(
            {
                "source FP32 checkpoint": str(fp32_path),
                "source QAT checkpoint": str(qat_path),
                "QAT SHA256": sha256_file(qat_path),
                "VAL image": str(image_path),
                "preprocessing": "cv2 decode -> LetterBox(640, auto=False, stride=32) -> BGR to RGB -> NCHW float32 / 255",
                "GPU": {"logical device": "cuda:0", "name": gpu_name},
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "versions": {"Python": platform.python_version(), "torch": torch.__version__, "nncf": nncf.__version__, "ultralytics": ultralytics.__version__},
            }
        )

        # Independent baseline proves that a second reconstruction itself did
        # not change output before testing either export-specific operation.
        _base_ctrl, base_model, base_restore = fresh_model(qat_path, fp32_path, device)
        reference = raw_prediction(base_model, image)
        report["native reference"] = {"restore": base_restore, "state": model_state(base_model), "output shape": list(reference.shape)}
        del base_model, _base_ctrl
        torch.cuda.empty_cache()

        cases: list[tuple[str, bool, bool]] = [
            ("fresh normal control", False, False),
            ("Detect.export only", True, False),
            ("NNCF prepare_for_export only", False, True),
            ("Detect.export plus prepare_for_export", True, True),
        ]
        results: dict[str, Any] = {}
        first_failure: str | None = None
        for label, detect_export, prepare in cases:
            controller, model, restored = fresh_model(qat_path, fp32_path, device)
            before = model_state(model)
            if detect_export:
                enable_ultralytics_detect_export(model)
            if prepare:
                controller.prepare_for_export()
            after = model_state(model)
            candidate = raw_prediction(model, image)
            result = {"restore": restored, "before": before, "after": after, "comparison": compare(reference, candidate)}
            results[label] = result
            if first_failure is None and not result["comparison"]["status"]:
                first_failure = label
            del model, controller
            torch.cuda.empty_cache()
            # This is a root-cause isolator, not an experiment sweep. Once a
            # state diverges, later states cannot make the first divergence
            # disappear and must not consume shared-server resources.
            if first_failure is not None:
                break
        report["cases"] = results
        report["first divergent state"] = first_failure
        if first_failure is None:
            report["status"] = "PASS: export-state operations preserve one representative native QAT forward."
            report["next step"] = "A single-image ONNX/OpenVINO retry may be justified before any 32-VAL acceptance run."
            print("QAT export-state diagnostic passed; no ONNX/OpenVINO work was performed.")
        else:
            report["status"] = "STOPPED: native QAT output changes before ONNX export."
            report["next step"] = "Do not retry ONNX/OpenVINO until the first divergent export state is diagnosed."
            print(f"QAT export-state diagnostic stopped at: {first_failure}", file=sys.stderr)
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        print(f"QAT export-state diagnostic failed: {report['error']}", file=sys.stderr)
    finally:
        write_json(ROOT / OUTPUT_PATH, report)
        print(f"Diagnostic: {ROOT / OUTPUT_PATH}")
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
