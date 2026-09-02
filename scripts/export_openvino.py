"""Export FP32 or diagnose unstripped true-NNCF-QAT YOLO11n OpenVINO paths.

The QAT diagnostic deliberately uses neither ``nncf.quantize`` nor an
Ultralytics ``int8`` export. Phase 7A proved that ``nncf.torch.strip`` changes
the restored QAT model numerically, so this script never invokes it for QAT.
It tests only direct conversion of the still-compressed model and the installed
controller's official ONNX export, with no calibration data or PTQ stage.

Run this only on the CUDA SSH host from the project root. QAT diagnostics use
exactly 32 deterministic VAL images and one deterministic in-domain synthetic
input; they never load TEST.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path.cwd().resolve()
load_dotenv(ROOT / ".env", override=False)

import cv2
import nncf
import openvino as ov
import torch
import torchvision
import ultralytics
import yaml
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, load_state
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.export.openvino import torch2openvino

from probe_qat_compatibility import (
    attach_ultralytics_training_args,
    require_cuda,
    require_project_root,
    restore_ultralytics_trainer_trainability,
)
from smoke_train_qat import quantizers, quantizers_enabled, sha256_file


FP32_CHECKPOINT = Path("artifacts/checkpoints/fp32/best.pt")
QAT_CHECKPOINT = Path("artifacts/checkpoints/qat/qat_best.pt")
DATA_YAML = Path("data/dataset/data.yaml")
OPENVINO_ROOT = Path("artifacts/openvino")
FP32_DIR = OPENVINO_ROOT / "fp32"
QAT_DIR = OPENVINO_ROOT / "qat_int8"
MANIFEST_PATH = OPENVINO_ROOT / "export_manifest.json"
GRAPH_REPORT_PATH = OPENVINO_ROOT / "graph_report.json"
QAT_DIAGNOSTIC_PATH = OPENVINO_ROOT / "qat_export_diagnostic.json"
CPU_CUDA_DIAGNOSTIC_PATH = OPENVINO_ROOT / "qat_cpu_cuda_diagnostic.json"
DIRECT_QAT_DIR = OPENVINO_ROOT / "debug/direct_qat"
CONTROLLER_QAT_DIR = OPENVINO_ROOT / "debug/controller_qat"
EXPECTED_NAMES = {0: "ball", 1: "gawang", 2: "robot"}
EXPECTED_QUANTIZERS = 184
INPUT_SHAPE = (1, 3, 640, 640)
REPRESENTATIVE_VAL_COUNT = 32
REPRESENTATIVE_SEED = 42


class ExportStop(RuntimeError):
    """Controlled error that prevents a misleading OpenVINO artifact."""


class TensorOutputExportWrapper(torch.nn.Module):
    """Use the single prediction tensor emitted by Ultralytics Detect export mode."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        # DetectionModel normally returns (prediction, feature-dict) outside
        # export mode. The first item is the official detect prediction tensor.
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"Expected a tensor export output, got {type(output).__name__}")
        return output


def enable_ultralytics_detect_export(model: torch.nn.Module) -> None:
    """Mirror the installed Ultralytics Exporter Detect-head export settings."""
    heads = 0
    for module in model.modules():
        if module.__class__.__name__ == "Detect":
            module.dynamic = False
            module.export = True
            module.format = "openvino"
            module.shape = None
            heads += 1
    if heads != 1:
        raise ExportStop(f"Expected one Ultralytics Detect head for export, found {heads}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("fp32", "qat-diagnostic"),
        default="qat-diagnostic",
        help="Preserve accepted FP32 export, or run isolated fail-closed QAT export diagnostics.",
    )
    return parser.parse_args()


def normalized_names(names: Any) -> dict[int, str]:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ExportStop(f"Unexpected model class mapping type: {type(names).__name__}")


def require_checkpoint(path: Path) -> Path:
    path = (ROOT / path).resolve()
    if not path.is_file():
        raise ExportStop(f"Required fixed checkpoint is missing: {path}")
    return path


def resolve_val_images() -> Path:
    data_yaml = (ROOT / DATA_YAML).resolve()
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ExportStop(f"Invalid dataset YAML: {data_yaml}")
    value = raw.get("val")
    if not isinstance(value, str) or not value:
        raise ExportStop("Canonical dataset YAML has no valid VAL path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        data_root = Path(raw.get("path") or data_yaml.parent).expanduser()
        if not data_root.is_absolute():
            data_root = (data_yaml.parent / data_root).resolve()
        path = data_root / path
    path = path.resolve()
    if not path.is_dir() or path.name != "images" or not path.is_relative_to(ROOT):
        raise ExportStop(f"VAL images do not resolve under the project root: {path}")
    return path


def preprocess_deployment_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """Phase 7 deployment sanity preprocessing: letterbox, RGB, NCHW float32 [0, 1]."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ExportStop(f"Unable to read image: {image_path}")
    image = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).contiguous().unsqueeze(0).float().div_(255.0)
    return tensor.to(device)


def val_input(val_images: Path, device: torch.device) -> tuple[torch.Tensor, str]:
    candidates = sorted(
        image for image in val_images.iterdir() if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not candidates:
        raise ExportStop(f"No image was found in VAL directory: {val_images}")
    image_path = candidates[0]
    return preprocess_deployment_image(image_path, device), str(image_path)


def representative_qat_inputs(val_images: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return the fixed Phase 7C-compatible [0, 1] synthetic and 32 VAL inputs."""
    candidates = sorted(
        image for image in val_images.iterdir() if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(candidates) < REPRESENTATIVE_VAL_COUNT:
        raise ExportStop(f"Need {REPRESENTATIVE_VAL_COUNT} VAL images, found {len(candidates)} at {val_images}")
    generator = torch.Generator(device="cpu").manual_seed(REPRESENTATIVE_SEED)
    inputs: dict[str, torch.Tensor] = {
        "in_domain_synthetic": torch.rand(INPUT_SHAPE, generator=generator, dtype=torch.float32).to(device)
    }
    val_names: list[str] = []
    for image_path in candidates[:REPRESENTATIVE_VAL_COUNT]:
        if image_path.name in inputs:
            raise ExportStop(f"Duplicate representative input key: {image_path.name}")
        inputs[image_path.name] = preprocess_deployment_image(image_path, device)
        val_names.append(image_path.name)
    return inputs, val_names


def tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensor_leaves(item)]
    raise ExportStop(f"Unsupported PyTorch output type during sanity check: {type(value).__name__}")


def output_summary(value: Any) -> dict[str, Any]:
    leaves = tensor_leaves(value)
    flat = torch.cat([leaf.detach().float().cpu().reshape(-1) for leaf in leaves])
    return {
        "tensor_shapes": [list(tensor.shape) for tensor in leaves],
        "element_count": int(flat.numel()),
        "finite": bool(torch.isfinite(flat).all().item()),
        "mean_absolute_value": float(flat.abs().mean().item()),
        "max_absolute_value": float(flat.abs().max().item()),
        "flat": flat,
    }


def compare_summaries(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Require a conservative numerical check; this is compatibility, not accuracy tuning."""
    same_elements = reference["element_count"] == candidate["element_count"]
    same_shapes = reference["tensor_shapes"] == candidate["tensor_shapes"]
    if same_elements:
        difference = (reference["flat"] - candidate["flat"]).abs()
        mean_abs_difference = float(difference.mean().item())
        max_abs_difference = float(difference.max().item())
        reference_mean_abs = float(reference["flat"].abs().mean().item())
        relative_mean_abs_difference = mean_abs_difference / max(reference_mean_abs, 1e-12)
        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(reference["flat"], candidate["flat"], dim=0).item()
        )
    else:
        mean_abs_difference = None
        max_abs_difference = None
        relative_mean_abs_difference = None
        cosine_similarity = None
    # FP32 conversion is normally far below this. A 1% relative MAE bound is
    # a compatibility guard, not a model-selection or tuning parameter.
    status = bool(
        reference["finite"]
        and candidate["finite"]
        and same_elements
        and same_shapes
        and relative_mean_abs_difference is not None
        and relative_mean_abs_difference <= 0.01
    )
    return {
        "status": status,
        "element_count_matches": same_elements,
        "tensor_shapes_match": same_shapes,
        "max_absolute_difference": max_abs_difference,
        "mean_absolute_difference": mean_abs_difference,
        "relative_mean_absolute_difference": relative_mean_abs_difference,
        "cosine_similarity": cosine_similarity,
        "maximum_allowed_relative_mean_absolute_difference": 0.01,
    }


def ov_output_summary(compiled_model: Any, input_tensor: torch.Tensor) -> dict[str, Any]:
    outputs = compiled_model([input_tensor.detach().cpu().numpy()])
    arrays = [torch.from_numpy(value).float().reshape(-1) for value in outputs.values()]
    flat = torch.cat(arrays)
    return {
        "tensor_shapes": [list(value.shape) for value in outputs.values()],
        "element_count": int(flat.numel()),
        "finite": bool(torch.isfinite(flat).all().item()),
        "mean_absolute_value": float(flat.abs().mean().item()),
        "max_absolute_value": float(flat.abs().max().item()),
        "flat": flat,
    }


def sanity_check(
    label: str,
    pytorch_model: torch.nn.Module,
    xml_path: Path,
    input_tensor: torch.Tensor,
) -> dict[str, Any]:
    pytorch_model.eval()
    with torch.inference_mode():
        torch_summary = output_summary(pytorch_model(input_tensor))
    core = ov.Core()
    ov_model = core.read_model(xml_path)
    compiled = core.compile_model(ov_model, "CPU")
    ov_summary = ov_output_summary(compiled, input_tensor)
    comparison = compare_summaries(torch_summary, ov_summary)
    for summary in (torch_summary, ov_summary):
        summary.pop("flat")
    return {
        "status": comparison["status"],
        "val_input": "One fixed VAL image only; no TEST image was loaded.",
        "pytorch": torch_summary,
        "openvino": ov_summary,
        **comparison,
    }


def graph_report(xml_path: Path) -> dict[str, Any]:
    model = ov.Core().read_model(xml_path)
    operations = model.get_ordered_ops()
    type_counts: dict[str, int] = {}
    element_type_counts: dict[str, int] = {}
    int8_related: list[dict[str, str]] = []
    for operation in operations:
        op_type = operation.get_type_name()
        type_counts[op_type] = type_counts.get(op_type, 0) + 1
        for index in range(operation.get_output_size()):
            element_type = str(operation.get_output_element_type(index))
            element_type_counts[element_type] = element_type_counts.get(element_type, 0) + 1
            if any(token in element_type.lower() for token in ("i8", "u8", "int8", "uint8")):
                int8_related.append({"op": op_type, "name": operation.get_friendly_name(), "element_type": element_type})
    return {
        "input_element_types": [str(port.get_element_type()) for port in model.inputs],
        "input_partial_shapes": [str(port.get_partial_shape()) for port in model.inputs],
        "output_element_types": [str(port.get_element_type()) for port in model.outputs],
        "output_partial_shapes": [str(port.get_partial_shape()) for port in model.outputs],
        "total_op_count": len(operations),
        "operation_type_counts": type_counts,
        "FakeQuantize_op_count": type_counts.get("FakeQuantize", 0),
        "Convert_op_count": type_counts.get("Convert", 0),
        "element_type_counts": element_type_counts,
        "int8_or_u8_graph_elements": int8_related,
        "int8_or_u8_graph_element_count": len(int8_related),
    }


def file_sizes(xml_path: Path) -> dict[str, int]:
    bin_path = xml_path.with_suffix(".bin")
    if not xml_path.is_file() or not bin_path.is_file():
        raise ExportStop(f"OpenVINO serialization did not produce both XML and BIN: {xml_path}")
    xml_bytes = xml_path.stat().st_size
    bin_bytes = bin_path.stat().st_size
    return {"XML bytes": xml_bytes, "BIN bytes": bin_bytes, "total bytes": xml_bytes + bin_bytes}


def save_openvino_ir(model: torch.nn.Module, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "model.xml"
    cpu_model = model.eval().cpu()
    example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    traced = torch.jit.trace(cpu_model, example, strict=False, check_trace=False)
    ov_model = ov.convert_model(traced, input=list(INPUT_SHAPE), example_input=example)
    # False preserves FP32 constants in FP32 export and preserves QAT graph data exactly as converted.
    ov.save_model(ov_model, xml_path, compress_to_fp16=False)
    return xml_path


def load_fp32(path: Path, device: torch.device) -> torch.nn.Module:
    model = YOLO(str(path)).model
    attach_ultralytics_training_args(model, ROOT)
    if normalized_names(model.names) != EXPECTED_NAMES:
        raise ExportStop("FP32 checkpoint class mapping differs from the fixed three-class mapping.")
    return model.to(device).eval()


def load_qat(path: Path, fp32_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "compression_state", "nncf_config"}
    missing = sorted(required.difference(saved))
    if missing:
        raise ExportStop(f"QAT checkpoint lacks NNCF reconstruction state: {missing}")
    fp32_model = YOLO(str(fp32_path)).model
    attach_ultralytics_training_args(fp32_model, ROOT)
    restore_ultralytics_trainer_trainability(fp32_model)
    _controller, qat_model = create_compressed_model(
        fp32_model,
        NNCFConfig.from_dict(saved["nncf_config"]),
        compression_state=saved["compression_state"],
        dump_graphs=False,
    )
    loaded_entries = load_state(qat_model, saved["model_state_dict"], is_resume=True)
    qat_model.to(device).eval()
    items = quantizers(qat_model)
    if len(items) != EXPECTED_QUANTIZERS:
        raise ExportStop(f"Expected {EXPECTED_QUANTIZERS} NNCF quantizers, found {len(items)}.")
    if not quantizers_enabled(items):
        raise ExportStop("QAT fake quantization is inactive before export.")
    if normalized_names(qat_model.names) != EXPECTED_NAMES:
        raise ExportStop("QAT checkpoint class mapping differs from the fixed three-class mapping.")
    return qat_model, {"loaded_state_entries": int(loaded_entries), "quantizer_count": len(items)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "flat"}


def controller_api_report(controller: Any) -> dict[str, Any]:
    export_method = getattr(controller, "export_model", None)
    return {
        "controller class": f"{type(controller).__module__}.{type(controller).__qualname__}",
        "public controller members": sorted(name for name in dir(controller) if not name.startswith("_")),
        "export_model available": callable(export_method),
        "export_model signature": str(inspect.signature(export_method)) if callable(export_method) else None,
        "export_model supported format from installed PTExporter": ["onnx", "onnx_<opset_version>"],
        "direct openvino.convert_model signature": str(inspect.signature(ov.convert_model)),
    }


def phase_7c_evidence() -> dict[str, Any]:
    """Carry audited representative-input evidence into the export diagnostic."""
    path = ROOT / CPU_CUDA_DIAGNOSTIC_PATH
    if not path.is_file():
        raise ExportStop(f"Phase 7C diagnostic is required before QAT export: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = "Mismatch is likely input-distribution-specific."
    if report.get("classification") != expected:
        raise ExportStop("Phase 7C did not establish representative CPU/CUDA compatibility.")
    return {
        "path": str(path),
        "classification": expected,
        "in-domain synthetic statistics": report.get("in-domain synthetic statistics"),
        "VAL relative-MAE summary": report.get("VAL relative-MAE summary"),
    }


def representative_summary(numerical: dict[str, dict[str, Any]], val_names: list[str]) -> dict[str, Any]:
    """Summarize the fixed acceptance population without using TEST data."""
    values = [numerical[name]["relative_mean_absolute_difference"] for name in val_names]
    if any(value is None for value in values):
        raise ExportStop("A candidate did not produce comparable output for every representative VAL input.")
    ordered = sorted(float(value) for value in values)
    p95_index = (len(ordered) - 1) * 0.95
    low, high = int(p95_index), min(int(p95_index) + 1, len(ordered) - 1)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (p95_index - low)
    return {
        "VAL input count": len(val_names),
        "minimum relative MAE": ordered[0],
        "median relative MAE": (ordered[15] + ordered[16]) / 2,
        "mean relative MAE": sum(ordered) / len(ordered),
        "p95 relative MAE": p95,
        "maximum relative MAE": ordered[-1],
        "count relative MAE <= 1 percent": sum(value <= 0.01 for value in ordered),
        "count relative MAE > 1 percent": sum(value > 0.01 for value in ordered),
        "in-domain synthetic relative MAE": numerical["in_domain_synthetic"]["relative_mean_absolute_difference"],
    }


def qat_cuda_cpu_equivalence(
    model: torch.nn.Module,
    input_cases: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compare the identical restored model on CUDA then CPU before any export."""
    # Phase 7C compared the restored model's native prediction tensor. Do not
    # mutate Detect.export before this check: that is export preparation, not
    # the validated CUDA/CPU reconstruction path.
    prediction_model = TensorOutputExportWrapper(model).eval()
    cuda_outputs: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for name, input_tensor in input_cases.items():
            cuda_outputs[name] = output_summary(prediction_model(input_tensor))
    model.cpu().eval()
    cpu_quantizers = quantizers(model)
    count_after_cpu = len(cpu_quantizers)
    enabled_after_cpu = quantizers_enabled(cpu_quantizers)
    cpu_outputs: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Any] = {}
    with torch.inference_mode():
        for name, input_tensor in input_cases.items():
            cpu_outputs[name] = output_summary(prediction_model(input_tensor.cpu()))
            comparisons[name] = {
                "CUDA reference": compact_summary(cuda_outputs[name]),
                "CPU candidate": compact_summary(cpu_outputs[name]),
                **compare_summaries(cuda_outputs[name], cpu_outputs[name]),
            }
    status = (
        count_after_cpu == EXPECTED_QUANTIZERS
        and enabled_after_cpu
        and all(item["status"] for item in comparisons.values())
    )
    return {
        "status": status,
        "quantizer count after model.cpu": count_after_cpu,
        "quantizers enabled after model.cpu": enabled_after_cpu,
        "per input": comparisons,
    }, cpu_outputs


def candidate_from_direct_openvino(
    model: torch.nn.Module,
    input_cases: dict[str, torch.Tensor],
    references: dict[str, dict[str, Any]],
    val_names: list[str],
) -> dict[str, Any]:
    """Candidate A: installed Ultralytics plain OpenVINO conversion of unstripped QAT."""
    report: dict[str, Any] = {
        "export path": "ultralytics.utils.export.openvino.torch2openvino(compressed_model wrapper, quantize=None)",
        "torch2openvino signature": str(inspect.signature(torch2openvino)),
        "quantize": None,
        "calibration_dataset": None,
        "int8_detect": False,
        "success": False,
        "calibration occurred": False,
        "PTQ used": False,
    }
    try:
        DIRECT_QAT_DIR.mkdir(parents=True, exist_ok=True)
        example = input_cases["in_domain_synthetic"].cpu()
        # The wrapper exposes only the raw YOLO prediction tensor and avoids
        # tracing Ultralytics' auxiliary feature dictionary. The compressed
        # NNCF model itself remains unstripped and fake quantization stays on.
        export_model = TensorOutputExportWrapper(model).eval().cpu()
        torch2openvino(
            model=export_model,
            im=example,
            output_dir=DIRECT_QAT_DIR,
            dynamic=False,
            quantize=None,
            calibration_dataset=None,
            int8_detect=False,
            prefix="QAT Candidate A:",
        )
        xml_path = DIRECT_QAT_DIR / "model.xml"
        report["files"] = file_sizes(xml_path)
        core = ov.Core()
        compiled = core.compile_model(core.read_model(xml_path), "CPU")
        numerical: dict[str, Any] = {}
        for name, input_tensor in input_cases.items():
            output = ov_output_summary(compiled, input_tensor.cpu())
            numerical[name] = {"candidate": compact_summary(output), **compare_summaries(references[name], output)}
        report["numerical checks"] = numerical
        report["representative numerical summary"] = representative_summary(numerical, val_names)
        numerical_ok = all(item["status"] for item in numerical.values())
        if not numerical_ok:
            report.update({"numerical equivalence": False, "rejection reason": "Representative numerical equivalence exceeds 1%."})
            return report
        report["graph"] = graph_report(xml_path)
        graph_ok = (
            report["graph"]["FakeQuantize_op_count"] > 0
            and report["graph"]["int8_or_u8_graph_element_count"] > 0
        )
        report.update(
            {
                "quantization graph visible": graph_ok,
                "numerical equivalence": numerical_ok,
                "success": graph_ok and numerical_ok,
            }
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def candidate_from_controller_export(
    controller: Any,
    model: torch.nn.Module,
    input_cases: dict[str, torch.Tensor],
    references: dict[str, dict[str, Any]],
    val_names: list[str],
) -> dict[str, Any]:
    """Candidate B: installed controller export_model() to ONNX, then OpenVINO conversion."""
    report: dict[str, Any] = {
        "export path": "compression_ctrl.export_model(onnx) -> ov.convert_model(onnx) without strip",
        "success": False,
        "calibration occurred": False,
        "PTQ used": False,
    }
    try:
        if not callable(getattr(controller, "export_model", None)):
            raise ExportStop("Installed compression controller has no callable export_model().")
        CONTROLLER_QAT_DIR.mkdir(parents=True, exist_ok=True)
        onnx_path = CONTROLLER_QAT_DIR / "model.onnx"
        controller.export_model(str(onnx_path), save_format="onnx")
        if not onnx_path.is_file():
            raise ExportStop("compression_ctrl.export_model() returned without producing ONNX.")
        ov_model = ov.convert_model(onnx_path)
        xml_path = CONTROLLER_QAT_DIR / "model.xml"
        ov.save_model(ov_model, xml_path, compress_to_fp16=False)
        report["files"] = {**file_sizes(xml_path), "ONNX bytes": onnx_path.stat().st_size}
        core = ov.Core()
        compiled = core.compile_model(core.read_model(xml_path), "CPU")
        numerical: dict[str, Any] = {}
        for name, input_tensor in input_cases.items():
            output = ov_output_summary(compiled, input_tensor.cpu())
            numerical[name] = {"candidate": compact_summary(output), **compare_summaries(references[name], output)}
        report["numerical checks"] = numerical
        report["representative numerical summary"] = representative_summary(numerical, val_names)
        numerical_ok = all(item["status"] for item in numerical.values())
        if not numerical_ok:
            report.update({"numerical equivalence": False, "rejection reason": "Representative numerical equivalence exceeds 1%."})
            return report
        report["graph"] = graph_report(xml_path)
        graph_ok = (
            report["graph"]["FakeQuantize_op_count"] > 0
            and report["graph"]["int8_or_u8_graph_element_count"] > 0
        )
        report.update(
            {
                "quantization graph visible": graph_ok,
                "numerical equivalence": numerical_ok,
                "success": graph_ok and numerical_ok,
            }
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def restore_qat_with_controller(qat_path: Path, fp32_path: Path, device: torch.device) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    saved = torch.load(qat_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "compression_state", "nncf_config"}
    missing = sorted(required.difference(saved))
    if missing:
        raise ExportStop(f"QAT checkpoint lacks NNCF reconstruction state: {missing}")
    fp32_model = YOLO(str(fp32_path)).model
    attach_ultralytics_training_args(fp32_model, ROOT)
    restore_ultralytics_trainer_trainability(fp32_model)
    controller, qat_model = create_compressed_model(
        fp32_model,
        NNCFConfig.from_dict(saved["nncf_config"]),
        compression_state=saved["compression_state"],
        dump_graphs=False,
    )
    loaded_entries = load_state(qat_model, saved["model_state_dict"], is_resume=True)
    qat_model.to(device).eval()
    items = quantizers(qat_model)
    if len(items) != EXPECTED_QUANTIZERS or not quantizers_enabled(items):
        raise ExportStop(f"Restored QAT quantizers invalid: count={len(items)}, enabled={quantizers_enabled(items)}")
    return controller, qat_model, {"loaded state entries": int(loaded_entries), "quantizer count": len(items)}


def run_qat_export_diagnostics(qat_path: Path, fp32_path: Path, device: torch.device, val_images: Path) -> dict[str, Any]:
    """Run only approved, unstripped candidates and promote solely on dual acceptance."""
    input_cases, val_names = representative_qat_inputs(val_images, device)
    controller, reference_model, restored = restore_qat_with_controller(qat_path, fp32_path, device)
    diagnostic: dict[str, Any] = {
        "restored QAT": restored,
        "pre-export model training state": bool(reference_model.training),
        "controller API inspection": controller_api_report(controller),
        "strip route": "disabled after Phase 7A semantic mismatch; not invoked",
        "Phase 7C evidence": phase_7c_evidence(),
        "representative inputs": {
            "in-domain synthetic": "torch.rand([1,3,640,640], seed=42, float32, [0,1])",
            "VAL image count": len(val_names),
            "VAL selection": "first 32 filenames in deterministic lexical sort",
            "preprocessing": "cv2 decode -> LetterBox(640, auto=False, stride=32) -> BGR to RGB -> NCHW float32 / 255",
            "TEST": "not loaded or used",
        },
        "calibration occurred": False,
        "PTQ used": False,
        "test split": "not loaded or used",
    }
    cpu_check, references = qat_cuda_cpu_equivalence(reference_model, input_cases)
    diagnostic["QAT CUDA to CPU equivalence"] = cpu_check
    if not cpu_check["status"]:
        diagnostic["selected export route"] = None
        diagnostic["status"] = "stopped: restored QAT CUDA-to-CPU mismatch"
        return diagnostic

    # Candidate A uses the same verified reference model after its CPU check;
    # no strip/replacement is performed.
    diagnostic["candidate A direct OpenVINO"] = candidate_from_direct_openvino(
        reference_model, input_cases, references, val_names
    )

    if diagnostic["candidate A direct OpenVINO"].get("success"):
        diagnostic["candidate B controller export"] = {
            "attempted": False,
            "reason": "Candidate A was accepted; Candidate B is reserved for a failed or unsupported Candidate A.",
        }
    else:
        # Candidate B gets a fresh reconstruction because export_model() may set
        # temporary NNCF export state internally.
        controller_b, model_b, _ = restore_qat_with_controller(qat_path, fp32_path, device)
        model_b.cpu().eval()
        diagnostic["candidate B controller export"] = candidate_from_controller_export(
            controller_b, model_b, input_cases, references, val_names
        )
    accepted = [
        ("direct OpenVINO", diagnostic["candidate A direct OpenVINO"], DIRECT_QAT_DIR),
        ("controller ONNX", diagnostic["candidate B controller export"], CONTROLLER_QAT_DIR),
    ]
    selected = next((item for item in accepted if item[1].get("success")), None)
    if selected is None:
        diagnostic["selected export route"] = None
        diagnostic["status"] = "no accepted QAT export candidate"
        return diagnostic
    route, _report, source_dir = selected
    QAT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".xml", ".bin"):
        shutil.copy2(source_dir / f"model{suffix}", ROOT / QAT_DIR / f"model{suffix}")
    diagnostic["selected export route"] = route
    diagnostic["status"] = "accepted and promoted"
    return diagnostic


def main() -> int:
    args = parse_args()
    manifest: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "started"}
    graph_reports: dict[str, Any] = {}
    try:
        require_project_root(ROOT)
        fp32_path = require_checkpoint(FP32_CHECKPOINT)
        qat_path = require_checkpoint(QAT_CHECKPOINT)
        device, gpu_name = require_cuda()
        if torch.cuda.device_count() != 1:
            raise ExportStop(f"Exactly one visible GPU is required, found {torch.cuda.device_count()}.")
        val_images = resolve_val_images()
        input_tensor, selected_val_image = val_input(val_images, device)
        manifest.update(
            {
                "source FP32 checkpoint": str(fp32_path),
                "FP32 SHA256": sha256_file(fp32_path),
                "source QAT checkpoint": str(qat_path),
                "QAT SHA256": sha256_file(qat_path),
                "versions": {
                    "Python": platform.python_version(),
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "ultralytics": ultralytics.__version__,
                    "nncf": nncf.__version__,
                    "openvino": ov.__version__,
                },
                "GPU": {"logical device": "cuda:0", "name": gpu_name},
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "input shape": list(INPUT_SHAPE),
                "sanity VAL image": selected_val_image,
                "FP32 export API/path": "torch.jit.trace -> openvino.convert_model -> openvino.save_model(compress_to_fp16=False)",
                "QAT export API/path": "Phase 7B diagnostics only: direct unstripped OpenVINO candidate and controller ONNX candidate",
                "QAT calibration occurred": False,
                "PTQ used": False,
                "test split": "not loaded or used",
            }
        )
        write_json(MANIFEST_PATH, manifest)

        if args.model == "fp32":
            fp32_model = load_fp32(fp32_path, device)
            enable_ultralytics_detect_export(fp32_model)
            fp32_export_model = TensorOutputExportWrapper(fp32_model)
            with torch.inference_mode():
                _ = fp32_export_model(input_tensor)  # Required 640 PyTorch forward before export.
            fp32_xml = save_openvino_ir(fp32_export_model, ROOT / FP32_DIR)
            graph_reports["FP32"] = graph_report(fp32_xml)
            fp32_export_model.to(device)  # save_openvino_ir traces the native model on CPU in-place.
            manifest["FP32"] = {
                "files": file_sizes(fp32_xml),
                "graph": graph_reports["FP32"],
                "sanity forward": sanity_check("FP32", fp32_export_model, fp32_xml, input_tensor),
            }
            if not manifest["FP32"]["sanity forward"]["status"]:
                raise ExportStop("FP32 PyTorch-to-OpenVINO numerical compatibility check failed.")
            del fp32_export_model, fp32_model
            torch.cuda.empty_cache()

        if args.model == "qat-diagnostic":
            diagnostic = run_qat_export_diagnostics(qat_path, fp32_path, device, val_images)
            write_json(QAT_DIAGNOSTIC_PATH, diagnostic)
            manifest["QAT diagnostic"] = {
                "path": str(ROOT / QAT_DIAGNOSTIC_PATH),
                "status": diagnostic["status"],
                "selected export route": diagnostic["selected export route"],
            }
            for name, key in (("direct_qat", "candidate A direct OpenVINO"), ("controller_qat", "candidate B controller export")):
                candidate = diagnostic.get(key, {})
                if "graph" in candidate:
                    graph_reports[name] = candidate["graph"]
            if diagnostic["status"] != "accepted and promoted":
                raise ExportStop(diagnostic["status"])
        manifest["status"] = "passed"
        print("OpenVINO export and graph verification passed (no PTQ, no calibration, no TEST usage).")
    except Exception as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        print(f"OpenVINO export failed closed: {manifest['error']}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        write_json(GRAPH_REPORT_PATH, graph_reports)
        write_json(MANIFEST_PATH, manifest)
        if 'diagnostic' in locals():
            write_json(QAT_DIAGNOSTIC_PATH, diagnostic)
        print(f"Manifest: {MANIFEST_PATH}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
