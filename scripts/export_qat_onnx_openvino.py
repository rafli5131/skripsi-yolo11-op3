"""Fail-closed Phase 7F export: restored NNCF QAT -> ONNX -> OpenVINO.

This is intentionally separate from :mod:`export_openvino`, which preserves
the historical direct-OpenVINO diagnostics.  It never invokes PTQ,
``nncf.quantize()``, ``nncf.torch.strip()``, an Ultralytics INT8 export, or
any TEST data.  A generated QAT IR is promoted only after both numerical and
graph-semantic acceptance checks pass.

Run this script only on the CUDA SSH host after the usual project-local NNCF
extension environment has been configured.  It uses 32 deterministic VAL
images, not a synthetic tensor, as the acceptance population.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path.cwd().resolve()
load_dotenv(ROOT / ".env", override=False)

import nncf
import openvino as ov
import torch
import torchvision
import ultralytics

# These are proven Phase 5--7 restoration/preprocessing helpers.  Importing
# them does not invoke the older export diagnostic's main() or mutate a model.
from export_openvino import (
    EXPECTED_QUANTIZERS,
    FP32_CHECKPOINT,
    INPUT_SHAPE,
    QAT_CHECKPOINT,
    ExportStop,
    enable_ultralytics_detect_export,
    preprocess_deployment_image,
    resolve_val_images,
    restore_qat_with_controller,
)
from probe_qat_compatibility import require_cuda, require_project_root
from smoke_train_qat import quantizers, quantizers_enabled, sha256_file


OPENVINO_ROOT = Path("artifacts/openvino")
DEBUG_ONNX_DIR = OPENVINO_ROOT / "debug/qat_onnx"
DEBUG_OV_DIR = OPENVINO_ROOT / "debug/qat_onnx_openvino"
FINAL_QAT_DIR = OPENVINO_ROOT / "qat_int8"
DIAGNOSTIC_PATH = OPENVINO_ROOT / "qat_onnx_export_diagnostic.json"
EXPECTED_STATE_ENTRIES = 1323
VAL_COUNT = 32
RELATIVE_MAE_LIMIT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-count",
        type=int,
        default=VAL_COUNT,
        help="Fixed deterministic VAL-image count for compatibility checks (default: 32).",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def raw_prediction(output: Any) -> torch.Tensor:
    """Extract the normal Ultralytics raw prediction tensor, never post-NMS."""
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ExportStop(f"Expected a raw tensor prediction, got {type(output).__name__}.")
    if tuple(output.shape) != (1, 7, 8400):
        raise ExportStop(f"Expected raw YOLO output [1,7,8400], got {list(output.shape)}.")
    if not torch.isfinite(output).all():
        raise ExportStop("Raw QAT prediction contains non-finite values.")
    return output.detach().float().cpu().contiguous()


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "status": False,
            "reference shape": list(reference.shape),
            "candidate shape": list(candidate.shape),
            "reason": "raw output shape mismatch",
        }
    delta = (reference - candidate).abs()
    mean_abs = float(delta.mean().item())
    ref_mean_abs = float(reference.abs().mean().item())
    relative = mean_abs / max(ref_mean_abs, 1e-12)
    cosine = float(torch.nn.functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0).item())
    return {
        "status": bool(torch.isfinite(candidate).all().item() and relative <= RELATIVE_MAE_LIMIT),
        "reference shape": list(reference.shape),
        "candidate shape": list(candidate.shape),
        "mean absolute difference": mean_abs,
        "mean absolute reference magnitude": ref_mean_abs,
        "relative MAE": relative,
        "maximum absolute difference": float(delta.max().item()),
        "cosine similarity": cosine,
        "maximum allowed relative MAE": RELATIVE_MAE_LIMIT,
    }


def summary(per_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = [float(value["relative MAE"]) for value in per_image.values() if value.get("relative MAE") is not None]
    if len(values) != len(per_image):
        raise ExportStop("Not every representative VAL output was comparable.")
    values.sort()
    p95_position = (len(values) - 1) * 0.95
    lower = int(p95_position)
    upper = min(lower + 1, len(values) - 1)
    p95 = values[lower] + (values[upper] - values[lower]) * (p95_position - lower)
    return {
        "VAL input count": len(values),
        "minimum relative MAE": values[0],
        "median relative MAE": values[len(values) // 2],
        "mean relative MAE": sum(values) / len(values),
        "p95 relative MAE": p95,
        "maximum relative MAE": values[-1],
        "count relative MAE <= 1 percent": sum(value <= RELATIVE_MAE_LIMIT for value in values),
        "count relative MAE > 1 percent": sum(value > RELATIVE_MAE_LIMIT for value in values),
    }


def deterministic_val_inputs(val_dir: Path, device: torch.device, count: int) -> list[tuple[str, torch.Tensor]]:
    paths = sorted(
        path for path in val_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(paths) < count:
        raise ExportStop(f"Need {count} VAL images but found {len(paths)} at {val_dir}.")
    return [(path.name, preprocess_deployment_image(path, device)) for path in paths[:count]]


def native_references(model: torch.nn.Module, inputs: list[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Reference is a distinct, unmodified normal-forward QAT model instance."""
    model.eval()
    if model.training:
        raise ExportStop("Native reference model unexpectedly remains in training mode.")
    references: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, tensor in inputs:
            references[name] = raw_prediction(model(tensor))
    return references


def controller_api(controller: Any) -> dict[str, Any]:
    method = getattr(controller, "export_model", None)
    source_file = inspect.getsourcefile(method) if callable(method) else None
    source = inspect.getsource(method) if callable(method) else ""
    return {
        "controller class": f"{type(controller).__module__}.{type(controller).__qualname__}",
        "export_model available": callable(method),
        "export_model signature": str(inspect.signature(method)) if callable(method) else None,
        "export_model source path": source_file,
        "export_model source summary": source.splitlines()[0] if source else None,
        "export-related public members": sorted(name for name in dir(controller) if "export" in name.lower()),
        "openvino.convert_model signature": str(inspect.signature(ov.convert_model)),
    }


def onnx_graph_report(path: Path) -> dict[str, Any]:
    import onnx  # imported only when already installed; never installed by this script

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    initializer_types = Counter(
        onnx.TensorProto.DataType.Name(initializer.data_type) for initializer in graph.initializer
    )
    return {
        "checker status": "passed",
        "IR version": model.ir_version,
        "opset imports": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "inputs": [value.name for value in graph.input],
        "outputs": [value.name for value in graph.output],
        "node count": len(graph.node),
        "operator counts": dict(sorted(op_counts.items())),
        "initializer element-type counts": dict(sorted(initializer_types.items())),
        "QuantizeLinear count": op_counts.get("QuantizeLinear", 0),
        "DequantizeLinear count": op_counts.get("DequantizeLinear", 0),
        "Clip count": op_counts.get("Clip", 0),
        "Round count": op_counts.get("Round", 0),
        "Cast count": op_counts.get("Cast", 0),
        "INT8 initializer count": initializer_types.get("INT8", 0),
        "UINT8 initializer count": initializer_types.get("UINT8", 0),
    }


def onnx_runtime_check(path: Path, inputs: list[tuple[str, torch.Tensor]], refs: dict[str, torch.Tensor]) -> dict[str, Any]:
    if importlib.util.find_spec("onnxruntime") is None:
        return {
            "available": False,
            "execution available": False,
            "status": "NOT AVAILABLE; no package installation was attempted.",
        }
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:
        # NNCF 2.13.0 exports trained fake quantization as the custom ONNX
        # operator org.openvinotoolkit::FakeQuantize. Stock ONNX Runtime has
        # no kernel for it. That is not a PyTorch->ONNX numerical result and
        # must not trigger a fallback that strips or PTQ-calibrates the graph.
        return {
            "available": True,
            "execution available": False,
            "status": "NOT EXECUTABLE by installed ONNX Runtime; custom NNCF operator is unsupported.",
            "error": f"{type(error).__name__}: {error}",
            "next step": "Continue only with OpenVINO ONNX import and native-QAT-to-OpenVINO comparison.",
        }
    input_name = session.get_inputs()[0].name
    per_image: dict[str, dict[str, Any]] = {}
    for name, tensor in inputs:
        outputs = session.run(None, {input_name: tensor.cpu().numpy()})
        if not outputs:
            raise ExportStop("ONNX Runtime produced no output tensors.")
        per_image[name] = comparison(refs[name], torch.from_numpy(outputs[0]).float())
    aggregate = summary(per_image)
    return {
        "available": True,
        "execution available": True,
        "providers": session.get_providers(),
        "per image": per_image,
        "summary": aggregate,
        "status": "passed" if aggregate["count relative MAE > 1 percent"] == 0 else "failed",
    }


def openvino_graph_report(xml_path: Path) -> dict[str, Any]:
    model = ov.Core().read_model(str(xml_path))
    operations = model.get_ordered_ops()
    op_counts = Counter(operation.get_type_name() for operation in operations)
    constant_types = Counter()
    all_output_types = Counter()
    for operation in operations:
        for index in range(operation.get_output_size()):
            element_type = str(operation.get_output_element_type(index))
            all_output_types[element_type] += 1
            if operation.get_type_name() == "Constant":
                constant_types[element_type] += 1
    i8_constants = sum(count for dtype, count in constant_types.items() if "i8" in dtype.lower())
    u8_constants = sum(count for dtype, count in constant_types.items() if "u8" in dtype.lower())
    return {
        "input shapes": [str(port.get_partial_shape()) for port in model.inputs],
        "output shapes": [str(port.get_partial_shape()) for port in model.outputs],
        "input element types": [str(port.get_element_type()) for port in model.inputs],
        "output element types": [str(port.get_element_type()) for port in model.outputs],
        "total operation count": len(operations),
        "operation counts": dict(sorted(op_counts.items())),
        "FakeQuantize count": op_counts.get("FakeQuantize", 0),
        "Convert count": op_counts.get("Convert", 0),
        "constant element-type counts": dict(sorted(constant_types.items())),
        "I8 constant count": i8_constants,
        "U8 constant count": u8_constants,
        "all output element-type counts": dict(sorted(all_output_types.items())),
        "quantization semantics visible": bool(op_counts.get("FakeQuantize", 0) > 0 or i8_constants > 0 or u8_constants > 0),
    }


def openvino_check(xml_path: Path, inputs: list[tuple[str, torch.Tensor]], refs: dict[str, torch.Tensor]) -> dict[str, Any]:
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(xml_path)), "CPU")
    per_image: dict[str, dict[str, Any]] = {}
    for name, tensor in inputs:
        result = compiled([tensor.cpu().numpy()])
        values = list(result.values())
        if not values:
            raise ExportStop("OpenVINO candidate produced no output tensors.")
        per_image[name] = comparison(refs[name], torch.from_numpy(values[0]).float())
    aggregate = summary(per_image)
    return {
        "device": "CPU",
        "per image": per_image,
        "summary": aggregate,
        "status": "passed" if aggregate["count relative MAE > 1 percent"] == 0 else "failed",
    }


def promote(onnx_path: Path, xml_path: Path) -> dict[str, str]:
    """Promotion is reached only after both acceptance gates have passed."""
    FINAL_QAT_DIR.mkdir(parents=True, exist_ok=True)
    target_onnx = ROOT / FINAL_QAT_DIR / "model.onnx"
    target_xml = ROOT / FINAL_QAT_DIR / "model.xml"
    target_bin = ROOT / FINAL_QAT_DIR / "model.bin"
    shutil.copy2(onnx_path, target_onnx)
    shutil.copy2(xml_path, target_xml)
    shutil.copy2(xml_path.with_suffix(".bin"), target_bin)
    sums = {
        "model.onnx": sha256_file(target_onnx),
        "model.xml": sha256_file(target_xml),
        "model.bin": sha256_file(target_bin),
    }
    (ROOT / FINAL_QAT_DIR / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sums.items()), encoding="utf-8"
    )
    return {"directory": str(ROOT / FINAL_QAT_DIR), **sums}


def main() -> int:
    args = parse_args()
    diagnostic: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "statement": "No PTQ, calibration, nncf.quantize(), held-out TEST data, or retraining was used during this QAT export.",
        "strip path": "disabled; historically rejected and not invoked",
        "test split": "not loaded or used",
        "calibration occurred": False,
        "PTQ used": False,
    }
    try:
        require_project_root(ROOT)
        if args.val_count != VAL_COUNT:
            raise ExportStop("Phase 7F requires exactly 32 deterministic VAL images; --val-count must remain 32.")
        fp32_path = (ROOT / FP32_CHECKPOINT).resolve()
        qat_path = (ROOT / QAT_CHECKPOINT).resolve()
        if not fp32_path.is_file() or not qat_path.is_file():
            raise ExportStop("Fixed FP32 or final QAT checkpoint is missing.")
        device, gpu_name = require_cuda()
        if torch.cuda.device_count() != 1:
            raise ExportStop(f"Exactly one visible logical GPU is required, found {torch.cuda.device_count()}.")
        val_images = resolve_val_images()
        native_inputs = deterministic_val_inputs(val_images, device, VAL_COUNT)

        # Native reference instance deliberately never sees Detect.export=True.
        reference_controller, reference_model, restored = restore_qat_with_controller(qat_path, fp32_path, device)
        reference_items = quantizers(reference_model)
        if restored["loaded state entries"] != EXPECTED_STATE_ENTRIES:
            raise ExportStop(f"Expected {EXPECTED_STATE_ENTRIES}/1323 restored entries, got {restored['loaded state entries']}.")
        if len(reference_items) != EXPECTED_QUANTIZERS or not quantizers_enabled(reference_items):
            raise ExportStop("Native QAT reference does not have 184 enabled fake quantizers.")
        diagnostic.update(
            {
                "source FP32 checkpoint": str(fp32_path),
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
                "representative inputs": {
                    "split": "VAL only",
                    "selection": "first 32 image filenames in deterministic lexical order",
                    "filenames": [name for name, _ in native_inputs],
                    "preprocessing": "cv2 decode -> LetterBox(640, auto=False, stride=32) -> BGR to RGB -> NCHW float32 / 255",
                },
                "NNCF state restore": restored,
                "reference quantizer count": len(reference_items),
                "reference enabled quantizer count": sum(item.is_enabled_quantization() for item in reference_items),
                "reference model mode": "eval" if not reference_model.training else "train",
                "NNCF controller API": controller_api(reference_controller),
                "ONNX package available": importlib.util.find_spec("onnx") is not None,
                "ONNX Runtime available": importlib.util.find_spec("onnxruntime") is not None,
            }
        )
        if not diagnostic["NNCF controller API"]["export_model available"]:
            raise ExportStop("FAILURE: NNCF → ONNX export; installed controller has no export_model().")
        references = native_references(reference_model, native_inputs)

        # Fresh instance: Detect.export can be scoped to NNCF's ONNX export without
        # changing the reference model or reference forward path.
        export_controller, export_model, export_restored = restore_qat_with_controller(qat_path, fp32_path, device)
        enable_ultralytics_detect_export(export_model)
        export_model.cpu().eval()
        export_items = quantizers(export_model)
        diagnostic["export model preparation"] = {
            "fresh model instance": True,
            "Detect.export scoped to export model only": True,
            "model mode": "eval" if not export_model.training else "train",
            "NNCF state restore": export_restored,
            "quantizer count after CPU transfer": len(export_items),
            "enabled quantizer count after CPU transfer": sum(item.is_enabled_quantization() for item in export_items),
        }
        if len(export_items) != EXPECTED_QUANTIZERS or not quantizers_enabled(export_items):
            raise ExportStop("FAILURE: QAT checkpoint restore; quantizers changed during export-only CPU preparation.")

        onnx_path = ROOT / DEBUG_ONNX_DIR / "model.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic["ONNX export mechanism"] = "QuantizationController.export_model(save_path, save_format='onnx')"
        try:
            export_controller.export_model(str(onnx_path), save_format="onnx")
        except Exception as error:
            raise ExportStop(f"FAILURE: NNCF → ONNX export; {type(error).__name__}: {error}") from error
        if not onnx_path.is_file():
            raise ExportStop("FAILURE: NNCF → ONNX export; controller returned without model.onnx.")
        diagnostic["ONNX"] = {"export status": "passed", "path": str(onnx_path), "file size bytes": onnx_path.stat().st_size}

        if importlib.util.find_spec("onnx") is None:
            raise ExportStop("FAILURE: ONNX graph validation; installed environment has no onnx package, so checker cannot run.")
        try:
            diagnostic["ONNX"].update(onnx_graph_report(onnx_path))
        except Exception as error:
            raise ExportStop(f"FAILURE: ONNX graph validation; {type(error).__name__}: {error}") from error

        ort_result = onnx_runtime_check(onnx_path, native_inputs, references)
        diagnostic["ONNX Runtime"] = ort_result
        if ort_result["execution available"] and ort_result["status"] != "passed":
            raise ExportStop("FAILURE: PyTorch QAT → ONNX numerical mismatch.")

        xml_path = ROOT / DEBUG_OV_DIR / "model.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ov_model = ov.convert_model(str(onnx_path))
            ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)
        except Exception as error:
            raise ExportStop(f"FAILURE: ONNX → OpenVINO conversion; {type(error).__name__}: {error}") from error
        if not xml_path.is_file() or not xml_path.with_suffix(".bin").is_file():
            raise ExportStop("FAILURE: ONNX → OpenVINO conversion; expected XML/BIN were not produced.")

        ov_numerical = openvino_check(xml_path, native_inputs, references)
        diagnostic["OpenVINO"] = {
            "conversion status": "passed",
            "path": str(xml_path),
            "XML bytes": xml_path.stat().st_size,
            "BIN bytes": xml_path.with_suffix(".bin").stat().st_size,
            "total bytes": xml_path.stat().st_size + xml_path.with_suffix(".bin").stat().st_size,
            "numerical": ov_numerical,
        }
        if ov_numerical["status"] != "passed":
            raise ExportStop("FAILURE: PyTorch QAT → OpenVINO numerical mismatch.")
        graph = openvino_graph_report(xml_path)
        diagnostic["OpenVINO"]["graph"] = graph
        if not graph["quantization semantics visible"]:
            raise ExportStop("FAILURE: OpenVINO quantization semantics missing.")
        diagnostic["promotion"] = {"status": "passed", **promote(onnx_path, xml_path)}
        diagnostic["selected export route"] = "NNCF QuantizationController.export_model(ONNX) -> OpenVINO"
        diagnostic["status"] = "VALID: numerically verified and promoted"
        print("QAT -> ONNX -> OpenVINO export passed and was promoted.")
    except Exception as error:
        diagnostic["status"] = "REJECTED / UNRESOLVED"
        diagnostic["failure"] = f"{type(error).__name__}: {error}"
        diagnostic.setdefault("promotion", {"status": "not promoted"})
        print(f"QAT ONNX/OpenVINO export failed closed: {diagnostic['failure']}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        write_json(ROOT / DIAGNOSTIC_PATH, diagnostic)
        print(f"Diagnostic: {ROOT / DIAGNOSTIC_PATH}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
