"""Run final true-QAT OpenVINO export with real-VAL numerical acceptance.

This launcher keeps scripts/export_openvino.py as the audited base implementation
and changes only the acceptance policy required for the frozen final QAT
checkpoint:

- the 1% relative-MAE threshold is unchanged;
- exactly 32 deterministic real VAL images are the acceptance population;
- synthetic inputs are still executed and recorded, but are diagnostic only;
- TEST is never loaded or used;
- PTQ/calibration/nncf.quantize are never used.

The CPU/CUDA diagnostic must belong to the current frozen QAT checkpoint and
all 32 VAL images must pass <=1% before export is attempted.
"""

from __future__ import annotations

import inspect
import json
import sys
from typing import Any

import torch

import export_openvino as base


SYNTHETIC_KEY = "in_domain_synthetic"


def _val_names(input_cases: dict[str, torch.Tensor]) -> list[str]:
    names = [name for name in input_cases if name != SYNTHETIC_KEY]
    if len(names) != base.REPRESENTATIVE_VAL_COUNT:
        raise base.ExportStop(
            f"Expected {base.REPRESENTATIVE_VAL_COUNT} representative VAL inputs, found {len(names)}."
        )
    return names


def phase_7c_evidence() -> dict[str, Any]:
    """Require current-checkpoint CPU/CUDA evidence on all representative VAL images."""
    path = base.ROOT / base.CPU_CUDA_DIAGNOSTIC_PATH
    if not path.is_file():
        raise base.ExportStop(f"Phase 7C diagnostic is required before QAT export: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))
    qat_path = (base.ROOT / base.QAT_CHECKPOINT).resolve()
    current_sha = base.sha256_file(qat_path)

    if report.get("QAT SHA256") != current_sha:
        raise base.ExportStop(
            "CPU/CUDA diagnostic belongs to a different QAT checkpoint; rerun "
            "scripts/diagnose_qat_cpu_cuda.py for the current frozen checkpoint."
        )

    val = report.get("VAL relative-MAE summary") or {}
    passed = int(val.get("count_relative_MAE_le_1_percent", 0))
    failed = int(val.get("count_relative_MAE_gt_1_percent", -1))

    if passed != base.REPRESENTATIVE_VAL_COUNT or failed != 0:
        raise base.ExportStop(
            "Current QAT checkpoint did not pass <=1% CPU/CUDA relative-MAE "
            "on all 32 representative VAL images."
        )

    comparisons = report.get("comparisons") or {}
    return {
        "path": str(path),
        "QAT SHA256": current_sha,
        "classification": report.get("classification"),
        "acceptance population": "32 deterministic real VAL images",
        "maximum allowed relative MAE": 0.01,
        "VAL relative-MAE summary": val,
        "synthetic inputs": "diagnostic only; not an acceptance criterion",
        "original synthetic diagnostic": comparisons.get("original_synthetic"),
        "in-domain synthetic diagnostic": comparisons.get(SYNTHETIC_KEY),
    }


def qat_cuda_cpu_equivalence(
    model: torch.nn.Module,
    input_cases: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compare CUDA vs CPU, accepting only on the fixed real-VAL population."""
    val_names = _val_names(input_cases)
    prediction_model = base.TensorOutputExportWrapper(model).eval()

    cuda_outputs: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for name, input_tensor in input_cases.items():
            cuda_outputs[name] = base.output_summary(prediction_model(input_tensor))

    model.cpu().eval()
    cpu_quantizers = base.quantizers(model)
    count_after_cpu = len(cpu_quantizers)
    enabled_after_cpu = base.quantizers_enabled(cpu_quantizers)

    cpu_outputs: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Any] = {}
    with torch.inference_mode():
        for name, input_tensor in input_cases.items():
            cpu_outputs[name] = base.output_summary(prediction_model(input_tensor.cpu()))
            comparisons[name] = {
                "CUDA reference": base.compact_summary(cuda_outputs[name]),
                "CPU candidate": base.compact_summary(cpu_outputs[name]),
                **base.compare_summaries(cuda_outputs[name], cpu_outputs[name]),
            }

    val_status = all(comparisons[name]["status"] for name in val_names)
    status = (
        count_after_cpu == base.EXPECTED_QUANTIZERS
        and enabled_after_cpu
        and val_status
    )

    return {
        "status": status,
        "acceptance population": "32 deterministic real VAL images",
        "maximum allowed relative MAE": 0.01,
        "synthetic inputs": "diagnostic only; not an acceptance criterion",
        "quantizer count after model.cpu": count_after_cpu,
        "quantizers enabled after model.cpu": enabled_after_cpu,
        "VAL pass count": sum(bool(comparisons[name]["status"]) for name in val_names),
        "VAL fail count": sum(not bool(comparisons[name]["status"]) for name in val_names),
        "per input": comparisons,
    }, cpu_outputs


def candidate_from_direct_openvino(
    model: torch.nn.Module,
    input_cases: dict[str, torch.Tensor],
    references: dict[str, dict[str, Any]],
    val_names: list[str],
) -> dict[str, Any]:
    """Candidate A: direct OpenVINO; accept numerically on all 32 real VAL images."""
    report: dict[str, Any] = {
        "export path": (
            "ultralytics.utils.export.openvino.torch2openvino("
            "compressed_model wrapper, quantize=None)"
        ),
        "torch2openvino signature": str(inspect.signature(base.torch2openvino)),
        "quantize": None,
        "calibration_dataset": None,
        "int8_detect": False,
        "success": False,
        "calibration occurred": False,
        "PTQ used": False,
        "acceptance population": "32 deterministic real VAL images",
        "synthetic inputs": "diagnostic only; not an acceptance criterion",
    }
    try:
        base.DIRECT_QAT_DIR.mkdir(parents=True, exist_ok=True)
        example = input_cases[SYNTHETIC_KEY].cpu()
        export_model = base.TensorOutputExportWrapper(model).eval().cpu()

        base.torch2openvino(
            model=export_model,
            im=example,
            output_dir=base.DIRECT_QAT_DIR,
            dynamic=False,
            quantize=None,
            calibration_dataset=None,
            int8_detect=False,
            prefix="QAT Candidate A:",
        )

        xml_path = base.DIRECT_QAT_DIR / "model.xml"
        report["files"] = base.file_sizes(xml_path)
        core = base.ov.Core()
        compiled = core.compile_model(core.read_model(xml_path), "CPU")

        numerical: dict[str, Any] = {}
        for name, input_tensor in input_cases.items():
            output = base.ov_output_summary(compiled, input_tensor.cpu())
            numerical[name] = {
                "candidate": base.compact_summary(output),
                **base.compare_summaries(references[name], output),
            }

        report["numerical checks"] = numerical
        report["representative numerical summary"] = base.representative_summary(
            numerical, val_names
        )
        numerical_ok = all(numerical[name]["status"] for name in val_names)

        if not numerical_ok:
            report.update(
                {
                    "numerical equivalence": False,
                    "rejection reason": (
                        "Representative VAL numerical equivalence exceeds 1%."
                    ),
                }
            )
            return report

        report["graph"] = base.graph_report(xml_path)
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
    """Candidate B: controller ONNX -> OpenVINO; accept on all 32 real VAL images."""
    report: dict[str, Any] = {
        "export path": "compression_ctrl.export_model(onnx) -> ov.convert_model(onnx) without strip",
        "success": False,
        "calibration occurred": False,
        "PTQ used": False,
        "acceptance population": "32 deterministic real VAL images",
        "synthetic inputs": "diagnostic only; not an acceptance criterion",
    }
    try:
        if not callable(getattr(controller, "export_model", None)):
            raise base.ExportStop(
                "Installed compression controller has no callable export_model()."
            )

        base.CONTROLLER_QAT_DIR.mkdir(parents=True, exist_ok=True)
        onnx_path = base.CONTROLLER_QAT_DIR / "model.onnx"
        controller.export_model(str(onnx_path), save_format="onnx")

        if not onnx_path.is_file():
            raise base.ExportStop(
                "compression_ctrl.export_model() returned without producing ONNX."
            )

        ov_model = base.ov.convert_model(onnx_path)
        xml_path = base.CONTROLLER_QAT_DIR / "model.xml"
        base.ov.save_model(ov_model, xml_path, compress_to_fp16=False)
        report["files"] = {
            **base.file_sizes(xml_path),
            "ONNX bytes": onnx_path.stat().st_size,
        }

        core = base.ov.Core()
        compiled = core.compile_model(core.read_model(xml_path), "CPU")

        numerical: dict[str, Any] = {}
        for name, input_tensor in input_cases.items():
            output = base.ov_output_summary(compiled, input_tensor.cpu())
            numerical[name] = {
                "candidate": base.compact_summary(output),
                **base.compare_summaries(references[name], output),
            }

        report["numerical checks"] = numerical
        report["representative numerical summary"] = base.representative_summary(
            numerical, val_names
        )
        numerical_ok = all(numerical[name]["status"] for name in val_names)

        if not numerical_ok:
            report.update(
                {
                    "numerical equivalence": False,
                    "rejection reason": (
                        "Representative VAL numerical equivalence exceeds 1%."
                    ),
                }
            )
            return report

        report["graph"] = base.graph_report(xml_path)
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


def main() -> int:
    # Patch only acceptance/equivalence policy; keep all other audited export
    # behavior in scripts/export_openvino.py unchanged.
    base.phase_7c_evidence = phase_7c_evidence
    base.qat_cuda_cpu_equivalence = qat_cuda_cpu_equivalence
    base.candidate_from_direct_openvino = candidate_from_direct_openvino
    base.candidate_from_controller_export = candidate_from_controller_export

    if len(sys.argv) == 1:
        sys.argv.extend(["--model", "qat-diagnostic"])

    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
