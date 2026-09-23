"""Audit fixed FP32/QAT artifacts before deployment on ROBOTIS OP3."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from op3_model_audit import (
    EXPECTED_CLASSES,
    lfs_pointer,
    ptq_validation_gate,
    qat_validation_gate,
    read_json,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/hardware_op3/model_inventory.json"


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_blob(relative_path: str, cache_path: Path | None = None) -> dict[str, Any]:
    path = ROOT / relative_path
    record: dict[str, Any] = {"repository_path": relative_path, "exists": path.is_file()}
    pointer = lfs_pointer(path) if path.is_file() else None
    if pointer:
        record.update({"sha256": pointer["sha256"], "size_bytes": pointer["size_bytes"], "storage": "Git LFS"})
    elif path.is_file():
        record.update({"sha256": sha256_file(path), "size_bytes": path.stat().st_size, "storage": "Git blob"})
    if cache_path:
        local = cache_path.resolve()
        cached = {"path": str(local), "exists": local.is_file()}
        if local.is_file():
            cached.update({"sha256": sha256_file(local), "size_bytes": local.stat().st_size})
            if record.get("sha256"):
                cached["matches_repository_lfs_sha256"] = cached["sha256"] == record["sha256"]
        record["runtime_copy"] = cached
    return record


def main() -> int:
    fp32_manifest = read_json(ROOT / "artifacts/checkpoints/fp32/manifest.json") or {}
    fp32_checkpoint_pointer = lfs_pointer(ROOT / "artifacts/checkpoints/fp32/best.pt")
    qat_manifest = read_json(ROOT / "artifacts/checkpoints/qat/manifest.json") or {}
    qat_gate = qat_validation_gate(ROOT)
    ptq_gate = ptq_validation_gate(ROOT)
    export_manifest = read_json(ROOT / "artifacts/openvino/export_manifest.json") or {}
    ptq_manifest = read_json(ROOT / "experiments/ptq/manifest.json") or {}
    ptq_comparison = read_json(ROOT / "experiments/ptq/comparison.json") or {}
    final_test = read_json(ROOT / "experiments/final_test/comparison.json") or {}

    fp32_xml_cache = ROOT / ".cache/hardware_models/fp32/model.xml"
    fp32_bin_cache = ROOT / ".cache/hardware_models/fp32/model.bin"
    fp32_native_cache = ROOT / ".cache/hardware_models/fp32_pytorch/best.pt"
    fp32_native = inspect_blob("artifacts/checkpoints/fp32/best.pt", fp32_native_cache)
    fp32_xml = inspect_blob("artifacts/openvino/fp32/model.xml", fp32_xml_cache)
    fp32_bin = inspect_blob("artifacts/openvino/fp32/model.bin", fp32_bin_cache)
    ptq_fp32 = ptq_manifest.get("PTQ artifact")
    fp32_export_valid = bool(
        isinstance(ptq_fp32, dict)
        and ptq_comparison.get("models", {}).get("FP32 OpenVINO", {}).get("status") == "VALID"
        and fp32_manifest.get("status") in {"completed", "passed"}
        and ptq_manifest.get("source FP32 SHA256")
        and fp32_checkpoint_pointer
        and fp32_checkpoint_pointer["sha256"] == ptq_manifest.get("source FP32 SHA256")
        and fp32_xml.get("sha256") == ptq_manifest.get("source FP32 OpenVINO SHA256")
        and fp32_bin.get("runtime_copy", {}).get("matches_repository_lfs_sha256") is True
        and fp32_xml.get("runtime_copy", {}).get("matches_repository_lfs_sha256") is True
    )
    fp32_native_valid = bool(
        fp32_manifest.get("status") in {"completed", "passed"}
        and fp32_manifest.get("class_mapping") == EXPECTED_CLASSES
        and fp32_checkpoint_pointer
        and fp32_native.get("sha256") == fp32_checkpoint_pointer.get("sha256")
        and fp32_native.get("runtime_copy", {}).get("matches_repository_lfs_sha256") is True
    )

    fp32_test = read_json(ROOT / "experiments/final_test/fp32_test_metrics.json") or {}
    qat_test = read_json(ROOT / "experiments/final_test/qat_test_metrics.json") or {}
    accuracy = {
        "source": "experiments/final_test/{fp32_test_metrics,qat_test_metrics}.json",
        "split": "test",
        "test_images": fp32_test.get("test_images"),
        "test_annotations": fp32_test.get("test_annotations"),
        "fp32": fp32_test.get("overall", {}),
        "qat_native_pytorch": qat_test.get("overall", {}),
        "note": "Dataset metrics only; not derived from live camera observations.",
    }
    inventory = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "expected_class_mapping": EXPECTED_CLASSES,
        "fp32_baseline": {
            "name": "YOLO11n FP32 OpenVINO IR",
            "format": "OpenVINO IR XML + BIN",
            "checkpoint": {
                "path": "artifacts/checkpoints/fp32/best.pt",
                "sha256": fp32_checkpoint_pointer.get("sha256") if fp32_checkpoint_pointer else None,
                "size_bytes": fp32_checkpoint_pointer.get("size_bytes") if fp32_checkpoint_pointer else None,
                "manifest_status": fp32_manifest.get("status"),
                "source": fp32_manifest.get("checkpoint_source"),
                "class_mapping": fp32_manifest.get("class_mapping"),
            },
            "quantization_method": "none (FP32)",
            "openvino_xml": fp32_xml,
            "openvino_bin": fp32_bin,
            "source_checkpoint_sha256_in_export": export_manifest.get("FP32 SHA256"),
            "validity": "valid" if fp32_export_valid else "unknown",
            "validity_evidence": [
                "FP32 checkpoint manifest status completed",
                "PTQ comparison manifest labels FP32 OpenVINO VALID",
                "FP32 XML SHA256 matches validation provenance",
                "downloaded XML/BIN match Git LFS OIDs",
            ],
        },
        "fp32_native_pytorch": {
            "name": "YOLO11n FP32 native PyTorch (no OpenVINO)",
            "format": "PyTorch checkpoint",
            "checkpoint": fp32_native,
            "quantization_method": "none (FP32)",
            "class_mapping": fp32_manifest.get("class_mapping"),
            "validity": "valid" if fp32_native_valid else "unknown",
            "validity_evidence": [
                "FP32 checkpoint manifest status completed",
                "class mapping matches expected three classes",
                "runtime PyTorch checkpoint copy SHA256 matches repository Git LFS OID",
            ],
        },
        "true_qat_checkpoint": {
            "name": "YOLO11n true NNCF QAT best checkpoint",
            "format": "PyTorch checkpoint",
            "path": "artifacts/checkpoints/qat/qat_best.pt",
            "sha256": qat_gate.get("checkpoint_sha256"),
            "size_bytes": (lfs_pointer(ROOT / "artifacts/checkpoints/qat/qat_best.pt") or {}).get("size_bytes"),
            "provenance_status": "confirmed by checkpoint manifest and matching LFS OID"
            if qat_gate.get("true_nncf_qat_provenance")
            else "unknown",
            "method": qat_manifest.get("QAT API"),
            "manifest_status": qat_manifest.get("status"),
            "quantizer_count": qat_manifest.get("quantizer count after reload"),
            "class_mapping": qat_manifest.get("class mapping"),
            "selection_manifest_present": bool(qat_gate.get("selection_manifest")),
        },
        "qat_openvino_int8": {
            "name": "YOLO11n QAT INT8 OpenVINO",
            "format": "OpenVINO IR XML + BIN",
            "xml": inspect_blob("artifacts/openvino/qat_int8/model.xml"),
            "bin": inspect_blob("artifacts/openvino/qat_int8/model.bin"),
            "quantization_method": "true NNCF QAT export candidate; no PTQ/calibration",
            "validity": "valid" if qat_gate["accepted"] else "rejected_or_unresolved",
            "gate": qat_gate,
        },
        "ptq_openvino_int8": {
            "name": "YOLO11n existing calibrated PTQ INT8 OpenVINO comparator",
            "format": "OpenVINO IR XML + BIN",
            "xml": inspect_blob("experiments/ptq/backend_models/ptq_openvino_model/model.xml"),
            "bin": inspect_blob("experiments/ptq/backend_models/ptq_openvino_model/model.bin"),
            "quantization_method": ptq_manifest.get("PTQ API"),
            "validity": "valid comparator" if ptq_gate["accepted"] else "blocked",
            "gate": ptq_gate,
            "accuracy_metrics": "existing VAL diagnostic/comparator only; not held-out TEST metrics",
        },
        "dataset_accuracy": accuracy,
        "ptq_comparison_status": ptq_comparison.get("models", {}).get("PTQ INT8", {}).get("status"),
        "warnings": [
            "The existing historical qat_int8 model is never eligible for benchmarking unless all QAT gate checks pass.",
            "PTQ metrics are not relabeled or used as QAT results.",
            "Dataset P/R/F1/mAP metrics are not derived from camera observations.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"FP32 OpenVINO: {inventory['fp32_baseline']['validity']}")
    print(f"FP32 native PyTorch: {inventory['fp32_native_pytorch']['validity']}")
    print(f"PTQ OpenVINO: {ptq_gate['status']}")
    print(f"QAT OpenVINO: {qat_gate['status']}")
    if not qat_gate["accepted"]:
        print("BLOCKED: no validated QAT OpenVINO artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
