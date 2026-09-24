"""Shared provenance checks for OP3 model inventory and benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_CLASSES = {"0": "ball", "1": "gawang", "2": "robot"}
RELATIVE_MAE_LIMIT = 0.01
PTQ_XML_RELATIVE_PATH = Path("experiments/ptq/backend_models/ptq_openvino_model/model.xml")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lfs_pointer(path: Path) -> dict[str, Any] | None:
    """Read a Git LFS pointer without confusing its SHA with a local file hash."""
    try:
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    if not content.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    oid_match = re.search(r"^oid sha256:([0-9a-f]{64})$", content, re.MULTILINE)
    size_match = re.search(r"^size (\d+)$", content, re.MULTILINE)
    if not oid_match or not size_match:
        return None
    return {"sha256": oid_match.group(1), "size_bytes": int(size_match.group(1))}


def _valid_numerical_summary(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    n_total = summary.get("VAL input count")
    n_passed = summary.get("count relative MAE <= 1 percent")
    n_failed = summary.get("count relative MAE > 1 percent")
    max_error = summary.get("maximum relative MAE")
    return bool(
        isinstance(n_total, int)
        and n_total > 0
        and n_passed == n_total
        and n_failed == 0
        and isinstance(max_error, (int, float))
        and max_error <= RELATIVE_MAE_LIMIT
    )


def qat_validation_gate(root: Path) -> dict[str, Any]:
    """Require selected true-QAT provenance and a passed export/equivalence gate."""
    qat_manifest_path = root / "artifacts/checkpoints/qat/manifest.json"
    qat_manifest = read_json(qat_manifest_path) or {}
    qat_hash = (
        qat_manifest.get("artifact freeze", {}).get("sha256", {}).get("qat_best.pt")
        if isinstance(qat_manifest.get("artifact freeze"), dict)
        else None
    )

    checkpoint_path = root / "artifacts/checkpoints/qat/qat_best.pt"
    pointer = lfs_pointer(checkpoint_path)
    materialized_hash = None
    if checkpoint_path.is_file() and pointer is None:
        try:
            materialized_hash = sha256_file(checkpoint_path)
        except OSError:
            materialized_hash = None

    checkpoint_hash_matches = bool(
        isinstance(qat_hash, str)
        and (
            (pointer is not None and pointer.get("sha256") == qat_hash)
            or materialized_hash == qat_hash
        )
    )

    qat_api = str(qat_manifest.get("QAT API", ""))
    reload_verification = qat_manifest.get("reload verification")
    reload_verification = reload_verification if isinstance(reload_verification, dict) else {}
    expected_quantizers = qat_manifest.get("expected quantizer count")
    reload_quantizers = reload_verification.get("quantizer_count")

    true_qat = bool(
        qat_manifest.get("status") == "passed"
        and "create_compressed_model" in qat_api
        and not qat_api.lstrip().startswith("nncf.quantize")
        and checkpoint_hash_matches
        and isinstance(expected_quantizers, int)
        and reload_quantizers == expected_quantizers
        and reload_verification.get("status") is True
        and reload_verification.get("fake_quant_enabled") is True
        and reload_verification.get("trained_weights_restored") is True
        and qat_manifest.get("quantizer count after training") == expected_quantizers
    )

    selection_candidates = [
        root / "artifacts/checkpoints/qat/selection_manifest.json",
        root / "experiments/qat/qat_best_search/selection_manifest.json",
    ]
    selection_path = next((path for path in selection_candidates if path.is_file()), None)
    selection = read_json(selection_path) if selection_path else None
    selection = selection if isinstance(selection, dict) else {}
    selection_status = str(selection.get("status", "")).lower()

    winner = selection.get("winner")
    winner = winner if isinstance(winner, dict) else {}
    promotion = selection.get("promotion")
    promotion = promotion if isinstance(promotion, dict) else {}
    tuning_candidate = qat_manifest.get("QAT tuning candidate")
    tuning_candidate = tuning_candidate if isinstance(tuning_candidate, dict) else {}
    selected_name = tuning_candidate.get("name")

    selection_accepted = bool(
        any(word in selection_status for word in ("passed", "accepted", "valid", "completed"))
        and isinstance(selected_name, str)
        and selected_name
        and winner.get("name") == selected_name
        and promotion.get("candidate") == selected_name
        and promotion.get("promoted") is True
        and selection.get("TEST split used for selection") is False
        and selection.get("PTQ metrics used for selection") is False
        and qat_manifest.get("TEST split used for selection") is False
        and qat_manifest.get("PTQ metrics used for selection") is False
    )

    export_candidates = [
        root / "artifacts/openvino/qat_export_diagnostic.json",
        root / "artifacts/openvino/qat_onnx_export_diagnostic.json",
    ]
    export_records = [(path, read_json(path)) for path in export_candidates if path.is_file()]
    accepted_statuses = {"passed", "accepted", "valid", "accepted and promoted"}
    export_record = next(
        (
            (path, value)
            for path, value in export_records
            if value and str(value.get("status", "")).lower() in accepted_statuses
        ),
        None,
    )

    export_accepted = False
    export_path = None
    export_route = None
    if export_record:
        export_path, diagnostic = export_record
        status = str(diagnostic.get("status", "")).lower()
        export_route = diagnostic.get("selected export route")
        phase_7c = diagnostic.get("Phase 7C evidence")
        phase_7c = phase_7c if isinstance(phase_7c, dict) else {}
        export_hash = (
            diagnostic.get("QAT SHA256")
            or diagnostic.get("QAT checkpoint SHA256")
            or phase_7c.get("QAT SHA256")
        )

        if status == "accepted and promoted":
            candidate_key = {
                "direct OpenVINO": "candidate A direct OpenVINO",
                "controller ONNX": "candidate B controller export",
            }.get(str(export_route))
            candidate = diagnostic.get(candidate_key, {}) if candidate_key else {}
            candidate = candidate if isinstance(candidate, dict) else {}
            graph = candidate.get("graph") if isinstance(candidate.get("graph"), dict) else {}
            summary = candidate.get("representative numerical summary")
            cpu_cuda = diagnostic.get("QAT CUDA to CPU equivalence")
            cpu_cuda = cpu_cuda if isinstance(cpu_cuda, dict) else {}
            export_accepted = bool(
                export_hash == qat_hash
                and diagnostic.get("calibration occurred") is False
                and diagnostic.get("PTQ used") is False
                and cpu_cuda.get("status") is True
                and cpu_cuda.get("VAL pass count") == 32
                and cpu_cuda.get("VAL fail count") == 0
                and candidate.get("success") is True
                and candidate.get("numerical equivalence") is True
                and candidate.get("quantization graph visible") is True
                and _valid_numerical_summary(summary)
                and isinstance(graph.get("FakeQuantize_op_count"), int)
                and graph.get("FakeQuantize_op_count", 0) > 0
                and isinstance(graph.get("int8_or_u8_graph_element_count"), int)
                and graph.get("int8_or_u8_graph_element_count", 0) > 0
            )
        else:
            promotion_record = diagnostic.get("promotion")
            export_accepted = bool(
                isinstance(promotion_record, dict)
                and str(promotion_record.get("status", "")).lower() in {"passed", "accepted", "promoted", "valid"}
                and export_hash == qat_hash
                and diagnostic.get("calibration occurred") is False
                and diagnostic.get("PTQ used") is False
            )
            if export_accepted:
                summary = diagnostic.get("OpenVINO numerical summary") or diagnostic.get("numerical summary")
                export_accepted = _valid_numerical_summary(summary)

    rejected_records = [
        {"path": str(path.relative_to(root)), "status": value.get("status")}
        for path, value in export_records
        if value and str(value.get("status", "")).lower() not in accepted_statuses
    ]

    accepted = bool(true_qat and selection_accepted and export_accepted)
    return {
        "accepted": accepted,
        "status": "VALID QAT INT8 OpenVINO" if accepted else "BLOCKED: no validated QAT OpenVINO artifact",
        "true_nncf_qat_provenance": true_qat,
        "checkpoint_sha256": qat_hash,
        "checkpoint_hash_matches_manifest": checkpoint_hash_matches,
        "checkpoint_lfs_pointer_matches_manifest": bool(pointer and pointer.get("sha256") == qat_hash),
        "checkpoint_materialized_sha256": materialized_hash,
        "reload_quantizer_count": reload_quantizers,
        "expected_quantizer_count": expected_quantizers,
        "selection_manifest": str(selection_path.relative_to(root)) if selection_path else None,
        "selected_candidate": selected_name,
        "selection_winner": winner.get("name"),
        "selection_promoted_candidate": promotion.get("candidate"),
        "selection_accepted": selection_accepted,
        "export_diagnostic": str(export_path.relative_to(root)) if export_path else None,
        "export_route": export_route,
        "export_equivalence_accepted": export_accepted,
        "rejected_or_unresolved_diagnostics": rejected_records,
        "reasons": [
            reason
            for condition, reason in (
                (true_qat, "true NNCF QAT checkpoint provenance is not confirmed"),
                (selection_accepted, "checkpoint selection manifest does not confirm the promoted validation-only winner"),
                (export_accepted, "latest QAT OpenVINO export lacks accepted numerical-equivalence evidence"),
            )
            if not condition
        ],
    }


def ptq_validation_gate(root: Path, model_xml: Path | None = None) -> dict[str, Any]:
    """Validate the repository's existing PTQ comparator without calibrating or exporting anything."""
    manifest = read_json(root / "experiments/ptq/manifest.json") or {}
    comparison = read_json(root / "experiments/ptq/comparison.json") or {}
    fp32_manifest = read_json(root / "artifacts/checkpoints/fp32/manifest.json") or {}
    fp32_pointer = lfs_pointer(root / "artifacts/checkpoints/fp32/best.pt")
    artifact_record = manifest.get("PTQ artifact", {})
    file_record = artifact_record.get("files", {}) if isinstance(artifact_record, dict) else {}
    candidate = (model_xml or (root / PTQ_XML_RELATIVE_PATH)).expanduser().resolve()
    candidate_bin = candidate.with_suffix(".bin")
    reasons: list[str] = []

    ptq_only = (
        manifest.get("PTQ only") is True
        and manifest.get("QAT checkpoint used") is False
        and manifest.get("QAT training") is False
        and manifest.get("TEST split loaded") is False
        and "quantize=8" in str(manifest.get("PTQ API", ""))
        and "calibration_dataset" in str(manifest.get("PTQ API", ""))
    )
    if not ptq_only:
        reasons.append("existing PTQ manifest does not prove PTQ-only provenance or TEST/QAT isolation")

    source_hash = manifest.get("source FP32 SHA256")
    source_valid = bool(
        fp32_manifest.get("status") in {"completed", "passed"}
        and fp32_pointer
        and fp32_pointer.get("sha256") == source_hash
        and fp32_manifest.get("class_mapping") == EXPECTED_CLASSES
    )
    if not source_valid:
        reasons.append("PTQ source FP32 checkpoint hash or class mapping is not confirmed")

    models = comparison.get("models", {})
    comparison_valid = models.get("PTQ INT8", {}).get("status") == "VALID comparator"
    if not comparison_valid:
        reasons.append("repository comparison does not mark this PTQ artifact as a valid comparator")

    expected_xml_hash = file_record.get("XML SHA256")
    expected_bin_hash = file_record.get("BIN SHA256")
    xml_hash = sha256_file(candidate) if candidate.is_file() else None
    bin_hash = sha256_file(candidate_bin) if candidate_bin.is_file() else None
    hashes_valid = bool(
        candidate.is_file()
        and candidate_bin.is_file()
        and xml_hash == expected_xml_hash
        and bin_hash == expected_bin_hash
        and candidate.stat().st_size == file_record.get("XML bytes")
        and candidate_bin.stat().st_size == file_record.get("BIN bytes")
    )
    if not hashes_valid:
        reasons.append("PTQ IR XML/BIN is missing or SHA256/size does not match the PTQ manifest")

    accepted = bool(ptq_only and source_valid and comparison_valid and hashes_valid)
    return {
        "accepted": accepted,
        "status": "VALID EXISTING PTQ INT8 COMPARATOR" if accepted else "BLOCKED: PTQ provenance/hash gate failed",
        "model_xml": str(candidate),
        "xml_sha256": xml_hash,
        "bin_sha256": bin_hash,
        "ptq_only_provenance": ptq_only,
        "qat_checkpoint_used": manifest.get("QAT checkpoint used"),
        "test_split_loaded": manifest.get("TEST split loaded"),
        "source_fp32_sha256": source_hash,
        "source_checkpoint_valid": source_valid,
        "comparison_status": models.get("PTQ INT8", {}).get("status"),
        "artifact_hashes_match_manifest": hashes_valid,
        "calibration_performed_during_benchmark": False,
        "reasons": reasons,
    }
