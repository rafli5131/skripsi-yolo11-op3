"""Shared provenance checks for OP3 model inventory and benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_CLASSES = {"0": "ball", "1": "gawang", "2": "robot"}
RELATIVE_MAE_LIMIT = 0.01


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


def qat_validation_gate(root: Path) -> dict[str, Any]:
    """Require selected true-QAT provenance and a passed export/equivalence gate."""
    qat_manifest_path = root / "artifacts/checkpoints/qat/manifest.json"
    qat_manifest = read_json(qat_manifest_path) or {}
    qat_hash = (
        qat_manifest.get("artifact freeze", {}).get("sha256", {}).get("qat_best.pt")
        if isinstance(qat_manifest.get("artifact freeze"), dict)
        else None
    )
    pointer = lfs_pointer(root / "artifacts/checkpoints/qat/qat_best.pt")
    qat_api = str(qat_manifest.get("QAT API", ""))
    true_qat = (
        qat_manifest.get("status") == "passed"
        and "create_compressed_model" in qat_api
        and not qat_api.lstrip().startswith("nncf.quantize")
        and isinstance(qat_hash, str)
        and pointer is not None
        and pointer["sha256"] == qat_hash
        and qat_manifest.get("quantizer count after reload") == qat_manifest.get("expected quantizer count")
    )

    selection_candidates = [
        root / "artifacts/checkpoints/qat/selection_manifest.json",
        root / "experiments/qat/qat_best_search/selection_manifest.json",
    ]
    selection_path = next((path for path in selection_candidates if path.is_file()), None)
    selection = read_json(selection_path) if selection_path else None
    selection_status = str((selection or {}).get("status", "")).lower()
    selection_accepted = bool(
        selection
        and any(word in selection_status for word in ("passed", "accepted", "valid", "completed"))
        and qat_hash in json.dumps(selection, sort_keys=True)
    )

    export_candidates = [
        root / "artifacts/openvino/qat_onnx_export_diagnostic.json",
        root / "artifacts/openvino/qat_export_diagnostic.json",
    ]
    export_records = [(path, read_json(path)) for path in export_candidates if path.is_file()]
    export_record = next(
        (
            (path, value)
            for path, value in export_records
            if value and str(value.get("status", "")).lower() in {"passed", "accepted", "valid"}
        ),
        None,
    )
    export_accepted = False
    export_path = None
    if export_record:
        export_path, diagnostic = export_record
        promotion = diagnostic.get("promotion")
        export_hash = diagnostic.get("QAT SHA256") or diagnostic.get("QAT checkpoint SHA256")
        export_accepted = bool(
            isinstance(promotion, dict)
            and str(promotion.get("status", "")).lower() in {"passed", "accepted", "promoted", "valid"}
            and export_hash == qat_hash
            and diagnostic.get("calibration occurred") is False
            and diagnostic.get("PTQ used") is False
        )
        if export_accepted:
            summary = diagnostic.get("OpenVINO numerical summary") or diagnostic.get("numerical summary")
            if not isinstance(summary, dict):
                export_accepted = False
            else:
                n_total = summary.get("VAL input count")
                n_passed = summary.get("count relative MAE <= 1 percent")
                n_failed = summary.get("count relative MAE > 1 percent")
                max_error = summary.get("maximum relative MAE")
                export_accepted = bool(
                    isinstance(n_total, int)
                    and n_total > 0
                    and n_passed == n_total
                    and n_failed == 0
                    and isinstance(max_error, (int, float))
                    and max_error <= RELATIVE_MAE_LIMIT
                )

    rejected_records = [
        {"path": str(path.relative_to(root)), "status": value.get("status")}
        for path, value in export_records
        if value and str(value.get("status", "")).lower() not in {"passed", "accepted", "valid"}
    ]
    accepted = bool(true_qat and selection_accepted and export_accepted)
    return {
        "accepted": accepted,
        "status": "VALID QAT INT8 OpenVINO" if accepted else "BLOCKED: no validated QAT OpenVINO artifact",
        "true_nncf_qat_provenance": true_qat,
        "checkpoint_sha256": qat_hash,
        "checkpoint_lfs_pointer_matches_manifest": bool(pointer and pointer.get("sha256") == qat_hash),
        "selection_manifest": str(selection_path.relative_to(root)) if selection_path else None,
        "selection_accepted": selection_accepted,
        "export_diagnostic": str(export_path.relative_to(root)) if export_path else None,
        "export_equivalence_accepted": export_accepted,
        "rejected_or_unresolved_diagnostics": rejected_records,
        "reasons": [
            reason
            for condition, reason in (
                (true_qat, "true NNCF QAT checkpoint provenance is not confirmed"),
                (selection_accepted, "checkpoint selection manifest is missing or does not accept this checkpoint"),
                (export_accepted, "latest QAT OpenVINO export lacks accepted numerical-equivalence evidence"),
            )
            if not condition
        ],
    }
