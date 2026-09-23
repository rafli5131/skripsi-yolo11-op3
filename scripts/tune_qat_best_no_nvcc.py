"""Run validation-only true-QAT tuning on a CUDA host without nvcc.

This launcher preserves ``scripts/tune_qat_best.py`` as the experiment
orchestrator, but injects the verified pure-PyTorch NNCF fake-quant backend into
every candidate subprocess before ``train_qat.main()`` starts.

Use this on managed servers where PyTorch CUDA works but CUDA development tools
or system Python development headers cannot be installed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import tune_qat_best as tuner  # noqa: E402


_original_child_code = tuner.child_code
_original_annotate_manifest = tuner.annotate_manifest


def _child_code_with_reference_backend(candidate, resume):
    code = _original_child_code(candidate, resume)
    marker = "import train_qat as q\n"
    if marker not in code:
        raise RuntimeError("Could not locate train_qat import in tuner child code; refusing unsafe injection.")
    injected = (
        marker
        + "from nncf_reference_quant import configure_nncf_reference_quantization\n"
        + "backend_report = configure_nncf_reference_quantization(force=True)\n"
        + "print('NNCF QAT execution backend:', backend_report, flush=True)\n"
    )
    return code.replace(marker, injected, 1)


def _annotate_manifest_with_backend(manifest_path: Path, candidate) -> None:
    _original_annotate_manifest(manifest_path, candidate)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["NNCF quantization execution backend"] = "pure-pytorch-nncf-reference"
    payload["NNCF native CUDA extension used"] = False
    payload["nvcc required for this run"] = False
    payload["QAT semantics"] = "NNCF fake quantization during training; not PTQ"
    payload["backend verification"] = (
        "scripts/verify_nncf_reference_quant.py performs numerical reference and CUDA autograd checks"
    )
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    # Make the experiment provenance explicit for subprocesses and logs.
    os.environ["NNCF_QAT_REFERENCE_BACKEND"] = "1"
    tuner.child_code = _child_code_with_reference_backend
    tuner.annotate_manifest = _annotate_manifest_with_backend
    return tuner.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except tuner.TuneStop as error:
        print(f"QAT tuning failed: {error}", file=sys.stderr)
        raise SystemExit(1)
