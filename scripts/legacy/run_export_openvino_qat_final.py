"""Run the final QAT OpenVINO export on managed CUDA hosts without nvcc."""

from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

from nncf_reference_quant import configure_nncf_reference_quantization
import step09_export_openvino


def main() -> int:
    backend = configure_nncf_reference_quantization(force=True)
    print("NNCF QAT execution backend:", backend, flush=True)
    return step09_export_openvino.main()


if __name__ == "__main__":
    raise SystemExit(main())
