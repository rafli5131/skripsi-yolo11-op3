"""Run the final QAT OpenVINO export on managed CUDA hosts without nvcc."""

from __future__ import annotations

from nncf_reference_quant import configure_nncf_reference_quantization
import export_openvino_qat_final


def main() -> int:
    backend = configure_nncf_reference_quantization(force=True)
    print("NNCF QAT execution backend:", backend, flush=True)
    return export_openvino_qat_final.main()


if __name__ == "__main__":
    raise SystemExit(main())
