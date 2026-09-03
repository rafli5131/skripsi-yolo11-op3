# Cookbook Perintah

## Tujuan

Menyediakan perintah reproduksi; jangan menjalankan tahap yang statusnya belum diizinkan.

## Status

**REFERENCE ONLY** — perintah training/evaluasi/export tidak dijalankan oleh dokumentasi ini.

```bash
cd /mnt/nas-hpg9/rafli5131/rafli/advance_vision
set -a; source .env; set +a                 # opsional; .env tidak di-Git
export UV_CACHE_DIR="$PWD/.cache/uv"
nvidia-smi
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/verify_torch_cuda.py
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/audit_dataset.py
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/train_fp32.py
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/train_fp32.py --resume
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/probe_qat_compatibility.py --train-images "$PWD/data/raw/advance_vision_human/train/images" --batch 2 --max-train-images 4 --init-samples 2
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/smoke_train_qat.py
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/train_qat.py
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/train_qat.py --resume
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/evaluate_final_test.py
# PTQ comparator: FP32 -> calibrated OpenVINO INT8; TRAIN calibration, VAL evaluation only
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/export_ptq_openvino.py --calibration-samples 300 --val-fraction 1.0 --overwrite
```

Untuk QAT CUDA extension, siapkan `CC=/usr/bin/gcc-11`, `CXX=/usr/bin/g++-11`, `CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH`, `TORCH_EXTENSIONS_DIR`, `TORCH_CUDA_ARCH_LIST=8.9`, dan `MPLCONFIGDIR` seperti run host tervalidasi. Diagnostic OpenVINO: `uv run python scripts/export_openvino.py --model qat-diagnostic`. Jangan masukkan API key ke command history atau dokumentasi.

## Artefak terkait

[README root](../README.md), [known issues](15_known_issues.md).
