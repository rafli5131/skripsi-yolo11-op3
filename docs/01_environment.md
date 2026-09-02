# Lingkungan Eksekusi

## Tujuan

Mencatat lingkungan yang dibuktikan oleh manifest dan lockfile.

## Status

**DONE** untuk host training; driver NVIDIA **UNRESOLVED** karena tidak direkam dalam manifest.

## Server dan paket

Project root: `/mnt/nas-hpg9/rafli5131/rafli/advance_vision`. Python 3.11.2; uv 0.12.8; torch 2.4.1+cu121; torchvision 0.19.1+cu121; CUDA runtime 12.1; Ultralytics 8.4.130; NNCF 2.13.0; OpenVINO 2024.4.0; Roboflow 1.2.16; W&B 0.29.0; OpenCV tersedia melalui lockfile. GPU yang direkam: NVIDIA L40, 45.386 MiB VRAM. Versi driver NVIDIA tidak direkam.

`CUDA_VISIBLE_DEVICES=1` memetakan GPU fisik 1 menjadi `cuda:0` logis. Source code hanya memakai device logis 0 dan tidak menetapkan GPU fisik.

## Keselamatan shared server

Satu GPU, tanpa DDP/multi-GPU, `workers=2`, `cache=false`, tanpa sudo, `uv` serta `.venv` dan cache proyek-lokal. Variabel `.env.example` tanpa nilai rahasia: `UV_CACHE_DIR`, `TORCH_HOME`, `YOLO_CONFIG_DIR`, `CUDA_VISIBLE_DEVICES`, `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_NAME`, `ROBOFLOW_API_KEY`, `ROBOFLOW_VERSION`.

## Artefak terkait

[`.env.example`](../.env.example), [`uv.lock`](../uv.lock), [manifest FP32](../artifacts/checkpoints/fp32/manifest.json), [manifest QAT](../artifacts/checkpoints/qat/manifest.json).
