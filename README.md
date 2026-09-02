# Skripsi YOLO11n QAT OpenVINO

Implementasi penelitian **Optimasi Deteksi Objek Real-Time pada Robot Humanoid menggunakan Quantization-Aware Training dan OpenVINO**.

Model yang digunakan adalah YOLO11n. Dataset object detection berasal dari Roboflow workspace `humanoid-atom`, project `advance_vision_human`, dengan kelas `0: ball`, `1: gawang`, dan `2: robot`.

## Environment

Project ini menggunakan `uv` dan virtual environment lokal `.venv`. Jalankan seluruh perintah dari root project dan arahkan cache ke dalam project:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
uv sync
uv run python scripts/diagnose_system.py
uv run python scripts/diagnose_gpu.py
```

Salin `.env.example` menjadi `.env` hanya bila diperlukan; jangan menyimpan API key di Git. `ROBOFLOW_VERSION` wajib ditentukan sebelum dataset diunduh.

## GPU verification from host SSH

Jalankan pemeriksaan berikut langsung pada terminal SSH host, bukan dari execution sandbox Codex:

```bash
cd /mnt/nas-hpg9/rafli5131/rafli/skripsi

export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"

nvidia-smi

# Contoh hanya jika GPU fisik 1 masih paling kosong.
export CUDA_VISIBLE_DEVICES=1

# ROS 2 pada host dapat mewariskan PYTHONPATH Python 3.12.
# Bersihkan hanya untuk command project ini; tidak mengubah shell global.
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/verify_torch_cuda.py

env -u PYTHONPATH -u VIRTUAL_ENV uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("logical cuda:0 =", torch.cuda.get_device_name(0))
PY
```

Pilih GPU yang paling kosong berdasarkan `nvidia-smi` sebelum memulai proses Python. Jangan hard-code pilihan GPU ke source code. Bila `CUDA_VISIBLE_DEVICES=1`, GPU fisik 1 dipetakan PyTorch menjadi logical `cuda:0`; konfigurasi penelitian tetap menggunakan `device: 0`.

Jika terminal telah menjalankan environment ROS 2, `PYTHONPATH` dapat menunjuk ke package Python 3.12/Micromamba di luar project dan mengalahkan package `.venv` Python 3.11. Gunakan prefiks `env -u PYTHONPATH -u VIRTUAL_ENV` di atas untuk setiap command Python project pada shell tersebut.

Training GPU belum boleh dilakukan sampai `nvidia-smi` dan `scripts/verify_torch_cuda.py` berhasil dari terminal SSH host. Dataset, training, QAT, conversion OpenVINO, dan benchmark belum dijalankan pada tahap ini.

## Phase 3: dataset and FP32 smoke test

Set `ROBOFLOW_API_KEY` and the exact published `ROBOFLOW_VERSION` in the shell (or in an uncommitted `.env` copied from `.env.example`). The download script uses the Roboflow `yolov11` export, validates the source class mapping, and creates `data/dataset/data.yaml` with explicit filesystem-verified paths.

```bash
cd /mnt/nas-hpg9/rafli5131/rafli/skripsi

export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
export ROBOFLOW_API_KEY='replace-with-your-key'
export ROBOFLOW_VERSION='replace-with-version'

nvidia-smi

# Select an idle physical GPU yourself before Python starts.
export CUDA_VISIBLE_DEVICES=1

env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/verify_torch_cuda.py
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/download_dataset.py
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/audit_dataset.py
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/smoke_train_fp32.py
```

The smoke test uses one epoch, 10% of train data, `val=False`, `save=False`, and `plots=False`. It is not the 100-epoch FP32 experiment. On success it writes `experiments/debug/fp32_smoke/manifest.json`.

## Phase 3.5: FP32 speed benchmark

The speed benchmark runs one epoch on `fraction=0.25` for batch sizes 16, 32, 64, and 128, comparing `cache=False` and `cache="ram"`. It measures only the train-epoch callbacks; final validation overhead is excluded. It does not change `configs/base.yaml`.

```bash
cd /mnt/nas-hpg9/rafli5131/rafli/skripsi

export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"

nvidia-smi
# Select an idle physical GPU before Python starts.
export CUDA_VISIBLE_DEVICES=1

env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/benchmark_fp32_speed.py
```

The script writes `experiments/debug/fp32_speed_benchmark.json` and reports the fastest successful configuration. CUDA OOM results are recorded and followed by a per-process CUDA cache cleanup; no other process is touched.

## Final FP32 baseline preparation

The final baseline is intentionally conservative for the shared server: `batch=16`, `workers=2`, `cache=false`, logical `device=0`, `imgsz=640`, and AMP. It uses the fixed 100-epoch AdamW configuration in `configs/base.yaml`; it never selects a physical GPU itself.

Start a new baseline only after checking that the selected physical GPU is sufficiently idle:

```bash
cd /mnt/nas-hpg9/rafli5131/rafli/skripsi

export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
export CUDA_VISIBLE_DEVICES=1

env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/train_fp32.py
```

The script saves `last.pt`, `best.pt`, metrics, plots, and a manifest under `experiments/fp32/fp32_baseline/`. To continue an interrupted run, use its saved optimizer state rather than a new model:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/train_fp32.py --resume
```

W&B is disabled by default. To enable the script's explicit epoch-level W&B logging, provide `WANDB_API_KEY` through the shell or uncommitted `.env`, optionally set `WANDB_PROJECT` and `WANDB_NAME`, then add `--enable-wandb`:

```bash
export WANDB_API_KEY='replace-with-your-key'
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/train_fp32.py --enable-wandb
```
