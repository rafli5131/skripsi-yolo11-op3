# Perintah Baca dan Verifikasi

Perintah di sini untuk memeriksa hasil yang sudah ada. Jalankan dari root repository dengan environment proyek. Training, export, PTQ calibration, dan FINAL TEST adalah tahap historis yang sudah selesai; jangan menjalankannya hanya untuk membaca laporan.

## Siapkan file LFS dan environment

```bash
git lfs pull
export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
env -u PYTHONPATH -u VIRTUAL_ENV uv run python -c 'import openvino as ov; print(ov.__version__)'
```

OpenVINO yang tercatat pada benchmark server adalah versi 2024.4.0. Bila model hanya berupa Git LFS pointer kecil, ulangi `git lfs pull` sebelum audit.

## Periksa provenance PTQ dan QAT tanpa mengubah artifact

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run python -c 'import sys; from pathlib import Path; sys.path.insert(0, "scripts"); from op3_model_audit import qat_validation_gate, ptq_validation_gate; root = Path.cwd(); print("QAT:", qat_validation_gate(root)); print("PTQ:", ptq_validation_gate(root, root / "experiments/ptq/backend_models/ptq_openvino_model/model.xml"))'
```

Kedua hasil harus memiliki `accepted: True` dan `reasons: []`. [Provenance](26_provenance_dan_verifikasi.md) menjelaskan syarat gate dan hash final.

## Lihat hasil

```bash
ls experiments/server_openvino/
ls experiments/hardware_op3/
git status --short
```

JSON server berisi ringkasan; CSV timing memiliki 300 baris data per model, dan CSV resource berisi sampling proses. Lihat [laporan server](23_server_openvino_benchmark.md) untuk tabel yang sudah dihitung dan [cara membaca hasil](27_cara_membaca_hasil.md) untuk batas perbandingan. Perintah benchmark ulang dan konfigurasi tepatnya ada di laporan server, tetapi menjalankannya lagi akan membuat pengukuran baru yang berbeda waktunya.
