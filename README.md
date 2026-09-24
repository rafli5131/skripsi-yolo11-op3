# YOLO11n FP32, PTQ, dan QAT untuk ROBOTIS OP3

Repository penelitian deteksi objek `ball`, `gawang`, dan `robot` dengan tiga model OpenVINO: FP32 baseline, PTQ INT8 comparator, dan true NNCF QAT INT8. Artifact final dan hasil evaluasi disimpan agar setiap klaim dapat ditelusuri ke checkpoint, split, dan perangkat pengukuran yang tepat.

## Status saat ini

| Bagian | Status | Mulai membaca |
| --- | --- | --- |
| FP32 dan QAT checkpoint | Final; TEST PyTorch sudah dievaluasi | [Final TEST](docs/08_final_test_evaluation.md) |
| PTQ INT8 OpenVINO | Comparator existing; gate provenance accepted | [Provenance](docs/26_provenance_dan_verifikasi.md) |
| QAT INT8 OpenVINO | Direct OpenVINO; gate equivalence/provenance accepted | [Provenance](docs/26_provenance_dan_verifikasi.md) |
| Benchmark CPU server | FP32/PTQ/QAT selesai; diagnostik host | [Laporan server](docs/23_server_openvino_benchmark.md) |
| Benchmark final ROBOTIS OP3 | Perlu pengukuran langsung pada OP3 | [Next steps](docs/18_next_steps.md) |

IR QAT lama yang ditolak telah dibersihkan dari worktree. JSON diagnostiknya tetap ada sebagai bukti; penjelasan kegagalan ada di [dokumen pembersihan](docs/25_kegagalan_dan_pembersihan.md). IR QAT final saat ini adalah artifact berbeda yang lolos audit.

[Tahapan eksperimen](docs/20_experiment_timeline.md) menempatkan seleksi QAT best sebelum FINAL TEST dan export final. [Ringkasan hasil server di samping raw JSON/CSV](experiments/server_openvino/README.md) memudahkan pemeriksaan angka tanpa membuka dokumentasi panjang.

## Peta cepat

```text
artifacts/checkpoints/fp32/       checkpoint baseline final
artifacts/checkpoints/qat/        checkpoint QAT final + manifest seleksi
artifacts/openvino/fp32/          IR FP32
artifacts/openvino/qat_int8/      IR QAT final yang diterima
experiments/ptq/                 PTQ comparator + hasil VAL
experiments/final_test/          held-out TEST PyTorch
experiments/server_openvino/     raw benchmark server + README hasil
experiments/hardware_op3/        hasil hardware OP3 yang terpisah
docs/                           metode, provenance, kegagalan, hasil
```

[Panduan repository](docs/24_panduan_repositori.md) menjelaskan susunan lengkap. [Cara membaca hasil](docs/27_cara_membaca_hasil.md) menjelaskan perbedaan VAL, TEST, server, dan OP3. [Indeks dokumentasi](docs/README.md) menghubungkan seluruh catatan fase.

## Environment dan audit baca-saja

Jalankan dari root repository. File checkpoint dan IR menggunakan Git LFS, sehingga materialize dahulu sebelum mengaudit atau inferensi.

```bash
git lfs pull
export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
env -u PYTHONPATH -u VIRTUAL_ENV uv run python -c 'import openvino as ov; print(ov.__version__)'
```

Hash dan perintah gate spesifik ada di [provenance](docs/26_provenance_dan_verifikasi.md). Jangan memakai hasil latency server sebagai klaim latency final OP3. Training, QAT tuning, PTQ calibration, dan FINAL TEST tidak diperlukan untuk membaca hasil yang sudah dibekukan.
