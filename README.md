# YOLO11n FP32, PTQ, dan QAT untuk ROBOTIS OP3

Repository penelitian deteksi objek `ball`, `gawang`, dan `robot` dengan tiga model OpenVINO: FP32 baseline, PTQ INT8 comparator, dan true NNCF QAT INT8. Artifact final dan hasil evaluasi disimpan agar setiap klaim dapat ditelusuri ke checkpoint, split, dan perangkat pengukuran yang tepat.

## Status saat ini

| Bagian | Status | Mulai membaca |
| --- | --- | --- |
| FP32 dan QAT checkpoint | Final; TEST PyTorch sudah dievaluasi | [Final TEST](docs/08_final_test_evaluation.md) |
| PTQ INT8 OpenVINO | Comparator existing; gate provenance accepted | [Provenance](docs/26_provenance_dan_verifikasi.md) |
| QAT INT8 OpenVINO | Direct OpenVINO; gate equivalence/provenance accepted | [Provenance](docs/26_provenance_dan_verifikasi.md) |
| Benchmark CPU server | FP32/PTQ/QAT selesai; diagnostik host | [Laporan server](docs/23_server_openvino_benchmark.md) |
| Evaluasi deteksi VAL server | FP32 native/OV, PTQ, QAT selesai | [Perbandingan precision/F1/mAP](docs/28_server_val_detection_comparison.md) |
| ROBOTIS OP3 (FP32/QAT ROS 2) | Kondisi diam dan berjalan didampingi operator: masing-masing 3 × 60 detik per model; stability QAT 600 detik selesai | [Hasil final ROS](experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/FP32_QAT_ROS_FINAL_RESULTS.md) |

**Status final eksperimen (19 Oktober 2026):** Semua pengukuran inti telah selesai dan tersimpan di `main`. [Snapshot final eksperimen `495a3a8`](https://github.com/rafli5131/skripsi-yolo11-op3/commit/495a3a893db3bfdfd24395cb25bf3a3cc300456c) mencakup kode, model, raw data, dan laporan pengujian. [Draft hasil pengujian](docs/draft_hasil_pengujian_akhir_skripsi_20261019.md) dan [rekap FP32–QAT ROS diam/berjalan](experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/FP32_QAT_ROS_FINAL_RESULTS.md) menjadi referensi terkini. Pengujian berjalan dilaksanakan dengan pendampingan operator, bukan pembuktian gait mandiri. QAT mempercepat inferensi tetapi FPS efektif ROS pada konfigurasi BEST_EFFORT/depth 5 tidak meningkat.

Eksperimen export awal dipertahankan sebagai riwayat pengembangan, sedangkan IR QAT final yang digunakan sekarang adalah artefak direct OpenVINO yang telah lolos validasi.

[Tahapan eksperimen](docs/20_experiment_timeline.md) menempatkan seleksi QAT best sebelum FINAL TEST dan export final. [Ringkasan hasil server di samping raw JSON/CSV](experiments/server_openvino/README.md) mencakup kualitas deteksi VAL empat model dan benchmark latency sintetis tiga IR sebagai dua run terpisah.

## Grafik untuk skripsi dan seminar hasil

[Galeri grafik hasil](docs/figures/README.md) menyediakan konteks batang dari nol,
tampilan titik berskala lokal, dan selisih bertanda agar perbedaan kecil terbaca.
[Catatan revisi dan cara membuat ulang](docs/32_revisi_visualisasi_grafik.md)
menjelaskan pemilihan skala. [Riset dan audit pemilihan grafik](docs/33_pemilihan_grafik_akademis.md) memetakan jenis data ke bentuk grafik dan sumber akademisnya. Seluruh grafik dibuat dari CSV/JSON yang sudah tersimpan,
tanpa menjalankan training atau pengukuran baru.

## Peta cepat

```text
artifacts/checkpoints/fp32/       checkpoint baseline final
artifacts/checkpoints/qat/        checkpoint QAT final + manifest seleksi
artifacts/openvino/fp32/          IR FP32
artifacts/openvino/qat_int8/      IR QAT final yang diterima
experiments/ptq/                 PTQ comparator + hasil VAL
experiments/final_test/          held-out TEST PyTorch
experiments/server_openvino/     raw benchmark server + README hasil
experiments/hardware_op3/        hasil OP3 diam/berjalan, stability, dan perbandingan model
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

Perintah gate dan verifikasi artefak ada di [provenance](docs/26_provenance_dan_verifikasi.md). Jangan memakai hasil latency server sebagai klaim latency final OP3. Training, QAT tuning, PTQ calibration, dan FINAL TEST tidak diperlukan untuk membaca hasil yang sudah dibekukan.

## Urutan script untuk laporan

Untuk penulisan BAB III dan lampiran kode, gunakan [urutan script bernomor](docs/29_urutan_skrip_laporan.md) dari `scripts/step01_download_dataset.py` sampai `scripts/step07_export_openvino.py`.
