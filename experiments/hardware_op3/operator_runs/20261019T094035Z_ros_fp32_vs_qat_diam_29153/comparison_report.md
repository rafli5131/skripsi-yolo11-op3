# Perbandingan ROS OpenVINO FP32 dan QAT INT8 — 19 Oktober 2026

Enam run berhasil diselesaikan: 3 × 60 detik per model. Semua `completed=true`, exception termasuk warmup = 0, dan `thermal_abort=false`. Jumlah baris CSV, frame, serta mean/P95 latency telah dicocokkan terhadap JSON.

## Protokol

- Hardware: CPU onboard OP3, Intel Core i5-1240P; ROS 2 Jazzy, Python 3.12.3, OpenVINO 2026.3.1.
- Kamera Logitech C920 `/dev/video0`, 640×480, format YUYV; setelan kamera 30 FPS. Framerate aktual sumber kamera tidak diukur secara independen.
- Model OpenVINO FP32 dan QAT INT8 memakai detector ROS yang sama; input 640×640, batch 1, confidence 0,25, IoU 0,45.
- Setiap run: 30 warmup inference dan 30 warmup frame kamera sebelum pengukuran 60 detik. Debug image dan benchmark telemetry dimatikan; Detection2DArray tetap dipublikasikan.
- Urutan: FP32 run 01–03, lalu QAT run 01–03. Kamera dimulai ulang saat berganti model. Adegan kamera langsung tidak dijamin identik.
- Kondisi `diam` mengikuti permintaan operator. Tidak ada telemetri gerak atau bukti fisik independen yang direkam; validitas kondisi diam tetap bergantung pada pengamatan operator.
- Isolasi: ROS_DOMAIN_ID=73 dan ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST untuk kedua model. Tidak ditemukan proses manager/gateway/fall detector/behavior pada pemeriksaan sebelum benchmark.
- Runner hanya memulai kamera dan detector; tidak memulai manager atau mengirim perintah aktuator.
- Commit: `a1bab9a7dc00989f86566c2bef2f16bb4bc81388`. SHA-256 XML/BIN sebelum pengukuran tersedia di `session_manifest.json` dan cocok setelah selesai.

## Hasil per run

Latency pipeline dihitung dari awal callback gambar sampai deteksi siap, sebelum membentuk dan memublikasikan pesan deteksi. Inference mencakup pemanggilan OpenVINO dan penyalinan output. `callback_end_to_end_ms` juga tersedia dalam JSON/CSV untuk bagian hingga setelah publikasi.

| Model | Run | Durasi (s) | Frame | Effective FPS | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 01 | 60.025 | 1492 | 24.86 | 30.52 / 47.80 | 34.04 / 52.68 |
| fp32 | 02 | 60.020 | 1420 | 23.66 | 31.20 / 51.86 | 34.92 / 57.86 |
| fp32 | 03 | 60.033 | 1496 | 24.92 | 30.72 / 48.42 | 34.24 / 54.02 |
| qat_int8 | 01 | 60.038 | 828 | 13.79 | 13.38 / 14.48 | 17.98 / 19.83 |
| qat_int8 | 02 | 60.015 | 793 | 13.21 | 13.48 / 14.81 | 18.26 / 20.42 |
| qat_int8 | 03 | 60.005 | 850 | 14.17 | 13.34 / 14.55 | 17.92 / 19.85 |

## Ringkasan tiga run

Angka berikut adalah mean ± simpangan baku sampel antar tiga run. P95 setiap run disajikan pada tabel di atas; tidak dianggap sebagai satu P95 gabungan.

| Metrik | FP32 | QAT INT8 |
|---|---:|---:|
| Mean inference (ms) | 30.81 ± 0.35 | 13.40 ± 0.07 |
| Mean pipeline (ms) | 34.40 ± 0.46 | 18.05 ± 0.18 |
| Mean callback hingga setelah publikasi (ms) | 34.59 ± 0.46 | 18.23 ± 0.18 |
| Effective FPS | 24.48 ± 0.71 | 13.72 ± 0.48 |
| CPU sistem (%) | 45.64 ± 0.86 | 6.44 ± 0.14 |
| CPU proses detector (%) | 713.52 ± 13.03 | 90.13 ± 2.33 |
| RAM sistem terpakai (MiB) | 5323.55 ± 51.55 | 5244.15 ± 8.80 |
| RSS proses detector (MiB) | 258.88 ± 0.56 | 245.69 ± 0.32 |
| Ukuran pasangan XML + BIN (MiB) | 10.259 | 3.250 |
| Suhu sensor tertinggi yang tersampel (°C) | 79.0 | 79.0 |

CPU proses mengikuti psutil: 100% kira-kira satu CPU logis dan dapat melampaui 100% karena multithreading. CPU sistem dirata-ratakan terhadap seluruh CPU logis. RAM sistem berasal dari `system_memory_used_bytes`; RSS hanya proses detector. Ukuran model mencakup XML dan BIN; `model.size_bytes` di metadata runner hanya menghitung XML sehingga tidak dipakai sebagai ukuran keseluruhan.

## Interpretasi

Mean inference QAT lebih rendah dengan rasio FP32/QAT 2.30×; mean latency pipeline dan penggunaan CPU juga lebih rendah. Namun effective FPS QAT pada sesi ini lebih rendah daripada FP32. Hasil ini tidak mendukung klaim bahwa QAT meningkatkan FPS ROS pada pengukuran tersebut.
Effective FPS adalah jumlah frame berhasil dibagi durasi pengukuran aktual, sehingga tidak sama dengan 1000 dibagi latency inference. Sumber pembatas throughput belum dapat ditetapkan dari rekaman ini; laju sumber kamera aktual dan frame yang hilang sebelum diterima detector tidak diukur secara independen. Tidak dilakukan benchmark tambahan.
Pengukuran ini menilai performa pipeline kamera ROS pada kondisi yang diminta operator. Tidak ada ground truth sehingga detection_count/confidence tidak dipakai sebagai precision, recall, F1, atau mAP. Hasil FINAL TEST, evaluasi OpenVINO, kamera standalone, berjalan dan stability tetap dilaporkan terpisah.

## Catatan lingkungan dan berkas

- Peringatan hook `op3_advanced_detector/.../pythonpath_develop.sh` yang hilang tetap tercatat pada console. Paket usb_cam dan detector tetap dapat berjalan.
- Log kamera mencatat file kalibrasi `default_cam.yaml` tidak tersedia. Detector memakai raw image dan tidak menggunakan CameraInfo; tidak ada kegagalan callback yang terekam.
- `fp32/` dan `qat_int8/`: protocol.json, environment JSON, logs, latency CSV, resource CSV, run_metadata JSON.
- `session_manifest.json`: protokol sesi, isolasi ROS, urutan model, ukuran dan checksum model.
- `comparison_summary.json`: hasil ringkas per run.
- `verified_summary.json`: hasil verifikasi data, agregasi, serta pemeriksaan proses setelah pengukuran.
- Seluruh hasil sesi berada di direktori baru; hasil penelitian lama tidak ditimpa. Tidak dilakukan commit atau push.
