# Hasil Benchmark Server OpenVINO

Benchmark diagnostik host untuk tiga model YOLO11n: FP32, PTQ INT8, dan QAT INT8. Pengukuran menggunakan OpenVINO CPU, input sintetis 640×640 yang sama, 30 warmup, dan 300 iterasi terukur per model. Ini **bukan benchmark final ROBOTIS OP3**. Metode, provenance, seluruh statistik, dan batasan dijelaskan dalam [laporan lengkap](../../docs/23_server_openvino_benchmark.md).

Direktori ini juga memuat [evaluasi kualitas deteksi VAL empat model](VAL_METRICS_REPORT.md): FP32 native PyTorch, FP32 OpenVINO, PTQ INT8, dan QAT INT8. Evaluasi VAL memakai 1.213 gambar berlabel dan berbeda dari benchmark latency sintetis di bawah.

## Kualitas deteksi pada VAL

| Metrik | FP32 native | FP32 OpenVINO | PTQ INT8 | QAT INT8 |
| --- | ---: | ---: | ---: | ---: |
| Precision | 0.9474 | 0.9362 | 0.9341 | 0.8657 |
| Recall | 0.9465 | 0.9576 | 0.9502 | 0.9305 |
| F1 | 0.9470 | 0.9468 | 0.9421 | 0.8970 |
| mAP50 | 0.9734 | 0.9730 | 0.9721 | 0.9614 |
| mAP50-95 | 0.7783 | 0.7713 | 0.7672 | 0.6549 |

Sumber lengkap: [laporan VAL](VAL_METRICS_REPORT.md), [comparison JSON](val_comparison.json), [overall CSV](val_comparison.csv), [per-class CSV](val_per_class.csv), dan empat file `val_<model>.json`. Semua model memakai VAL, CPU, 640, batch 1, IoU 0.7, dan confidence default Ultralytics. FINAL TEST tidak dijalankan ulang. Hasil QAT lebih rendah daripada PTQ pada metrik deteksi VAL ini; perbandingan kecepatan server berada pada tabel benchmark terpisah di bawah.

## Benchmark latency sintetis

| Metrik | FP32 | PTQ INT8 | QAT INT8 |
| --- | ---: | ---: | ---: |
| Mean inference latency (ms) | 17.160 | 13.097 | 12.829 |
| Median inference latency (ms) | 15.517 | 10.696 | 10.907 |
| P95 inference latency (ms) | 22.773 | 25.855 | 23.120 |
| P99 inference latency (ms) | 44.885 | 41.992 | 26.895 |
| Inference FPS | 58.276 | 76.351 | 77.951 |
| Effective end-to-end FPS | 50.661 | 61.385 | 63.675 |
| Process CPU mean (%) | 3231.921 | 2865.058 | 2974.009 |
| Process RSS mean (MiB) | 254.582 | 233.798 | 238.006 |
| Model XML + BIN (MiB) | 10.259 | 3.250 | 3.319 |

FPS inference dihitung dari `1000 / mean inference latency`; effective FPS mencakup preprocess dan postprocess. Process CPU bisa melebihi 100% karena 100% setara dengan satu logical CPU. RSS adalah RAM fisik proses; ukuran XML + BIN adalah ukuran file di disk.

## Perbandingan

| Dibanding FP32 | PTQ INT8 | QAT INT8 |
| --- | ---: | ---: |
| Penurunan mean latency | 23.67% | 25.24% |
| Kenaikan inference FPS | 31.02% | 33.76% |
| Penurunan ukuran model | 68.32% | 67.64% |

Pada satu run server ini, QAT memiliki mean latency 0.269 ms dan p95 2.735 ms lebih rendah daripada PTQ, serta FPS 1.600 lebih tinggi. PTQ memakai process CPU mean 108.950 poin persen dan RSS mean 4.208 MiB lebih rendah daripada QAT; artifact PTQ lebih kecil 0.070 MiB. PTQ juga memiliki median latency sedikit lebih rendah. Perbedaan kecil dalam satu run tidak membuktikan keunggulan umum di ROBOTIS OP3.

## File raw

| Model | Ringkasan JSON | 300 iterasi CSV | Sampling resource CSV |
| --- | --- | --- | --- |
| FP32 | [JSON](server_benchmark_fp32.json) | [CSV](server_benchmark_fp32.csv) | [CSV](resources_server_benchmark_fp32.csv) |
| PTQ INT8 | [JSON](server_benchmark_ptq_int8.json) | [CSV](server_benchmark_ptq_int8.csv) | [CSV](resources_server_benchmark_ptq_int8.csv) |
| QAT INT8 | [JSON](server_benchmark_qat_int8.json) | [CSV](server_benchmark_qat_int8.csv) | [CSV](resources_server_benchmark_qat_int8.csv) |

Setiap JSON mencatat timestamp, Git commit saat run, model path dan SHA256, konfigurasi, statistik latency, FPS, serta ringkasan resource. Ketiga artifact lolos gate provenance saat benchmark; lihat [cara verifikasi model](../../docs/26_provenance_dan_verifikasi.md). Tidak ada accuracy yang dihitung ulang dari input sintetis ini. Angka TEST PyTorch berada di `experiments/final_test/` dan tidak boleh digabungkan seolah berasal dari run server yang sama.

Pengukuran OP3 dan bukti terkait berada terpisah di `experiments/hardware_op3/`. Tahap yang masih diperlukan dijelaskan pada [timeline](../../docs/20_experiment_timeline.md) dan [langkah berikutnya](../../docs/18_next_steps.md).
