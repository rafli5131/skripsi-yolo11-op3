# Hasil Benchmark Server OpenVINO

Benchmark diagnostik host untuk tiga model YOLO11n: FP32, PTQ INT8, dan QAT INT8. Pengukuran menggunakan OpenVINO CPU, input sintetis 640×640 yang sama, 30 warmup, dan 300 iterasi terukur per model. Ini **bukan benchmark final ROBOTIS OP3**. Metode, provenance, seluruh statistik, dan batasan dijelaskan dalam [laporan lengkap](../../docs/23_server_openvino_benchmark.md).

## Ringkasan hasil

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
