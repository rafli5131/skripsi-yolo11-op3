# Hasil akhir ROS FP32 dan QAT pada OP3

## 5A. ROS FP32 — robot berjalan dan perbandingan dengan QAT

FP32 berjalan menyusul sesi QAT dengan manager dan parameter gait yang sama, tanpa init pose ulang. Operator mengonfirmasi berjalan stabil dengan penopangan yang sama sebelum pengukuran, lalu mengonfirmasi kondisi sesudah sesi. Masing-masing run menggunakan 30 warmup inference + 30 frame, CPU, input 640×640/batch 1, C920 YUYV 640×480 pada 30 FPS, confidence 0,25, IoU 0,45, sampling 0,2 detik, BEST_EFFORT/depth 5, dan domain vision 73. Protocol JSON FP32–QAT berjalan cocok pada seluruh field selain path model.

Semua run FP32 memiliki `completed=true`, durasi aktual ≥60 detik, `thermal_abort=false`, nol exception termasuk warmup, serta jumlah frame/resource yang cocok dengan raw CSV.

| Model / kondisi | Frame | Durasi (s) | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) | Effective FPS | Exception |
|---|---:|---:|---:|---:|---:|---:|
| FP32 berjalan run_01 | 1231 | 60,0065 | 33,9576 / 55,8076 | 36,4623 / 59,9422 | 20,5144 | 0 |
| FP32 berjalan run_02 | 1190 | 60,0198 | 34,1952 / 58,8548 | 36,7979 / 63,3219 | 19,8268 | 0 |
| FP32 berjalan run_03 | 1234 | 60,0166 | 34,2168 / 58,8241 | 36,6580 / 63,2791 | 20,5610 | 0 |
| FP32 berjalan gabungan | 3655 | 180,0429 | 34,1225 / 58,3080 | 36,6377 / 62,6153 | 20,3007 | 0 |
| QAT berjalan gabungan | 1807 | 180,0622 | 12,9642 / 14,8962 | 16,6935 / 19,3917 | 10,0354 | 0 |

### Resource FP32 berjalan per run dan perbandingan model

| Model / run | CPU sistem / proses mean (%) | RAM sistem mean (MB) | RSS mean / peak (MB) | Suhu mean / maksimum (°C) | Sampel |
|---|---:|---:|---:|---:|---:|
| run_01 | 43,7956 / 662,3734 | 3521,5546 | 271,3356 / 271,3559 | 59,2660 / 67,0000 | 297 |
| run_02 | 42,7404 / 646,1448 | 3501,6018 | 271,8799 / 271,8925 | 61,9293 / 68,0000 | 297 |
| run_03 | 44,3108 / 668,5165 | 3507,4229 | 272,0933 / 272,1178 | 63,3771 / 69,0000 | 297 |
| fp32_berjalan gabungan | 43,6156 / 659,0116 | 3510,1931 | 271,7696 / 272,1178 | 61,5241 / 69,0000 | 891 |
| qat_berjalan gabungan | 6,7054 / 62,1569 | 3455,5512 | 257,1154 / 257,3476 | 44,8670 / 54,0000 | 887 |

### Variasi antar tiga run FP32 berjalan

| Metrik | Rata-rata nilai per run | SD sampel antar-run |
|---|---:|---:|
| Frame | 1218,3333 | 24,5832 |
| Effective FPS | 20,3007 | 0,4111 |
| Mean inference (ms) | 34,1232 | 0,1438 |
| Mean pipeline (ms) | 36,6394 | 0,1686 |
| CPU sistem mean (%) | 43,6156 | 0,8005 |
| CPU detector mean (%) | 659,0116 | 11,5585 |
| RAM sistem mean (MB) | 3510,1931 | 10,2608 |
| RSS detector mean (MB) | 271,7696 | 0,3907 |
| Suhu mean (°C) | 61,5241 | 2,0853 |

### Ringkasan ROS diam dan berjalan — dataset 19 Oktober

Tabel berikut memakai sesi diam 19 Oktober untuk kedua model, terpisah dari referensi QAT diam 7 Oktober (2.442 frame) pada Bagian 4–5. Tiap baris adalah gabungan tiga run dari datasetnya sendiri; tidak ada pooling lintas model/kondisi/tanggal.

| Model / kondisi | Frame | Durasi (s) | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) | Effective FPS | Exception |
|---|---:|---:|---:|---:|---:|---:|
| FP32 diam | 4408 | 180,0777 | 30,8054 / 49,3312 | 34,3875 / 55,1134 | 24,4783 | 0 |
| QAT diam | 2471 | 180,0576 | 13,3977 / 14,5546 | 18,0472 / 19,9430 | 13,7234 | 0 |
| FP32 berjalan | 3655 | 180,0429 | 34,1225 / 58,3080 | 36,6377 / 62,6153 | 20,3007 | 0 |
| QAT berjalan | 1807 | 180,0622 | 12,9642 / 14,8962 | 16,6935 / 19,3917 | 10,0354 | 0 |

Pada sesi berjalan, mean inference FP32 34,1225 ms dan QAT 12,9642 ms (rasio FP32/QAT 2,6320). Namun effective FPS FP32 20,3007 dan QAT 10,0354. Jadi penurunan latency inference tidak diikuti kenaikan throughput kamera ROS pada konfigurasi ini. Nilai FPS berasal dari raw frame/durasi, bukan 1000/mean inference. Frame yang tidak diterima callback tidak tercatat sebagai exception. Tidak ada eksperimen QoS tambahan dilakukan.

Perbandingan bersifat deskriptif: sesi berjalan berurutan QAT lalu FP32, adegan dan beban latar tidak sepenuhnya sama. Dukungan operator dikonfirmasi sama, tetapi tingkat penopangan/kontak lantai tidak diukur rinci; tidak tersedia video atau konfirmasi fisik terpisah setiap run. Ini bukan evaluasi gait berjalan mandiri atau bukti sebab-akibat gerakan robot. Sesi diam dan berjalan memakai tanggal/waktu serta commit metadata yang berbeda; nilai historis tidak diubah.

Sumber: [raw FP32 berjalan](..), [rekap JSON](ros_fp32_vs_qat_recap.json), [CSV model/kondisi](ros_fp32_vs_qat_conditions.csv), [konfirmasi operator](../operator_confirmation_end.json).

## Keterbatasan tambahan

- Walking sessions were sequential, QAT before FP32, without randomized order or identical visual scenes; ambient/background system conditions may differ.
- Operator confirmed the same support for FP32 as QAT. Support amount and floor contact were not separately measured; no independent video or per-run physical ground truth. This is inference during supported movement, not autonomous gait validation.
- No motion gateway/fall-detector process was active; physical safety was monitored by the operator. Controller init was not repeated and gait/offsets were unchanged between the two walking sessions.
- Technical completion and zero exceptions do not prove all camera frames were delivered. BEST_EFFORT/depth5 was retained for both models as requested; higher FPS cannot be inferred from lower inference latency.
- Stationary measurements are earlier 19 October sessions with different background loads; legacy QoS was not explicitly recorded in per-run metadata. The QAT stationary reference from 7 October remains a separate dataset in the thesis.
- Detector process CPU/RSS exclude camera/manager. System CPU/RAM include other processes. MB is decimal; temperature is the maximum thermal-zone reading per sample, not necessarily an identified CPU package sensor.
- Historical git_commit measurement fields and all old raw data remain unchanged. New FP32 and prior QAT commits differ due to reporting commits; relevant pipeline code was not changed.
- No accuracy metrics are inferred from live camera confidence; frozen TEST results remain separate.
