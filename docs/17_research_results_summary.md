# Ringkasan Hasil Penelitian

## Tujuan

Merangkum metrik validasi dan held-out TEST tanpa menarik kesimpulan deployment.

## Status

**FP32/QAT TEST selesai; benchmark diagnostik server FP32/PTQ/QAT selesai; benchmark final OP3 tetap terpisah.**

| Ruang evaluasi | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| QAT best validation (epoch 1) | 0,8810 | 0,9393 | 0,9585 | 0,6513 |
| QAT final validation (epoch 5) | 0,8748 | 0,9448 | 0,9610 | 0,6510 |
| FP32 TEST | 0,9313 | 0,9506 | 0,9642 | 0,7799 |
| QAT TEST | 0,8542 | 0,9375 | 0,9512 | 0,6567 |

## Perbandingan OpenVINO (VAL diagnostik)

| Metrik | FP32 OpenVINO | PTQ INT8 | Delta PTQ |
|---|---:|---:|---:|
| Precision | 0,936223 | 0,934116 | −0,225% |
| Recall | 0,957564 | 0,950208 | −0,768% |
| mAP50 | 0,972973 | 0,972145 | −0,085% |
| mAP50-95 | 0,771344 | 0,767179 | −0,540% |
| F1 | 0,946773 | 0,942093 | −0,494% |
| Latency host (ms/image) | 21,965 | 13,231 | −39,763% |
| FPS host diagnostik | 45,527 | 75,580 | +66,011% |
| Ukuran IR | 10.756.936 B | 3.407.480 B | −68,323% |

Tabel VAL PTQ di atas dibuat saat hanya IR QAT lama yang rejected tersedia. IR QAT final sekarang berbeda dan valid melalui direct OpenVINO; lihat [provenance](26_provenance_dan_verifikasi.md). Angka QAT final tidak ditambahkan ke tabel VAL historis ini karena tidak dievaluasi dalam run PTQ yang sama. Timing tabel adalah diagnostik host lama, bukan benchmark final ROBOTIS OP3.

TEST delta QAT terhadap FP32: P -0,0772 (-8,29%), R -0,0131 (-1,37%), mAP50 -0,0130 (-1,35%), mAP50-95 -0,1232 (-15,80%), F1 -0,0470 (-4,99%). Per-class mAP50-95: ball 0,9130/0,7470; gawang 0,5683/0,5282; robot 0,8585/0,6948 (FP32/QAT).

Benchmark server OpenVINO CPU terbaru mencatat latency, FPS, CPU, dan RSS ketiga IR pada [laporan server](23_server_openvino_benchmark.md). Angka itu tidak membuktikan manfaat deployment OP3; pengukuran final perlu dilakukan langsung pada robot.

## Artefak terkait

[QAT final](07_final_qat_training.md), [TEST](08_final_test_evaluation.md), [comparison JSON](../experiments/final_test/comparison.json).
