# Ringkasan Hasil Penelitian

## Tujuan

Merangkum metrik validasi dan held-out TEST tanpa menarik kesimpulan deployment.

## Status

**FP32/QAT TEST selesai; benchmark latency server dan evaluasi deteksi VAL server selesai; benchmark final OP3 tetap terpisah.**

| Ruang evaluasi | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| QAT best validation (epoch 1) | 0,8810 | 0,9393 | 0,9585 | 0,6513 |
| QAT final validation (epoch 5) | 0,8748 | 0,9448 | 0,9610 | 0,6510 |
| FP32 TEST | 0,9313 | 0,9506 | 0,9642 | 0,7799 |
| QAT TEST | 0,9341 | 0,9502 | 0,9721 | 0,7672 |

## Evaluasi server pada VAL (empat model)

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 native PyTorch | 0,9474 | 0,9465 | 0,9470 | 0,9734 | 0,7783 |
| FP32 OpenVINO | 0,9362 | 0,9576 | 0,9468 | 0,9730 | 0,7713 |
| PTQ INT8 OpenVINO | 0,8657 | 0,9305 | 0,8970 | 0,9614 | 0,6549 |
| QAT INT8 OpenVINO | 0,9341 | 0,9502 | 0,9421 | 0,9721 | 0,7672 |

Sumber: [raw VAL comparison](../experiments/server_openvino/val_comparison.json) dan [analisis](28_server_val_detection_comparison.md). Ini VAL CPU, bukan FINAL TEST di tabel sebelumnya dan bukan benchmark latency sintetis.

## Perbandingan OpenVINO (VAL diagnostik)

Tabel berikut berasal dari eksperimen PTQ historis. Evaluasi VAL empat model terbaru dengan konfigurasi sama dan FP32 native tambahan tersedia di [laporan server VAL](28_server_val_detection_comparison.md).

| Metrik | FP32 OpenVINO | PTQ INT8 | Delta PTQ |
|---|---:|---:|---:|
| Precision | 0,936223 | 0,887335 | −5,222% |
| Recall | 0,957564 | 0,935881 | −2,264% |
| mAP50 | 0,972973 | 0,961287 | −1,201% |
| mAP50-95 | 0,771344 | 0,651042 | −15,596% |
| F1 | 0,946773 | 0,910962 | −3,782% |
| Latency host (ms/image) | 21,965 | 13,947 | −36,504% |
| FPS host diagnostik | 45,527 | 71,701 | +57,491% |
| Ukuran IR | 10.756.936 B | 3.483.079 B | −67,621% |

Tabel VAL PTQ di atas dibuat saat hanya IR QAT lama yang rejected tersedia. IR QAT final sekarang berbeda dan valid melalui direct OpenVINO; lihat [provenance](26_provenance_dan_verifikasi.md). Angka QAT final tidak ditambahkan ke tabel VAL historis ini karena tidak dievaluasi dalam run PTQ yang sama. Timing tabel adalah diagnostik host lama, bukan benchmark final ROBOTIS OP3.

TEST delta QAT terhadap FP32: P +0,0028 (+0,30%), R -0,0004 (-0,04%), mAP50 +0,0079 (+0,82%), mAP50-95 -0,0127 (-1,63%), F1 +0,0012 (+0,13%). Per-class mAP50-95: ball 0,9130/0,8796; gawang 0,5683/0,5970; robot 0,8585/0,8250 (FP32/QAT).

Benchmark server OpenVINO CPU terbaru mencatat latency, FPS, CPU, dan RSS ketiga IR pada [laporan server](23_server_openvino_benchmark.md). Angka itu tidak membuktikan manfaat deployment OP3; pengukuran final perlu dilakukan langsung pada robot.

## Artefak terkait

[QAT final](07_final_qat_training.md), [TEST](08_final_test_evaluation.md), [comparison JSON](../experiments/final_test/comparison.json).
