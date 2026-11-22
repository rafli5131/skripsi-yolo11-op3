# Lampiran hasil pengujian PTQ INT8

Hasil PTQ INT8 digunakan sebagai pembanding tambahan di lampiran. Hasil utama penelitian membandingkan FP32 dan QAT. Grafik benchmark yang menyertakan PTQ tersedia di [lampiran grafik](figures/appendix_ptq/README.md).

| Pengujian tambahan | Tabel dan data |
| --- | --- |
| VAL historis | Tabel pada lampiran ini |
| VAL server | [Laporan metrik deteksi](28_server_val_detection_comparison.md) |
| Benchmark sintetis CPU server | [Laporan latensi, FPS, dan sumber daya](23_server_openvino_benchmark.md) |
| Benchmark standalone OP3, mode sintetis dan kamera | [Rekap pengukuran](../experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv) |

## Peran dan sumber

PTQ adalah comparator yang terpisah dari true NNCF QAT. PTQ historis dibuat dari checkpoint FP32 `artifacts/checkpoints/fp32/best.pt` melalui `torch2openvino(..., quantize=8, calibration_dataset=...)` dan `nncf.quantize()` dengan subset deterministik TRAIN. Manifest menyatakan `QAT checkpoint used=false` dan `TEST split loaded=false`. Benchmark server terbaru **memakai artifact PTQ existing**, tanpa quantization atau calibration baru.

Canonical IR untuk pembandingan sekarang adalah `experiments/ptq/backend_models/ptq_openvino_model/model.{xml,bin}`. Pemeriksaan `ptq_validation_gate()` dijelaskan dalam [provenance](26_provenance_dan_verifikasi.md).

## Hambatan loader awal

Percobaan evaluator pertama berhenti karena loader Ultralytics mengharapkan direktori bernama `*_openvino_model`, bukan argumen langsung `model.xml`. Staging lokal ke `backend_models/ptq_openvino_model/` menyelesaikan masalah loader tanpa membuat IR PTQ baru. Ini masalah cara memuat artifact, bukan bukti PTQ numeriknya gagal.

## Hasil VAL diagnostik historis

| Metrik | FP32 OpenVINO | PTQ INT8 | Delta PTQ |
| --- | ---: | ---: | ---: |
| Precision | 0,936223 | 0,887335 | −5,222% |
| Recall | 0,957564 | 0,935881 | −2,264% |
| mAP50 | 0,972973 | 0,961287 | −1,201% |
| mAP50-95 | 0,771344 | 0,651042 | −15,596% |
| F1 | 0,946773 | 0,910962 | −3,782% |
| Latency host lama (ms/image) | 21,965 | 13,947 | −36,504% |
| FPS host lama | 45,527 | 71,701 | +57,491% |

Sumber: `experiments/ptq/{fp32_openvino_val_metrics,ptq_val_metrics}.json`, `comparison.json`, dan `manifest.json`. Angka timing tersebut berasal dari evaluasi VAL historis, bukan benchmark server tiga model terbaru dan bukan final OP3. Laporan benchmark server yang adil untuk tiga IR ada di [23 Server OpenVINO benchmark](23_server_openvino_benchmark.md).

Comparison PTQ historis dibuat dalam konteks eksperimen yang berbeda dari artefak QAT final. IR QAT final yang digunakan sekarang telah lolos audit direct OpenVINO dan harus dijadikan rujukan untuk status QAT.

## Batas perbandingan

VAL PTQ historis, TEST PyTorch final, benchmark server, dan pengujian hardware OP3 berasal dari konteks berbeda. Sebut split, evaluator, model, dan host setiap kali memakai angka. Jangan menafsirkan latency VAL lama sebagai latency final robot. Baca [cara membaca hasil](27_cara_membaca_hasil.md) sebelum menyusun tabel tesis.
