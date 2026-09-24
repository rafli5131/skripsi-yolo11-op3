# Server VAL Detection Metrics — FP32 Native vs FP32 OpenVINO vs PTQ vs QAT

Evaluasi diagnostik pada host server dengan **split VAL** yang sama: 1.213 gambar dan 1.525 anotasi. Empat model memakai evaluator Ultralytics 8.4.130 pada CPU, input 640, batch 1, IoU 0.7, dan confidence default (`conf=None`). Tidak ada training, export, quantization, calibration, atau FINAL TEST dalam run ini.

## Hasil keseluruhan

| Metric | FP32 native | FP32 OpenVINO | PTQ INT8 | QAT INT8 |
| --- | ---: | ---: | ---: | ---: |
| precision | 0.9474 | 0.9362 | 0.9341 | 0.8657 |
| recall | 0.9465 | 0.9576 | 0.9502 | 0.9305 |
| F1 | 0.9470 | 0.9468 | 0.9421 | 0.8970 |
| mAP50 | 0.9734 | 0.9730 | 0.9721 | 0.9614 |
| mAP50-95 | 0.7783 | 0.7713 | 0.7672 | 0.6549 |

F1 dihitung sebagai harmonic mean dari precision dan recall keseluruhan yang dilaporkan evaluator. Nilai lain berasal langsung dari evaluator; angka dibulatkan untuk tabel, sedangkan JSON/CSV menyimpan presisi aslinya.

## Selisih terhadap FP32 native

| Metric (poin persentase) | FP32 OpenVINO | PTQ INT8 | QAT INT8 |
| --- | ---: | ---: | ---: |
| precision | -1.12 | -1.33 | -8.17 |
| recall | +1.11 | +0.37 | -1.60 |
| F1 | -0.02 | -0.49 | -5.00 |
| mAP50 | -0.05 | -0.13 | -1.21 |
| mAP50-95 | -0.69 | -1.11 | -12.33 |

## Per kelas

| Class | Metric | FP32 native | FP32 OpenVINO | PTQ INT8 | QAT INT8 |
| --- | --- | ---: | ---: | ---: | ---: |
| ball | precision | 0.9676 | 0.9513 | 0.9430 | 0.8623 |
| ball | recall | 1.0000 | 1.0000 | 1.0000 | 0.9945 |
| ball | F1 | 0.9835 | 0.9751 | 0.9707 | 0.9237 |
| ball | mAP50 | 0.9903 | 0.9891 | 0.9892 | 0.9846 |
| ball | mAP50-95 | 0.8985 | 0.8835 | 0.8796 | 0.7119 |
| gawang | precision | 0.8955 | 0.8913 | 0.8988 | 0.9195 |
| gawang | recall | 0.8833 | 0.8998 | 0.8811 | 0.8389 |
| gawang | F1 | 0.8894 | 0.8956 | 0.8899 | 0.8774 |
| gawang | mAP50 | 0.9477 | 0.9479 | 0.9458 | 0.9431 |
| gawang | mAP50-95 | 0.6034 | 0.5983 | 0.5970 | 0.5558 |
| robot | precision | 0.9792 | 0.9660 | 0.9606 | 0.8154 |
| robot | recall | 0.9562 | 0.9729 | 0.9695 | 0.9582 |
| robot | F1 | 0.9675 | 0.9694 | 0.9650 | 0.8810 |
| robot | mAP50 | 0.9823 | 0.9818 | 0.9814 | 0.9565 |
| robot | mAP50-95 | 0.8329 | 0.8322 | 0.8250 | 0.6971 |

## Provenance dan batasan

Sumber model dan SHA256 ada di masing-masing `val_<model>.json`. PTQ dan QAT harus lolos provenance gate sebelum evaluasi; IR OpenVINO dijalankan melalui symlink sementara tanpa mengubah artifact aslinya. Dataset VAL canonical dibaca dari `data/dataset/data.yaml`; TEST tidak dimuat. JSON model mencatat waktu, host, commit, versi software, dan konfigurasi evaluator.

Ini adalah kualitas deteksi **VAL pada server**, bukan accuracy FINAL TEST dan bukan benchmark latency/FPS final ROBOTIS OP3. Waktu yang dikeluarkan evaluator hanya diagnostik; bandingkan performa host memakai [benchmark server terkontrol](README.md) dan performa robot dari pengujian OP3 yang terpisah.

Raw result: [comparison JSON](val_comparison.json), [overall CSV](val_comparison.csv), [per-class CSV](val_per_class.csv), serta `val_fp32_native.json`, `val_fp32_openvino.json`, `val_ptq_int8.json`, dan `val_qat_int8.json` di direktori ini.
