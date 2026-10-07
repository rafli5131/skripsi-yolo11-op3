# Urutan Script Implementasi untuk Laporan

Untuk laporan skripsi, pipeline utama disederhanakan menjadi tujuh langkah.
Probe kompatibilitas dan smoke test tetap disimpan di `scripts/diagnostics/`
sebagai bukti teknis, tetapi tidak dicampur dengan alur metodologi utama.

| No. | Tahap | Script utama | Keluaran utama |
| ---: | --- | --- | --- |
| 1 | Akuisisi dataset | `scripts/step01_download_dataset.py` | Dataset YOLO11 |
| 2 | Audit dataset | `scripts/step02_audit_dataset.py` | Dataset tervalidasi |
| 3 | Training FP32 | `scripts/step03_train_fp32.py` | Baseline YOLO11n FP32 |
| 4 | Training QAT | `scripts/step04_train_qat.py` | Checkpoint true NNCF QAT |
| 5 | Seleksi QAT | `scripts/step05_select_qat_best.py` | Checkpoint QAT final berbasis VAL |
| 6 | Evaluasi final | `scripts/step06_evaluate_final_test.py` | Precision, recall, F1, mAP |
| 7 | Konversi OpenVINO | `scripts/step07_export_openvino.py` | OpenVINO IR QAT INT8 |

## Pemetaan BAB III

```text
3.x.1 Akuisisi Dataset
3.x.2 Audit dan Validasi Dataset
3.x.3 Pelatihan YOLO11n FP32
3.x.4 Quantization-Aware Training
3.x.5 Pemilihan Model QAT Terbaik
3.x.6 Evaluasi Model
3.x.7 Konversi Model ke OpenVINO
```

Verifikasi kompatibilitas QAT dan smoke test dapat dijelaskan singkat pada
subbab implementasi QAT, dengan source di
`scripts/diagnostics/probe_qat_compatibility.py` dan
`scripts/diagnostics/smoke_train_qat.py`.

Setelah STEP 07, pekerjaan berlanjut ke deployment ROBOTIS OP3 sesuai
`docs/18_next_steps.md`.
