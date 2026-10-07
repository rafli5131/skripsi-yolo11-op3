# Urutan Script Implementasi untuk Laporan

Dokumen ini memetakan implementasi penelitian ke urutan script yang lebih mudah digunakan pada laporan skripsi. Entry point berurutan tersedia di [`scripts/report_steps/`](../scripts/report_steps/README.md).

## Urutan implementasi

| No. | Tahap | Script laporan | Keluaran utama |
| ---: | --- | --- | --- |
| 1 | Akuisisi dataset | `scripts/report_steps/01_download_dataset.py` | Dataset YOLO11 dari Roboflow |
| 2 | Audit dataset | `scripts/report_steps/02_audit_dataset.py` | Dataset tervalidasi dan siap training |
| 3 | Training baseline FP32 | `scripts/report_steps/03_train_fp32.py` | Checkpoint YOLO11n FP32 |
| 4 | Verifikasi QAT | `scripts/report_steps/04_probe_qat.py` | Bukti forward, loss, backward, dan quantizer NNCF |
| 5 | Smoke test QAT | `scripts/report_steps/05_smoke_qat.py` | Bukti integrasi QAT dengan lifecycle trainer |
| 6 | Training QAT | `scripts/report_steps/06_train_qat.py` | Checkpoint true NNCF QAT |
| 7 | Seleksi QAT terbaik | `scripts/report_steps/07_select_qat_best.py` | Checkpoint QAT final terpilih berbasis VAL |
| 8 | Evaluasi final | `scripts/report_steps/08_evaluate_final.py` | Precision, recall, F1, mAP50, dan mAP50-95 |
| 9 | Konversi OpenVINO | `scripts/report_steps/09_export_openvino.py` | OpenVINO IR FP32 dan QAT INT8 |

## Pemetaan ke metodologi

Urutan tersebut dapat digunakan langsung sebagai subbagian implementasi pada BAB III:

```text
3.x.1 Akuisisi Dataset
3.x.2 Audit dan Validasi Dataset
3.x.3 Pelatihan YOLO11n FP32
3.x.4 Persiapan Quantization-Aware Training
3.x.5 Verifikasi Integrasi QAT
3.x.6 Pelatihan Model QAT
3.x.7 Pemilihan Model QAT Terbaik
3.x.8 Evaluasi Model
3.x.9 Konversi Model ke OpenVINO
```

Script lama di `scripts/` tetap menjadi implementasi canonical. Folder `report_steps/` hanya memberi penomoran dan urutan yang jelas agar kode mudah dirujuk pada naskah skripsi tanpa mengubah jalur eksperimen yang sudah tervalidasi.

## Status akhir

Pipeline sampai konversi OpenVINO sudah selesai. Artefak QAT final menggunakan true NNCF QAT dan direct OpenVINO yang telah tervalidasi pada input VAL representatif. Tahap setelah dokumen ini adalah deployment dan benchmark pada ROBOTIS OP3.
