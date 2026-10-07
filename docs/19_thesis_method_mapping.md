# Pemetaan Metode terhadap Skripsi

## Tujuan

Memetakan implementasi ke metodologi dan pelaporan skripsi.

## Status

**REFERENCE**.

| Tahap | Implementasi | Dukungan BAB III | Dukungan BAB IV/V |
|---|---|---|---|
| Dataset | `scripts/step01_download_dataset.py`, `scripts/step02_audit_dataset.py` | sumber, split, audit | statistik/limitasi data |
| Baseline FP32 | `scripts/step03_train_fp32.py` | konfigurasi training | metrik baseline |
| QAT | `scripts/step04_train_qat.py` | true NNCF QAT | checkpoint dan metrik |
| Seleksi QAT | `scripts/step05_select_qat_best.py` | pemilihan berbasis VAL | checkpoint final |
| Evaluasi final | `scripts/step06_evaluate_final_test.py` | protokol TEST read-only | precision, recall, F1, mAP |
| OpenVINO | `scripts/step07_export_openvino.py` | konversi dan validasi | IR QAT final |

Gunakan [decision log](16_decision_log.md) untuk justifikasi perubahan konfigurasi dan [timeline](20_experiment_timeline.md) untuk kronologi. Ini bukan naskah BAB lengkap.


Urutan lengkap untuk kebutuhan penulisan laporan tersedia di [29 Urutan Script Laporan](29_urutan_skrip_laporan.md).
