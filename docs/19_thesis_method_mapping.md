# Pemetaan Metode terhadap Skripsi

## Tujuan

Memetakan implementasi ke metodologi dan pelaporan skripsi.

## Status

**REFERENCE**.

| Tahap | Implementasi | Dukungan BAB III | Dukungan BAB IV/V |
|---|---|---|---|
| Dataset | `report_steps/01_download_dataset.py`, `report_steps/02_audit_dataset.py` | sumber, split, audit | statistik/limitasi data |
| Baseline FP32 | `report_steps/03_train_fp32.py` | konfigurasi training | metrik validasi/test |
| Probe QAT | `report_steps/04_probe_qat.py` | verifikasi API | bukti forward/loss/backward |
| Trainer smoke | `report_steps/05_smoke_qat.py` | integrasi lifecycle | diagnostic, bukan hasil utama |
| QAT final | `report_steps/06_train_qat.py`, `report_steps/07_select_qat_best.py` | hyperparameter dan seleksi QAT | checkpoint dan metrik |
| Held-out | `report_steps/08_evaluate_final.py` | protokol test read-only | tabel perbandingan |
| OpenVINO | `report_steps/09_export_openvino.py` | jalur konversi | bukti export dan validasi |

Gunakan [decision log](16_decision_log.md) untuk justifikasi perubahan konfigurasi dan [timeline](20_experiment_timeline.md) untuk kronologi. Ini bukan naskah BAB lengkap.


Urutan lengkap untuk kebutuhan penulisan laporan tersedia di [29 Urutan Script Laporan](29_urutan_skrip_laporan.md).
