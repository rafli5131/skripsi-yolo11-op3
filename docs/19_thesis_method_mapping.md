# Pemetaan Metode terhadap Skripsi

## Tujuan

Memetakan implementasi ke metodologi dan pelaporan skripsi.

## Status

**REFERENCE**.

| Tahap | Implementasi | Dukungan BAB III | Dukungan BAB IV/V |
|---|---|---|---|
| Dataset | `scripts/step01_download_dataset.py`, `scripts/step02_audit_dataset.py` | sumber, split, audit | statistik/limitasi data |
| Baseline FP32 | `scripts/step03_train_fp32.py` | konfigurasi training | metrik validasi/test |
| Probe QAT | `scripts/step04_probe_qat_compatibility.py` | verifikasi API | bukti forward/loss/backward |
| Trainer smoke | `scripts/step05_smoke_train_qat.py` | integrasi lifecycle | diagnostic, bukan hasil utama |
| QAT final | `scripts/step06_train_qat.py`, `scripts/step07_select_qat_best.py` | hyperparameter dan seleksi QAT | checkpoint dan metrik |
| Held-out | `scripts/step08_evaluate_final_test.py` | protokol test read-only | tabel perbandingan |
| OpenVINO | `scripts/step09_export_openvino.py` | jalur konversi | bukti export dan validasi |

Gunakan [decision log](16_decision_log.md) untuk justifikasi perubahan konfigurasi dan [timeline](20_experiment_timeline.md) untuk kronologi. Ini bukan naskah BAB lengkap.


Urutan lengkap untuk kebutuhan penulisan laporan tersedia di [29 Urutan Script Laporan](29_urutan_skrip_laporan.md).
