# Pemetaan Metode terhadap Skripsi

## Tujuan

Memetakan implementasi ke metodologi dan pelaporan skripsi.

## Status

**REFERENCE**.

| Tahap | Implementasi | Dukungan BAB III | Dukungan BAB IV/V |
|---|---|---|---|
| Dataset | `download_dataset.py`, `audit_dataset.py` | sumber, split, audit | statistik/limitasi data |
| Baseline FP32 | `train_fp32.py` | konfigurasi training | metrik validasi/test |
| Probe QAT | `probe_qat_compatibility.py` | verifikasi API | bukti forward/loss/backward |
| Trainer smoke | `smoke_train_qat.py` | integrasi lifecycle | diagnostic, bukan hasil utama |
| QAT final | `train_qat.py` | hyperparameter QAT | checkpoint dan metrik |
| Held-out | `evaluate_final_test.py` | protokol test read-only | tabel perbandingan |
| OpenVINO | `export_openvino.py` | jalur konversi | VALID/REJECTED export evidence |

Gunakan [decision log](16_decision_log.md) untuk justifikasi perubahan konfigurasi dan [timeline](20_experiment_timeline.md) untuk kronologi. Ini bukan naskah BAB lengkap.
