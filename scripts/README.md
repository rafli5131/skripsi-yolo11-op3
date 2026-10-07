# Struktur Script Penelitian

Direktori ini memakai **satu jalur canonical bernomor** untuk alur utama skripsi.

## Pipeline utama

| Step | File | Fungsi |
| ---: | --- | --- |
| 01 | `step01_download_dataset.py` | Akuisisi dataset Roboflow |
| 02 | `step02_audit_dataset.py` | Audit dataset dan label |
| 03 | `step03_train_fp32.py` | Training baseline YOLO11n FP32 |
| 04 | `step04_probe_qat_compatibility.py` | Verifikasi kompatibilitas true NNCF QAT |
| 05 | `step05_smoke_train_qat.py` | Smoke test lifecycle QAT |
| 06 | `step06_train_qat.py` | Training true QAT |
| 07 | `step07_select_qat_best.py` | Seleksi checkpoint QAT terbaik menggunakan VAL |
| 08 | `step08_evaluate_final_test.py` | Evaluasi final model terpilih |
| 09 | `step09_export_openvino.py` | Export QAT final ke OpenVINO |

File `step01` sampai `step09` adalah **kode eksperimen utama yang sebenarnya**, bukan file pembungkus.

## Helper dan alat pendukung

File tanpa prefix `stepXX` adalah helper, diagnostic, comparator historis, atau alat deployment. File tersebut tidak membentuk urutan metode utama BAB III dan tidak perlu dimasukkan sebagai Kode Program utama dalam laporan.

Helper penting yang dipakai pipeline:

- `export_openvino.py`: fungsi inti export dan validasi OpenVINO.
- `nncf_reference_quant.py`: helper backend quantization NNCF.
- `op3_model_audit.py`: helper audit artefak untuk tahap deployment.

## Menjalankan pipeline

Contoh:

```bash
uv run python scripts/step01_download_dataset.py
uv run python scripts/step02_audit_dataset.py
uv run python scripts/step03_train_fp32.py
# ...
uv run python scripts/step09_export_openvino.py
```

Tidak perlu menjalankan ulang training atau FINAL TEST hanya untuk dokumentasi. Gunakan hasil yang sudah dibekukan.
