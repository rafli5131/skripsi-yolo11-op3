# Struktur Script Penelitian

Direktori `scripts/` sudah dipisahkan menjadi **pipeline utama**, **shared helper**, **diagnostic**, dan **deployment** agar kode untuk laporan tidak bercampur dengan eksperimen lama.

## Pipeline utama skripsi

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
| 09 | `step09_export_openvino.py` | Konversi QAT final ke OpenVINO |

File `step01` sampai `step09` adalah **implementasi canonical langsung**, bukan wrapper.

## Struktur direktori

```text
scripts/
├── README.md
├── step01_download_dataset.py
├── step02_audit_dataset.py
├── step03_train_fp32.py
├── step04_probe_qat_compatibility.py
├── step05_smoke_train_qat.py
├── step06_train_qat.py
├── step07_select_qat_best.py
├── step08_evaluate_final_test.py
├── step09_export_openvino.py
├── _shared/
│   ├── openvino_export_core.py
│   ├── nncf_reference_quant.py
│   └── op3_model_audit.py
├── diagnostics/
│   └── ... alat diagnosis dan validasi tambahan
└── deployment/
    └── ... tool pengujian ROBOTIS OP3
```

Eksperimen lama yang tidak lagi menjadi jalur utama dipindahkan ke
`archive/legacy_scripts/`.

## Aturan penggunaan

Untuk BAB III/lampiran kode, gunakan **hanya STEP 01–09** sebagai urutan implementasi utama. Folder `_shared/` adalah fungsi pendukung internal. Folder `diagnostics/` bukan hasil utama. Folder `deployment/` digunakan setelah model selesai dikonversi pada STEP 09.

## Contoh menjalankan pipeline

```bash
uv run python scripts/step01_download_dataset.py
uv run python scripts/step02_audit_dataset.py
uv run python scripts/step03_train_fp32.py
uv run python scripts/step04_probe_qat_compatibility.py
uv run python scripts/step05_smoke_train_qat.py
uv run python scripts/step06_train_qat.py
uv run python scripts/step07_select_qat_best.py
uv run python scripts/step08_evaluate_final_test.py
uv run python scripts/step09_export_openvino.py
```

Tidak perlu menjalankan ulang training atau FINAL TEST hanya untuk dokumentasi. Gunakan artefak hasil eksperimen yang sudah dibekukan.
