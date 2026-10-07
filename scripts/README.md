# Struktur Script Penelitian

Direktori `scripts/` memakai satu pipeline utama yang ringkas untuk laporan.
Probe, smoke test, dan diagnosis dipisahkan dari langkah metodologi utama.

## Pipeline utama skripsi

| Step | File | Fungsi |
| ---: | --- | --- |
| 01 | `step01_download_dataset.py` | Akuisisi dataset Roboflow |
| 02 | `step02_audit_dataset.py` | Audit dan validasi dataset |
| 03 | `step03_train_fp32.py` | Training baseline YOLO11n FP32 |
| 04 | `step04_train_qat.py` | Training true Quantization-Aware Training |
| 05 | `step05_select_qat_best.py` | Seleksi checkpoint QAT terbaik menggunakan VAL |
| 06 | `step06_evaluate_final_test.py` | Evaluasi final FP32 dan QAT |
| 07 | `step07_export_openvino.py` | Konversi QAT final ke OpenVINO |

Ketujuh file tersebut adalah **implementasi canonical langsung**, bukan wrapper.

## Struktur direktori

```text
scripts/
├── README.md
├── step01_download_dataset.py
├── step02_audit_dataset.py
├── step03_train_fp32.py
├── step04_train_qat.py
├── step05_select_qat_best.py
├── step06_evaluate_final_test.py
├── step07_export_openvino.py
├── _shared/
│   ├── openvino_export_core.py
│   ├── nncf_reference_quant.py
│   └── op3_model_audit.py
├── diagnostics/
│   ├── probe_qat_compatibility.py
│   ├── smoke_train_qat.py
│   └── ... alat diagnosis tambahan
└── deployment/
    └── ... tool pengujian ROBOTIS OP3
```

Eksperimen lama yang tidak lagi menjadi jalur utama berada di
`archive/legacy_scripts/`.

## Aturan untuk laporan

Untuk BAB III dan lampiran program, gunakan **STEP 01–07** sebagai urutan
implementasi utama. `diagnostics/` hanya menjelaskan verifikasi teknis
tambahan dan tidak perlu menjadi langkah utama metodologi.

## Menjalankan

```bash
uv run python scripts/step01_download_dataset.py
uv run python scripts/step02_audit_dataset.py
uv run python scripts/step03_train_fp32.py
uv run python scripts/step04_train_qat.py
uv run python scripts/step05_select_qat_best.py
uv run python scripts/step06_evaluate_final_test.py
uv run python scripts/step07_export_openvino.py
```

Training dan FINAL TEST yang sudah selesai tidak perlu dijalankan ulang hanya
untuk dokumentasi.
