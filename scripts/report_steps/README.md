# Urutan Script untuk Laporan Skripsi

Folder ini menyediakan entry point berurutan dari persiapan dataset sampai model selesai dikonversi ke OpenVINO.

File di folder ini adalah **entry point laporan**. Implementasi utama tetap berada di `scripts/` agar eksperimen yang sudah tervalidasi tidak diduplikasi dan tidak berubah perilakunya.

| Step | Script laporan | Implementasi utama | Tujuan |
| ---: | --- | --- | --- |
| 01 | `01_download_dataset.py` | `scripts/download_dataset.py` | Mengunduh dataset dari Roboflow |
| 02 | `02_audit_dataset.py` | `scripts/audit_dataset.py` | Memeriksa struktur, label, split, dan konsistensi dataset |
| 03 | `03_train_fp32.py` | `scripts/train_fp32.py` | Melatih YOLO11n FP32 sebagai baseline |
| 04 | `04_probe_qat.py` | `scripts/probe_qat_compatibility.py` | Memastikan NNCF QAT kompatibel dengan YOLO11n |
| 05 | `05_smoke_qat.py` | `scripts/smoke_train_qat.py` | Menguji lifecycle QAT sebelum training final |
| 06 | `06_train_qat.py` | `scripts/train_qat.py` | Menjalankan training true QAT |
| 07 | `07_select_qat_best.py` | `scripts/tune_qat_best.py` | Menyeleksi checkpoint QAT terbaik menggunakan VAL |
| 08 | `08_evaluate_final.py` | `scripts/evaluate_final_test.py` | Evaluasi final FP32 dan QAT pada held-out TEST |
| 09 | `09_export_openvino.py` | `scripts/export_openvino_qat_final.py` | Mengonversi model final menjadi OpenVINO |

## Alur ringkas

```text
01 Dataset download
        ↓
02 Dataset audit
        ↓
03 FP32 baseline training
        ↓
04 QAT compatibility probe
        ↓
05 QAT smoke test
        ↓
06 QAT training
        ↓
07 QAT best selection
        ↓
08 Final evaluation
        ↓
09 OpenVINO export
```

## Menjalankan

Jalankan dari root repository.

```bash
uv run python scripts/report_steps/01_download_dataset.py
uv run python scripts/report_steps/02_audit_dataset.py
uv run python scripts/report_steps/03_train_fp32.py
uv run python scripts/report_steps/04_probe_qat.py
uv run python scripts/report_steps/05_smoke_qat.py
uv run python scripts/report_steps/06_train_qat.py
uv run python scripts/report_steps/07_select_qat_best.py
uv run python scripts/report_steps/08_evaluate_final.py
uv run python scripts/report_steps/09_export_openvino.py
```

Argumen CLI dari masing-masing implementasi tetap dapat digunakan karena entry point meneruskan eksekusi ke script canonical.

## Catatan untuk laporan

Untuk BAB III, gunakan nomor step ini sebagai urutan implementasi. Untuk BAB IV, hasil utama tetap diambil dari artefak dan manifest eksperimen yang sudah selesai. Held-out TEST tidak perlu dijalankan ulang hanya untuk dokumentasi.
