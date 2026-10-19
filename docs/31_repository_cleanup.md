# Audit dan Pembersihan Repository (19 Oktober 2026)

## Peta kode

- `scripts/step01_*.py`–`step07_*.py`: pipeline metodologi utama (akuisisi dan audit dataset, FP32, QAT, seleksi checkpoint, TEST, export OpenVINO).
- `scripts/_shared/`: modul yang digunakan lintas skrip.
- `scripts/diagnostics/`: probe dan pengujian teknis, bukan pipeline inti BAB III.
- `scripts/deployment/`, `scripts/run_op3.sh`: benchmark, audit, dan kontrol operator untuk ROBOTIS OP3.
- `ros2_ws/src/yolo11_op3_vision/`: node inference/detection ROS 2, integrasi soccer, dan pengukuran.
- `artifacts/`, `experiments/`: checkpoint, OpenVINO IR, manifest, hasil TEST, pengukuran server, dan bukti OP3.

## Perubahan cleanup

1. Tiga salinan `experiments/final_test/qat_plots/val_batch*_labels.jpg` dihapus setelah dicek identik pada level Git blob dengan file `experiments/final_test/fp32_plots/`. Label ground truth tetap tersedia satu kali; gambar prediksi FP32 dan QAT tidak diubah.
2. Dua berkas Markdown besar `final_qat/resources/resources_all.md` dan `final_qat/tables/table_3_2_resources_raw.md` dihapus. Keduanya hanya representasi tabel dari raw CSV, bukan data baru. CSV gabungan dan CSV per-run tetap ada.
3. Generator `build_op3_thesis_tables.py` tidak lagi membuat representasi Markdown untuk tabel raw sumber daya yang berukuran besar. Markdown ringkasan (latency dan resource summary) tetap dibuat.
4. `.gitignore` diperluas untuk cache/test dan lingkungan Python lokal, sementara README menampilkan status pengujian OP3 sesuai bukti terkini.

## Data yang sengaja dipertahankan

**Jangan hapus secara otomatis:** `artifacts/checkpoints/`, `artifacts/openvino/`, `experiments/ptq/`, `experiments/final_test/`, `experiments/server_openvino/`, dan raw `experiments/hardware_op3/`. Beberapa file memang berulang karena provenance per eksperimen; identitas checkpoint, run, split VAL/TEST, dan sumber hardware perlu tetap dapat ditelusuri. Tidak ada model, konfigurasi training, atau pengukuran asli yang diubah.

`archive/legacy_scripts/` menyimpan referensi historis, bukan entrypoint utama. Folder `experiments/debug/` berguna untuk bukti probe/smoke test, namun tidak boleh digunakan sebagai hasil final.

## Verifikasi

```bash
# Dari root repository
python3 -m unittest discover -s scripts/deployment -p 'test_op3_thesis_tables.py'
python3 -m compileall -q scripts ros2_ws/src/yolo11_op3_vision/yolo11_op3_vision
git lfs pull
git status --short
```

Uji unit generator memakai direktori sementara, tidak menjalankan training dan tidak mengubah hasil pengukuran final. Uji integrasi ROS 2, model OpenVINO, dan operasi motor memerlukan lingkungan/perangkat ROBOTIS OP3 yang sesuai.

Status pengukuran ROBOTIS OP3: 3 run QAT ROS kondisi **diam** dan stability 600 detik sudah tercatat; kondisi **berjalan** belum. Baca [laporan hardware](../experiments/hardware_op3/HARDWARE_TEST_REPORT.md).