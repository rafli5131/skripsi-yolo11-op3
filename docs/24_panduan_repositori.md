# Panduan Membaca Repository

## Mulai dari sini

Repository ini mencatat penelitian deteksi objek YOLO11n untuk ROBOTIS OP3. Ada tiga jalur model: FP32 baseline, PTQ INT8 sebagai comparator, dan true NNCF QAT INT8. Jalur QAT final yang berlaku sekarang adalah checkpoint `qat_best.pt` hasil seleksi VAL dan IR direct OpenVINO yang lolos audit. Kegagalan export QAT lama adalah bagian sejarah, bukan status IR final saat ini.

| Jika ingin mengetahui… | Baca |
| --- | --- |
| Ringkasan status dan susunan direktori | Dokumen ini |
| Kenapa percobaan lama gagal dan apa yang dibersihkan | [25 Kegagalan dan pembersihan](25_kegagalan_dan_pembersihan.md) |
| Model mana yang boleh dipakai dan cara memeriksanya | [26 Provenance dan verifikasi](26_provenance_dan_verifikasi.md) |
| Perbedaan VAL, TEST, server, dan OP3 | [27 Cara membaca hasil](27_cara_membaca_hasil.md) |
| Angka benchmark server FP32/PTQ/QAT | [Ringkasan dekat raw result](../experiments/server_openvino/README.md) dan [laporan lengkap](23_server_openvino_benchmark.md) |
| Precision, recall, F1, dan mAP empat model pada VAL | [Laporan evaluasi VAL](28_server_val_detection_comparison.md) |
| Hasil final TEST PyTorch | [08 Final TEST](08_final_test_evaluation.md) |
| Riwayat pemilihan checkpoint QAT | [23 QAT best selection](23_qat_best_selection.md) |

## Peta direktori

| Direktori | Isi | Perlakuan |
| --- | --- | --- |
| `configs/` | Konfigurasi eksperimen | Baca sebelum menjalankan skrip; jangan mengubah konfigurasi hasil final secara diam-diam |
| `scripts/` | Training, export, audit, evaluasi, dan benchmark | Gunakan environment `uv`; beberapa skrip adalah alat penelitian historis |
| `artifacts/checkpoints/fp32/` | Checkpoint FP32 final | Beku; sumber baseline dan PTQ |
| `artifacts/checkpoints/qat/` | Checkpoint QAT final dan manifest | Beku; sumber IR QAT valid |
| `artifacts/checkpoints/qat_search/` | Kandidat pencarian QAT | Bukti seleksi VAL; kandidat yang kalah bukan berarti gagal |
| `artifacts/openvino/fp32/` dan `qat_int8/` | IR FP32 dan QAT final | Artifact inference yang diaudit |
| `artifacts/openvino/ptq_int8/` | Salinan PTQ historis | Identik hash dengan comparator canonical di `experiments/ptq/backend_models/ptq_openvino_model/` |
| `artifacts/openvino/*diagnostic.json` | Bukti export, termasuk percobaan yang ditolak | Simpan untuk provenance; baca status di dalamnya |
| `experiments/final_test/` | Evaluasi held-out TEST PyTorch yang dibekukan | Bukan benchmark server/OP3 |
| `experiments/ptq/` | PTQ comparator dan evaluasi VAL | Canonical PTQ IR ada di `backend_models/ptq_openvino_model/` |
| `experiments/server_openvino/` | Raw benchmark latency sintetis, evaluasi deteksi VAL, dan laporan hasil | Keduanya diagnostik server; bukan hasil deployment OP3 |
| `experiments/hardware_op3/` | Hasil pengujian hardware yang sudah ada | Jangan campur dengan benchmark server |
| `experiments/debug/` | Smoke test dan eksplorasi awal yang berhasil/berguna | Bukan final model |
| `docs/` | Penjelasan metode, status, kegagalan, dan hasil | Dokumen bertanggal fase lama perlu dibaca bersama status terkini |

`data/raw/`, `.venv/`, `.cache/`, `runs/`, dan `wandb/` adalah data/environment lokal yang diabaikan Git. Jangan stage direktori itu saat mengubah dokumentasi.

## Status saat ini

FP32, PTQ comparator, dan QAT direct OpenVINO tersedia. Gate PTQ dan QAT accepted. Benchmark server untuk ketiganya selesai dan tersimpan terpisah. FINAL TEST PyTorch FP32/QAT telah dilakukan sebelumnya dan tidak dijalankan ulang dalam pembersihan repository ini. Benchmark final langsung pada ROBOTIS OP3 tetap pekerjaan berikutnya.

## Urutan membaca yang disarankan

1. Baca [provenance](26_provenance_dan_verifikasi.md) untuk tahu artifact mana yang sah.
2. Baca [kegagalan](25_kegagalan_dan_pembersihan.md) agar tidak mengulang jalur export yang ditolak.
3. Baca [cara membaca hasil](27_cara_membaca_hasil.md) sebelum membandingkan angka dari split atau perangkat berbeda.
4. Baca [ringkasan hasil server](../experiments/server_openvino/README.md), lalu [laporan lengkap](23_server_openvino_benchmark.md) untuk metode, latency, FPS, CPU, dan RSS.
