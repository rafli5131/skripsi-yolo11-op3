# Indeks Dokumentasi

Mulai dari [panduan repository](24_panduan_repositori.md), lalu [provenance](26_provenance_dan_verifikasi.md), [catatan eksperimen dan pembersihan](25_kegagalan_dan_pembersihan.md), dan [cara membaca hasil](27_cara_membaca_hasil.md). Dokumen fase lama mencatat keadaan pada waktu penulisannya; status artifact final saat ini ada pada provenance dan manifest terkini.

## Status terkini

| Bidang | Status |
| --- | --- |
| FP32 baseline dan held-out TEST | Selesai |
| True NNCF QAT checkpoint dan held-out TEST | Selesai |
| PTQ INT8 comparator | Valid; provenance gate accepted |
| QAT INT8 direct OpenVINO | Valid; export/equivalence gate accepted |
| Benchmark server OpenVINO CPU | Selesai; diagnostik, terpisah dari OP3 |
| Evaluasi deteksi VAL empat model di server | Selesai; FP32 native/OV, PTQ, QAT |
| Benchmark final ROBOTIS OP3 | Selesai: FP32/QAT diam dan berjalan didampingi operator (masing-masing 3 × 60 detik), serta stability QAT 600 detik |

Urutan tahap final ada di [timeline eksperimen](20_experiment_timeline.md). [Ringkasan di direktori raw server](../experiments/server_openvino/README.md) dapat dibuka saat menelusuri JSON/CSV tanpa berpindah ke `docs/`.

## Hasil final skripsi dan ROBOTIS OP3

- [Draft hasil pengujian akhir](draft_hasil_pengujian_akhir_skripsi_20261019.md) — ringkasan akurasi TEST, model OpenVINO, benchmark OP3, ROS diam/berjalan, dan stability.
- [Hasil akhir ROS FP32 vs QAT, diam dan berjalan](../experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/FP32_QAT_ROS_FINAL_RESULTS.md) — pengukuran 3 × 60 detik per model/kondisi dan keterbatasan bukti gerak yang didampingi operator.
- [Pengujian QAT berjalan](../experiments/hardware_op3/operator_runs/20261019T131319Z_qat_berjalan_final/analysis/WALKING_RESULTS.md) — tiga run valid dan konfirmasi operator.
- [Laporan OP3 7 Oktober (historis)](../experiments/hardware_op3/HARDWARE_TEST_REPORT.md) — tetap memuat hasil pada tanggal pengujian tersebut.
- [Snapshot eksperimen `495a3a8`](https://github.com/rafli5131/skripsi-yolo11-op3/commit/495a3a893db3bfdfd24395cb25bf3a3cc300456c) — kode, konfigurasi, model, dan data pengukuran yang dibekukan. Perapian dokumentasi setelahnya tidak mengubah hasil penelitian.

**Interpretasi:** QAT mengurangi ukuran artefak dan waktu inferensi, tetapi tidak meningkatkan FPS efektif ROS pada seluruh pengujian BEST_EFFORT/depth 5. Gerak berjalan dilakukan dengan pendampingan operator, bukan validasi berjalan mandiri.

## Memahami proyek

- [00 Gambaran proyek](00_project_overview.md)
- [01 Lingkungan](01_environment.md)
- [02 Dataset dan split](02_dataset.md)
- [04 Desain true QAT](04_qat_design.md)
- [12 Reproducibility](12_reproducibility.md)
- [14 Inventaris artifact](14_artifact_inventory.md)
- [19 Pemetaan ke tesis](19_thesis_method_mapping.md)
- [24 Panduan repository](24_panduan_repositori.md)
- [26 Provenance dan verifikasi](26_provenance_dan_verifikasi.md)
- [27 Cara membaca hasil](27_cara_membaca_hasil.md)

## Training, pemilihan, dan evaluasi

- [03 FP32 training](03_fp32_training.md)
- [05 Implementasi QAT](05_qat_implementation.md)
- [06 Smoke test QAT](06_qat_smoke_and_validation.md)
- [07 QAT final historis](07_final_qat_training.md)
- [23 Seleksi QAT best terbaru](23_qat_best_selection.md)
- [08 Held-out FINAL TEST](08_final_test_evaluation.md)
- [17 Ringkasan hasil](17_research_results_summary.md)
- [28 Perbandingan precision, recall, F1, dan mAP server VAL](28_server_val_detection_comparison.md)
- [22 PTQ comparator](22_ptq_comparison.md)

## Export, validasi, dan deployment

- [09 Export OpenVINO saat ini](09_openvino_export.md)
- [10 Catatan export QAT](10_openvino_export_failure_analysis.md)
- [11 Diagnostik QAT CPU/CUDA](11_cpu_cuda_qat_diagnostic.md)
- [15 Known issues](15_known_issues.md)
- [16 Decision log](16_decision_log.md)
- [20 Timeline eksperimen](20_experiment_timeline.md)
- [21 Percobaan ONNX historis](21_qat_onnx_openvino_export.md)
- [23 Benchmark server FP32/PTQ/QAT](23_server_openvino_benchmark.md)
- [Ringkasan hasil di direktori server](../experiments/server_openvino/README.md)
- [Raw laporan evaluasi VAL empat model](../experiments/server_openvino/VAL_METRICS_REPORT.md)
- [25 Catatan eksperimen dan pembersihan](25_kegagalan_dan_pembersihan.md)
- [18 Rencana implementasi OP3 (historis, per 7 Oktober 2026)](18_next_steps.md)

## Referensi operasional

- [13 Perintah proyek](13_commands.md)
- [Analisis dukungan Ultralytics QAT](ultralytics_qat_support_analysis.md)

- [29 Urutan script untuk laporan](29_urutan_skrip_laporan.md) — entry point bernomor dari dataset sampai OpenVINO.

## Pemeliharaan repository

- [31 Audit dan cleanup repository](31_repository_cleanup.md) — berkas mana yang dipertahankan, duplikasi yang dibersihkan, dan cara verifikasi.
