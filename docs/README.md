# Dokumentasi Teknis Proyek

## Tujuan

Indeks dokumentasi teknis proyek `advance_vision`.

## Status saat ini

| Tahap | Status |
|---|---|
| FP32 training | **DONE** |
| FP32 held-out TEST | **DONE** |
| True NNCF QAT | **DONE** |
| QAT held-out TEST | **DONE** |
| OpenVINO FP32 export | **VALID** |
| OpenVINO QAT INT8 export | **UNRESOLVED**; artefak strip sebelumnya **REJECTED** |
| CPU deployment benchmark | **NOT YET EXECUTED** |
| ROS2 / ROBOTIS OP3 deployment | **NOT YET EXECUTED** |

## Indeks

- [00 Gambaran proyek](00_project_overview.md) — tujuan, pipeline, kelas, dan batas PTQ/QAT.
- [01 Lingkungan](01_environment.md) — server, versi, GPU mapping, dan safety.
- [02 Dataset](02_dataset.md) — struktur, audit, anomali, isolasi split.
- [03 FP32 training](03_fp32_training.md) — baseline dan konfigurasi.
- [04 QAT design](04_qat_design.md) — konsep dan API true QAT.
- [05 QAT implementation](05_qat_implementation.md) — DetectionTrainer dan checkpoint NNCF.
- [06 Smoke QAT](06_qat_smoke_and_validation.md) — bukti integrasi diagnostik.
- [07 Final QAT](07_final_qat_training.md) — konfigurasi dan hasil QAT final.
- [08 Final TEST](08_final_test_evaluation.md) — held-out comparison yang dibekukan.
- [09 OpenVINO export](09_openvino_export.md) — jalur FP32 valid dan QAT saat ini.
- [10 Export failure](10_openvino_export_failure_analysis.md) — strip QAT yang ditolak.
- [11 CPU/CUDA diagnostic](11_cpu_cuda_qat_diagnostic.md) — evidence representatif.
- [12 Reproducibility](12_reproducibility.md) — seed, lockfile, hash, batas.
- [13 Commands](13_commands.md) — cookbook host.
- [14 Artifact inventory](14_artifact_inventory.md) — status file penting.
- [15 Known issues](15_known_issues.md) — masalah dan workaround.
- [16 Decision log](16_decision_log.md) — keputusan penelitian.
- [17 Results summary](17_research_results_summary.md) — tabel hasil.
- [18 Next steps](18_next_steps.md) — pekerjaan tersisa.
- [19 Thesis mapping](19_thesis_method_mapping.md) — hubungan artefak dan bab skripsi.
- [20 Timeline](20_experiment_timeline.md) — kronologi fase.
- [21 QAT ONNX → OpenVINO](21_qat_onnx_openvino_export.md) — jalur Phase 7F yang terisolasi dan fail-closed.
