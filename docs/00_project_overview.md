# Gambaran Proyek

## Tujuan

Mendokumentasikan penelitian **Optimasi Deteksi Objek Real-Time pada Robot Humanoid menggunakan Teknik Quantization-Aware Training dan Akselerasi OpenVINO**.

## Status

**IN PROGRESS** — training dan evaluasi held-out selesai; export OpenVINO QAT valid belum tersedia.

## Ruang lingkup

Pipeline penelitian adalah `Dataset -> YOLO11n FP32 -> NNCF QAT -> OpenVINO -> ROS2 -> ROBOTIS OP3`. Baseline adalah YOLO11n FP32; optimisasi adalah true QAT NNCF; target deployment adalah CPU/OpenVINO pada ROBOTIS OP3. Kelas: `0 ball`, `1 gawang`, `2 robot`.

QAT memakai `nncf.torch.create_compressed_model()`, bukan `nncf.quantize()`. API terakhir adalah PTQ dan tidak digunakan dalam penelitian ini.

## Artefak terkait

[konfigurasi](../configs/base.yaml), [FP32](03_fp32_training.md), [QAT](04_qat_design.md), dan [status hasil](17_research_results_summary.md).
