# Pengujian Hardware OP3 — FP32, PTQ dan QAT

> **Catatan status historis (7 Oktober 2026):** Isi tabel dan protokol di bawah mendokumentasikan pengujian pada tanggal tersebut; pernyataan “berjalan belum diuji” hanya berlaku saat laporan ini dibuat. Pengujian ROS **FP32 dan QAT** pada kondisi diam dan berjalan **dengan pendampingan operator**, masing-masing 3 × 60 detik per model/kondisi, sudah selesai pada **19 Oktober 2026**. Lihat [hasil final ROS](operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/FP32_QAT_ROS_FINAL_RESULTS.md), [hasil QAT berjalan](operator_runs/20261019T131319Z_qat_berjalan_final/analysis/WALKING_RESULTS.md), dan [draft hasil akhir skripsi](../../docs/draft_hasil_pengujian_akhir_skripsi_20261019.md). Gerak tersebut bukan pembuktian gait mandiri. Snapshot eksperimen: [`495a3a8`](https://github.com/rafli5131/skripsi-yolo11-op3/commit/495a3a893db3bfdfd24395cb25bf3a3cc300456c).

Pengujian tanggal 7 Oktober 2026 memuat perbandingan baru **FP32 PyTorch tanpa OpenVINO, FP32 OpenVINO, PTQ INT8 OpenVINO, dan QAT INT8 OpenVINO**. IR dibaca langsung dari `artifacts/openvino/`. Perbandingan standalone dipisahkan dari tiga run ROS QAT dan stability QAT. Tabel skripsi khusus FP32/QAT tersedia tanpa baris PTQ. Laporan campuran historical tetap ada di [arsip](archive/HARDWARE_TEST_REPORT_MIXED_20261007.md).

## 1. Hardware

ROBOTIS OP3 dengan onboard NUC12WSHi5, Intel Core i5-1240P, 12 core / 16 logical CPU. RAM total 7875014656 bytes. Ubuntu 24.04; kamera Logitech C920 pada device audited `/dev/video0`. Robot diam dikonfirmasi operator. Snapshot lengkap: [environment.json](final_qat/environment.json) dan [environment.md](final_qat/environment.md).

## 2. Software

Python 3.12.3, OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3, ROS 2 jazzy. Runtime Python project `.venv-op3-vision`. Package skripsi terpisah `ros2_ws/src/yolo11_op3_vision`; package robot existing tidak diubah.

## 3. Model provenance

| Model | Runtime | Path aktual |
| --- | --- | --- |
| FP32 tanpa OpenVINO | PyTorch + Ultralytics CPU | `artifacts/checkpoints/fp32/best.pt` |
| FP32 OpenVINO | OpenVINO CPU | `artifacts/openvino/fp32/model.xml` + `model.bin` |
| PTQ INT8 OpenVINO | OpenVINO CPU | `artifacts/openvino/ptq_int8/model.xml` + `model.bin` |
| QAT INT8 OpenVINO | OpenVINO CPU | `artifacts/openvino/qat_int8/model.xml` + `model.bin` |

Label QAT/PTQ mengikuti model yang ditetapkan pengguna. Tidak ada pemeriksaan hash; manifest/checkpoint penelitian tidak diubah. Mapping tetap `0: ball`, `1: gawang`, `2: robot`. Ketiga IR berhasil compile/inference CPU dengan output finite `[1,7,8400]`. Native FP32 tidak menggunakan OpenVINO. Tidak ada training, export ulang, calibration, atau tuning.

## 4. Experimental setup

Bagian ini menjelaskan protokol ROS QAT. Protokol perbandingan standalone empat model ada pada bagian 5–6 dan `comparison_models/20261007_four_models_run1/protocol.json`.

Protokol dibekukan sebelum pengukuran dalam [protocol.json](final_qat/protocol.json): 3 run × 60 detik, batch 1, device CPU, input model 640×640, capture 640×480 YUYV dengan permintaan 30 Hz. Threshold confidence 0.25 dan IoU 0.45. Setiap run memakai 30 inference dummy dan membuang 30 frame kamera sukses sebelum jendela pengukuran. Resource sampling 0.2 detik; debug image, render evidence, dan telemetry OFF pada run resmi.

Latency utama dihitung dari awal callback sampai postprocess selesai, sebelum publishing. FPS efektif = jumlah frame sukses / durasi aktual. Statistik ROS/tabel menggunakan SD sampel (`ddof=1`); helper benchmark engine existing menggunakan SD populasi. CPU utama adalah CPU seluruh sistem. MB = bytes / 1.000.000; RSS proses terpisah dari RAM sistem.

## 5. Standalone benchmark — empat model

Perbandingan baru memakai CPU, batch 1, input 640×640, seed synthetic 42, 30 warmup dan 300 iteration per model. Loading/compile tidak dihitung dalam inference. Resource diukur setiap 0.2 detik. Semua baris berikut adalah run baru dengan protokol yang sama, bukan hasil historical yang diganti labelnya.

| mode | model | inference_mean_ms | inference_p95_ms | pipeline_mean_ms | effective_fps | system_cpu_mean_percent | process_cpu_mean_percent | system_memory_mean_mb | rss_mean_mb | rss_peak_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | FP32 PyTorch (tanpa OpenVINO) | 52.7003 | 54.3355 | 55.3120 | 18.0694 | 49.8659 | 800.2293 | 4285.1645 | 375.5225 | 391.3851 |
| synthetic | FP32 OpenVINO | 28.0504 | 29.6411 | 29.8745 | 33.4559 | 55.2955 | 893.2841 | 4174.0696 | 240.7455 | 240.8325 |
| synthetic | PTQ INT8 OpenVINO | 11.5404 | 11.8309 | 13.1660 | 75.8624 | 20.5895 | 343.8316 | 4155.2372 | 230.6080 | 230.6908 |
| synthetic | QAT INT8 OpenVINO | 11.2265 | 11.4352 | 12.9337 | 77.2287 | 20.3842 | 341.1632 | 4137.5352 | 225.3613 | 225.4029 |

## 6. Camera benchmark — empat model

Kamera C920 `/dev/video0`, capture 640×480, input model 640×640, warmup 30 frame, 60 detik per model, confidence 0.25, IoU 0.45, visualization OFF. Native PyTorch memakai `rect=False`, sehingga frame 640×480 tetap di-letterbox ke 640×640.

| mode | model | inference_mean_ms | inference_p95_ms | pipeline_mean_ms | effective_fps | system_cpu_mean_percent | process_cpu_mean_percent | system_memory_mean_mb | rss_mean_mb | rss_peak_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camera | FP32 PyTorch (tanpa OpenVINO) | 50.9233 | 52.9269 | 56.8538 | 17.5780 | 49.3805 | 783.5617 | 4310.9365 | 388.9649 | 407.6462 |
| camera | FP32 OpenVINO | 37.1724 | 59.8114 | 48.4690 | 20.6250 | 44.8789 | 706.8708 | 4152.4328 | 242.7902 | 243.3475 |
| camera | PTQ INT8 OpenVINO | 12.5824 | 15.3498 | 43.3076 | 23.0824 | 8.4631 | 127.3879 | 4154.7287 | 233.9171 | 234.5247 |
| camera | QAT INT8 OpenVINO | 12.0950 | 14.8793 | 41.1139 | 24.3139 | 8.5440 | 128.3852 | 4142.1401 | 228.3978 | 229.0237 |

Keempat run dilakukan berurutan; scene/frame kamera tidak dijamin identik. Preprocessing/postprocess masing-masing runtime berbeda. Native stage timing berasal dari Ultralytics; OpenVINO memakai perf_counter_ns. Pipeline wall latency dan FPS efektif mencakup biaya implementasi/capture. MB memakai bytes/1.000.000. CPU sistem dan CPU proses dipisahkan; CPU proses dapat >100% akibat multi-core.

Laporan lengkap dan raw measurements: [MODEL_COMPARISON_REPORT.md](comparison_models/20261007_four_models_run1/MODEL_COMPARISON_REPORT.md). Bukti visual native tanpa OpenVINO: [FP32 PyTorch](comparison_models/20261007_four_models_run1/evidence/live_detection_fp32_pytorch.png); QAT: [deteksi C920](functional_qat/20261007_topic_check/functional_run_01_detection.png).

## 7. ROS 2 benchmark

Node skripsi subscribe `/thesis/image_raw` dan publish `vision_msgs/Detection2DArray` pada `/vision/detections`. Ketiga run diam memakai model QAT dari path yang sama.

| condition | run_id | frames | duration_seconds | latency_mean_ms | latency_sd_ms | fps |
| --- | --- | --- | --- | --- | --- | --- |
| diam | run_01 | 809 | 60.0034 | 18.0768 | 2.0378 | 13.4826 |
| diam | run_02 | 904 | 60.0047 | 17.4554 | 2.5103 | 15.0655 |
| diam | run_03 | 729 | 60.0038 | 18.5108 | 1.7577 | 12.1492 |

Gabungan: **2442 frame**, latency mean **17.9763 ms**, FPS efektif **13.5658**, nol exception. Raw per-frame dan metadata tersedia di `final_qat/latency/` dan `final_qat/run_metadata/`.

Run functional terpisah memverifikasi penerimaan 122 pesan deteksi, 122 telemetry, dan 3 debug image. Bukti: [topic_verification.json](functional_qat/20261007_topic_check/topic_verification.json). Visualisasi/telemetry ON hanya pada run functional ini.

## 8. CPU utilization — ROS QAT

885 sampel periodik pada tiga run diam. CPU sistem mean **6.4163%**, SD sampel **1.5655%**, peak **11.5000%**. CPU proses mean **88.6172%**; nilai proses dapat melebihi 100% karena multi-core.

Grafik actual: [resource run diam](final_qat/evidence/resources_diam_run_01.png). Raw CSV: `final_qat/resources/diam_run_*.csv`.

## 9. RAM utilization — ROS QAT

RAM sistem used mean **4083.7804 MB**, usage mean **56.5762%**. Persentase berasal langsung dari psutil.virtual_memory().percent. RSS detector mean **257.5252 MB**, peak **257.8186 MB**. RAM sistem mencakup proses OS/ROS lain dan tidak disamakan dengan RSS detector.

Rekap lengkap: [Tabel 3.3](final_qat/tables/table_3_3_resources_summary.md).

## 10. Stability — ROS QAT

QAT ROS + real C920 selesai **600.0162 detik**, **8260 frame**, FPS **13.7663**, latency mean **17.6222 ms**, exception **0**. RSS awal/akhir 257945600/258379776 bytes; growth 434176 bytes sebelum alokasi summary. Suhu puncak 59.0 C.

Raw data dan detail: [STABILITY_REPORT.md](stability_qat/20261007_ros_qat/STABILITY_REPORT.md). Perubahan RSS kecil pada run ini bukan bukti mutlak ada/tidaknya memory leak.

## 11. Limitations

- Pada sesi **7 Oktober 2026**, kondisi berjalan belum diuji sesuai instruksi operator; script disiapkan tanpa mengirim command motion. **Pembaruan:** hasil berjalan 3 × 60 detik per model sudah tersedia pada sesi 19 Oktober 2026 (tautan laporan final di bagian atas).
- Kamera diminta 30 Hz, tetapi throughput frame aktual lebih rendah. Auto exposure dan dynamic framerate teramati aktif; kemungkinan pengaruhnya belum diuji secara terisolasi. Setting tidak diubah selama pengukuran.
- Dropped frame kamera tidak diketahui dari callback yang diterima saja.
- Pengukuran memakai onboard NUC12 aktual dan versi software yang tercatat di metadata.
- Accuracy live-camera tidak dihitung tanpa ground truth. Accuracy native PyTorch berikut tidak otomatis merupakan accuracy IR runtime.

## 12. Final results

Untuk penulisan skripsi gunakan [comparison_thesis_fp32_qat.csv](comparison_models/20261007_four_models_run1/comparison_thesis_fp32_qat.csv), yang berisi FP32 tanpa OpenVINO, FP32 OpenVINO, dan QAT OpenVINO. PTQ tetap tersedia sebagai pembanding tambahan dalam [comparison_all.csv](comparison_models/20261007_four_models_run1/comparison_all.csv).

### Accuracy checkpoint native PyTorch — TEST yang dibekukan

| model | split | precision | recall | F1 | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 native PyTorch checkpoint | TEST | 0.9313 | 0.9506 | 0.9409 | 0.9642 | 0.7799 |
| QAT native PyTorch checkpoint | TEST | 0.9341 | 0.9502 | 0.9421 | 0.9721 | 0.7672 |

Accuracy berasal dari `experiments/final_test/{fp32,qat}_test_metrics.json`. Accuracy itu milik checkpoint native PyTorch, bukan hasil kamera dan bukan evaluasi ulang accuracy IR. Tidak ada tuning/evaluasi ulang TEST.

### ROS QAT diam

| Model runtime | Kondisi | Latency deteksi mean (ms) | FPS efektif | CPU sistem mean (%) | RAM sistem mean (MB) | RSS detector mean (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| QAT INT8 OpenVINO | Diam, 3 × 60 detik | 17.9763 | 13.5658 | 6.4163 | 4083.7804 | 257.5252 |

## 13. Reproduction commands

Jalankan dari root repository. Run diam yang sudah ada tidak boleh ditimpa; runner menolak overwrite.

```bash
bash scripts/run_op3_qat_measurements.sh --condition diam --operator-confirmed --runs 3 --seconds 60 --camera-device /dev/video0
# Contoh perintah historis; pengujian berjalan telah selesai pada 19 Oktober 2026. Reproduksi hanya dengan persetujuan operator:
bash scripts/run_op3_qat_measurements.sh --condition berjalan --operator-confirmed --runs 3 --seconds 60 --camera-device /dev/video0
python3 scripts/deployment/build_op3_thesis_tables.py --root experiments/hardware_op3/final_qat
```

Path model default runner adalah `artifacts/openvino/qat_int8/model.xml`. Snapshot command, commit pengukuran, hostname, CPU/RAM, versi runtime dan kondisi worktree tersimpan pada `final_qat/environment*.json` dan per-run metadata. Laporan/tabel yang bisa dibangun ulang: [FINAL_OP3_QAT_REPORT.md](final_qat/FINAL_OP3_QAT_REPORT.md).

### Reproduksi perbandingan empat model

```bash
# Membuat direktori baru; data lama tidak ditimpa.
bash scripts/run_op3_model_comparison.sh
# Bangun ulang tabel dari delapan JSON/raw CSV yang sudah diukur:
python3 scripts/deployment/build_op3_model_comparison.py --root experiments/hardware_op3/comparison_models/20261007_four_models_run1
```

Protokol perbandingan: `comparison_models/20261007_four_models_run1/protocol.json`. Command, model path, commit pengukuran, versi runtime, CPU/RAM dan raw CSV tercatat pada JSON setiap model/mode. Perbandingan standalone ini tidak disebut sebagai perbandingan ROS empat model; pengujian ROS terbaru dan stability hanya QAT.

Grafik perbandingan empat model dari run kamera actual: [four_model_camera_comparison.png](comparison_models/20261007_four_models_run1/evidence/four_model_camera_comparison.png).
