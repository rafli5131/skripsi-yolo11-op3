# Pengujian Hardware OP3 — QAT INT8 OpenVINO

Laporan pengujian QAT tanggal 7 Oktober 2026. Semua hasil runtime di laporan ini berasal dari **`artifacts/openvino/qat_int8/model.xml` dan `model.bin`**. Path tercatat pada protocol dan metadata setiap run. Laporan campuran terdahulu disimpan di [arsip](archive/HARDWARE_TEST_REPORT_MIXED_20261007.md); angka historical tidak diganti labelnya.

## 1. Hardware

ROBOTIS OP3 dengan onboard NUC12WSHi5, Intel Core i5-1240P, 12 core / 16 logical CPU. RAM total 7875014656 bytes. Ubuntu 24.04; kamera Logitech C920 pada device audited `/dev/video0`. Robot diam dikonfirmasi operator. Snapshot lengkap: [environment.json](final_qat/environment.json) dan [environment.md](final_qat/environment.md).

## 2. Software

Python 3.12.3, OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3, ROS 2 jazzy. Runtime Python project `.venv-op3-vision`. Package skripsi terpisah `ros2_ws/src/yolo11_op3_vision`; package robot existing tidak diubah.

## 3. Model provenance

Model penelitian mengikuti penetapan pengguna sebagai QAT INT8 OpenVINO. Runtime membaca XML/BIN langsung dari `artifacts/openvino/qat_int8`, tanpa pemeriksaan hash. Manifest/checkpoint penelitian tetap dipertahankan; identitas payload tidak diperiksa secara independen.

- Model runtime: `artifacts/openvino/qat_int8/model.xml` + `model.bin`.
- Checkpoint QAT penelitian: `artifacts/checkpoints/qat/qat_best.pt`.
- Mapping: `0: ball`, `1: gawang`, `2: robot`.
- CPU compile/inference berhasil; input `[1,3,640,640]`, output finite `[1,7,8400]`.
- Tidak ada training, export ulang, calibration, atau tuning di robot.

## 4. Experimental setup

Protokol dibekukan sebelum pengukuran dalam [protocol.json](final_qat/protocol.json): 3 run × 60 detik, batch 1, device CPU, input model 640×640, capture 640×480 YUYV dengan permintaan 30 Hz. Threshold confidence 0.25 dan IoU 0.45. Setiap run memakai 30 inference dummy dan membuang 30 frame kamera sukses sebelum jendela pengukuran. Resource sampling 0.2 detik; debug image, render evidence, dan telemetry OFF pada run resmi.

Latency utama dihitung dari awal callback sampai postprocess selesai, sebelum publishing. FPS efektif = jumlah frame sukses / durasi aktual. Statistik ROS/tabel menggunakan SD sampel (`ddof=1`); helper benchmark engine existing menggunakan SD populasi. CPU utama adalah CPU seluruh sistem. MB = bytes / 1.000.000; RSS proses terpisah dari RAM sistem.

## 5. Standalone benchmark

Benchmark engine QAT memakai 30 warmup dan 300 iteration. Loading/compile dikeluarkan dari waktu inference; kamera dan ROS tidak termasuk.

| Model | Inference mean (ms) | Inference P95 (ms) | FPS dari mean inference | FPS efektif loop |
| --- | ---: | ---: | ---: | ---: |
| QAT INT8 OpenVINO | 11.2795 | 11.6004 | 88.6562 | 76.7968 |

Data: [JSON](benchmark_qat_int8_engine_20261007.json), [raw timing CSV](benchmark_qat_int8_engine_20261007.csv), [raw resources CSV](resources_benchmark_qat_int8_engine_20261007.csv).

## 6. Camera benchmark

Pengukuran resmi terbaru memakai kamera C920 melalui ROS, dengan hasil di bagian 7. Run standalone GUI terdahulu `camera_benchmark_qat_int8.json` adalah functional check dengan visualisasi ON, sehingga tidak dicampur ke tabel resmi.

Screenshot actual QAT: [deteksi C920](functional_qat/20261007_topic_check/functional_run_01_detection.png). Screenshot menampilkan kelas/confidence dan bukan evaluasi accuracy.

## 7. ROS 2 benchmark

Node skripsi subscribe `/thesis/image_raw` dan publish `vision_msgs/Detection2DArray` pada `/vision/detections`. Ketiga run diam memakai model QAT dari path yang sama.

| condition | run_id | frames | duration_seconds | latency_mean_ms | latency_sd_ms | fps |
| --- | --- | --- | --- | --- | --- | --- |
| diam | run_01 | 809 | 60.0034 | 18.0768 | 2.0378 | 13.4826 |
| diam | run_02 | 904 | 60.0047 | 17.4554 | 2.5103 | 15.0655 |
| diam | run_03 | 729 | 60.0038 | 18.5108 | 1.7577 | 12.1492 |

Gabungan: **2442 frame**, latency mean **17.9763 ms**, FPS efektif **13.5658**, nol exception. Raw per-frame dan metadata tersedia di `final_qat/latency/` dan `final_qat/run_metadata/`.

Run functional terpisah memverifikasi penerimaan 122 pesan deteksi, 122 telemetry, dan 3 debug image. Bukti: [topic_verification.json](functional_qat/20261007_topic_check/topic_verification.json). Visualisasi/telemetry ON hanya pada run functional ini.

## 8. CPU utilization

885 sampel periodik pada tiga run diam. CPU sistem mean **6.4163%**, SD sampel **1.5655%**, peak **11.5000%**. CPU proses mean **88.6172%**; nilai proses dapat melebihi 100% karena multi-core.

Grafik actual: [resource run diam](final_qat/evidence/resources_diam_run_01.png). Raw CSV: `final_qat/resources/diam_run_*.csv`.

## 9. RAM utilization

RAM sistem used mean **4083.7804 MB**, usage mean **56.5762%**. Persentase berasal langsung dari psutil.virtual_memory().percent. RSS detector mean **257.5252 MB**, peak **257.8186 MB**. RAM sistem mencakup proses OS/ROS lain dan tidak disamakan dengan RSS detector.

Rekap lengkap: [Tabel 3.3](final_qat/tables/table_3_3_resources_summary.md).

## 10. Stability

QAT ROS + real C920 selesai **600.0162 detik**, **8260 frame**, FPS **13.7663**, latency mean **17.6222 ms**, exception **0**. RSS awal/akhir 257945600/258379776 bytes; growth 434176 bytes sebelum alokasi summary. Suhu puncak 59.0 C.

Raw data dan detail: [STABILITY_REPORT.md](stability_qat/20261007_ros_qat/STABILITY_REPORT.md). Perubahan RSS kecil pada run ini bukan bukti mutlak ada/tidaknya memory leak.

## 11. Limitations

- Berjalan belum diuji sesuai instruksi operator; script siap dan tidak mengirim command motion.
- Kamera diminta 30 Hz, tetapi throughput frame aktual lebih rendah. Auto exposure dan dynamic framerate teramati aktif; kemungkinan pengaruhnya belum diuji secara terisolasi. Setting tidak diubah selama pengukuran.
- Dropped frame kamera tidak diketahui dari callback yang diterima saja.
- Pengukuran memakai onboard NUC12 aktual dan versi software yang tercatat di metadata.
- Accuracy live-camera tidak dihitung tanpa ground truth. Accuracy native PyTorch berikut tidak otomatis merupakan accuracy IR runtime.

## 12. Final results

| Model runtime | Kondisi | Latency deteksi mean (ms) | FPS efektif | CPU sistem mean (%) | RAM sistem mean (MB) | RSS detector mean (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| QAT INT8 OpenVINO | Diam, 3 × 60 detik | 17.9763 | 13.5658 | 6.4163 | 4083.7804 | 257.5252 |

Accuracy checkpoint QAT native PyTorch dari evaluasi TEST yang telah dibekukan, 592 images; tidak dievaluasi ulang:

| Model evaluasi dataset | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QAT native PyTorch checkpoint | 0.934116 | 0.950208 | 0.942093 | 0.972145 | 0.767179 |

Sumber accuracy: `experiments/final_test/qat_test_metrics.json` dan manifest evaluasi status passed. Tabel accuracy dan runtime dipisahkan karena objek evaluasinya berbeda.

## 13. Reproduction commands

Jalankan dari root repository. Run diam yang sudah ada tidak boleh ditimpa; runner menolak overwrite.

```bash
bash scripts/run_op3_qat_measurements.sh --condition diam --operator-confirmed --runs 3 --seconds 60 --camera-device /dev/video0
# Berjalan nanti setelah operator menjalankan walking dan memastikan kondisi aktual:
bash scripts/run_op3_qat_measurements.sh --condition berjalan --operator-confirmed --runs 3 --seconds 60 --camera-device /dev/video0
python3 scripts/deployment/build_op3_thesis_tables.py --root experiments/hardware_op3/final_qat
```

Path model default runner adalah `artifacts/openvino/qat_int8/model.xml`. Snapshot command, commit pengukuran, hostname, CPU/RAM, versi runtime dan kondisi worktree tersimpan pada `final_qat/environment*.json` dan per-run metadata. Laporan/tabel yang bisa dibangun ulang: [FINAL_OP3_QAT_REPORT.md](final_qat/FINAL_OP3_QAT_REPORT.md).
