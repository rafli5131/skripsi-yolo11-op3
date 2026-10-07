# Perbandingan empat model pada CPU OP3

Pengukuran baru menggunakan FP32 PyTorch tanpa OpenVINO, FP32 OpenVINO, PTQ INT8 OpenVINO, dan QAT INT8 OpenVINO. Semua IR diambil dari artifacts/openvino/{fp32,ptq_int8,qat_int8}/model.xml; FP32 native dari artifacts/checkpoints/fp32/best.pt. Label kuantisasi mengikuti artefak pengguna, tanpa pemeriksaan hash.

## Protokol

CPU, batch 1, model input 640×640, 30 warmup, synthetic 300 iteration dengan seed 42, kamera 60 detik per model. Confidence 0.25, IoU 0.45, sampling resource 0.2 detik, visualization OFF. Native PyTorch memakai rect=False agar kamera 640×480 tetap di-letterbox ke 640×640. Model loading/compile dikeluarkan dari waktu inference.

## Standalone synthetic

| mode | model | inference_mean_ms | inference_p95_ms | pipeline_mean_ms | effective_fps | system_cpu_mean_percent | process_cpu_mean_percent | system_memory_mean_mb | rss_mean_mb | rss_peak_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | FP32 PyTorch (tanpa OpenVINO) | 52.7003 | 54.3355 | 55.3120 | 18.0694 | 49.8659 | 800.2293 | 4285.1645 | 375.5225 | 391.3851 |
| synthetic | FP32 OpenVINO | 28.0504 | 29.6411 | 29.8745 | 33.4559 | 55.2955 | 893.2841 | 4174.0696 | 240.7455 | 240.8325 |
| synthetic | PTQ INT8 OpenVINO | 11.5404 | 11.8309 | 13.1660 | 75.8624 | 20.5895 | 343.8316 | 4155.2372 | 230.6080 | 230.6908 |
| synthetic | QAT INT8 OpenVINO | 11.2265 | 11.4352 | 12.9337 | 77.2287 | 20.3842 | 341.1632 | 4137.5352 | 225.3613 | 225.4029 |


## Standalone kamera

| mode | model | inference_mean_ms | inference_p95_ms | pipeline_mean_ms | effective_fps | system_cpu_mean_percent | process_cpu_mean_percent | system_memory_mean_mb | rss_mean_mb | rss_peak_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camera | FP32 PyTorch (tanpa OpenVINO) | 50.9233 | 52.9269 | 56.8538 | 17.5780 | 49.3805 | 783.5617 | 4310.9365 | 388.9649 | 407.6462 |
| camera | FP32 OpenVINO | 37.1724 | 59.8114 | 48.4690 | 20.6250 | 44.8789 | 706.8708 | 4152.4328 | 242.7902 | 243.3475 |
| camera | PTQ INT8 OpenVINO | 12.5824 | 15.3498 | 43.3076 | 23.0824 | 8.4631 | 127.3879 | 4154.7287 | 233.9171 | 234.5247 |
| camera | QAT INT8 OpenVINO | 12.0950 | 14.8793 | 41.1139 | 24.3139 | 8.5440 | 128.3852 | 4142.1401 | 228.3978 | 229.0237 |


## Tabel untuk skripsi: FP32 dan QAT

| mode | model | inference_mean_ms | inference_p95_ms | pipeline_mean_ms | effective_fps | system_cpu_mean_percent | process_cpu_mean_percent | system_memory_mean_mb | rss_mean_mb | rss_peak_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | FP32 PyTorch (tanpa OpenVINO) | 52.7003 | 54.3355 | 55.3120 | 18.0694 | 49.8659 | 800.2293 | 4285.1645 | 375.5225 | 391.3851 |
| synthetic | FP32 OpenVINO | 28.0504 | 29.6411 | 29.8745 | 33.4559 | 55.2955 | 893.2841 | 4174.0696 | 240.7455 | 240.8325 |
| synthetic | QAT INT8 OpenVINO | 11.2265 | 11.4352 | 12.9337 | 77.2287 | 20.3842 | 341.1632 | 4137.5352 | 225.3613 | 225.4029 |
| camera | FP32 PyTorch (tanpa OpenVINO) | 50.9233 | 52.9269 | 56.8538 | 17.5780 | 49.3805 | 783.5617 | 4310.9365 | 388.9649 | 407.6462 |
| camera | FP32 OpenVINO | 37.1724 | 59.8114 | 48.4690 | 20.6250 | 44.8789 | 706.8708 | 4152.4328 | 242.7902 | 243.3475 |
| camera | QAT INT8 OpenVINO | 12.0950 | 14.8793 | 41.1139 | 24.3139 | 8.5440 | 128.3852 | 4142.1401 | 228.3978 | 229.0237 |


PTQ tersedia sebagai pembanding tambahan dalam comparison_all.csv; tidak termasuk comparison_thesis_fp32_qat.csv.

## Accuracy dataset yang dibekukan

| model | split | precision | recall | F1 | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 native PyTorch checkpoint | TEST | 0.9313 | 0.9506 | 0.9409 | 0.9642 | 0.7799 |
| QAT native PyTorch checkpoint | TEST | 0.9341 | 0.9502 | 0.9421 | 0.9721 | 0.7672 |


Accuracy di atas milik checkpoint native PyTorch pada TEST, bukan hasil kamera atau pengukuran ulang accuracy IR. Tidak ada training/calibration/tuning/evaluasi ulang TEST.

## Batas perbandingan

Pengujian runtime dilakukan berurutan pada host yang sama. Kamera live tidak menjamin frame/scene identik antar model. OpenVINO memakai preprocessing/postprocess custom existing; native memakai Ultralytics. Timing inference native berasal dari profiler Ultralytics, OpenVINO memakai perf_counter_ns; pipeline wall time/FPS lebih mewakili biaya keseluruhan implementasi. Jumlah thread mengikuti default masing-masing runtime dan dicatat pada JSON jika tersedia. CPU sistem dan CPU proses terpisah; CPU proses dapat >100%. RAM MB = bytes/1000000.

Hasil ROS QAT/stability tersimpan terpisah dalam final_qat/ dan stability_qat/. Tabel standalone ini tidak disebut sebagai perbandingan ROS empat model.

## Reproduksi

bash scripts/run_op3_model_comparison.sh <direktori-output-baru>

python3 scripts/deployment/build_op3_model_comparison.py --root /home/humanoid/ros2_jazzy/skripsi-yolo11-op3/experiments/hardware_op3/comparison_models/20261007_four_models_run1

Raw timing/resource CSV, timestamp, commit, hostname, model path, versi runtime dan command aktual ada pada setiap JSON benchmark. Protocol dibekukan di protocol.json sebelum run.

Grafik dari hasil actual: [four_model_camera_comparison.png](evidence/four_model_camera_comparison.png).
