# Final OP3 QAT runtime report

Model label: QAT INT8 OpenVINO, berdasarkan folder model yang diberikan pengguna. Identitas artefak tidak diverifikasi dengan hash.

Hardware dan versi software: lihat environment_*.json. CPU onboard aktual NUC12; batch 1, input 640 × 640, CPU.

Protocol dibekukan sebelum run di protocol.json. Tidak ada training, calibration, atau tuning.

## Latency dan FPS

| condition | run_id | frames | duration_seconds | latency_mean_ms | latency_sd_ms | fps |
| --- | --- | --- | --- | --- | --- | --- |
| diam | run_01 | 809 | 60.0034 | 18.0768 | 2.0378 | 13.4826 |
| diam | run_02 | 904 | 60.0047 | 17.4554 | 2.5103 | 15.0655 |
| diam | run_03 | 729 | 60.0038 | 18.5108 | 1.7577 | 12.1492 |


## CPU dan RAM

| condition | samples | runs | cpu_usage_percent_mean | cpu_usage_percent_sd | cpu_usage_percent_peak | memory_used_mb_mean | memory_used_mb_sd | memory_used_mb_peak | memory_usage_percent_mean | memory_usage_percent_sd | memory_usage_percent_peak | detector_rss_mb_mean | detector_rss_mb_sd | detector_rss_mb_peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diam | 885 | 3 | 6.4163 | 1.5655 | 11.5000 | 4083.7804 | 8.3941 | 4153.6840 | 56.5762 | 0.1082 | 57.5000 | 257.5252 | 0.3085 | 257.8186 |
| berjalan | 0 | 0 | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia | belum diukur / tidak tersedia |


CPU utama adalah CPU sistem. CPU proses dapat >100% karena multi-core. RAM utama adalah sistem dalam MB desimal; RSS proses disimpan terpisah.

SD memakai ddof=1. Sampel resource dipool per kondisi; frame warmup dikecualikan. Kamera/ROS publishing aktif, debug image dan benchmark telemetry OFF.

## Keterbatasan

Kondisi berjalan belum diukur; script disiapkan untuk operator. Script tidak mengirim perintah walking. Dropped frame kamera tidak dapat ditentukan hanya dari callback yang diterima.

Precision, Recall, F1 dan mAP tidak dihitung dari live camera. Laporan ini hanya mengukur runtime dan resource.

Run dengan exception/thermal abort tidak dianggap berhasil oleh runner. Tabel hanya mengambil run completed; CSV raw dan metadata disimpan.

## Reproduksi

Lihat command aktual dalam environment_*.json dan tiap run_metadata/*.json. Jalankan scripts/deployment/build_op3_thesis_tables.py --root /home/humanoid/ros2_jazzy/skripsi-yolo11-op3/experiments/hardware_op3/final_qat

## Stability dan functional evidence

Stability real camera 600 detik: 8260 frame, 13.7663 FPS, latency mean 17.6222 ms, zero exception. RSS growth 434176 bytes sebelum alokasi summary; thermal peak 59 C. Lihat ../stability_qat/20261007_ros_qat/STABILITY_REPORT.md.

Functional ROS terpisah memverifikasi reception 122 detections, 122 telemetry, dan 3 debug image; lihat ../functional_qat/20261007_topic_check/topic_verification.json dan functional_run_01_detection.png. Debug/evidence OFF pada tiga run resmi.

Accuracy PyTorch QAT yang telah dibekukan: Precision 0.934116, Recall 0.950208, F1 0.942093, mAP50 0.972145, mAP50-95 0.767179, sumber experiments/final_test/qat_test_metrics.json dan manifest.json (status passed, 592 TEST images). Angka itu milik checkpoint QAT native PyTorch; tidak dinyatakan sebagai accuracy live-camera atau accuracy IR yang sedang digunakan. TEST tidak dievaluasi ulang.
