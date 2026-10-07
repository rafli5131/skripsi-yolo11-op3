## Stability dan functional evidence

Stability real camera 600 detik: 8260 frame, 13.7663 FPS, latency mean 17.6222 ms, zero exception. RSS growth 434176 bytes sebelum alokasi summary; thermal peak 59 C. Lihat ../stability_qat/20261007_ros_qat/STABILITY_REPORT.md.

Functional ROS terpisah memverifikasi reception 122 detections, 122 telemetry, dan 3 debug image; lihat ../functional_qat/20261007_topic_check/topic_verification.json dan functional_run_01_detection.png. Debug/evidence OFF pada tiga run resmi.

Accuracy PyTorch QAT yang telah dibekukan: Precision 0.934116, Recall 0.950208, F1 0.942093, mAP50 0.972145, mAP50-95 0.767179, sumber experiments/final_test/qat_test_metrics.json dan manifest.json (status passed, 592 TEST images). Angka itu milik checkpoint QAT native PyTorch; tidak dinyatakan sebagai accuracy live-camera atau accuracy IR yang sedang digunakan. TEST tidak dievaluasi ulang.
