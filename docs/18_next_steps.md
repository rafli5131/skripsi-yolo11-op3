# Langkah Berikutnya

FP32, PTQ INT8 comparator, dan QAT INT8 direct OpenVINO sudah tersedia dan lolos pemeriksaan provenance. FINAL TEST PyTorch telah dilakukan setelah pemilihan QAT best. Benchmark latency sintetis server tiga IR dan evaluasi kualitas deteksi VAL empat model juga selesai; lihat [ringkasan hasil di direktori eksperimen](../experiments/server_openvino/README.md), [laporan latency](23_server_openvino_benchmark.md), dan [laporan precision/F1/mAP](28_server_val_detection_comparison.md). Urutan tahapan ada di [timeline](20_experiment_timeline.md). Tidak ada kebutuhan untuk mengulang training, QAT tuning, FINAL TEST, PTQ, atau calibration hanya untuk melanjutkan tahap deployment.

## Pengukuran pada ROBOTIS OP3

1. Materialize artifact FP32, PTQ INT8, dan QAT INT8 di OP3; verifikasi hash serta gate yang relevan.
2. Jalankan benchmark synthetic OpenVINO CPU dengan konfigurasi sama untuk ketiga model.
3. Jalankan benchmark kamera nyata dengan kondisi yang dicatat jelas.
4. Ukur latency mean dan p95/p99, FPS, process CPU, RSS/RAM, serta stabilitas run panjang.
5. Buat perbandingan akhir OP3 yang terpisah dari [hasil server](../experiments/server_openvino/README.md) dan jelas mencantumkan model, perangkat, serta konfigurasi.
6. Susun tabel dan kesimpulan tesis dari artifact valid, dengan split dan perangkat disebut eksplisit.

Hasil di `experiments/hardware_op3/` yang sudah ada tetap terpisah. Tahap berikutnya adalah perbandingan final ketiga model pada OP3; keberadaan data hardware lama tidak berarti perbandingan final FP32/PTQ/QAT sudah selesai. Jalur ONNX lama sudah ditolak dan tidak perlu diulang untuk memakai QAT IR final direct OpenVINO yang accepted. Jika ada kebutuhan penelitian baru untuk menelusuri akar mismatch ONNX, baca [analisis kegagalan](25_kegagalan_dan_pembersihan.md).
