# Langkah Berikutnya

FP32, PTQ INT8 comparator, dan QAT INT8 direct OpenVINO sudah tersedia dan lolos pemeriksaan provenance. FINAL TEST PyTorch telah dilakukan sebelumnya. Benchmark diagnostik server ketiga IR juga selesai; lihat [laporan server](23_server_openvino_benchmark.md). Tidak ada kebutuhan untuk mengulang training, QAT tuning, FINAL TEST, PTQ, atau calibration hanya untuk melanjutkan tahap deployment.

## Pengukuran pada ROBOTIS OP3

1. Materialize artifact FP32, PTQ INT8, dan QAT INT8 di OP3; verifikasi hash serta gate yang relevan.
2. Jalankan benchmark synthetic OpenVINO CPU dengan konfigurasi sama untuk ketiga model.
3. Jalankan benchmark kamera nyata dengan kondisi yang dicatat jelas.
4. Ukur latency mean dan p95/p99, FPS, process CPU, RSS/RAM, serta stabilitas run panjang.
5. Buat perbandingan akhir OP3 yang terpisah dari [hasil server](27_cara_membaca_hasil.md).
6. Susun tabel dan kesimpulan tesis dari artifact valid, dengan split dan perangkat disebut eksplisit.

Jalur ONNX lama sudah ditolak dan tidak perlu diulang untuk memakai QAT IR final direct OpenVINO yang accepted. Jika ada kebutuhan penelitian baru untuk menelusuri akar mismatch ONNX, baca [analisis kegagalan](25_kegagalan_dan_pembersihan.md); jangan campurkan percobaan itu dengan benchmark deployment.
