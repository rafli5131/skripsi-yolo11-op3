# Next Steps

## Tujuan

Mencatat pekerjaan yang belum dilakukan dan dependensinya.

## Status

**NOT YET EXECUTED** untuk seluruh item berikut.

1. Jalankan ulang Phase 7F dengan ONNX Runtime untuk mengisolasi mismatch PyTorch QAT -> ONNX atau ONNX -> OpenVINO.
2. Terima/promo hanya jika equivalence representatif dan graph quantization lulus.
3. Setelah kandidat IR valid, benchmark OpenVINO FP32 versus PTQ INT8 pada deployment CPU: latency, FPS, CPU, RAM.
4. Integrasikan ROS2, lalu deploy/ukur pada ROBOTIS OP3.
5. Jalankan eksperimen jarak/pencahayaan jika disyaratkan rencana penelitian.
6. Susun tabel/figur dan kesimpulan tesis dari artefak valid.

Dependency utama: PTQ hanya comparator terisolasi, bukan fallback untuk mengubah hasil QAT. Jangan memakai QAT IR rejected, melakukan retrain, atau retune menggunakan TEST.
