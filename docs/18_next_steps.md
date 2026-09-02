# Next Steps

## Tujuan

Mencatat pekerjaan yang belum dilakukan dan dependensinya.

## Status

**NOT YET EXECUTED** untuk seluruh item berikut.

1. Jalankan Phase 7D direct QAT export pada host CUDA; Candidate B hanya bila A gagal.
2. Terima/promo hanya jika equivalence representatif dan graph quantization lulus.
3. Setelah IR QAT valid, benchmark OpenVINO FP32 versus QAT INT8 pada deployment CPU: latency, FPS, CPU, RAM.
4. Integrasikan ROS2, lalu deploy/ukur pada ROBOTIS OP3.
5. Jalankan eksperimen jarak/pencahayaan jika disyaratkan rencana penelitian.
6. Susun tabel/figur dan kesimpulan tesis dari artefak valid.

Dependency utama: langkah 3--6 tidak boleh memakai QAT IR rejected. Jangan PTQ, retrain, retune TEST, atau calibration export sebagai fallback.
