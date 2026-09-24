# Tahapan dan Timeline Eksperimen

Dokumen ini menunjukkan urutan bukti yang menghasilkan artifact final saat ini. Istilah `selesai`, `ditolak`, dan `berikutnya` merujuk pada status artifact, bukan instruksi untuk menjalankan eksperimen ulang. Detail provenance dan hash ada di [panduan verifikasi](26_provenance_dan_verifikasi.md).

| Tahap | Pekerjaan dan sumber | Hasil yang berlaku | Status |
| --- | --- | --- | --- |
| 1. Setup dan dataset | Environment `uv`, split TRAIN/VAL/TEST, audit label | Dataset siap; TEST terpisah dari training dan seleksi | Selesai |
| 2. Baseline FP32 | Training dari YOLO11n; `artifacts/checkpoints/fp32/` | Checkpoint FP32 final sebagai sumber IR FP32 dan PTQ | Selesai |
| 3. Verifikasi QAT | Probe API, smoke test, lalu run QAT 5 epoch awal | True NNCF QAT dan reload terverifikasi; run 5 epoch adalah riwayat sebelum pemilihan best | Selesai, historis |
| 4. Export QAT strip | Konversi awal QAT ke OpenVINO | Output tidak ekuivalen secara numerik; kandidat lama ditolak | Ditolak |
| 5. Cabang PTQ comparator | PTQ dari FP32 dengan calibration TRAIN historis; evaluasi VAL | Comparator PTQ canonical valid; tidak memakai QAT/TEST | Selesai |
| 6. Diagnostik ONNX | QAT → ONNX → OpenVINO pada checkpoint lama | OpenVINO gagal 31/32 input VAL; kandidat tidak dipromosikan | Ditolak |
| 7. Seleksi QAT best | Pencarian dengan TRAIN dan pemilihan hanya dari VAL | `qat_lr1e4_r256_e15` dipromosikan ke checkpoint QAT final | Selesai |
| 8. FINAL TEST | Evaluasi held-out FP32 dan QAT terpilih pada PyTorch | Hasil TEST dibekukan; bukan input tuning atau export | Selesai |
| 9. Export model final | FP32 IR, PTQ comparator existing, QAT direct OpenVINO dari checkpoint terpilih | Gate PTQ dan QAT accepted; QAT 32/32 VAL lulus numerical equivalence | Selesai |
| 10. Benchmark server | Tiga IR pada OpenVINO CPU, 640×640, 30 warmup, 300 iterasi | Raw result dan [ringkasan di direktori server](../experiments/server_openvino/README.md) | Selesai, diagnostik |
| 10b. Evaluasi deteksi server | FP32 native, FP32 OpenVINO, PTQ, QAT pada VAL yang sama | Precision, recall, F1, mAP50, mAP50-95 dan per kelas; [laporan VAL](../experiments/server_openvino/VAL_METRICS_REPORT.md) | Selesai, diagnostik VAL |
| 11. Perbandingan final OP3 | Tiga IR di ROBOTIS OP3: synthetic, kamera, resource, stabilitas | Hasil final tiga model pada robot belum tersedia | Berikutnya |

Tahap 4 dan 6 menjelaskan kegagalan historis; file model kandidatnya telah dibersihkan, sedangkan JSON diagnostiknya tetap ada. Kegagalan tersebut **tidak** mengubah status IR QAT direct OpenVINO final di tahap 9. [Analisis kegagalan](25_kegagalan_dan_pembersihan.md) menjelaskan sebab yang diketahui dan yang belum pasti.

Manifest FINAL TEST saat ini bertanggal 2026-09-24 03:35 UTC dan mencatat hash QAT final `b9e520...35430d`. Export direct OpenVINO tercatat 05:04 UTC, diikuti benchmark server mulai 05:38 UTC. Ini menempatkan FINAL TEST final **setelah** seleksi QAT best dan **sebelum** export/benchmark server. PTQ adalah cabang comparator terpisah; evaluasi VAL historisnya tidak disamakan dengan TEST.

`experiments/hardware_op3/` menyimpan hasil hardware yang sudah ada. Tahap 11 merujuk khusus pada perbandingan final **FP32 vs PTQ vs QAT** langsung pada OP3 dengan konfigurasi sepadan. Benchmark server tahap 10 tetap berada di `experiments/server_openvino/` dan tidak menggantikan tahap 11. Lihat [langkah berikutnya](18_next_steps.md).

Tahap 10b memakai split VAL berlabel dan evaluator deteksi. Tahap 10 memakai input sintetis untuk latency. Keduanya tidak mengulang FINAL TEST tahap 8 dan tidak boleh digabung sebagai satu run.
