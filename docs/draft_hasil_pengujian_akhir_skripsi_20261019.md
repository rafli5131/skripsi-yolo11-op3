# Draft hasil pengujian akhir skripsi

Pembaruan: 2026-10-19T10:56:33.171701+00:00. Branch sumber akhir: `main`.

Pembaruan ini menghitung ulang raw CSV dan JSON yang telah tersimpan. Training, seleksi checkpoint, export, dan FINAL TEST tidak diulang. Pengujian berjalan baru belum dijalankan karena kesiapan keselamatan/operator belum terverifikasi. Tahap eksperimen berjalan belum dinyatakan selesai.

## Ringkasan status

- Perubahan QoS commit `53d971f` telah di-fast-forward ke main dan push normal ke origin/main.
- FINAL TEST native PyTorch FP32/QAT, benchmark standalone OpenVINO, ROS QAT diam 3 × 60 detik, dan stability QAT 600 detik tersedia.
- Pengukuran tambahan ROS FP32/QAT 3 × 60 detik per model pada 19 Oktober tersedia dan dilaporkan terpisah dari referensi QAT diam 7 Oktober.
- Pengujian berjalan baru 3 × 60 detik: **BELUM TERUKUR**. Tidak ada angka berjalan final yang diisi dari perkiraan atau dari run yang belum memiliki bukti kondisi fisik.
- Uji QoS 15 detik hanya bukti perbaikan implementasi; bukan hasil utama penelitian. Pengujian berjalan utama tetap memakai BEST_EFFORT depth 5 sesuai instruksi dan protokol referensi legacy.
- Tidak ada evaluasi akurasi berlabel OpenVINO. Confidence dan detection_count kamera live bukan precision, recall, F1, atau mAP.

Perangkat yang tercatat pada pengukuran runtime: ROBOTIS OP3, Intel NUC12WSHi5/Core i5-1240P, Ubuntu 24.04, ROS 2 Jazzy, Python 3.12.3, OpenVINO 2026.3.1, C920 640 × 480; input model 640 × 640, batch 1, CPU.

## 1. FINAL TEST — checkpoint native PyTorch

Angka berikut berasal dari JSON TEST beku: 592 gambar dan 734 anotasi. Evaluator menggunakan confidence 0,001 dan IoU 0,70, yang berbeda dari threshold deployment. Waktu evaluator GPU tidak dipakai sebagai latency deployment OP3.

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| FP32 PyTorch | 0,931347 | 0,950572 | 0,940861 | 0,964232 | 0,779908 |
| NNCF QAT PyTorch | 0,934116 | 0,950208 | 0,942093 | 0,972145 | 0,767179 |

QAT mencatat precision, F1, dan mAP50 sedikit lebih tinggi, sementara recall dan mAP50-95 sedikit lebih rendah. Hasil berlaku untuk checkpoint native PyTorch dan TEST beku, bukan sebagai evaluasi akurasi OpenVINO.

Sumber: [metrik FP32](../experiments/final_test/fp32_test_metrics.json), [metrik QAT](../experiments/final_test/qat_test_metrics.json).

## 2. Ukuran artefak OpenVINO

Ukuran model adalah jumlah XML + BIN, dalam MB desimal; ukuran checkpoint pelatihan tidak disamakan dengan ukuran artefak deployment.

| Model | XML (byte) | BIN (byte) | Total (MB) |
|---|---:|---:|---:|
| FP32 | 324944 | 10431992 | 10,7569 |
| QAT INT8 | 666400 | 2741080 | 3,4075 |

Pasangan XML/BIN QAT berukuran 68,32% lebih kecil. Checksum saat finalisasi sesuai dengan snapshot sebelum merge. Metadata runtime lama hanya mencatat path dan tidak memverifikasi identitas payload; keterbatasan historis ini tidak diubah secara retroaktif.

## 3. Benchmark OpenVINO standalone pada CPU OP3

Ini pipeline standalone tanpa ROS. Synthetic memakai 300 iterasi, sedangkan kamera sekitar 60 detik. Keduanya memakai 30 warmup dan sampling resource 0,2 detik. Pipeline standalone diukur menggunakan loop_wall_ms; mode kamera mencakup akuisisi gambar. Effective FPS dihitung dari frame/iterasi berhasil dibagi durasi aktual, bukan dari kebalikan mean inference.

| Mode | Model | Frame/iterasi | Inference mean / P95 (ms) | Loop mean / P95 (ms) | Effective FPS | CPU sistem / proses mean (%) | RAM sistem / RSS mean (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| synthetic | fp32 | 300 | 28,0504 / 29,6411 | 29,8745 / 31,4893 | 33,4559 | 55,2955 / 893,2841 | 4174,0696 / 240,7455 |
| synthetic | qat_int8 | 300 | 11,2265 / 11,4352 | 12,9337 / 13,2231 | 77,2287 | 20,3842 / 341,1632 | 4137,5352 / 225,3613 |
| camera | fp32 | 1238 | 37,1724 / 59,8114 | 48,4690 / 96,1018 | 20,6250 | 44,8789 / 706,8708 | 4152,4328 / 242,7902 |
| camera | qat_int8 | 1459 | 12,0950 / 14,8793 | 41,1139 / 70,4028 | 24,3139 | 8,5440 / 128,3852 | 4142,1401 / 228,3978 |

Sumber: [raw benchmark standalone](../experiments/hardware_op3/comparison_models/20261007_four_models_run1/). Nilai di tabel dihitung ulang dari CSV frame/resource dan durasi JSON.

## 4. ROS QAT INT8 — robot diam, referensi utama

Konfigurasi referensi: C920 640 × 480 YUYV, CPU, input 640 × 640, confidence 0,25, IoU 0,45, batch 1, warmup 30 inference + 30 frame kamera, sampling 0,2 detik, debug image/telemetry benchmark dimatikan. Kamera berjalan kontinu selama ketiga run pada runner lama.

QoS legacy BEST_EFFORT depth 5 berasal dari penggunaan qos_profile_sensor_data pada kode legacy. Metadata lama tidak menyimpan field QoS secara eksplisit. Pengukuran baru akan mencatat QoS tersebut secara eksplisit; keterbatasan bukti historis tetap dinyatakan.

| Run | Frame | Durasi (s) | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) | Effective FPS | Exception |
|---|---:|---:|---:|---:|---:|---:|
| run_01 | 809 | 60,0034 | 13,4770 / 14,9848 | 18,0768 / 20,3608 | 13,4826 | 0 |
| run_02 | 904 | 60,0047 | 13,1792 / 14,8562 | 17,4554 / 20,2106 | 15,0655 | 0 |
| run_03 | 729 | 60,0038 | 13,6519 / 15,1802 | 18,5108 / 21,2428 | 12,1492 | 0 |
| Gabungan | 2442 | 180,0119 | 13,4190 / 14,9848 | 17,9763 / 20,6444 | 13,5658 | 0 |

Latency pipeline diukur dari awal callback sampai deteksi siap sebelum publish. Callback_end_to_end_ms juga mencakup pembentukan/publikasi pesan. Camera_delivery_ms dicatat sesudah pemrosesan, sehingga bukan murni latency transport kamera.

### Resource per run dan gabungan

| Run | CPU sistem / proses mean (%) | RAM sistem mean (MB) | RSS mean / peak (MB) | Suhu mean / maksimum (°C) | Sampel |
|---|---:|---:|---:|---:|---:|
| run_01 | 6,4417 / 88,6464 | 4086,1780 | 257,7877 / 257,8186 | 44,05 / 53,0 | 295 |
| run_02 | 6,7339 / 95,0075 | 4080,1290 | 257,6942 / 257,7162 | 44,88 / 56,0 | 295 |
| run_03 | 6,0732 / 82,1976 | 4085,0343 | 257,0937 / 257,1182 | 45,18 / 52,0 | 295 |
| Gabungan | 6,4163 / 88,6172 | 4083,7804 | 257,5252 / 257,8186 | 44,71 / 56,0 | 885 |

### Variasi antar tiga run

| Metrik | Rata-rata nilai per run | SD sampel antar-run |
|---|---:|---:|
| Effective FPS | 13,5658 | 1,4599 |
| Mean inference (ms) | 13,4360 | 0,2390 |
| Mean pipeline (ms) | 18,0143 | 0,5305 |
| CPU sistem mean (%) | 6,4163 | 0,3311 |
| CPU proses mean (%) | 88,6172 | 6,4050 |
| RAM sistem mean (MB) | 4083,7804 | 3,2135 |
| RSS mean (MB) | 257,5252 | 0,3766 |

Mean/SD antar-run tidak disamakan dengan statistik gabungan frame. P95 gabungan menggunakan seluruh latency frame berhasil dengan interpolasi linear. Resource gabungan menggunakan seluruh sampel periodik, bukan rata-rata yang dibobot berdasarkan durasi.

Sumber: [metadata dan raw QAT diam](../experiments/hardware_op3/final_qat/), [rekap hasil terverifikasi](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/existing_results_recap.json).

## 5. ROS QAT INT8 — robot diam versus berjalan

| Kondisi | Status | Run sah | Frame gabungan | Durasi aktual (s) | Pipeline mean / P95 (ms) | Effective FPS | Exception |
|---|---|---:|---:|---:|---:|---:|---:|
| Robot diam | Terukur, referensi | 3 | 2442 | 180,0119 | 17,9763 / 20,6444 | 13,5658 | 0 |
| Robot berjalan | **BELUM TERUKUR — keselamatan/operator belum terverifikasi** | — | — | — | — | — | — |

Tidak ada angka berjalan baru yang diisi. Pengamatan terbaru tidak menerima status gateway/fall detector pada domain 0 maupun 73. Kamera /dev/video0 masih dipakai proses lain dengan format yuyv2rgb. Adaptor FTDI ttyUSB0 terdeteksi, tetapi kesehatan komunikasi Dynamixel dan kesiapan mekanis belum dapat dibuktikan dari keberadaan perangkat.

Run tidak lengkap tetap dikecualikan:

| Sumber lokal | Durasi (s) | Frame | completed | Keputusan |
|---|---:|---:|---|---|
| final_qat | 12,5234 | 171 | false | Bukan hasil final |
| 20261019T092114_qat_int8_ros_berjalan_27433 | 16,6991 | 216 | false | Bukan hasil final |

Run operator 20261019T092437_qat_int8_ros_berjalan_27824 memiliki tiga pengukuran yang secara teknis completed=true. Bukti operator bahwa robot benar-benar berjalan sepanjang setiap run belum tersedia, sehingga tidak dimasukkan sebagai hasil berjalan final dan tidak menggantikan pengujian baru yang diminta. Seluruh berkasnya dipertahankan secara lokal.

Rencana pengukuran baru: tepat tiga run 60 detik dengan BEST_EFFORT depth 5, kamera YUYV, dan parameter referensi di atas; operator menjalankan dan menghentikan walking melalui kontrol existing. Tiap run harus memiliki catatan waktu serta konfirmasi gerak nyata sepanjang interval, tanpa fall/kehilangan komunikasi/kondisi tidak aman. Label condition atau operator-confirmed saja tidak cukup.

Perbedaan/keterbatasan yang harus tetap dicatat: commit sumber lama dan baru berbeda, QoS/domain lama tidak dicatat eksplisit, identitas hash model saat pengukuran lama tidak diverifikasi, beban proses kontrol/adegan kamera dapat berbeda. Jika format kamera atau QoS berubah, perbandingan tidak dinyatakan sepenuhnya setara.

Berkas persiapan: [rencana walking](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/walking_plan.json), [snapshot keselamatan](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/safety_preflight.json), [tabel diam versus berjalan](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/qat_diam_vs_berjalan.csv).

## 6. Stability QAT 600 detik

Stability adalah satu run terpisah; tidak dipool dengan run 60 detik.

| Durasi (s) | Frame | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) | FPS | Exception | CPU sistem / proses mean (%) | RAM sistem / RSS mean (MB) | Suhu maksimum (°C) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 600,0162 | 8260 | 13,2500 / 14,4723 | 17,6222 / 19,6381 | 13,7663 | 0 | 6,3661 / 88,2798 | 4090,4133 / 258,3648 | 59,0 |

RSS awal 257,9456 MB, akhir 258,3798 MB, growth 434176 byte. Satu run ini tidak membuktikan ada/tidaknya memory leak. Sampel resource: 2957.

Sumber: [raw stability QAT](../experiments/hardware_op3/stability_qat/20261007_ros_qat/). Stability FP32 lama tetap merupakan [arsip terpisah](../experiments/hardware_op3/stability_test.json), memakai implementasi/path model dan metadata sumber berbeda; tidak digunakan sebagai pasangan perbandingan langsung dengan stability QAT.

## 7. Perbandingan tambahan ROS FP32 dan QAT — 19 Oktober, protokol awal

Sesi ini sudah tersedia sebelum finalisasi: masing-masing model 3 × 60 detik, pipeline ROS yang sama, QoS legacy BEST_EFFORT/5 dan domain 73. Angka dihitung ulang dari raw masing-masing model; tidak dipool dengan referensi QAT diam 7 Oktober. Kondisi diam mengikuti instruksi operator, tanpa bukti fisik independen tersimpan.

| Model | Frame | Durasi (s) | Inference mean / P95 gabungan (ms) | Pipeline mean / P95 gabungan (ms) | FPS gabungan | CPU sistem / proses mean (%) | RAM sistem / RSS mean (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 4408 | 180,0777 | 30,8054 / 49,3312 | 34,3875 / 55,1134 | 24,4783 | 45,6369 / 713,5325 | 5582,1647 / 271,4608 |
| qat_int8 | 2471 | 180,0576 | 13,3977 / 14,5546 | 18,0472 / 19,9430 | 13,7234 | 6,4410 / 90,1301 | 5498,8889 / 257,6271 |

QAT memiliki latency inference lebih rendah, tetapi FPS kamera ROS pada sesi awal lebih rendah. Pengiriman frame ROS diketahui dapat melewatkan frame sebelum callback; nol exception bukan jaminan seluruh frame kamera diterima. Data dipertahankan sebagaimana terukur dan tidak diganti dengan nilai FPS teoritis.
Sumber: [raw ROS FP32/QAT protokol awal](../experiments/hardware_op3/operator_runs/20261019T094035Z_ros_fp32_vs_qat_diam_29153/).

## 8. Perbaikan QoS implementasi — bukan hasil utama penelitian

Profil 15 detik BEST_EFFORT/5, RELIABLE/5, dan RELIABLE/1 beserta verifikasi script asli telah dilaporkan dalam [laporan perbaikan QoS](../experiments/hardware_op3/operator_runs/20261019T101101Z_ros_qos_ab_31903/QOS_FIX_REPORT.md). Opsi QoS tersedia di main. Profil singkat ini tidak dimasukkan ke tabel utama diam/berjalan, tidak menggantikan benchmark 3 × 60 detik, dan tidak dipakai untuk memperkirakan angka berjalan. Pengukuran berjalan yang diminta tetap memakai BEST_EFFORT/5.

## 9. Definisi metrik dan pembahasan

- Inference latency mengikuti interval OpenVINO infer dan pembacaan/copy output yang diukur kode; load/compile dan warmup tidak dimasukkan.
- Pipeline ROS: awal callback sampai deteksi siap sebelum publish. Callback_end_to_end juga mencakup pembentukan/publikasi pesan. Latency tidak disamakan dengan keseluruhan waktu fisik sensor-ke-konsumen.
- Effective FPS adalah jumlah frame berhasil dibagi durasi aktual. Frame yang hilang sebelum callback tidak muncul sebagai exception. Jangan gunakan 1000/mean inference sebagai effective FPS kamera.
- CPU sistem dirata-ratakan atas CPU logis; CPU proses dapat melampaui 100% karena multithreading. CPU proses yang tercatat adalah detector/benchmark, bukan total kamera + seluruh stack robot.
- RAM sistem mencakup proses lain. RSS hanya proses detector/benchmark. Semua ukuran memori tabel ini memakai MB desimal (byte/1.000.000), bukan MiB.
- SD menggunakan ddof=1. Statistik pooled frame dan SD antar-run dilaporkan terpisah. P95 gabungan dihitung dari frame mentah, bukan rata-rata P95 per run.
- Temperatur adalah maksimum pembacaan thermal_zone pada setiap sampel. Label sensor historis tidak disimpan; angka tidak selalu dapat disebut suhu CPU package.
- TEST PyTorch, OpenVINO standalone, ROS 60 detik, stability 600 detik, dan diagnostik QoS adalah protokol berbeda. Metrik tidak digabung sebagai satu pengukuran.
- QAT mengurangi ukuran pasangan model dan latency inference pada benchmark yang ada. Keunggulan inference tidak menjamin kenaikan FPS kamera jika input dibatasi laju kamera/pengiriman ROS.
- Tidak ada ground truth pada kamera live. Precision, recall, F1, dan mAP hanya berasal dari TEST berlabel yang dibekukan.
- Metadata commit historis pada JSON lama dipertahankan persis, termasuk hash sebelum rewrite metadata Git. Hash adalah jejak pada saat pengukuran.

## 10. Checklist operator sebelum walking

- [ ] Operator di lokasi memastikan mekanik, postur, penopangan, dan area eksperimen aman.
- [ ] Kontrol existing aktif dan domain ROS diketahui; status gateway/fall detector fresh dan recovery/interlock normal.
- [ ] Komunikasi Dynamixel sehat menurut kontrol/log existing; keberadaan ttyUSB0 saja tidak cukup.
- [ ] Operator mengetahui dan sudah memverifikasi kontrol stop yang efektif pada owner gerak aktif. CMD_STOP gateway melewati validasi permission epoch dan jalur stop walking; perintah itu tidak otomatis menyelesaikan recovery/action.
- [ ] Operator menutup sesi kamera lain agar runner memakai C920 640×480 YUYV sesuai referensi. Codex tidak mematikan sesi/proses kontrol atau melepas interlock.
- [ ] Operator mengaktifkan walking melalui antarmuka existing, lalu mengonfirmasi gerak nyata pada setiap interval run; Codex hanya menjalankan pengukuran.
- [ ] Jika fall/putus komunikasi/gerak tidak aman terjadi, operator menghentikan motion melalui kontrol terverifikasi; benchmark dihentikan dan run tidak sah dikecualikan.
- [ ] Output baru, BEST_EFFORT depth 5, domain vision-only 73, serta konfirmasi sebelum/sesudah tiap run siap dicatat.

## Artefak rekap dan status akhir sementara

- [Rekap JSON dari sumber mentah](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/existing_results_recap.json)
- [Rekap per run ROS](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/ros_per_run_recap.csv)
- [Tabel diam versus berjalan](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/qat_diam_vs_berjalan.csv)
- [Checksum sumber perhitungan](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/source_sha256.json)
- [Snapshot keselamatan dan preservasi](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/safety_preflight.json)
- Kode QoS telah berada di main. Pengujian berjalan belum selesai; finalisasi penelitian masih memerlukan tiga run yang sah setelah operator memenuhi syarat keselamatan.
