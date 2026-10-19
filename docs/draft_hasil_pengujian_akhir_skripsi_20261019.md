# Draft hasil pengujian akhir skripsi

Pembaruan: 2026-10-19T13:32:25.660498+00:00. Branch sumber akhir: `main`.

Pembaruan ini menghitung ulang raw CSV dan JSON yang telah tersimpan. Training, seleksi checkpoint, export, dan FINAL TEST tidak diulang. Pengujian QAT dan FP32 dengan robot bergerak didampingi operator masing-masing telah dilakukan tiga run × 60 detik. Kondisi fisik dan keterbatasan bukti dijelaskan pada Bagian 5 dan 5A.

## Ringkasan status

- Perubahan QoS commit `53d971f` telah di-fast-forward ke main dan push normal ke origin/main.
- FINAL TEST native PyTorch FP32/QAT, benchmark standalone OpenVINO, ROS QAT diam 3 × 60 detik, dan stability QAT 600 detik tersedia.
- Pengukuran tambahan ROS FP32/QAT 3 × 60 detik per model pada 19 Oktober tersedia dan dilaporkan terpisah dari referensi QAT diam 7 Oktober.
- Pengujian QAT berjalan dengan pendampingan/penopangan operator 3 × 60 detik selesai: 1.807 frame, tiga `completed=true`, nol exception. Konfirmasi gerak tersedia sebelum dan sesudah sesi; rincian penopangan/kontak lantai serta bukti video tidak tersedia.
- FP32 berjalan dengan pendampingan operator juga selesai 3 × 60 detik: 3655 frame, tiga `completed=true`, nol exception. Perbandingan FP32–QAT berjalan dan rekap empat kondisi tersedia pada Bagian 5A.
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

QoS legacy BEST_EFFORT depth 5 berasal dari penggunaan qos_profile_sensor_data pada kode legacy. Metadata lama tidak menyimpan field QoS secara eksplisit. Pengukuran berjalan baru mencatat QoS tersebut secara eksplisit; keterbatasan bukti historis tetap dinyatakan.

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

Tiga run berjalan baru telah selesai pada 19 Oktober 2026. Kondisi gerak dikonfirmasi operator sebelum sesi; setelah sesi operator menyatakan robot tidak jatuh dan sudah berhenti. Robot didampingi/ditopang operator; jumlah penopangan dan kontak kaki dengan lantai tidak didokumentasikan eksplisit. Hasil ini merupakan pengujian inferensi saat robot bergerak dengan pendampingan, bukan validasi gait berjalan mandiri. Konfirmasi terpisah setiap run dan video tidak tersedia.

Seluruh run berjalan memiliki `completed=true`, durasi aktual ≥60 detik, `thermal_abort=false`, dan nol exception termasuk warmup. Pemeriksaan raw CSV sesuai jumlah frame/resource pada JSON. Warmup 30 inference + 30 frame kamera per run, C920 YUYV 640×480, input 640×640/batch 1, CPU, confidence 0,25, IoU 0,45, sampling 0,2 detik, BEST_EFFORT depth 5, debug image/telemetry dimatikan. Kamera kontinu selama tiga run; detector baru untuk setiap run. Domain vision 73 dipisahkan dari kontrol domain 0.

| Kondisi / run | Frame | Durasi (s) | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) | Effective FPS | Exception |
|---|---:|---:|---:|---:|---:|---:|
| Berjalan run_01 | 566 | 60,0197 | 13,0494 / 14,8944 | 16,8577 / 19,2927 | 9,4302 | 0 |
| Berjalan run_02 | 603 | 60,0334 | 12,7908 / 14,7780 | 16,4298 / 19,1687 | 10,0444 | 0 |
| Berjalan run_03 | 638 | 60,0090 | 13,0526 / 14,9911 | 16,7970 / 19,9950 | 10,6317 | 0 |
| Diam gabungan | 2442 | 180,0119 | 13,4190 / 14,9848 | 17,9763 / 20,6444 | 13,5658 | 0 |
| Berjalan gabungan | 1807 | 180,0622 | 12,9642 / 14,8962 | 16,6935 / 19,3917 | 10,0354 | 0 |

### Resource berjalan per run dan perbandingan gabungan

| Kondisi / run | CPU sistem / proses mean (%) | RAM sistem mean (MB) | RSS mean / peak (MB) | Suhu mean / maksimum (°C) | Sampel |
|---|---:|---:|---:|---:|---:|
| run_01 | 6,6476 / 59,0855 | 3458,5560 | 256,8211 / 256,8479 | 44,6351 / 54,0000 | 296 |
| run_02 | 6,6530 / 61,3828 | 3454,2332 | 257,1934 / 257,2165 | 44,8480 / 52,0000 | 296 |
| run_03 | 6,8159 / 66,0156 | 3453,8588 | 257,3326 / 257,3476 | 45,1186 / 51,0000 | 295 |
| diam gabungan | 6,4163 / 88,6172 | 4083,7804 | 257,5252 / 257,8186 | 44,7051 / 56,0000 | 885 |
| berjalan gabungan | 6,7054 / 62,1569 | 3455,5512 | 257,1154 / 257,3476 | 44,8670 / 54,0000 | 887 |

### Variasi antar tiga run berjalan

| Metrik | Rata-rata nilai per run | SD sampel antar-run |
|---|---:|---:|
| Frame | 602,3333 | 36,0046 |
| Effective FPS | 10,0355 | 0,6008 |
| Mean inference (ms) | 12,9643 | 0,1502 |
| Mean pipeline (ms) | 16,6949 | 0,2315 |
| CPU sistem mean (%) | 6,7055 | 0,0956 |
| CPU detector mean (%) | 62,1613 | 3,5300 |
| RAM sistem mean (MB) | 3455,5493 | 2,6106 |
| RSS detector mean (MB) | 257,1157 | 0,2645 |
| Suhu mean (°C) | 44,8673 | 0,2423 |

FPS gabungan diam 13,5658 dan berjalan 10,0354; perubahan relatif -26,0239%. Mean inference berjalan 12,9642 ms dan pipeline 16,6935 ms. FPS berjalan lebih rendah meskipun latency frame yang diproses tidak meningkat. Pengiriman frame sebelum callback, adegan kamera dan beban sistem tidak dikendalikan sepenuhnya; data ini tidak membuktikan bahwa gerak robot sendiri menyebabkan penurunan FPS. CPU proses yang lebih rendah dapat sejalan dengan lebih sedikit frame yang diproses, tetapi penyebabnya tidak diuji lebih lanjut.

Perbandingan ini bersifat deskriptif. Parameter utama terverifikasi cocok melalui protocol.json dan setup JSON, serta format kamera melalui log. Namun tanggal/commit, adegan dan beban kontrol berbeda; QoS/domain lama tidak dicatat eksplisit dan identitas model lama tidak diverifikasi. Perbandingan tidak dinyatakan sepenuhnya setara atau sebagai efek kausal gerakan robot.

Run berjalan lama 12,5234 detik (`final_qat`, 171 frame, `completed=false`) dan 16,6991 detik (`20261019T092114...`, 216 frame, `completed=false`) tetap dikecualikan. Sesi `20261019T092437...` tetap arsip; tidak dicampurkan ke tiga run baru. Metadata commit historis tidak diubah.

Sumber: [raw CSV/JSON dan log](../experiments/hardware_op3/operator_runs/20261019T131319Z_qat_berjalan_final/), [rekap JSON](../experiments/hardware_op3/operator_runs/20261019T131319Z_qat_berjalan_final/analysis/qat_diam_vs_berjalan_recap.json), [tabel perbandingan CSV](../experiments/hardware_op3/operator_runs/20261019T131319Z_qat_berjalan_final/analysis/qat_diam_vs_berjalan.csv), [konfirmasi sebelum sesi](../experiments/hardware_op3/operator_runs/20261019T131319Z_qat_berjalan_final/operator_confirmation_start.json), [konfirmasi sesudah sesi](../experiments/hardware_op3/operator_runs/20261019T131319Z_qat_berjalan_final/operator_confirmation_end.json).

## 5A. ROS FP32 — robot berjalan dan perbandingan dengan QAT

FP32 berjalan menyusul sesi QAT dengan manager dan parameter gait yang sama, tanpa init pose ulang. Operator mengonfirmasi berjalan stabil dengan penopangan yang sama sebelum pengukuran, lalu mengonfirmasi kondisi sesudah sesi. Masing-masing run menggunakan 30 warmup inference + 30 frame, CPU, input 640×640/batch 1, C920 YUYV 640×480 pada 30 FPS, confidence 0,25, IoU 0,45, sampling 0,2 detik, BEST_EFFORT/depth 5, dan domain vision 73. Protocol JSON FP32–QAT berjalan cocok pada seluruh field selain path model.

Semua run FP32 memiliki `completed=true`, durasi aktual ≥60 detik, `thermal_abort=false`, nol exception termasuk warmup, serta jumlah frame/resource yang cocok dengan raw CSV.

| Model / kondisi | Frame | Durasi (s) | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) | Effective FPS | Exception |
|---|---:|---:|---:|---:|---:|---:|
| FP32 berjalan run_01 | 1231 | 60,0065 | 33,9576 / 55,8076 | 36,4623 / 59,9422 | 20,5144 | 0 |
| FP32 berjalan run_02 | 1190 | 60,0198 | 34,1952 / 58,8548 | 36,7979 / 63,3219 | 19,8268 | 0 |
| FP32 berjalan run_03 | 1234 | 60,0166 | 34,2168 / 58,8241 | 36,6580 / 63,2791 | 20,5610 | 0 |
| FP32 berjalan gabungan | 3655 | 180,0429 | 34,1225 / 58,3080 | 36,6377 / 62,6153 | 20,3007 | 0 |
| QAT berjalan gabungan | 1807 | 180,0622 | 12,9642 / 14,8962 | 16,6935 / 19,3917 | 10,0354 | 0 |

### Resource FP32 berjalan per run dan perbandingan model

| Model / run | CPU sistem / proses mean (%) | RAM sistem mean (MB) | RSS mean / peak (MB) | Suhu mean / maksimum (°C) | Sampel |
|---|---:|---:|---:|---:|---:|
| run_01 | 43,7956 / 662,3734 | 3521,5546 | 271,3356 / 271,3559 | 59,2660 / 67,0000 | 297 |
| run_02 | 42,7404 / 646,1448 | 3501,6018 | 271,8799 / 271,8925 | 61,9293 / 68,0000 | 297 |
| run_03 | 44,3108 / 668,5165 | 3507,4229 | 272,0933 / 272,1178 | 63,3771 / 69,0000 | 297 |
| fp32_berjalan gabungan | 43,6156 / 659,0116 | 3510,1931 | 271,7696 / 272,1178 | 61,5241 / 69,0000 | 891 |
| qat_berjalan gabungan | 6,7054 / 62,1569 | 3455,5512 | 257,1154 / 257,3476 | 44,8670 / 54,0000 | 887 |

### Variasi antar tiga run FP32 berjalan

| Metrik | Rata-rata nilai per run | SD sampel antar-run |
|---|---:|---:|
| Frame | 1218,3333 | 24,5832 |
| Effective FPS | 20,3007 | 0,4111 |
| Mean inference (ms) | 34,1232 | 0,1438 |
| Mean pipeline (ms) | 36,6394 | 0,1686 |
| CPU sistem mean (%) | 43,6156 | 0,8005 |
| CPU detector mean (%) | 659,0116 | 11,5585 |
| RAM sistem mean (MB) | 3510,1931 | 10,2608 |
| RSS detector mean (MB) | 271,7696 | 0,3907 |
| Suhu mean (°C) | 61,5241 | 2,0853 |

### Ringkasan ROS diam dan berjalan — dataset 19 Oktober

Tabel berikut memakai sesi diam 19 Oktober untuk kedua model, terpisah dari referensi QAT diam 7 Oktober (2.442 frame) pada Bagian 4–5. Tiap baris adalah gabungan tiga run dari datasetnya sendiri; tidak ada pooling lintas model/kondisi/tanggal.

| Model / kondisi | Frame | Durasi (s) | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) | Effective FPS | Exception |
|---|---:|---:|---:|---:|---:|---:|
| FP32 diam | 4408 | 180,0777 | 30,8054 / 49,3312 | 34,3875 / 55,1134 | 24,4783 | 0 |
| QAT diam | 2471 | 180,0576 | 13,3977 / 14,5546 | 18,0472 / 19,9430 | 13,7234 | 0 |
| FP32 berjalan | 3655 | 180,0429 | 34,1225 / 58,3080 | 36,6377 / 62,6153 | 20,3007 | 0 |
| QAT berjalan | 1807 | 180,0622 | 12,9642 / 14,8962 | 16,6935 / 19,3917 | 10,0354 | 0 |

Pada sesi berjalan, mean inference FP32 34,1225 ms dan QAT 12,9642 ms (rasio FP32/QAT 2,6320). Namun effective FPS FP32 20,3007 dan QAT 10,0354. Jadi penurunan latency inference tidak diikuti kenaikan throughput kamera ROS pada konfigurasi ini. Nilai FPS berasal dari raw frame/durasi, bukan 1000/mean inference. Frame yang tidak diterima callback tidak tercatat sebagai exception. Tidak ada eksperimen QoS tambahan dilakukan.

Perbandingan bersifat deskriptif: sesi berjalan berurutan QAT lalu FP32, adegan dan beban latar tidak sepenuhnya sama. Dukungan operator dikonfirmasi sama, tetapi tingkat penopangan/kontak lantai tidak diukur rinci; tidak tersedia video atau konfirmasi fisik terpisah setiap run. Ini bukan evaluasi gait berjalan mandiri atau bukti sebab-akibat gerakan robot. Sesi diam dan berjalan memakai tanggal/waktu serta commit metadata yang berbeda; nilai historis tidak diubah.

Sumber: [raw FP32 berjalan](../experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/), [rekap JSON](../experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/ros_fp32_vs_qat_recap.json), [CSV model/kondisi](../experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/ros_fp32_vs_qat_conditions.csv), [konfirmasi operator](../experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/operator_confirmation_end.json).

### Penjelasan perbedaan FPS FP32 dan QAT

Kedua sesi berjalan benar-benar memakai BEST_EFFORT depth 5. QAT lebih cepat secara inference (12,9642 ms versus FP32 34,1225 ms), tetapi mean jarak header frame yang diterima lebih besar: 99,5814 ms versus 49,2730 ms. Ini menjelaskan FPS teramati yang lebih rendah tanpa menyatakan komputasi QAT lebih lambat. Tidak ditemukan pembatas skip-frame berbeda pada jalur detector.

Diagnostik lama sekitar 15 detik memperlihatkan QAT BEST_EFFORT/5 13,5192 FPS dan verifikasi runner RELIABLE/1 29,9259 FPS; FP32 RELIABLE/1 29,3988 FPS. Angka diagnostik tersebut hanya bukti pengaruh konfigurasi implementasi, bukan hasil utama atau pengganti run berjalan. Dugaan terkuat adalah interaksi pengiriman ROS/QoS/DDS/antrean/scheduling dengan beban model; penyebab spesifik belum terbukti. Jumlah buffer kamera dan packet loss tidak diukur pada sesi final, sehingga pengaruh laju kamera/exposure belum dapat dieliminasi.

Analisis lengkap: [diagnosis FPS berbasis data tersimpan](../experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/FPS_DIAGNOSIS.md). Tidak ada eksperimen QoS baru atau perubahan protokol/model setelah pengukuran.

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
- [ ] Kontrol existing dan domain ROS diketahui. Bila gateway/fall detector digunakan, status harus fresh dan recovery/interlock normal. Sesi berjalan ini memakai manager langsung tanpa gateway/fall detector; keselamatan dipantau operator.
- [ ] Komunikasi Dynamixel sehat menurut kontrol/log existing; keberadaan ttyUSB0 saja tidak cukup.
- [ ] Operator mengetahui dan sudah memverifikasi kontrol stop yang efektif pada owner gerak aktif. CMD_STOP gateway melewati validasi permission epoch dan jalur stop walking; perintah itu tidak otomatis menyelesaikan recovery/action.
- [ ] Sesi kamera lain ditutup atas izin operator agar runner memakai C920 640×480 YUYV sesuai referensi; interlock tidak dilewati.
- [ ] Aktivasi manager/init/walking melalui kontrol existing membutuhkan persetujuan operator di lokasi. Dalam sesi ini perintah dikirim melalui run_op3.sh atas persetujuan eksplisit operator; operator mengonfirmasi postur stabil, gerak, dan berhenti. Gait/offset tidak diubah.
- [ ] Jika fall/putus komunikasi/gerak tidak aman terjadi, operator menghentikan motion melalui kontrol terverifikasi; benchmark dihentikan dan run tidak sah dikecualikan.
- [ ] Output baru, BEST_EFFORT depth 5, domain vision-only 73, serta konfirmasi sebelum/sesudah tiap run siap dicatat.

## Artefak rekap dan status akhir

- [Rekap JSON dari sumber mentah](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/existing_results_recap.json)
- [Rekap per run ROS](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/ros_per_run_recap.csv)
- [Tabel diam versus berjalan](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/qat_diam_vs_berjalan.csv)
- [Checksum sumber perhitungan](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/source_sha256.json)
- [Snapshot keselamatan dan preservasi](../experiments/hardware_op3/finalization/20261019T105633Z_main_and_walking_preflight/safety_preflight.json)
- Kode QoS berada di main. Tiga run inferensi per model QAT dan FP32 selama gerak didampingi operator telah selesai dan direkap dari raw data; keterbatasan kondisi fisik tetap berlaku.
- [Hasil berjalan dan rekap terbaru](../experiments/hardware_op3/operator_runs/20261019T131319Z_qat_berjalan_final/analysis/WALKING_RESULTS.md)
- [CSV diam versus berjalan terbaru](../experiments/hardware_op3/operator_runs/20261019T131319Z_qat_berjalan_final/analysis/qat_diam_vs_berjalan.csv)
- [Checksum sumber rekap terbaru](../experiments/hardware_op3/operator_runs/20261019T131319Z_qat_berjalan_final/analysis/source_sha256.json)
- Artefak preflight bertanggal 10:56 UTC di atas dipertahankan sebagai arsip persiapan, bukan status pengukuran terbaru.

- [Rekap FP32–QAT ROS diam/berjalan terbaru](../experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/FP32_QAT_ROS_FINAL_RESULTS.md)
- [CSV empat dataset ROS terbaru](../experiments/hardware_op3/operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/ros_fp32_vs_qat_conditions.csv)
