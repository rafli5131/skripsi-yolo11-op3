# Langkah Berikutnya — Implementasi ke ROBOTIS OP3 dan Pengambilan Data Final

## 1. Tujuan tahap ini

Training YOLO11n FP32, true NNCF QAT, pemilihan checkpoint QAT terbaik, evaluasi FINAL TEST, dan konversi QAT ke OpenVINO sudah selesai. Tahap berikutnya **bukan training ulang**, melainkan implementasi artefak QAT INT8 OpenVINO ke ROBOTIS OP3 melalui ROS 2 dan pengambilan data penelitian pada perangkat robot.

Dokumen ini menjadi instruksi utama untuk Codex ketika bekerja pada robot. Implementasi dan data yang diambil harus mengikuti proposal skripsi, khususnya:

- Rumusan masalah 1: kualitas deteksi berdasarkan precision, recall, F1-score, mAP@0.5, dan mAP@0.5:0.95.
- Rumusan masalah 2: waktu komputasi pada ROBOTIS OP3 berdasarkan latensi dan FPS.
- Rumusan masalah 3: beban komputasi pada ROBOTIS OP3 berdasarkan utilisasi CPU dan penggunaan memori.
- Pengujian sumber daya dilakukan pada dua kondisi: **robot diam dengan inferensi aktif** dan **robot berjalan dengan inferensi aktif**.
- Kamera yang digunakan adalah Logitech C920. Akuisisi kamera pada proposal menggunakan resolusi 640×480, sedangkan input model tetap diproses ke 640×640.
- Perangkat target adalah CPU onboard Intel NUC12WSHi5 pada ROBOTIS OP3.
- ROS 2 menjadi lingkungan integrasi modul persepsi dan OpenVINO menjadi runtime inferensi.
- Penelitian utama tidak melakukan studi komparatif QAT vs PTQ. PTQ yang pernah ada di repository adalah eksperimen/comparator historis dan **tidak menjadi data utama tahap deployment skripsi ini**.

Target akhir tahap ini adalah satu pipeline ROS 2 yang dapat menerima frame kamera C920, menjalankan QAT INT8 OpenVINO pada CPU onboard, menghasilkan deteksi kelas `ball`, `gawang`, dan `robot`, serta merekam seluruh data yang diperlukan untuk Tabel 3.1, Tabel 3.2, dan Tabel 3.3 pada proposal.

---

## 2. Prinsip yang tidak boleh diubah oleh Codex

Codex harus memperlakukan artefak berikut sebagai model final yang sudah dibekukan:

```text
artifacts/openvino/qat_int8/model.xml
artifacts/openvino/qat_int8/model.bin
```

Codex **tidak boleh** melakukan training ulang, QAT tuning, PTQ, calibration, perubahan bobot, perubahan dataset, atau pemilihan checkpoint baru hanya untuk deployment.

Held-out TEST juga tidak boleh dijalankan ulang. Hasil akurasi FINAL TEST yang sudah dibekukan tetap menjadi sumber angka precision, recall, F1-score, mAP@0.5, dan mAP@0.5:0.95. Kamera live di robot tidak memiliki ground truth, sehingga jumlah deteksi dan confidence dari kamera live **tidak boleh** diperlakukan sebagai precision, recall, F1, atau mAP.

Validasi OpenVINO yang sudah dilakukan tetap dipakai sebagai bukti bahwa artefak deployment merepresentasikan model QAT final. Tahap robot hanya mengambil data waktu komputasi dan sumber daya sesuai proposal.

---

## 3. Tahapan kerja Codex pada robot

### 3.1 Audit awal perangkat

Sebelum mengubah workspace ROS 2, Codex harus membaca kondisi robot dan mencatatnya. Minimal periksa:

| Item | Yang harus dicatat |
| --- | --- |
| Robot | ROBOTIS OP3 |
| CPU onboard | model CPU aktual; target proposal Intel Core i5-1240P pada NUC12WSHi5 |
| RAM | total RAM |
| OS | Ubuntu 24.04 |
| ROS 2 | distro yang aktif, target Jazzy |
| OpenVINO | versi runtime aktual |
| Kamera | Logitech C920 dan device yang benar, misalnya `/dev/video0` |
| Resolusi kamera | 640×480 |
| Model input | 1×3×640×640 |
| Device inferensi | CPU |
| Class mapping | 0 ball, 1 gawang, 2 robot |

Gunakan `scripts/deployment/audit_hardware_op3.py` sebagai titik awal apabila masih sesuai. Audit ini hanya untuk memastikan lingkungan dapat digunakan dan tidak boleh mengubah model.

Simpan hasil audit final ke:

```text
experiments/hardware_op3/final_qat/environment.json
experiments/hardware_op3/final_qat/environment.md
```

### 3.2 Periksa workspace ROS 2 yang sudah ada

Repository skripsi belum menyimpan source package ROS 2 deployment secara lengkap. Karena pada robot mungkin sudah terdapat `usb_cam`, detector node, message package, atau launch file, Codex harus **inspect dulu** workspace robot sebelum membuat package baru.

Urutan keputusan:

```text
inspect workspace ROS 2
        |
        +-- detector OpenVINO yang ada dapat dipakai?
        |       |
        |       +-- ya -> rapikan/adaptasi seminimal mungkin
        |
        +-- tidak -> buat package deployment khusus QAT
```

Jangan membuat stack paralel jika implementasi yang sudah ada dapat dipakai dengan aman.

Jika perlu package baru, gunakan package Python ROS 2 yang jelas, misalnya:

```text
op3_qat_vision/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
├── config/
│   └── qat_detector.yaml
├── launch/
│   └── qat_openvino_camera.launch.py
└── op3_qat_vision/
    ├── __init__.py
    └── qat_openvino_detector_node.py
```

Source package yang dibuat untuk robot harus ikut disimpan di repository agar implementasi skripsi dapat direproduksi.

### 3.3 Pipeline ROS 2 yang harus diimplementasikan

Pipeline final:

```text
Logitech C920
640×480
    |
    v
usb_cam / camera node
    |
    v
/image_raw
    |
    v
QAT OpenVINO detector node
    |
    +--> preprocessing
    |       resize/letterbox 640×640
    |       BGR -> RGB bila diperlukan
    |       NCHW
    |       float32 / 255
    |
    +--> OpenVINO CPU inference
    |
    +--> YOLO postprocess
    |       confidence filter
    |       NMS
    |       coordinate restore
    |
    +--> detection output
    |
    +--> benchmark telemetry
```

Detector node harus memuat XML/BIN QAT final sekali ketika startup. Jangan load/compile model untuk setiap frame.

Parameter minimal yang harus tersedia melalui ROS parameter atau YAML:

| Parameter | Nilai/tujuan |
| --- | --- |
| `model_path` | path `artifacts/openvino/qat_int8/model.xml` |
| `device` | `CPU` |
| `input_topic` | topic kamera |
| `input_size` | 640 |
| `confidence_threshold` | mengikuti konfigurasi evaluasi/deployment yang sudah ditetapkan |
| `iou_threshold` | mengikuti konfigurasi evaluasi/deployment yang sudah ditetapkan |
| `publish_debug_image` | default `false` saat pengukuran |
| `publish_benchmark` | `true` saat pengambilan data |

Codex tidak boleh mengubah confidence/IoU hanya untuk membuat hasil live terlihat lebih baik. Jika nilainya belum terdokumentasi, ambil dari implementasi evaluasi yang sudah digunakan dan tulis nilainya pada metadata eksperimen.

### 3.4 Topic benchmark

Detector harus menerbitkan telemetry setiap frame yang berhasil diproses. Bila workspace sudah memiliki `/vision/benchmark`, pertahankan topic tersebut. Format boleh memakai message yang sudah ada; bila tidak ada message khusus, JSON melalui `std_msgs/msg/String` dapat dipakai untuk telemetry penelitian.

Minimal data per frame:

```text
timestamp
frame_index
preprocess_ms
inference_ms
postprocess_ms
latency_ms
detection_count
error
```

Definisi **latency_ms** untuk data utama proposal:

```text
latency_ms =
waktu setelah hasil deteksi untuk frame siap
-
waktu saat detector mulai memproses frame tersebut
```

Dengan demikian latency utama mencakup:

```text
preprocess + OpenVINO inference + postprocess
```

`inference_ms` tetap dicatat sebagai data diagnostik, tetapi **Tabel 3.1 menggunakan latency_ms**, karena proposal mendefinisikan latensi sebagai waktu pemrosesan satu frame sampai menghasilkan keluaran deteksi.

---

## 4. Warm-up dan pengukuran waktu

Proposal mewajibkan tahap warm-up sebelum pengambilan data agar inisialisasi model, alokasi memori, dan kompilasi OpenVINO tidak memengaruhi hasil.

Proposal tidak menentukan jumlah frame warm-up. Oleh karena itu Codex harus membuat nilai ini configurable dan mencatat nilai aktual pada metadata. Implementasi saat ini boleh memakai default:

```text
warmup_frames = 30
```

tetapi angka tersebut harus diperlakukan sebagai **parameter implementasi**, bukan angka yang diklaim berasal dari proposal.

Setelah warm-up selesai, setiap run harus mencatat:

```text
run_id
jumlah frame berhasil diproses
waktu pengujian aktual (detik)
latency setiap frame (ms)
rata-rata latency run (ms)
FPS = jumlah frame / waktu pengujian aktual
```

Jumlah pengulangan `n` dan durasi per run juga harus configurable. Sebelum eksperimen final, tetapkan nilainya sekali lalu gunakan konfigurasi yang sama pada seluruh run. Jangan mengganti durasi atau jumlah pengulangan setelah melihat hasil.

Output raw:

```text
experiments/hardware_op3/final_qat/latency/
├── run_01_frames.csv
├── run_02_frames.csv
├── ...
├── runs_summary.csv
└── metadata.json
```

### Format raw latency per frame

| Kolom | Keterangan |
| --- | --- |
| `run_id` | nomor pengujian |
| `frame_index` | nomor frame pada run |
| `timestamp` | waktu pengambilan sampel |
| `preprocess_ms` | waktu preprocessing |
| `inference_ms` | waktu OpenVINO inferensi |
| `postprocess_ms` | waktu postprocessing |
| `latency_ms` | total waktu pemrosesan detector per frame |
| `detection_count` | jumlah deteksi pada frame; diagnostik |
| `error` | kosong bila sukses |

### Tabel 3.1 yang harus dibuat otomatis

**Tabel 3.1 Hasil Pengujian Waktu Komputasi Sistem Deteksi**

| Pengujian | Jumlah Frame | Waktu Pengujian (s) | Rata-Rata Latensi (ms) | FPS |
| ---: | ---: | ---: | ---: | ---: |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| ... | ... | ... | ... | ... |
| n |  |  |  |  |
| **Rata-rata** |  |  |  |  |

Rumus yang harus digunakan:

```text
Li = t_akhir,i - t_awal,i

L_rata-rata = sum(Li) / n_frame

FPS = N / T
```

FPS utama **bukan** `1000 / mean inference_ms`. FPS proposal adalah jumlah frame yang berhasil diproses dibagi total waktu pengujian.

---

## 5. Pengujian CPU dan memori sesuai proposal

Pengujian sumber daya harus dilakukan pada **dua kondisi yang sama persis dengan proposal**:

### Kondisi A — Robot diam, inferensi aktif

- Robot dalam keadaan diam.
- Kamera dan detector QAT OpenVINO tetap aktif.
- ROS 2 stack yang dibutuhkan tetap berjalan.
- Tidak ada gerakan walking.
- CPU dan memori dicatat periodik selama jendela pengujian.

### Kondisi B — Robot berjalan, inferensi aktif

- Detector dan kamera tetap berjalan dengan konfigurasi yang sama.
- ROBOTIS OP3 melakukan gerakan berjalan.
- Walking controller yang normal digunakan pada robot harus aktif.
- CPU dan memori dicatat periodik dengan metode dan interval yang sama seperti kondisi diam.

Kedua kondisi harus memakai model, kamera, resolusi, threshold, runtime, durasi, dan interval sampling yang sama. Perbedaannya hanya keadaan robot diam atau berjalan.

Proposal hanya menyebut pencatatan periodik dan tidak menentukan interval. Codex harus membuat interval configurable dan mencatat nilai aktual. Default teknis yang dapat dipakai:

```text
resource_sample_interval = 0.2 s
```

### Metrik sumber daya utama

Karena proposal membahas **beban komputasi sistem pada robot**, nilai utama Tabel 3.2 dan Tabel 3.3 harus menggunakan metrik sistem:

```text
CPU Usage (%)      = utilisasi CPU sistem
Memory Used (MB)   = RAM sistem yang sedang digunakan
Memory Usage (%)   = persentase RAM sistem
```

Codex boleh merekam `process_cpu_percent` dan process RSS detector sebagai data tambahan untuk diagnosis, tetapi angka tersebut tidak menggantikan kolom utama Tabel 3.2/3.3.

Raw output:

```text
experiments/hardware_op3/final_qat/resources/
├── diam_run_01.csv
├── diam_run_02.csv
├── ...
├── berjalan_run_01.csv
├── berjalan_run_02.csv
├── ...
├── resources_all.csv
├── resources_summary.csv
└── metadata.json
```

### Format raw CSV sumber daya

| Kolom | Keterangan |
| --- | --- |
| `sample_no` | nomor sampel |
| `run_id` | nomor pengujian |
| `condition` | `diam` atau `berjalan` |
| `timestamp` | waktu sampling |
| `cpu_usage_percent` | CPU sistem (%) |
| `memory_used_mb` | RAM sistem yang digunakan (MB) |
| `memory_usage_percent` | RAM sistem (%) |
| `detector_process_cpu_percent` | tambahan diagnostik |
| `detector_rss_mb` | tambahan diagnostik |

### Tabel 3.2 yang harus dibuat otomatis

**Tabel 3.2 Hasil Pengujian Penggunaan CPU dan Memori**

| No. | Kondisi Robot | CPU Usage (%) | Memory Used (MB) | Memory Usage (%) |
| ---: | --- | ---: | ---: | ---: |
| 1 | Diam |  |  |  |
| ... | Diam |  |  |  |
| n | Diam |  |  |  |
| 1 | Berjalan |  |  |  |
| ... | Berjalan |  |  |  |
| n | Berjalan |  |  |  |

Tabel ini harus berasal langsung dari raw CSV. Jangan mengetik angka secara manual ke laporan jika generator tabel dapat mengambil angka dari data mentah.

---

## 6. Rekapitulasi sumber daya

Proposal menggunakan rata-rata dan **standar deviasi sampel**. Untuk standar deviasi, denominator harus `n - 1`.

Rumus:

```text
CPU_mean = sum(CPU_i) / n

s_CPU = sqrt(
    sum((CPU_i - CPU_mean)^2) / (n - 1)
)

MEM_mean = sum(MEM_i) / n

s_MEM = sqrt(
    sum((MEM_i - MEM_mean)^2) / (n - 1)
)

delta_CPU = CPU_mean_berjalan - CPU_mean_diam

delta_MEM = MEM_mean_berjalan - MEM_mean_diam
```

**Penting:** jika memakai NumPy, gunakan `np.std(values, ddof=1)` untuk tabel proposal. Jangan memakai `np.std(values)` default karena itu memakai `ddof=0`.

### Tabel 3.3 yang harus dibuat otomatis

**Tabel 3.3 Rekapitulasi Penggunaan Sumber Daya Komputasi**

| Parameter | Robot Diam | Robot Berjalan | Selisih |
| --- | ---: | ---: | ---: |
| Rata-rata CPU (%) |  |  |  |
| Standar deviasi CPU (%) |  |  | — |
| Rata-rata memori (MB) |  |  |  |
| Standar deviasi memori (MB) |  |  | — |
| Rata-rata penggunaan memori (%) |  |  |  |

Untuk `Rata-rata CPU` gunakan `delta_CPU` pada kolom Selisih. Untuk `Rata-rata memori (MB)` gunakan `delta_MEM`. Selisih standar deviasi tidak diperlukan karena proposal menandainya dengan tanda `-`. Bila kolom selisih rata-rata penggunaan memori (%) diisi, tulis sebagai perbedaan **percentage point** dan beri penjelasan pada laporan.

---

## 7. Data akurasi yang masuk laporan

Data akurasi **tidak diambil dari kamera live robot**.

Data yang harus masuk ke bagian pengujian kualitas model:

| Model penelitian | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n QAT INT8 final | gunakan hasil FINAL TEST yang sudah dibekukan | gunakan hasil FINAL TEST | gunakan hasil FINAL TEST | gunakan hasil FINAL TEST | gunakan hasil FINAL TEST |

Sumber angka harus berasal dari artefak evaluasi final yang sudah ada di:

```text
experiments/final_test/
```

Jangan menghitung ulang metrik dari live camera karena tidak ada anotasi ground truth.

Untuk pelaporan, jelaskan dua bukti secara terpisah:

```text
akurasi model
= held-out TEST yang sudah dibekukan

kesiapan deployment OpenVINO
= validasi numerik OpenVINO yang sudah diselesaikan

kinerja real-time pada robot
= data latency/FPS/CPU/RAM dari tahap ini
```

Jangan menggabungkan ketiganya seolah berasal dari run yang sama.

---

## 8. Script yang harus dibuat/dirapikan oleh Codex

Codex harus memanfaatkan script yang sudah ada bila memungkinkan, khususnya `scripts/deployment/benchmark_op3_ros2.py`, tetapi script final harus mengikuti tabel proposal dan hanya fokus pada QAT final.

Target implementasi:

```text
scripts/
├── deploy_op3_qat.py
├── benchmark_op3_qat_latency.py
├── benchmark_op3_qat_resources.py
└── build_op3_thesis_tables.py
```

Jika fungsi yang sudah ada di `benchmark_op3_ros2.py` dapat dipakai, pindahkan fungsi umum ke helper tanpa merusak data lama. Hindari copy-paste besar.

### `deploy_op3_qat.py`

Tugas:

- audit bahwa XML/BIN QAT tersedia;
- memastikan OpenVINO CPU tersedia;
- menampilkan konfigurasi deployment;
- membantu menyalin/materialize model ke path robot bila diperlukan;
- tidak melakukan konversi ulang;
- tidak melakukan QAT/PTQ/training.

### `benchmark_op3_qat_latency.py`

Tugas:

- membaca telemetry ROS 2 detector;
- melewati warm-up;
- menjalankan `n` run;
- mencatat raw latency per frame;
- menghitung jumlah frame, waktu aktual, mean latency, dan FPS;
- menghasilkan `table_3_1_latency.csv` dan versi Markdown.

### `benchmark_op3_qat_resources.py`

Tugas:

- menerima `--condition diam|berjalan`;
- merekam CPU/RAM secara periodik;
- menulis raw CSV;
- tidak menggerakkan robot secara otomatis kecuali Codex sudah memahami dan memverifikasi interface walking robot yang ada;
- untuk kondisi berjalan, operator boleh memulai walking melalui mekanisme robot yang sudah tervalidasi, sedangkan logger hanya merekam data;
- menghitung summary setelah kedua kondisi tersedia.

**Keselamatan robot lebih penting daripada otomatisasi eksperimen.** Jangan membuat perintah walking baru hanya untuk benchmark jika interface gerak robot belum diverifikasi.

### `build_op3_thesis_tables.py`

Tugas:

- hanya membaca raw CSV;
- tidak menjalankan model;
- membuat Tabel 3.1, 3.2, dan 3.3;
- menggunakan `ddof=1` untuk standar deviasi;
- menghasilkan CSV dan Markdown;
- tidak membulatkan data mentah; pembulatan hanya untuk tabel presentasi.

Output akhir:

```text
experiments/hardware_op3/final_qat/
├── environment.json
├── environment.md
├── latency/
│   ├── run_*.csv
│   ├── runs_summary.csv
│   └── metadata.json
├── resources/
│   ├── diam_run_*.csv
│   ├── berjalan_run_*.csv
│   ├── resources_all.csv
│   ├── resources_summary.csv
│   └── metadata.json
├── tables/
│   ├── table_3_1_latency.csv
│   ├── table_3_1_latency.md
│   ├── table_3_2_resources_raw.csv
│   ├── table_3_2_resources_raw.md
│   ├── table_3_3_resources_summary.csv
│   └── table_3_3_resources_summary.md
└── FINAL_OP3_QAT_REPORT.md
```

---

## 9. Metadata yang wajib disimpan

Setiap eksperimen final robot harus mencatat konfigurasi agar data dapat dipertanggungjawabkan:

| Metadata | Contoh isi |
| --- | --- |
| tanggal/waktu | UTC dan/atau waktu lokal |
| robot | ROBOTIS OP3 |
| CPU | model CPU aktual |
| RAM total | nilai aktual |
| OS | Ubuntu 24.04 |
| ROS distro | Jazzy |
| OpenVINO version | versi aktual robot |
| camera | Logitech C920 |
| camera device | contoh `/dev/video0` |
| camera resolution | 640×480 |
| model | QAT INT8 OpenVINO final |
| model input | 640×640, batch 1 |
| inference device | CPU |
| confidence threshold | nilai aktual |
| IoU threshold | nilai aktual |
| warm-up frames | nilai aktual |
| duration per run | nilai aktual |
| number of runs | nilai aktual |
| resource interval | nilai aktual |
| condition | diam / berjalan |

Tidak perlu menambahkan pemeriksaan file hash.

---

## 10. Kondisi lingkungan eksperimen

Proposal menetapkan pengujian di Ruang Kerja Humanoid, Gedung A FILKOM UB, pada lingkungan yang dapat menyerupai KRSBI-H dengan rumput sintetis dan pencahayaan yang dijaga stabil.

Untuk setiap run final, Codex/operator harus menjaga sebisa mungkin:

- posisi/arah kamera konsisten;
- pencahayaan tidak sengaja diubah antar-run;
- konfigurasi C920 sama;
- aplikasi berat yang tidak berkaitan ditutup;
- model dan ROS launch sama;
- mode daya CPU tidak diubah di tengah eksperimen;
- robot memiliki kondisi baterai yang memadai;
- kondisi diam dan berjalan menggunakan jendela pengukuran dengan durasi yang sama.

Jika kondisi tersebut tidak dapat dibuat identik, tulis perbedaannya di metadata; jangan menyembunyikannya.

Variasi jarak dan pencahayaan sudah menjadi pertimbangan pada penyusunan dataset proposal. Jangan menambahkan eksperimen jarak/pencahayaan baru sebagai variabel utama skripsi kecuali metodologi proposal direvisi.

---

## 11. Urutan eksekusi final di ROBOTIS OP3

```text
STEP 1
Audit robot + kamera + ROS 2 + OpenVINO
        |
        v
STEP 2
Pastikan model QAT final dapat di-load OpenVINO CPU
        |
        v
STEP 3
Jalankan C920 640×480
        |
        v
STEP 4
Jalankan QAT detector ROS 2
        |
        v
STEP 5
Functional check:
ball / gawang / robot dapat menghasilkan output
        |
        v
STEP 6
Warm-up
        |
        v
STEP 7
Ambil data latency + FPS
        |
        v
STEP 8
Ambil resource:
robot DIAM + inference aktif
        |
        v
STEP 9
Ambil resource:
robot BERJALAN + inference aktif
        |
        v
STEP 10
Generate Tabel 3.1, 3.2, 3.3
        |
        v
STEP 11
Buat FINAL_OP3_QAT_REPORT.md
        |
        v
STEP 12
Gunakan tabel tersebut untuk BAB IV
```

---

## 12. Definition of Done untuk Codex

Tahap robot baru dianggap selesai bila semua kondisi berikut terpenuhi:

| Kriteria | Harus terpenuhi |
| --- | --- |
| QAT OpenVINO dapat di-load di CPU onboard | Ya |
| C920 640×480 dapat menyediakan frame | Ya |
| ROS 2 detector menerima frame secara kontinu | Ya |
| Output kelas ball/gawang/robot tersedia | Ya |
| Warm-up dilakukan sebelum pencatatan | Ya |
| Raw latency per frame tersimpan | Ya |
| Jumlah frame dan waktu aktual tersimpan | Ya |
| FPS dihitung sebagai N/T | Ya |
| Raw CPU/RAM kondisi diam tersimpan | Ya |
| Raw CPU/RAM kondisi berjalan tersimpan | Ya |
| Tabel 3.1 dibuat otomatis | Ya |
| Tabel 3.2 dibuat otomatis | Ya |
| Tabel 3.3 dibuat otomatis | Ya |
| Standar deviasi memakai n-1 / ddof=1 | Ya |
| Metadata perangkat dan konfigurasi tersimpan | Ya |
| Tidak ada training/PTQ/calibration baru | Ya |
| Held-out TEST tidak dijalankan ulang | Ya |
| Tidak ada angka live-camera yang diklaim sebagai precision/recall/mAP | Ya |

---

## 13. Instruksi ringkas untuk Codex

Ketika Codex menerima tugas implementasi di robot, tujuan utamanya adalah:

> Implementasikan model final YOLO11n true QAT INT8 OpenVINO ke CPU onboard ROBOTIS OP3 melalui ROS 2 menggunakan Logitech C920 640×480. Jangan melakukan training, tuning, PTQ, calibration, atau evaluasi ulang held-out TEST. Gunakan artefak final `artifacts/openvino/qat_int8/model.xml` dan `model.bin`. Audit terlebih dahulu workspace ROS 2 yang sudah ada dan reuse node/launch yang layak. Implementasikan telemetry per-frame untuk latency detector dan logger CPU/RAM. Setelah warm-up, ambil data latency/FPS dalam beberapa run identik. Kemudian ambil data CPU dan memori secara periodik pada dua kondisi proposal: robot diam dengan inferensi aktif dan robot berjalan dengan inferensi aktif. Simpan semua raw data ke CSV. Bangun otomatis Tabel 3.1, Tabel 3.2, dan Tabel 3.3 sesuai format proposal. Gunakan FPS=N/T dan standar deviasi sampel `ddof=1`. Jangan menggunakan camera confidence sebagai metrik akurasi. Hasil accuracy tetap berasal dari FINAL TEST yang sudah dibekukan. Jangan mengubah interface walking robot tanpa audit dan verifikasi keselamatan.

Dokumen ini menjadi sumber kerja tahap deployment robot. Jika implementasi aktual membutuhkan perubahan teknis, perubahan tersebut boleh dilakukan selama **tidak mengubah variabel penelitian, model final, definisi metrik, atau struktur data yang telah ditetapkan proposal**.
