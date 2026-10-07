# Langkah Berikutnya — Implementasi ke ROBOTIS OP3 dan Pengambilan Data Final

## 0. Status kode saat ini (7 Oktober 2026)

Dokumen ini membedakan implementasi yang tersedia dari pekerjaan final yang masih perlu dilakukan. Pengukuran dilakukan pada workspace `main` commit `01af397` dengan perubahan implementasi lokal; metadata menyimpan kondisi worktree tersebut. Logger final diimplementasikan; 3 run diam dan stability ROS 10 menit selesai tanpa exception. Data historical dipertahankan.

| Komponen | Status aktual | Bukti/lokasi |
| --- | --- | --- |
| Hardware | NUC12WSHi5, Intel Core i5-1240P, Ubuntu 24.04, ROS 2 Jazzy, C920 | `experiments/hardware_op3/system_info.json` dan `.md` |
| QAT OpenVINO CPU | Load, compile, dan inference dummy berhasil; output finite `[1,7,8400]` | `experiments/hardware_op3/openvino_sanity.json` |
| Identitas model | Mengikuti folder yang dikonfirmasi pengguna; tidak ada pemeriksaan SHA256 | `experiments/hardware_op3/model_inventory.json` |
| Visual QAT standalone | Run kamera dengan visualisasi ON tersedia; functional check, bukan pengukuran final | `scripts/run_qat_visual.sh`, `camera_benchmark_qat_int8.json` |
| Package ROS skripsi | Package terpisah dibuild; 3 run diam QAT folder selesai tanpa exception | `ros2_ws/src/yolo11_op3_vision/` |
| Stability | ROS QAT folder 600 detik, 8260 frame, nol exception | `experiments/hardware_op3/stability_qat/20261007_ros_qat/` |
| Logging node ROS | CSV timing dan resource tersedia saat `benchmark_seconds > 0` | `yolo11_op3_vision/detector.py` |
| Topic telemetry final | Telemetry opsional tersedia; OFF untuk run resmi | Parameter `publish_benchmark` |
| Kondisi diam/berjalan | Logger kondisi/multi-run tersedia; berjalan ditunda operator | `run_op3_qat_experiments.py` |
| Tabel 3.1–3.3 | Generator tersedia, SD sampel `ddof=1` | `scripts/deployment/build_op3_thesis_tables.py` |
| Output final | 3 run diam selesai; tabel dan raw CSV tersedia | `experiments/hardware_op3/final_qat/` |

Data FP32/PTQ dan stability FP32 yang sudah ada adalah hasil terdahulu. Jangan menganggapnya sebagai pengujian final QAT diam/berjalan. Data mentah lama tetap dipertahankan.

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

Laporan validasi numerik OpenVINO yang sudah ada tetap menjadi bukti tahap export yang dilaporkan repository. Deployment menggunakan model yang dilatih/diunggah dan dikonfirmasi pengguna. Identitas payload tidak diverifikasi secara independen dan tidak ada pemeriksaan hash. Sanity runtime CPU menunjukkan model dapat dijalankan; sanity ini bukan evaluasi ulang akurasi atau numerical equivalence. Tahap robot mengambil data waktu komputasi dan sumber daya sesuai proposal.

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

Gunakan `scripts/deployment/audit_hardware_op3.py` dan `scripts/deployment/audit_op3_models.py`. Audit saat ini menulis `experiments/hardware_op3/system_info.json`, `system_info.md`, dan `model_inventory.json`; script tersebut belum menyediakan argumen output ke `final_qat`. Audit hanya membaca model dan mencatat lingkungan.

Snapshot final tersedia di environment.json/.md dan environment_<timestamp>.json; command aktual dan kondisi worktree disimpan bersama metadata run:

```text
experiments/hardware_op3/final_qat/environment.json
experiments/hardware_op3/final_qat/environment.md
```

### 3.2 Periksa workspace ROS 2 yang sudah ada

Pengguna meminta package skripsi terpisah dari package robot yang sudah ada. Implementasinya sudah tersedia; lanjutkan package ini dan jangan membuat detector paralel baru:

```text
ros2_ws/src/yolo11_op3_vision/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/yolo11_op3_vision
├── config/params.yaml
├── launch/detector.launch.py
└── yolo11_op3_vision/
    ├── __init__.py
    └── detector.py
```

Package menggunakan `sensor_msgs/msg/Image` dan `vision_msgs/msg/Detection2DArray`. Kamera dijalankan melalui publisher eksternal seperti `usb_cam`; launch detector belum memulai kamera. Audit device/topic kamera sebelum menjalankan publisher.

Build dari root repository:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-select yolo11_op3_vision
source install/setup.bash
```

Source package tersedia di repository skripsi; hasil pengujian final disimpan bersama raw data. Build/install/log tidak ikut disimpan ke Git.

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
| `imgsz` | 640; ini nama parameter aktual, bukan `input_size` |
| `confidence_threshold` | default kode/config 0.25 |
| `iou_threshold` | default kode/config 0.45 |
| `publish_debug_image` | default `false` saat pengukuran |
| `debug_image_topic` | default `/vision/debug_image` |
| `benchmark_seconds` | default 0.0; set positif untuk logging timed run |
| `warmup_iterations` | default 30 inference tensor dummy saat startup |
| `resource_interval_seconds` | default 0.2 |
| `output_dir` | wajib diset bila `benchmark_seconds > 0` |
| `result_stem` | default `ros2_benchmark`; gunakan nama unik agar tidak overwrite run |
| `evidence_image_path` | default kosong; biarkan kosong saat benchmark resmi |

Launch juga menerima `python_prefix` untuk interpreter yang mempunyai OpenVINO dan modul ROS Jazzy. Parameter `publish_benchmark` tersedia, default false. Run final menggunakan logging internal.

`config/params.yaml` menyimpan default package, tetapi `detector.launch.py` saat ini membentuk parameter langsung dari launch arguments. Mengedit YAML saja belum mengubah parameter launch tersebut; gunakan launch arguments atau tambahkan pemuatan YAML pada implementasi berikutnya.

Output saat ini: `/vision/detections` dengan `class_id` seperti `0:ball`, confidence pada hypothesis, dan bounding box melalui center/size. Debug image bersifat opsional. `x1/y1/x2/y2` tersedia pada hasil postprocess internal dan dapat diturunkan dari bounding box message.

Codex tidak boleh mengubah confidence/IoU hanya untuk membuat hasil live terlihat lebih baik. Jika nilainya belum terdokumentasi, ambil dari implementasi evaluasi yang sudah digunakan dan tulis nilainya pada metadata eksperimen.

### 3.4 Topic benchmark

Node kini mendukung telemetry opsional `/vision/benchmark` melalui `std_msgs/String`. Run resmi memakai logging CSV internal dan telemetry OFF agar tidak menggandakan logger. Logger lama masih memerlukan PID eksplisit bila dipakai dengan node skripsi. CSV ditulis streaming, bukan menumpuk frame di RAM saat inference.

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

CSV final mencatat `latency_ms` sejak callback dimulai sampai postprocess selesai, sebelum publishing. Preprocess mencakup konversi ROS image. `callback_end_to_end_ms` tetap terpisah dan mencakup publishing/debug bila aktif. Data historical tidak diganti namanya.

---

## 4. Warm-up dan pengukuran waktu

Proposal mewajibkan tahap warm-up sebelum pengambilan data agar inisialisasi model, alokasi memori, dan kompilasi OpenVINO tidak memengaruhi hasil.

Proposal tidak menentukan jumlah frame warm-up. Oleh karena itu Codex harus membuat nilai ini configurable dan mencatat nilai aktual pada metadata. Implementasi saat ini boleh memakai default:

```text
warmup_frames = 30
```

tetapi angka tersebut harus diperlakukan sebagai **parameter implementasi**, bukan angka yang diklaim berasal dari proposal.

Node menjalankan 30 inference dummy saat startup dan membuang 30 frame kamera sukses (`warmup_camera_frames`). Pengukuran dimulai pada frame berikutnya. Runner membekukan 3 run × 60 detik dan sampling 0.2 detik dalam `protocol.json`, dengan konfigurasi sama untuk diam/berjalan. Robot diam telah dikonfirmasi pengguna; berjalan disiapkan untuk operator nanti.

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

**Kode yang tersedia:** sampler final menulis condition, run_id, sample_no, system CPU, system memory used/available, memory_usage_percent langsung dari psutil, CPU proses, RSS/VMS, CPU per core, dan suhu. CSV disimpan streaming setiap 0.2 detik; MB memakai bytes/1,000,000. Data historical tetap memakai schema lamanya.

Untuk logger final, ambil penggunaan RAM persen langsung dari `psutil.virtual_memory().percent`. Jangan menyamakannya dengan `used / total` karena definisi memori Linux juga memperhitungkan memori yang tersedia/cache. Konversi `memory_used_mb = system_memory_used_bytes / 1_000_000` untuk MB; nilai RSS MiB pada laporan lama menggunakan pembagi `1_048_576` dan harus tetap diberi label MiB. Setiap run harus mencatat kondisi aktual dari operator, bukan sekadar memberi label `berjalan` pada log ketika robot belum berjalan.

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

Statistik historical memakai `ddof=0`. Node/generator final memakai `ddof=1` dari raw CSV. Jangan mengubah summary eksperimen lama agar terlihat sudah sesuai. Jika hanya tersedia satu sampel, standar deviasi sampel harus ditandai tidak tersedia, bukan diisi nol.

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

Codex harus melanjutkan package skripsi dan helper existing. Path script saat ini sudah berada di `scripts/deployment/`; jangan memakai path lama di root `scripts/`. Script final harus mengikuti tabel proposal dan fokus pada QAT yang ditetapkan pengguna.

| Kode tersedia | Fungsi aktual | Batas penggunaan |
| --- | --- | --- |
| `scripts/deployment/audit_hardware_op3.py` | Audit sistem, kamera, runtime | Output fixed pada `experiments/hardware_op3/system_info.*` |
| `scripts/deployment/audit_op3_models.py` | Inventory file dan bukti manifest tanpa hash | Bukti manifest tidak memverifikasi identitas payload |
| `scripts/deployment/benchmark_op3.py` | Synthetic atau kamera OpenVINO, raw timing/resource | Single run, belum schema proposal diam/berjalan |
| `scripts/deployment/benchmark_op3_ros2.py` | Subscriber JSON `/vision/benchmark`, logger resource | Memerlukan telemetry dan penyesuaian PID node skripsi |
| `scripts/deployment/capture_op3_evidence.py` | Functional evidence kamera/resource | Evidence bukan evaluasi akurasi |
| `scripts/run_qat_visual.sh` | OpenCV visual QAT melalui SSH X | Visualisasi ON; bukan benchmark resmi |
| `ros2_ws/src/yolo11_op3_vision/yolo11_op3_vision/detector.py` | Detector ROS, publisher deteksi, CSV timed run | Telemetry opsional, raw CSV streaming, kondisi/run_id; runner multi-run tersedia |

Implementasi final memakai satu runner latency/resource: `scripts/deployment/run_op3_qat_experiments.py`, dengan wrapper `scripts/run_op3_qat_measurements.sh`. Ini memakai node detector existing, bukan membuat tiga pipeline paralel. Runner memulai kamera, mencatat kondisi dan multi-run, membekukan protocol, memantau kegagalan kamera, serta membersihkan hanya process miliknya. Script tidak menggerakkan robot.

### `build_op3_thesis_tables.py`

Status: tersedia; diuji untuk ddof=1, FPS wall-time, exclusion error frame, dan walking yang belum diukur. Tabel historical tetap terpisah.

Tugas:

- hanya membaca raw CSV;
- tidak menjalankan model;
- membuat Tabel 3.1, 3.2, dan 3.3;
- menggunakan `ddof=1` untuk standar deviasi;
- menghasilkan CSV dan Markdown;
- tidak membulatkan data mentah; pembulatan hanya untuk tabel presentasi.

Output final dibuat saat run aktual: raw frame CSV mempunyai run_id/condition/error, resource CSV mempunyai condition, dan JSON per run ada di run_metadata/. Mode final_experiment menggunakan layout final; mode legacy tetap menggunakan result_stem.

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
| git commit | commit dan catatan perubahan lokal saat run |
| model label source | folder/model path yang dikonfirmasi pengguna |
| identity verification | tidak dilakukan; tidak ada pemeriksaan hash |
| timestamp scope | UTC untuk metadata; monotonic/perf_counter untuk durasi |
| visualization | debug image dan evidence rendering OFF pada run resmi |

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

Sanity CPU dan functional ROS C920 berhasil. Node final mempunyai telemetry opsional, warmup kamera, kondisi, dan tabel otomatis. Walking menunggu operator; urutan berikut tetap bisa dipakai untuk functional deployment.

Contoh menjalankan detector ROS **yang sudah ada**, setelah publisher C920 aktif pada `/image_raw`. Jalankan dari root repository dan gunakan runtime OpenVINO yang sudah disiapkan pada robot:

```bash
OP3_REPO_ROOT="$PWD"
source /opt/ros/jazzy/setup.bash
source "$OP3_REPO_ROOT/ros2_ws/install/setup.bash"
ros2 launch yolo11_op3_vision detector.launch.py \
  model_path:="$OP3_REPO_ROOT/artifacts/openvino/qat_int8/model.xml" \
  input_topic:=/image_raw device:=CPU imgsz:=640 \
  confidence_threshold:=0.25 iou_threshold:=0.45 \
  publish_debug_image:=false \
  python_prefix:="$OP3_REPO_ROOT/../.venv-op3-vision/bin/python"
```

Ini functional run tanpa timed logging (`benchmark_seconds=0`). Untuk timed check existing, tambahkan `benchmark_seconds:=60`, `output_dir` yang tersedia, dan `result_stem` unik. Biarkan `evidence_image_path` kosong saat pengukuran. Untuk multi-run dan kondisi gunakan run_op3_qat_experiments.py melalui wrapper pengukuran. Ganti `input_topic` sesuai topic hasil audit; build workspace terpisah terlebih dahulu bila belum dibangun.

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

| Kriteria wajib | Status kode/data saat ini |
| --- | --- |
| QAT OpenVINO dapat di-load di CPU onboard | Selesai: sanity CPU load/compile/inference |
| C920 640×480 dapat menyediakan frame | Selesai pada functional run terdahulu |
| ROS 2 detector menerima frame secara kontinu | Package sudah diuji dengan FP32; ROS QAT final masih perlu diuji |
| Output kelas ball/gawang/robot tersedia | Mapping dan publisher tersedia; functional observation bukan evaluasi accuracy |
| Warm-up dilakukan sebelum pencatatan | 30 inference dummy + 30 frame kamera sukses; metadata aktual tersedia |
| Raw latency per frame sesuai definisi proposal | latency_ms detection-ready tersedia; callback_end_to_end terpisah |
| Jumlah frame dan waktu aktual tersimpan | Runner multi-run tersedia; jumlah frame dan durasi aktual per run |
| FPS dihitung sebagai N/T | Existing effective FPS tersedia; jendela final dimulai setelah warm-up kamera |
| Raw CPU/RAM kondisi diam tersimpan | Sampling 0.2 detik dan condition tersedia; lihat raw final_qat |
| Raw CPU/RAM kondisi berjalan tersimpan | Belum diuji; operator/interface walking perlu disiapkan |
| Tabel 3.1 dibuat otomatis | Belum ada generator |
| Tabel 3.2 dibuat otomatis | Belum ada generator |
| Tabel 3.3 dibuat otomatis | Belum ada generator |
| Standar deviasi memakai n-1 / ddof=1 | Belum pada generator final; statistik existing masih ddof=0 |
| Metadata final perangkat/run/condition tersimpan | Environment snapshot dan run_metadata tersedia |
| Tidak ada training/PTQ/calibration baru | Dipertahankan |
| Held-out TEST tidak dijalankan ulang | Dipertahankan |
| Tidak ada angka live-camera yang diklaim sebagai precision/recall/mAP | Dipertahankan |

---

## 13. Instruksi ringkas untuk Codex

Ketika Codex menerima tugas implementasi di robot, tujuan utamanya adalah:

> Lanjutkan package skripsi terpisah `ros2_ws/src/yolo11_op3_vision` untuk deployment model QAT yang ditetapkan pengguna pada CPU NUC12 ROBOTIS OP3 dengan C920 640×480. Gunakan `artifacts/openvino/qat_int8/model.xml` dan `model.bin` tanpa pemeriksaan hash. Jangan melakukan training, tuning, PTQ, calibration, atau evaluasi ulang held-out TEST. Sanity CPU sudah selesai; lanjutkan ROS QAT dan perbaiki pencatatan detection-ready latency, warm-up kamera, telemetry/logging, multi-run, serta logger CPU/RAM sistem untuk kondisi diam/berjalan. Reuse kode `scripts/deployment/` dan node existing di package skripsi; jangan membuat detector/package paralel. Simpan seluruh raw CSV dan metadata run. Implementasikan generator Tabel 3.1–3.3 dengan FPS=N/T, MB yang eksplisit, dan standar deviasi sampel ddof=1. Pisahkan functional GUI run, hasil lama FP32/PTQ, dan eksperimen final. Accuracy tetap memakai FINAL TEST yang dibekukan. Walking dilakukan lewat interface yang sudah diaudit dan operator; logger tidak memulai gerak robot sendiri.

Dokumen ini menjadi sumber kerja tahap deployment robot. Jika implementasi aktual membutuhkan perubahan teknis, perubahan tersebut boleh dilakukan selama **tidak mengubah variabel penelitian, model final, definisi metrik, atau struktur data yang telah ditetapkan proposal**.

## Perintah pengukuran

```bash
# Diam, setelah kondisi fisik dikonfirmasi
bash scripts/run_op3_qat_measurements.sh --condition diam --operator-confirmed --runs 3 --seconds 60 --camera-device /dev/video0
# Berjalan NANTI; operator menjalankan walking melalui interface existing
bash scripts/run_op3_qat_measurements.sh --condition berjalan --operator-confirmed --runs 3 --seconds 60 --camera-device /dev/video0
python3 scripts/deployment/build_op3_thesis_tables.py --root experiments/hardware_op3/final_qat
```

Runner menolak overwrite CSV/metadata dan protocol berbeda. Stability memakai output terpisah dan `--condition stability --runs 1 --seconds 600`. Parameter baru: `experiment_condition`, `run_id`, `final_experiment`, `warmup_camera_frames`, `publish_benchmark`, `benchmark_topic`, `thermal_limit_c`.

## Hasil aktual tahap ini

Tiga run diam: 2442 frame; latency gabungan 17.9763 ms; FPS efektif gabungan 13.5658. CPU sistem mean 6.4163%, RAM sistem mean 4083.7804 MB, RSS detector mean 257.5252 MB. Raw per-frame dan 885 sampel resource ada di final_qat. Nilai accuracy tidak diambil dari kamera.

Stability ROS 600 detik: 8260 frame, 13.7663 FPS, latency 17.6222 ms, nol exception, RSS growth 434176 bytes sebelum alokasi summary, suhu puncak 59 C. Ini observasi stability, bukan bukti mutlak tidak ada memory leak.

Walking belum dilakukan sesuai instruksi pengguna; script siap. Jangan mengisi hasil berjalan sebelum ada run aktual. Laporan: experiments/hardware_op3/final_qat/FINAL_OP3_QAT_REPORT.md dan experiments/hardware_op3/stability_qat/20261007_ros_qat/STABILITY_REPORT.md.

## Pembanding runtime FP32/PTQ/QAT (7 Oktober 2026)

Atas permintaan pengguna, empat model diukur ulang secara standalone: FP32 native PyTorch tanpa OpenVINO, FP32 OpenVINO, PTQ INT8 OpenVINO, QAT INT8 OpenVINO. Seluruh IR berasal dari artifacts/openvino dan native dari artifacts/checkpoints/fp32/best.pt. Protocol: CPU batch 1, 640×640, warmup 30, synthetic 300 iteration, real C920 60 detik/model, visualization OFF. Native memakai rect=False agar input kamera tetap 640×640.

Hasil lengkap: experiments/hardware_op3/comparison_models/20261007_four_models_run1/MODEL_COMPARISON_REPORT.md. Tabel skripsi comparison_thesis_fp32_qat.csv mengecualikan PTQ. Accuracy native checkpoint tetap dipisahkan dari runtime. ROS QAT/stability dan walking workflow di atas tetap terpisah; tidak ada klaim ROS benchmark empat model.
