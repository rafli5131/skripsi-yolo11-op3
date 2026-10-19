# Uji perbaikan pengiriman gambar ROS — 19 Oktober 2026

Konfigurasi yang dipilih untuk perbandingan FP32/QAT berikutnya: **RELIABLE, KEEP_LAST, depth 1**. Opsi sudah diimplementasikan pada detector dan runner di branch `fix/ros-camera-qos-20261019`, kemudian diverifikasi memakai script deployment asli. Saat pengukuran dan laporan disusun, perubahan belum di-commit/push; metadata tetap merekam kondisi tersebut.

## Protokol dan status data

- Uji diagnostik singkat, 15 detik per profil; bukan pengganti benchmark final 3 × 60 detik.
- Setiap profil memakai 30 warmup inference dan 30 warmup frame kamera, kamera C920 640×480 YUYV dengan setelan 30 FPS, input 640×640, CPU, confidence 0,25 dan IoU 0,45.
- ROS_DOMAIN_ID=73 dan discovery LOCALHOST. Tidak ditemukan proses manager/gateway/fall detector/behavior pada pemeriksaan sebelum mulai; kamera sedang bebas.
- Profil diuji berurutan, satu run per kandidat: FP32 BEST_EFFORT/5, QAT BEST_EFFORT/5, QAT RELIABLE/5, FP32 RELIABLE/5. Setelah ditemukan delay FP32, diuji FP32 dan QAT RELIABLE/1. Terakhir, keduanya diverifikasi dengan script deployment asli yang sudah mendukung opsi QoS.
- Tidak ada perintah gerak, startup manager, perubahan model/checkpoint, training atau evaluasi TEST. Kondisi diam mengikuti instruksi operator; tidak ada verifikasi gerak fisik independen.
- Satu peluncuran awal gagal sebelum kamera dimulai karena objek QoS rclpy tidak mendukung deepcopy. Folder gagal `20261019T101007Z_ros_qos_ab_31809` dipertahankan dan tidak dipakai sebagai hasil.
- Seluruh delapan profil yang menghasilkan pengukuran selesai, nol exception, dan tanpa thermal abort. Checksum XML/BIN sesuai dengan manifest awal.

## Hasil diagnostik kandidat

Umur frame dihitung dari timestamp header kamera. Kolom setelah publish berarti sampai setelah pemanggilan publish pesan deteksi, bukan sampai pesan diterima node konsumen. Field lama `camera_delivery_ms` digunakan dengan arti umur frame setelah publish; tidak disebut sebagai waktu transport murni.

| Model | QoS | Depth | FPS | Pipeline mean / P95 (ms) | Umur frame saat callback mean / P95 (ms) | Umur frame setelah publish mean / P95 (ms) |
|---|---|---:|---:|---:|---:|---:|
| fp32 | best_effort | 5 | 25.39 | 33.63 / 42.34 | 55.05 / 87.41 | 88.90 / 120.19 |
| qat_int8 | best_effort | 5 | 13.52 | 18.43 / 19.76 | 33.21 / 35.00 | 51.92 / 54.94 |
| qat_int8 | reliable | 5 | 29.97 | 13.77 / 14.32 | 37.34 / 44.30 | 51.32 / 58.60 |
| fp32 | reliable | 5 | 29.49 | 32.26 / 34.74 | 114.51 / 186.77 | 146.99 / 219.58 |
| fp32 | reliable | 1 | 29.44 | 32.16 / 34.02 | 47.53 / 62.15 | 79.91 / 94.84 |
| qat_int8 | reliable | 1 | 29.95 | 13.68 / 14.25 | 42.71 / 44.36 | 56.60 / 58.70 |

RELIABLE/5 memperbaiki FPS tetapi menambah umur frame FP32: P95 setelah publish 219,58 ms, dibanding BEST_EFFORT/5 120,19 ms. RELIABLE/1 menurunkannya ke 94,84 ms dalam profil kandidat dengan FPS 29,44. Pada QAT, RELIABLE/1 mencapai 29,95 FPS; P95 umur frame setelah publish 58,70 ms dibanding baseline 54,94 ms. Selisih sekitar 4 ms pada QAT dicatat sebagai tradeoff, bukan dihilangkan dari pelaporan.

## Verifikasi implementasi dengan runner asli

Profil berikut memakai opsi CLI baru, tanpa override atau monkeypatch QoS. Protokol dan metadata memuat nilai QoS yang sama. CSV mentah cocok dengan jumlah frame JSON.

| Model | FPS | Pipeline mean / P95 (ms) | Umur frame setelah publish mean / P95 (ms) | Exception |
|---|---:|---:|---:|---:|
| fp32 | 29.40 | 32.16 / 33.82 | 82.58 / 94.60 | 0 |
| qat_int8 | 29.93 | 13.71 / 14.41 | 56.43 / 58.74 | 0 |

Hasil mendukung pemilihan RELIABLE/1 untuk uji final yang sebanding. Keduanya mendekati laju kamera 30 FPS; peningkatan kemampuan inference QAT tidak otomatis membuat FPS kamera melampaui laju input. Namun ini tetap diagnosis 15 detik dan belum membuktikan stability jangka panjang atau signifikansi statistik.

## Perubahan kode minimal

- `detector.py`: parameter `image_qos_reliability` dan `image_qos_depth`, validasi, QoS subscription, log startup, serta metadata konfigurasi.
- `run_op3_qat_experiments.py`: opsi CLI baru, forwarding ke node, dan nilai QoS pada protocol.json.
- `run_op3.sh`: opsi CLI baru diteruskan pada action ros; dapat dilihat dengan dry-run.
- Default BEST_EFFORT/5 dipertahankan agar pemilihan protokol baru eksplisit. Pakai dua opsi RELIABLE/1 pada kedua model. Metadata lama tidak diedit; direktori dengan protokol lama tidak boleh digunakan untuk protokol baru.
- Branch dibuat dari HEAD yang sama. Hook post-checkout melaporkan git-lfs tidak tersedia pada PATH, tetapi branch aktif sudah diverifikasi dan model XML/BIN tetap materialized dengan checksum sama. Tidak ada hook atau lingkungan Git LFS yang diubah.
- Pemeriksaan: sintaks Python, bash -n, git diff --check, kesetaraan QoS default terhadap sensor profile lama, dry-run forwarding kedua model, dan profil kamera aktual melalui runner asli.

## Perintah benchmark final yang disiapkan

Belum dieksekusi sebagai bagian diagnosis ini. Jalankan ketika operator siap melakukan perbandingan final dengan protokol baru. Tiap perintah membuat direktori timestamp baru dan meminta konfirmasi UKUR. Kedua model harus memakai kondisi diam serta opsi yang sama.

```bash
ROS_DOMAIN_ID=73 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
bash scripts/run_op3.sh --model fp32 --action ros --condition diam \
  --camera-device /dev/video0 --image-qos-reliability reliable --image-qos-depth 1

ROS_DOMAIN_ID=73 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
bash scripts/run_op3.sh --model qat --action ros --condition diam \
  --camera-device /dev/video0 --image-qos-reliability reliable --image-qos-depth 1
```

Untuk memeriksa perintah tanpa membuka kamera, tambahkan `--dry-run`. Ctrl-C menghentikan benchmark kamera/detector melalui runner; script benchmark tidak mengirim stop ke aktuator. Domain ROS terpisah harus tetap dipakai untuk pengujian vision-only ini.

## Berkas dan akhir sesi

- `qos_ab_results.json`: empat profil awal.
- `fp32_reliable_depth1/`, `qat_int8_reliable_depth1/`: profil antrean satu frame.
- `implementation_check_20261019T101759Z/`: verifikasi script asli, source SHA-256, protokol, metadata, log dan CSV.
- `decision_summary.json`: keputusan, seluruh ringkasan, status kamera dan checksum.
- `implementation.patch`: salinan diff perubahan source untuk ditinjau.
- Kamera bebas setelah pengukuran: True. Proses diagnostik tersisa: 0.
- Hasil final lama dan seluruh data penelitian lama dipertahankan. Jangan mencampur angka protokol BEST_EFFORT/5, diagnosis singkat, dan pengukuran final RELIABLE/1 yang akan datang.
