# Mengapa FPS FP32 lebih tinggi daripada QAT pada sesi berjalan?

Analisis read-only setelah kedua sesi selesai. Tidak ada eksperimen QoS baru, perubahan konfigurasi, replay inference, atau perubahan model.

## Temuan dari raw final berjalan

| Metrik | FP32 | QAT INT8 |
|---|---:|---:|
| Frame berhasil / detik aktual | 3655 / 180,0429 | 1807 / 180,0622 |
| Effective FPS | 20,3007 | 10,0354 |
| Mean inference (ms) | 34,1225 | 12,9642 |
| Mean pipeline (ms) | 36,6377 | 16,6935 |
| Mean jarak header frame yang diterima (ms) | 49,2730 | 99,5814 |
| Median jarak header frame (ms) | 40,0945 | 83,9255 |
| P95 jarak header frame (ms) | 84,1044 | 251,3154 |
| Jumlah durasi callback / durasi ukur (%) | 74,8321 | 16,9565 |
| QoS subscription | BEST_EFFORT / depth 5 | BEST_EFFORT / depth 5 |
| Exception termasuk warmup | 0 | 0 |

QAT menghabiskan sekitar 16,96% durasi ukur di interval callback yang diukur, sedangkan FP32 sekitar 74,83%. Sisa durasi bukan otomatis waktu tunggu kamera atau idle CPU: dapat mencakup scheduling, penerimaan/serialisasi ROS, timer, dan pencatatan di luar interval callback. Tidak ada penghitung drop atau rate buffer driver selama run final. Jarak header lebih besar menunjukkan frame yang diamati detector lebih jarang, bukan menghitung packet loss DDS secara langsung.

## Pemeriksaan kode dan protokol

- Runner dan log kamera sama-sama mencatat C920 640×480 YUYV, setelan 30 FPS. Setelan bukan bukti laju capture aktual sepanjang run; exposure, adegan dan scheduling belum dipisahkan.
- Detector memakai jalur callback yang sama, satu synchronous infer per gambar yang diterima. Tidak ditemukan skip setiap-N-frame, sleep, rate limiter berdasarkan model, atau threshold frame berbeda antara FP32 dan QAT pada jalur tersebut.
- Kedua protocol.json dan setup metadata eksplisit memakai BEST_EFFORT depth 5. Perbedaan FPS bukan karena satu sesi diam-diam memakai RELIABLE. Default BEST_EFFORT tetap dipakai untuk menjaga protokol utama sesuai instruksi, meskipun opsi RELIABLE sudah tersedia di main.
- Kode publisher usb_cam memakai rclcpp::QoS{100}; default reliability ROS 2 adalah RELIABLE. Publisher RELIABLE dan subscriber BEST_EFFORT kompatibel. Ini bukan mismatch QoS yang sepenuhnya memutus koneksi, tetapi subscriber BEST_EFFORT tidak menjamin seluruh sampel diterima. [Dokumentasi resmi ROS 2 Jazzy](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst).
- Compile OpenVINO memakai core.compile_model(model, "CPU") tanpa menetapkan jumlah thread atau pinning. Pembacaan properti pada diagnosis lama menunjukkan konfigurasi otomatis berbeda menurut model (FP32 12 thread/pinning false, QAT 4 thread/pinning true). Itu bukti historis dan bukan metadata runtime final; belum membuktikan penyebab.

## Bukti diagnostik lama — bukan data utama skripsi

| Model | Profil diagnostik | Frame | Durasi (s) | FPS |
|---|---|---:|---:|---:|
| fp32 | best_effort/depth5 | 381 | 15,0045 | 25,3924 |
| qat_int8 | best_effort/depth5 | 203 | 15,0157 | 13,5192 |
| fp32 | reliable/depth1 | 442 | 15,0347 | 29,3988 |
| qat_int8 | reliable/depth1 | 449 | 15,0037 | 29,9259 |

Angka BEST_EFFORT dihitung dari frame/durasi JSON lama; RELIABLE/1 diverifikasi lagi dari CSV dan metadata runner asli yang telah tersimpan. Profil ini sekitar 15 detik, robot diam, dilakukan berurutan, dan tidak dicampur dengan data berjalan 3 × 60 detik. [Laporan diagnostik QoS lama](../../20261019T101101Z_ros_qos_ab_31903/QOS_FIX_REPORT.md).

Diagnosis capture lama juga mencatat 299 buffer DQBUF sukses dan 165 frame detector dalam jendela sekitar 9,98 detik. Bukti ini mendukung masalah jalur setelah capture pada diagnosis tersebut; tidak membuktikan jumlah kehilangan frame dalam sesi final. [Salinan analisis historis](historical_diagnostics/fps_diagnosis_analysis_20261019T100731Z.json).

## Kesimpulan dan batas kepastian

**Terukur:** QAT memiliki latency inference lebih rendah, tetapi detector menerima/memproses frame lebih jarang pada konfigurasi ROS BEST_EFFORT/5. Karena effective FPS dihitung dari jumlah frame sukses per durasi nyata, FPS QAT menjadi lebih rendah. Ini bukan bukti bahwa komputasi QAT lebih lambat.

**Paling mungkin:** interaksi jalur pengiriman/penerimaan gambar ROS (QoS/DDS/antrean/scheduling) dengan beban runtime model. Bukti terkuat adalah perubahan FPS pada diagnostik QoS lama dan gap besar sebelum callback di raw final. Namun QoS yang sama saja belum menjelaskan secara pasti mengapa dua model mengalami laju penerimaan berbeda.

**Belum terbukti:** komponen DDS tertentu, packet loss aktual, thread pinning tertentu, ataupun exposure sebagai penyebab perbedaan final. Tidak tepat menyatakan QoS sebagai satu-satunya penyebab atau mengklaim kamera final pasti mengirim 30 FPS.

Hasil final BEST_EFFORT/5 dipertahankan. Jangan mengganti FPS dengan 1000/mean inference atau memasukkan angka diagnostik RELIABLE sebagai hasil final berjalan. Tidak diperlukan perubahan kode/model untuk melaporkan keterbatasan ini.

Sumber hitung tambahan: [interval frame JSON](frame_gap_analysis.json), [rekap QoS historis JSON](historical_qos_evidence.json), [checksum sumber diagnosis](diagnosis_source_sha256.json).
