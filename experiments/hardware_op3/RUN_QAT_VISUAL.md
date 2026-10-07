# Menjalankan visualisasi QAT di OP3 melalui SSH X

Runner ini membuka kamera dan jendela OpenCV untuk functional check. Overlay menampilkan nama kelas dan confidence. Model memakai `artifacts/openvino/qat_int8/model.xml` sesuai penetapan pengguna. Tidak ada pemeriksaan SHA256. Bukti manifest tetap dibaca, dan file model harus tersedia sebagai XML/BIN yang dapat dimuat. Mode visual bukan benchmark latency resmi karena tampilan X dapat menambah beban.

## Dari komputer pengguna

Hubungkan SSH dengan X forwarding:

```bash
ssh -X <user>@<alamat-OP3>
```

Jika X forwarding tidak bekerja, pastikan X server berjalan di komputer pengguna. Di sesi SSH, `echo "$DISPLAY"` harus menampilkan nilai seperti `localhost:10.0`.

## Di OP3

```bash
cd ~/ros2_jazzy/skripsi-yolo11-op3
bash scripts/run_qat_visual.sh
```

Tekan `q` pada jendela kamera untuk keluar. Jika Python virtual environment berada di lokasi lain, pilih interpreter secara eksplisit:

```bash
OP3_PYTHON=/path/ke/venv/bin/python bash scripts/run_qat_visual.sh
```

Script benchmark menerima parameter kamera yang sama jika diperlukan, misalnya `--camera-device /dev/video0`; bantuan CLI berada di `scripts/deployment/benchmark_op3.py --help`.

Runner memakai `/dev/video0`, sesuai audit C920 sebelumnya. Jika kamera terdaftar pada path lain, pilih path hasil audit dengan `OP3_CAMERA_DEVICE`, contohnya:

```bash
OP3_CAMERA_DEVICE=/dev/video1 bash scripts/run_qat_visual.sh
```

## Status artefak

Pengguna mengonfirmasi bahwa model sudah dilatih dan diunggah sendiri. Label QAT mengikuti direktori yang ditetapkan pengguna; identitas payload tidak diverifikasi secara independen. Payload Git LFS sudah tersedia. NUC12 adalah hardware pengujian aktual dan tidak menghalangi deployment. Catatan runtime tersimpan di `experiments/hardware_op3/openvino_sanity.json` setelah sanity test dilakukan.
