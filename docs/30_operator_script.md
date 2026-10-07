# Satu script operator OP3

Dari root repository:

```bash
bash scripts/run_op3.sh
```

Pilih model: **1 FP32 OpenVINO**, **2 QAT INT8 OpenVINO**, **3 PTQ INT8 OpenVINO**, atau **4 FP32 PyTorch tanpa OpenVINO**. IR memakai `artifacts/openvino/{fp32,qat_int8,ptq_int8}/model.xml`; native memakai `artifacts/checkpoints/fp32/best.pt`. Memilih model tidak mengaktifkan gerak.

Menu operasi:

| Pilihan | Operasi |
| --- | --- |
| 1 | Visual kamera 120 detik; membutuhkan SSH `-X`/DISPLAY; `q` atau Esc untuk selesai |
| 2 | Benchmark CPU synthetic, warmup 30 + 300 iteration |
| 3 | Benchmark kamera standalone 60 detik; pilih kondisi aktual diam/berjalan |
| 4 | Benchmark ROS OpenVINO, 3 × 60 detik; pilih kondisi aktual diam/berjalan |
| 5 | Jalankan controller hardware tanpa auto init pose; konfirmasi `MANAGER` |
| 6 | Init pose; konfirmasi `INIT`, topang robot saat bergerak |
| 7 | Aktifkan walking dengan parameter existing; konfirmasi `JALAN` setelah posisi stabil |
| 8 | Stop walking; tidak memutus daya motor |
| 9 | Ganti model |
| 0 | Keluar; kirim stop untuk walking yang dimulai dari menu ini |

Untuk pengukuran berjalan: pilih model → 5 (jika manager belum aktif) → 6 → tunggu robot stabil → 7 → 3 atau 4 → pilih `berjalan` → konfirmasi `UKUR`. Setelah selesai pilih 8. Script tidak mengganti parameter gait dan menolak start/init langsung jika motion gateway existing aktif, agar tidak ada dua pemilik kontrol gerak.

FP32 tanpa OpenVINO memakai menu 1/2/3. Menu 4 menggunakan node OpenVINO; script tidak menyebut pengujian native PyTorch sebagai ROS benchmark. Benchmark tidak otomatis menginisialisasi atau memulai walking.

Hasil baru per model/action/condition disimpan di `experiments/hardware_op3/operator_runs/<timestamp>_<model>_<action>_<condition>_<pid>/`. Raw data final sebelumnya tetap dipertahankan. Visualisasi adalah functional check dan ditandai bukan benchmark resmi. Metadata standalone mencatat kondisi yang dikonfirmasi operator; ROS raw CSV mempunyai condition/run_id.

Controller yang dijalankan menu 5 berjalan sebagai process terpisah, dengan PID/log di `experiments/hardware_op3/operator_sessions/`. Keluar dari menu menghentikan walking yang dimulai lewat menu, tetapi tidak mematikan manager atau motor. Selama inference/benchmark berjalan di foreground, gunakan terminal kedua untuk stop segera:

```bash
bash scripts/run_op3.sh --action stop
```

Shortcut visual:

```bash
bash scripts/run_op3.sh --model qat --action visual
bash scripts/run_op3.sh --model non_openvino --action visual
```

Shortcut benchmark berjalan, setelah robot benar-benar berjalan:

```bash
bash scripts/run_op3.sh --model ptq --action ros --condition berjalan
bash scripts/run_op3.sh --model non_openvino --action camera --condition berjalan
```

Nama model CLI: `fp32`, `qat`, `ptq`, `non_openvino`. Set kamera audited dengan `--camera-device /dev/video0` atau `OP3_CAMERA_DEVICE`. Runtime bisa diatur melalui `OP3_PYTHON` dan `OP3_TORCH_PYTHON`; lama visual melalui `OP3_VISUAL_SECONDS` (default 120). Jika memakai sparse checkout, sertakan `ros2_ws` melalui `git sparse-checkout add ros2_ws`.

Preview perintah tanpa menjalankan inference atau motor:

```bash
bash scripts/run_op3.sh --model qat --action start --dry-run
bash scripts/run_op3.sh --model fp32 --action ros --condition berjalan --dry-run
```

Validasi implementation: route empat model dan pergantian menu diuji dengan dry-run; native `--show` diuji menggunakan kamera C920 dan Qt offscreen selama 3 detik. Record ada di `experiments/hardware_op3/operator_ui_check/native_visual_offscreen/`. Pengujian offscreen tidak membuktikan koneksi SSH X pengguna. Tidak ada command manager/init/start/stop aktual yang dieksekusi selama pengembangan script.
