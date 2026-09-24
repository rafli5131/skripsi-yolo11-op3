# Server OpenVINO Benchmark — FP32 vs PTQ INT8 vs QAT INT8

## 1. Tujuan

Benchmark ini adalah diagnostic benchmark pada host server: sanity check runtime OpenVINO CPU, pembanding awal FP32/PTQ/QAT, dan persiapan pengujian di ROBOTIS OP3. Hasil ini **bukan** hasil deployment final OP3. Ketiga model menjalankan input sintetis yang sama; accuracy tidak dihitung ulang di sini.

[Ringkasan hasil dan tautan seluruh raw file](../experiments/server_openvino/README.md) tersedia langsung di direktori eksperimen.

## 2. Model yang Dibandingkan

| Model | Precision | Method | Artifact XML |
| --- | --- | --- | --- |
| FP32 | FP32 | Baseline | `artifacts/openvino/fp32/model.xml` |
| PTQ INT8 | INT8 | Post-Training Quantization | `experiments/ptq/backend_models/ptq_openvino_model/model.xml` |
| QAT INT8 | INT8 | Quantization-Aware Training | `artifacts/openvino/qat_int8/model.xml` |

PTQ memakai existing canonical comparator; tidak ada calibration atau quantization baru selama benchmark. QAT berasal dari true NNCF QAT dan diekspor melalui jalur direct OpenVINO.

| Model | XML SHA256 | BIN SHA256 | XML bytes | BIN bytes | Total bytes | Total MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FP32 | `233bb872e218ab3bf525188b0e105eea9f2a79e60ffecb4b767e63939ba257e7` | `e85f45438b5f7c8858b1a168a03f61b12a73b8159391529d19b84e09ff854026` | 324,944 | 10,431,992 | 10,756,936 | 10.259 |
| PTQ INT8 | `10cb9dfa3539650a0bc46d60ad708f9ff79fc1e2dc79c5ac4a73ff4943326657` | `82246726edcd780685657f673caa2b40719a8c22b67e3702ca3d1f226b8fcf81` | 666,400 | 2,741,080 | 3,407,480 | 3.250 |
| QAT INT8 | `0a97a52f7a72ae9667c6f501ac1e0f44ff04bb5a860ac072e4f39ad406c424b2` | `4a88143019b418edd879932820589c18557dfe9954196fa92025749505bc6352` | 735,797 | 2,744,884 | 3,480,681 | 3.319 |

BIN masing-masing berada di samping XML dengan nama `model.bin`. MiB = bytes / 1,048,576; ini adalah ukuran XML + BIN pada disk, bukan alokasi runtime.

## 3. Provenance

### FP32

Baseline IR berasal dari `artifacts/checkpoints/fp32/best.pt`; [catatan export FP32](09_openvino_export.md) mencatat jalur `torch.jit.trace → openvino.convert_model → openvino.save_model(compress_to_fp16=False)`. Checkpoint FP32 sumber yang materialized memiliki SHA256 `e05b2b21f7faa68f65671f0aaae37ad962aa5e25c6b8b26853fd0b389d420b8c`.

### PTQ INT8

`ptq_validation_gate()` menerima canonical artifact: `ptq_only_provenance=true`, source FP32 SHA256 cocok, class mapping cocok, `QAT checkpoint used=false`, `TEST split loaded=false`, dan hash/ukuran XML serta BIN cocok dengan `experiments/ptq/manifest.json`. Calibration PTQ historis menggunakan data yang tercatat di manifest tersebut; TEST tidak dipakai untuk calibration/tuning PTQ. Tidak ada calibration selama benchmark ini. Audit tooling diperbaiki agar menerima checkpoint FP32 materialized dengan SHA256 yang cocok, selain pointer Git LFS yang cocok; syarat provenance lain tetap sama.

### QAT INT8

`qat_validation_gate()` menerima true NNCF QAT (`create_compressed_model`, 184 quantizer yang terverifikasi), winner dipilih hanya dengan VAL tanpa TEST atau metrik PTQ, dan checkpoint final `artifacts/checkpoints/qat/qat_best.pt` cocok dengan SHA256 `b9e520cbd5d56597014c932701f471505d2243e02763a2245a51fd83fc35430d`. Diagnostic export menerima jalur **direct OpenVINO** dengan numerical equivalence pada 32 input VAL, `PTQ used=false`, `calibration occurred=false`, dan TEST tidak dimuat/digunakan saat export. Diagnostic jalur ONNX yang ditolak tidak digunakan. Gate QAT juga dijalankan oleh skrip saat benchmark QAT.

## 4. Server Environment

| Item | Nilai |
| --- | --- |
| Hostname | `labkc-l40-debian` |
| OS | Debian GNU/Linux 12 (bookworm), x86_64 |
| Kernel | `6.1.0-43-amd64` |
| CPU model | AMD EPYC 7413 24-Core Processor (nama yang dilaporkan host) |
| CPU cores terlihat oleh OS | 80 physical, 80 logical (`psutil.cpu_count`) |
| RAM total | 304,266,973,184 bytes (283.37 GiB) |
| Python | 3.11.2 (environment project melalui `uv`) |
| OpenVINO | `2024.4.0-16579-c3152d32c9c-releases/2024/4` |
| Benchmark script SHA256 | `6c278137e4aaf9c01a8a8603bc171062c3ca300ee0f0cbc61fc051b069da78ed` |
| Git commit saat pengukuran | `57b456032be3cee3b4cc389b36d5184e251dfed2` |
| Waktu mulai FP32 / PTQ / QAT (UTC) | 2026-09-24 05:38:37 / 05:38:51 / 05:39:24 |

Commit pengukuran adalah revisi sebelum laporan dan perbaikan audit ini di-commit. File model sendiri diidentifikasi secara terpisah oleh SHA256 di atas. GPU tidak dipakai untuk inference.

## 5. Benchmark Configuration

Ketiganya menggunakan OpenVINO `CPU`, input `[1,3,640,640]` (640×640), batch 1, 30 warmup, dan 300 timed iterations. Mode adalah `synthetic` dengan frame acak deterministik dari seed 42. Confidence 0.25 dan IoU 0.45 adalah default skrip. Tidak ada pengaturan thread atau performance hint khusus per model. Compile dan warmup dikecualikan dari timing. Proses dijalankan berurutan di server yang sama.

Stem `server_benchmark_*` diarahkan oleh skrip ke `experiments/server_openvino/`, terpisah dari hasil pengujian hardware ROBOTIS OP3 di `experiments/hardware_op3/`.

Environment: `UV_CACHE_DIR="$PWD/.cache/uv"`, `TORCH_HOME="$PWD/.cache/torch"`, `YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"`; setiap perintah berikut dijalankan melalui `env -u PYTHONPATH -u VIRTUAL_ENV uv run python` dengan tiga variabel tersebut diset.

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/benchmark_op3.py --model-kind fp32 --model-xml artifacts/openvino/fp32/model.xml --device CPU --warmup 30 --iterations 300 --result-stem server_benchmark_fp32
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/benchmark_op3.py --model-kind ptq_int8 --model-xml experiments/ptq/backend_models/ptq_openvino_model/model.xml --device CPU --warmup 30 --iterations 300 --result-stem server_benchmark_ptq_int8
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/benchmark_op3.py --model-kind qat_int8 --model-xml artifacts/openvino/qat_int8/model.xml --device CPU --warmup 30 --iterations 300 --result-stem server_benchmark_qat_int8
```

Angka latency berikut adalah `latency_ms.inference_ms` (hanya inference). FPS = `1000 / mean inference latency`; `effective end-to-end FPS` memakai total waktu loop yang juga mencakup preprocess dan postprocess. Semua nilai resource berasal dari sampling selama timed iterations.

## 6. Hasil Benchmark Utama

| Metric | FP32 | PTQ INT8 | QAT INT8 |
| --- | ---: | ---: | ---: |
| Mean latency (ms) | 17.160 | 13.097 | 12.829 |
| Median latency (ms) | 15.517 | 10.696 | 10.907 |
| Minimum latency (ms) | 11.650 | 9.036 | 9.116 |
| Maximum latency (ms) | 131.191 | 122.209 | 147.565 |
| P50 latency (ms) | 15.517 | 10.696 | 10.907 |
| P90 latency (ms) | 20.395 | 18.563 | 14.795 |
| P95 latency (ms) | 22.773 | 25.855 | 23.120 |
| P99 latency (ms) | 44.885 | 41.992 | 26.895 |
| Inference FPS | 58.276 | 76.351 | 77.951 |
| Effective end-to-end FPS | 50.661 | 61.385 | 63.675 |
| Process CPU mean (%) | 3231.921 | 2865.058 | 2974.009 |
| Process CPU p95 / peak (%) | 3515.180 / 3532.700 | 3147.300 / 3261.700 | 3341.330 / 3433.200 |
| System CPU mean (%) | 46.176 | 41.375 | 42.430 |
| RSS mean (MiB) | 254.582 | 233.798 | 238.006 |
| RSS p95 / peak (MiB) | 258.464 / 259.562 | 236.070 / 236.070 | 239.234 / 239.305 |
| VMS mean (MiB) | 26580.606 | 25879.618 | 26451.398 |
| System memory used mean (MiB) | 36433.719 | 36259.794 | 36372.683 |
| Resource samples | 29 | 24 | 23 |
| Model size XML + BIN (MiB) | 10.259 | 3.250 | 3.319 |

`psutil.Process.cpu_percent()` dapat melebihi 100% pada proses multithread: 100% setara dengan satu logical CPU yang sibuk. VMS adalah ruang alamat virtual, bukan RAM fisik yang dipakai proses; RSS lebih tepat untuk perbandingan RAM proses. System memory dan system CPU juga mencakup beban host lain.

## 7. Improvement terhadap FP32

| Metric | PTQ vs FP32 | QAT vs FP32 |
| --- | ---: | ---: |
| Mean latency reduction | +23.67% (−4.062 ms) | +25.24% (−4.331 ms) |
| Inference FPS increase | +31.02% (+18.075 FPS) | +33.76% (+19.675 FPS) |
| Process CPU mean reduction | +11.35% (−366.862 poin %) | +7.98% (−257.912 poin %) |
| Process RSS mean reduction | +8.16% (−20.784 MiB) | +6.51% (−16.576 MiB) |
| Model size reduction | +68.32% (−7.009 MiB) | +67.64% (−6.939 MiB) |

Untuk latency, CPU, RAM, dan ukuran model, tanda positif berarti nilai model INT8 lebih rendah daripada FP32: `(FP32 − INT8) / FP32 × 100%`. Untuk FPS, tanda positif berarti lebih tinggi: `(INT8 − FP32) / FP32 × 100%`. CPU dibandingkan dari process CPU mean; RAM dari process RSS mean.

## 8. PTQ vs QAT

| Metric | PTQ INT8 | QAT INT8 | Better pada run ini |
| --- | ---: | ---: | --- |
| Mean latency (ms) | 13.097 | 12.829 | QAT (0.269 ms lebih rendah; 2.05% vs PTQ) |
| P95 latency (ms) | 25.855 | 23.120 | QAT (2.735 ms lebih rendah; 10.58%) |
| Inference FPS | 76.351 | 77.951 | QAT (1.600 FPS lebih tinggi; 2.10%) |
| Process CPU mean (%) | 2865.058 | 2974.009 | PTQ (108.950 poin % lebih rendah; 3.80%) |
| RSS mean (MiB) | 233.798 | 238.006 | PTQ (4.208 MiB lebih rendah; 1.80%) |
| Model size (MiB) | 3.250 | 3.319 | PTQ (0.070 MiB lebih kecil; 2.15%) |

PTQ juga memiliki median 0.211 ms lebih rendah daripada QAT. QAT memiliki p99 15.097 ms lebih rendah, tetapi maximum latency 25.357 ms lebih tinggi. Perbedaan ini dilaporkan tanpa memilih satu statistik sebagai pengganti statistik lain.

## 9. Analisis

Pada host ini kedua INT8 meningkatkan throughput inference dibanding FP32, sambil mengurangi ukuran artifact sekitar dua pertiga. QAT memiliki mean inference 0.269 ms lebih cepat daripada PTQ pada run ini, tetapi keunggulan kecil tersebut belum cukup untuk menyatakan superioritas umum tanpa pengulangan dan pengukuran di OP3. PTQ memakai process CPU mean dan RSS mean lebih rendah daripada QAT; keduanya lebih rendah daripada FP32. Ukuran PTQ lebih kecil sedikit daripada QAT.

Tail latency tidak bergerak seragam: p95 PTQ 25.855 ms dan QAT 23.120 ms, keduanya di atas FP32 22.773 ms, sementara p99 kedua INT8 justru di bawah FP32. QAT memiliki maximum tertinggi. Variasi beban host, penjadwalan CPU, dan sampling resource yang berbeda jumlahnya dapat memengaruhi satu run berurutan ini. Tidak ada klaim signifikansi statistik atau kestabilan deployment dari hasil tersebut.

Accuracy PTQ/QAT tidak dihitung ulang dalam benchmark server ini. Hasil accuracy terdahulu, bila dibahas, harus mengacu pada evaluasi beserta split yang sah secara terpisah; angka TEST accuracy tidak dicampur dengan angka latency server ini.

## 10. Validitas dan Batasan

> Hasil ini merupakan benchmark diagnostik pada server dan tidak merepresentasikan kinerja deployment akhir pada ROBOTIS OP3 karena terdapat perbedaan arsitektur CPU, konfigurasi hardware, sistem operasi, beban sistem, serta lingkungan runtime. Benchmark final latency, FPS, CPU, dan RAM tetap harus dilakukan langsung pada ROBOTIS OP3.

Server memiliki hardware berbeda dari OP3. GPU tidak digunakan untuk inference; semua pengukuran memakai OpenVINO CPU. Input sintetis tidak mencerminkan distribusi kamera nyata dan run ini tidak mengukur accuracy. FINAL TEST tidak dijalankan ulang. Tidak ada training, QAT tuning, PTQ baru, quantization, atau calibration baru. Checkpoint, artifact model, dan dataset split tidak diubah.

Raw JSON, CSV timing, dan CSV resource masing-masing tersedia di `experiments/server_openvino/` dengan stem `server_benchmark_fp32`, `server_benchmark_ptq_int8`, dan `server_benchmark_qat_int8`; CSV resource memakai prefiks `resources_`.

## 11. Kesimpulan

Pada host server, PTQ INT8 menghasilkan mean latency 13.097 ms dan inference FPS 76.351; QAT INT8 menghasilkan 12.829 ms dan 77.951 FPS. Dibanding FP32 (17.160 ms; 58.276 FPS), keduanya lebih cepat dan ukuran modelnya lebih kecil. Antara PTQ dan QAT, QAT unggul tipis pada mean latency, p95, p99, dan FPS run ini; PTQ unggul pada CPU mean, RSS mean, ukuran model, dan median latency. Hasil ini hanya berlaku sebagai diagnosis server sampai benchmark OP3 dilakukan.

## 12. Next Step

Deploy FP32, PTQ INT8, dan QAT INT8 ke ROBOTIS OP3 → benchmark synthetic CPU OP3 → benchmark kamera nyata → latency → FPS → CPU → RAM → stability benchmark → final comparison.
