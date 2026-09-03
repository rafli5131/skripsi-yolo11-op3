# Eksperimen Pembanding PTQ INT8

## Tujuan

Dokumen ini mendefinisikan eksperimen pembanding **Post-Training
Quantization (PTQ)** terhadap baseline FP32 dan model true-QAT. PTQ adalah
jalur terpisah dan tidak menggantikan QAT.

## Status

**DONE (VAL diagnostic)** — kalibrasi PTQ dan evaluasi perbandingan telah
berhasil. Percobaan pertama berhenti pada loader karena Ultralytics
memerlukan direktori bernama `*_openvino_model`, bukan path `model.xml`
langsung; skrip kemudian diperbaiki dengan staging lokal tanpa mengubah IR
sumber.

## Metode

Jalur PTQ yang digunakan:

```text
artifacts/checkpoints/fp32/best.pt
    -> Ultralytics torch2openvino(quantize=8)
    -> NNCF nncf.quantize (PTQ calibration)
    -> OpenVINO INT8 IR
```

Kalibrasi memakai subset deterministik dari **TRAIN** saja. Evaluasi
diagnostik memakai **VAL** saja. TEST tidak dibaca dan tidak digunakan untuk
kalibrasi, pemilihan konfigurasi, atau tuning.

API `nncf.quantize()` sengaja hanya muncul pada eksperimen PTQ ini. API itu
tetap dilarang pada jalur true-QAT yang menggunakan
`nncf.torch.create_compressed_model()`.

## Skrip dan keluaran

Skrip utama adalah [export_ptq_openvino.py](../scripts/export_ptq_openvino.py).
Default-nya mengikuti rekomendasi exporter Ultralytics terpasang dengan 300
gambar TRAIN untuk kalibrasi dan seluruh VAL untuk metrik.

Keluaran yang diharapkan:

| Artefak | Tujuan |
|---|---|
| `artifacts/openvino/ptq_int8/model.xml` dan `.bin` | PTQ OpenVINO IR |
| `experiments/ptq/ptq_val_metrics.json` | Metrik PTQ pada VAL |
| `experiments/ptq/fp32_openvino_val_metrics.json` | Metrik baseline FP32 OpenVINO pada VAL |
| `experiments/ptq/comparison.json/.csv/.md` | Tabel FP32/PTQ/QAT diagnostik |
| `experiments/ptq/manifest.json` | Konfigurasi, hash, graph, dan status |

## Hasil keseluruhan (VAL)

| Metrik | FP32 OpenVINO | PTQ INT8 | Delta PTQ terhadap FP32 |
|---|---:|---:|---:|
| Precision | 0,936223 | 0,934116 | −0,002107 (−0,225%) |
| Recall | 0,957564 | 0,950208 | −0,007356 (−0,768%) |
| mAP50 | 0,972973 | 0,972145 | −0,000827 (−0,085%) |
| mAP50-95 | 0,771344 | 0,767179 | −0,004166 (−0,540%) |
| F1 | 0,946773 | 0,942093 | −0,004680 (−0,494%) |
| Latency host (ms/image) | 21,965 | 13,231 | −8,734 (−39,763%) |
| FPS host diagnostik | 45,527 | 75,580 | +30,053 (+66,011%) |
| Ukuran IR | 10.756.936 B | 3.407.480 B | −68,323% |

Kolom QAT INT8 pada tabel perbandingan tetap `—`. Satu-satunya IR QAT yang
tersedia berstatus **REJECTED** karena gagal uji equivalence numerik; nilainya
tidak boleh diperlakukan sebagai hasil deployment QAT. Metrik QAT yang sah
tersedia pada [evaluasi TEST PyTorch](08_final_test_evaluation.md), bukan pada
IR OpenVINO yang ditolak.

## Metrik

Perbandingan menggunakan preprocessing dan evaluator yang sama untuk:

- Precision, Recall, F1;
- mAP50 dan mAP50-95;
- ukuran XML/BIN dan struktur graph quantization;
- timing CPU host sebagai diagnostik, bukan benchmark final OP3.

Benchmark latency/FPS/CPU/RAM final harus dilakukan terpisah pada perangkat
deployment ROBOTIS OP3 dengan kondisi yang sama untuk FP32 dan PTQ.

Evaluasi ulang tanpa membuat PTQ IR dari awal:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  UV_CACHE_DIR="$PWD/.cache/uv" \
  uv run python scripts/export_ptq_openvino.py --evaluate-existing
```

Mode ini hanya membaca IR PTQ yang sudah ada dan menjalankan evaluasi VAL.
Kolom QAT INT8 akan diberi status `REJECTED / diagnostic only` dan bernilai
`—`, karena IR QAT yang tersedia sebelumnya gagal uji equivalence numerik.
Nilai tersebut tidak boleh dianggap sebagai hasil deployment QAT yang valid;
metrik QAT yang sah tetap berasal dari evaluasi QAT PyTorch pada
`experiments/final_test/`.

Jika ingin menampilkan metrik deteksi dari IR QAT yang ditolak (tanpa
mengubah status penolakan), tambahkan opsi eksplisit berikut:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  UV_CACHE_DIR="$PWD/.cache/uv" \
  uv run python scripts/export_ptq_openvino.py \
  --evaluate-existing --evaluate-rejected-qat
```

Angka QAT dari mode ini adalah **diagnostik VAL saja**, bukan bukti bahwa
artefak tersebut ekuivalen dengan model QAT PyTorch atau aman untuk deploy.

## Batasan

PTQ adalah comparator metodologis. Hasil PTQ tidak boleh dipakai untuk
menyembunyikan atau menggantikan status **UNRESOLVED** pada export QAT
OpenVINO. Timing di atas adalah diagnostik host dan bukan benchmark akhir
ROBOTIS OP3. CPU utilization dan RAM belum diukur.

## Artefak terkait

- [Desain QAT](04_qat_design.md)
- [Ringkasan hasil](17_research_results_summary.md)
- [Langkah berikutnya](18_next_steps.md)
