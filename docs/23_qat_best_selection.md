# QAT Best-Model Selection

> Status terkini: selection manifest berstatus passed; `qat_lr1e4_r256_e15` dipromosikan ke checkpoint final SHA256 `b9e520cbd5d56597014c932701f471505d2243e02763a2245a51fd83fc35430d`. FINAL TEST PyTorch dan export direct OpenVINO berikutnya sudah dilakukan. Bagian perintah di bawah menjelaskan workflow historis, bukan pekerjaan yang perlu diulang untuk memakai artifact final.

## Tujuan

Eksperimen utama penelitian diarahkan ke **true Quantization-Aware Training (QAT)**. PTQ tetap dipertahankan hanya sebagai eksperimen pembanding dan provenance lama; hasil PTQ tidak digunakan sebagai hasil QAT.

Kondisi QAT sebelumnya memakai fine-tuning 5 epoch dan range initialization 16 sampel. Hasil terbaik VAL pada run tersebut berada di sekitar mAP50-95 0,651. Karena itu, pipeline baru melakukan pencarian hyperparameter QAT menggunakan TRAIN + VAL agar checkpoint QAT yang dipilih merupakan hasil QAT terbaik yang benar-benar terukur.

## Prinsip evaluasi

1. Semua kandidat dimulai dari checkpoint FP32 yang sama: `artifacts/checkpoints/fp32/best.pt`.
2. Semua kandidat memakai jalur true-QAT yang sudah diverifikasi: `nncf.torch.create_compressed_model()` dengan fake quantizer aktif.
3. `nncf.quantize()` tidak dipakai pada proses ini karena API tersebut adalah jalur PTQ.
4. Pemilihan model dilakukan hanya dari validation set.
5. Metrik utama pemilihan adalah mAP50-95. Tie breaker: mAP50, precision, lalu recall.
6. TEST split tidak boleh dipakai untuk memilih hyperparameter atau kandidat.
7. Setelah kandidat terbaik dipromosikan ke `artifacts/checkpoints/qat`, TEST dievaluasi satu kali untuk pelaporan akhir.
8. Hasil PTQ tidak disalin, diubah label, atau dipakai sebagai metrik QAT.

## Search space

Konfigurasi kandidat berada di:

```text
configs/qat_best_search.yaml
```

Search awal mengeksplorasi kombinasi berikut:

- 10–20 epoch QAT fine-tuning;
- learning rate `1e-5` sampai `1e-4`;
- LR final ratio `0.10` atau `0.20`;
- range initialization 128, 256, dan 512 sampel TRAIN;
- batch 16, imgsz 640, AdamW, seed 42, AMP disabled.

Search space dapat diperluas jika seluruh kandidat masih belum memulihkan akurasi dengan baik. Perubahan search space harus tetap menggunakan validation set, bukan TEST.

## Menjalankan tuning

Dari root repository pada host CUDA:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"

nvidia-smi
export CUDA_VISIBLE_DEVICES=1

env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/verify_torch_cuda.py
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/tune_qat_best.py --dry-run
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/tune_qat_best.py
```

Jika kandidat terhenti di tengah jalan dan `qat_last.pt` sudah tersedia, script otomatis mencoba resume kandidat tersebut. Gunakan `--force` hanya bila kandidat memang ingin dilatih ulang dari awal.

## Output

Run kandidat:

```text
experiments/qat/qat_best_search/runs/<candidate>/
```

Artefak kandidat:

```text
artifacts/checkpoints/qat_search/<candidate>/
```

Ringkasan seleksi:

```text
experiments/qat/qat_best_search/selection_manifest.json
```

Kandidat pemenang dipromosikan ke lokasi QAT canonical:

```text
artifacts/checkpoints/qat/
```

Promosi juga menulis ulang `SHA256SUMS.txt` sehingga hash checkpoint dan manifest canonical sesuai dengan artefak yang benar-benar dipilih.

## Evaluasi akhir

Setelah tuning dan promosi selesai, jalankan TEST sekali:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/evaluate_final_test.py
```

Hasil TEST baru inilah yang dipakai untuk membandingkan FP32 vs QAT pada laporan akhir. Jangan memilih ulang hyperparameter berdasarkan hasil TEST.

## Catatan OpenVINO

Pemilihan checkpoint QAT terbaik tidak dengan sendirinya menyelesaikan masalah ekuivalensi QAT PyTorch -> OpenVINO. Validasi numerik export tetap harus dilakukan terpisah sebelum mengklaim deployment INT8 QAT di ROBOTIS OP3. Checkpoint QAT terbaik dipilih lebih dahulu berdasarkan akurasi QAT native; kemudian jalur export harus lolos pemeriksaan ekuivalensi dan benchmark perangkat.
