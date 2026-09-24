# Decision Log

## Tujuan

Merekam keputusan kronologis, alasan, bukti, dan dampaknya.

## Status

**CURRENT**.

| Phase | Keputusan | Alasan/bukti | Dampak |
|---|---|---|---|
| Setup | YOLO11n dan dataset detection native | config dan class mapping | baseline konsisten |
| Dataset | pipeline segmentation ditinggalkan; satu polygon dikoreksi | audit passed | label valid, provenance file terbatas |
| Server | uv, batch 16, workers 2, cache false | shared-server safety | run lebih konservatif |
| FP32 | 100 epoch AdamW | manifest FP32 | baseline dibekukan |
| 5A | tolak `nncf.quantize`, pilih `create_compressed_model` | docstring/API NNCF | true QAT |
| 5B | integrate DetectionTrainer dan NNCF-aware checkpoint | smoke/reload passed | QAT lifecycle valid |
| 5C | LR 1e-5, no warmup, lrf 1.0 | 5 epoch dari FP32 converged | QAT final terkunci |
| 6 | model dibekukan sebelum TEST; TEST sekali | manifest final test | tidak ada test tuning |
| 7A | tolak strip artifact | divergence ~36,24% | no deploy artifact |
| 7C/7D | acceptance memakai `[0,1]` + 32 VAL | CPU/CUDA representative pass | export QAT fail-closed |
| 7F historis | Uji `QuantizationController.export_model(..., save_format="onnx")` | ONNX/checker lulus, tetapi OpenVINO gagal numerik pada 31/32 VAL | Kandidat ONNX tidak dipromosikan |
| Seleksi QAT | Pilih `qat_lr1e4_r256_e15` dari VAL | Selection manifest passed; TEST/PTQ metrics tidak dipakai memilih | Checkpoint final SHA256 `b9e520...35430d` |
| Export final | Pakai direct OpenVINO untuk QAT terpilih | 32/32 VAL lulus equivalence, graph quantized, gate accepted | QAT IR final valid |
| Benchmark server | Ukur FP32/PTQ/QAT pada CPU host | 30 warmup, 300 iterasi per model | Hasil diagnostik server terpisah dari OP3 |
| Pembersihan | Hapus binary kandidat rejected, simpan JSON diagnostic | Model gagal tidak boleh tertukar dengan final | Repository lebih jelas tanpa kehilangan bukti keputusan |
